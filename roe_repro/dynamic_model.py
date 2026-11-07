"""Unchanged release generation/scoring with restored official hard routing."""
import json
from pathlib import Path
import time

import torch
from lmms_eval.models.llava import Llava
from lmms_eval import utils
from . import routing_backend


class DynamicLlava(Llava):
    def __init__(self, roe_stats_dir, **kwargs):
        super().__init__(**kwargs)
        assert self.batch_size==1 and self.world_size==1 and self.use_cache
        self.stats_dir=Path(roe_stats_dir)
        self.stats_dir.mkdir(parents=True,exist_ok=True)
        assert not (self.stats_dir/'routes.jsonl').exists(),'Refusing to mix route traces'
        self.layers=list(self.model.model.layers)
        assert len(self.layers)==32
        routing_backend.install(self.model)
        self.stats={'samples':0,'skipped':0,'total':0,'full':0,
            'per_layer_skip':[0]*32,'router_calls':0,'attention_calls':0,
            'adapter_calls':0,'finite_logits_checks':0,'model_generate_seconds':0.,
            'mode':'official_hard_routing_with_skip_cache_length_compatibility',
            'precision':'FP16 unchanged','force_first_layer_full':False,
            'fixed_skip_fraction':False,'precision_retries':0,'skipped_samples':0}
        self.trace=(self.stats_dir/'routes.jsonl').open('w')
        self._decisions=None
        for i,layer in enumerate(self.layers):
            layer.adapter_MHA_router.register_forward_hook(self.router_hook(i))
            layer.self_attn.register_forward_hook(self.count_hook('attention_calls'))
            layer.adapter_MHA.register_forward_hook(self.count_hook('adapter_calls'))
        self.model.lm_head.register_forward_hook(self.logits_hook)
        original_generate=self.model.generate

        def observed_generate(*args,**kw):
            assert kw.get('num_beams',1)==1 and kw.get('use_cache',True)
            self._decisions=[None]*32
            torch.cuda.synchronize()
            start=time.perf_counter()
            try:
                output=original_generate(*args,**kw)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                if any(v is None for v in self._decisions):
                    raise RuntimeError('Missing per-layer prefill routing decision')
                index=self.stats['samples']
                task,split,doc_id=self.request_order[index]
                skip=[not value for value in self._decisions]
                n=sum(skip)
                self.stats['samples']+=1
                self.stats['skipped']+=n;self.stats['full']+=32-n;self.stats['total']+=32
                self.stats['model_generate_seconds']+=elapsed
                for i,value in enumerate(skip):self.stats['per_layer_skip'][i]+=int(value)
                self.trace.write(json.dumps({'task':task,'split':split,'doc_id':doc_id,
                    'skip':skip,'output_tokens':int(output.shape[-1]),'seconds':elapsed})+'\n')
                if self.stats['samples']%32==0:self.save_stats('generating')
                return output
            except Exception as error:
                self.save_stats('failed',error=repr(error))
                raise
        self.model.generate=observed_generate
        self.save_stats('loaded')

    def count_hook(self,kind):
        def hook(module,inputs,output):self.stats[kind]+=1
        return hook

    def router_hook(self,i):
        def hook(module,inputs,output):
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError(f'Nonfinite FP16 router logits at layer {i}; no fallback or sample skipping')
            if output.shape[0]!=1 or output.shape[-1]!=2:
                raise RuntimeError(f'Unexpected router shape {list(output.shape)}')
            if self._decisions[i] is not None:
                raise RuntimeError(f'Route recomputed during cached decode at layer {i}')
            self._decisions[i]=bool(output[0,-1,0]>output[0,-1,1])
            self.stats['router_calls']+=1
        return hook

    def logits_hook(self,module,inputs,output):
        if not bool(torch.isfinite(output[:,-1,:]).all()):
            raise RuntimeError('Nonfinite final-position FP16 logits; no fallback or sample skipping')
        self.stats['finite_logits_checks']+=1

    def save_stats(self,state,**extra):
        data=dict(self.stats,state=state,**extra)
        data['actual_skip_rate']=self.stats['skipped']/self.stats['total'] if self.stats['total'] else None
        data['per_layer_skip_rate']=[n/self.stats['samples'] for n in self.stats['per_layer_skip']] if self.stats['samples'] else None
        data['timing_note']='generate time includes routing/numeric observation; not a matched dense speedup measurement'
        p=self.stats_dir/'summary.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
        self.trace.flush()

    def generate_until(self,requests):
        def key(x):return -len(self.tok_encode(x[0])),x[0]
        collator=utils.Collator([r.args for r in requests],key,grouping=True)
        self.request_order=[(chunk[0][4],chunk[0][5],chunk[0][3])
            for chunk in collator.get_batched(n=1,batch_fn=None)]
        self._stats_start=time.perf_counter()
        try:
            result=super().generate_until(requests)
            assert self.stats['samples']==len(requests)
            assert self.stats['router_calls']==32*len(requests)
            assert len(result)==len(requests)
            self.stats['generation_loop_wall_seconds']=time.perf_counter()-self._stats_start
            self.stats['empty_answers']=sum(not answer.strip() for answer in result)
            self.save_stats('generated')
            return result
        except Exception as error:
            self.save_stats('failed',error=repr(error))
            raise
        finally:
            self.trace.close()
