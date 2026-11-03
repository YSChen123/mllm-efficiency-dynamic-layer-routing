"""Official inference plus passive timing/routing instrumentation in an isolated run."""
import json
import os
from pathlib import Path
import runpy
import sys
import time
import torch
import datasets
from local_dataset import load_prepared_dataset
from eval_policy import track_requests, expected_layers, evaluation_module

ROLE=os.environ['ADA_ROLE']
OUT=Path(os.environ['ADA_MEASUREMENTS']).parent
manifest=json.loads(Path(os.environ['ADA_DATASET_MANIFEST']).read_text())
model_manifest=json.loads(Path(os.environ['ADA_MODEL_MANIFEST']).read_text())
original_load_dataset=datasets.load_dataset

def local_load_dataset(path,name=None,*args,**kwargs):
    if path!=manifest['repo']:return original_load_dataset(path,name,*args,**kwargs)
    if name!=manifest['config'] or kwargs.get('revision')!=manifest['revision']:
        raise ValueError('Dataset config/revision does not match pinned manifest')
    data=load_prepared_dataset(manifest)
    return data[kwargs['split']] if kwargs.get('split') else data

datasets.load_dataset=local_load_dataset
from src.adallava.eval.adallava_wrapper import AdaLlava
from lmms_eval.models.llava import Llava

if ROLE=='dense':
    pos=sys.argv.index('--model');sys.argv[pos+1]='llava'
    pos=sys.argv.index('--model_args')
    sys.argv[pos+1]=','.join(part for part in sys.argv[pos+1].split(',') if not part.startswith('latency='))
(OUT/'effective_invocation.json').write_text(json.dumps({'argv':sys.argv,'role':ROLE,
    'model_path':model_manifest['model_path'],'measurement_scope':'model.generate: vision + LM prefill + decode; excludes CPU image preprocessing/tokenization'},indent=2))

cls=Llava if ROLE=='dense' else AdaLlava
original_init=cls.__init__
original_generate_until=cls.generate_until
errors=[]

def tracked_generate_until(self,requests):
    with track_requests(requests,lambda doc:setattr(self,'_active_doc',doc)):
        return original_generate_until(self,requests)

def measured_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    loaded={'role':ROLE,'model_path':model_manifest['model_path'],'dtype':str(self.model.dtype),
        'attention':self.model.config._attn_implementation,'verified_master_samples':[],
        'gpu_index':os.environ['ADA_GPU'],'gpu_name':torch.cuda.get_device_name(0)}
    if ROLE=='reproduced':
        params=dict(self.model.named_parameters())
        for name,values in model_manifest['master_samples'].items():
            actual=params[name].detach().flatten()[:len(values)].cpu()
            expected=torch.tensor(values,dtype=torch.bfloat16).to(actual.dtype)
            assert torch.equal(actual,expected),('Loaded weights differ from final export',name)
            loaded['verified_master_samples'].append(name)
    if ROLE=='dense':assert not hasattr(self.model,'scheduler'),'Dense baseline loaded adaptive model'
    else:assert self.model.config.scheduler_type=='L' and self.model.config.num_prefix_layers==16
    (OUT/'MODEL_LOADED.json').write_text(json.dumps(loaded,indent=2))
    self._active_doc={}
    observed={}
    call_count=0
    original_generate=self.model.generate

    def first_forward_pre(module,args,kwargs):
        if not observed.get('generating') or observed.get('prefill_started'):return
        observed['prefill_started']=True
        start=torch.cuda.Event(enable_timing=True);start.record()
        observed['prefill_start']=start
        embeds=kwargs.get('inputs_embeds')
        inputs=kwargs.get('input_ids')
        if embeds is not None:observed['expanded_prompt_tokens']=int(embeds.shape[1])
        elif inputs is not None:observed['expanded_prompt_tokens']=int(inputs.shape[1])

    def first_forward_post(module,args,kwargs,result):
        if not observed.get('generating') or observed.get('prefill_end') is not None:return
        end=torch.cuda.Event(enable_timing=True);end.record();observed['prefill_end']=end

    self.model.register_forward_pre_hook(first_forward_pre,with_kwargs=True)
    self.model.register_forward_hook(first_forward_post,with_kwargs=True)
    if ROLE!='dense':
        for name in ['forward','latency_encoding']:
            original=getattr(self.model.scheduler,name)
            def timed(*a,_name=name,_original=original,**kw):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                if _name=='forward':observed['scheduler_observed_budget']=float(a[1].detach().float().item())
                result=_original(*a,**kw)
                end.record();observed.setdefault('scheduler_events',{}).setdefault(_name,[]).append((start,end))
                return result
            setattr(self.model.scheduler,name,timed)

    def generate(*generation_args,**generation_kwargs):
        nonlocal call_count
        observed.clear();observed['generating']=True
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        started=time.perf_counter();start.record()
        try:output=original_generate(*generation_args,**generation_kwargs)
        except Exception as error:
            errors.append(repr(error));raise
        end.record();torch.cuda.synchronize()
        wall=time.perf_counter()-started
        observed['generating']=False
        assert self._active_doc,'Generation has no tracked benchmark doc_id'
        assert 'prefill_end' in observed,'First LM forward was not observed'
        if ROLE=='dense':
            plan=[32]*32;target=1.;seen=1.
            sequences=getattr(output,'sequences',output)
            generated=int(sequences.shape[1]-1)
        else:
            plan=[int(value) for value in output.execution_plan]
            target=float(generation_kwargs['latency']);seen=observed['scheduler_observed_budget']
            generated=int(output.gen_len)
        mask=[int(value>=0) for value in plan]
        row={**self._active_doc,'role':ROLE,'target_budget':target,'scheduler_observed_budget':seen,
             'generation_wall_seconds':wall,'generation_cuda_event_ms':start.elapsed_time(end),
             'lm_prefill_cuda_ms':observed['prefill_start'].elapsed_time(observed['prefill_end']),
             'vision_plus_lm_prefill_cuda_ms':start.elapsed_time(observed['prefill_end']),
             'expanded_prompt_tokens':observed.get('expanded_prompt_tokens'),
             'generated_len':generated,'executed_layers':sum(mask),'total_layers':len(mask),
             'layer_skipping_ratio':1-sum(mask)/len(mask),'execution_plan_heads_per_layer':plan,
             'layer_execution_mask':mask,'effective_layer_fraction':sum(mask)/len(mask),
             'budget_satisfied':sum(mask)<=expected_layers(target),
             'maximal_quantized_budget_utilization':sum(mask)==expected_layers(target),
             'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
             'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
             'gpu_index':os.environ['ADA_GPU'],'cold_first_sample':call_count==0,
             'warmup_exclusion_candidate':call_count<3,'shared_gpu':False,
             'phase':'full_generate_including_vision_prefill_decode'}
        for name,pairs in observed.get('scheduler_events',{}).items():
            row['scheduler_'+name+'_cuda_ms']=sum(a.elapsed_time(b) for a,b in pairs)
        with Path(os.environ['ADA_MEASUREMENTS']).open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        print('ADA_INFERENCE '+json.dumps(row),flush=True)
        call_count+=1
        return output
    self.model.generate=generate

cls.__init__=measured_init
cls.generate_until=tracked_generate_until
runpy.run_module(evaluation_module(ROLE),run_name='__main__')
if errors:raise RuntimeError('Official wrapper caught generation errors: '+str(errors[:3]))
