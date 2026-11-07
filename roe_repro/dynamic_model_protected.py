"""Optional FP32 recovery; kept distinct from strict FP16 release diagnostics."""
import copy
import json
import time
import torch
from .dynamic_model import DynamicLlava
from .precision_retry import guarded_generate


class DynamicProtectedLlava(DynamicLlava):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        original_observed=self.model.generate
        self.stats['precision']='FP16; only nonfinite requests retried once in FP32'
        self.stats['mode']='official_hard_routing_cache_compatible_selective_fp32_recovery'
        self.stats['failed_fp16_seconds']=0.

        def generate(*args,**kw):
            before=copy.deepcopy(self.stats)
            started=time.perf_counter()
            event=None
            def invoke():
                call_kw=dict(kw)
                dtype=next(self.model.parameters()).dtype
                images=call_kw.get('images')
                if isinstance(images,list):call_kw['images']=[x.to(dtype=dtype) for x in images]
                elif images is not None:call_kw['images']=images.to(dtype=dtype)
                return original_observed(*args,**call_kw)
            def on_retry(error):
                nonlocal event
                task,split,doc_id=self.request_order[before['samples']]
                event={'task':task,'split':split,'doc_id':doc_id,
                    'fp16_error':str(error),'fallback_precision':'FP32'}
                self.stats=copy.deepcopy(before)
                self.stats['precision_retries']+=1
                self.stats['failed_fp16_seconds']+=time.perf_counter()-started
            try:
                result,retried=guarded_generate(self.model,invoke,on_retry)
                if retried:
                    event['status']='recovered'
                    with (self.stats_dir/'precision_retries.jsonl').open('a') as file:
                        file.write(json.dumps(event)+'\n')
                    self.save_stats('generating')
                return result
            except Exception as error:
                if event is not None:
                    event.update(status='failed',recovery_error=repr(error))
                    with (self.stats_dir/'precision_retries.jsonl').open('a') as file:
                        file.write(json.dumps(event)+'\n')
                self.save_stats('failed',error=repr(error))
                raise
        self.model.generate=generate
        self.save_stats('loaded')
