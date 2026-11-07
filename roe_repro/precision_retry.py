"""Explicit, separately labelled numerical compatibility for failed FP16 requests."""
import torch


def guarded_generate(model,call,on_retry):
    try:
        return call(),False
    except RuntimeError as error:
        if not str(error).startswith(('Nonfinite FP16 router logits at layer ',
                'Nonfinite final-position FP16 logits;')):
            raise
        on_retry(error)
    parameter_dtypes={name:p.dtype for name,p in model.named_parameters()}
    buffers=[(module,name,None if value is None else value.detach().clone())
        for module in model.modules() for name,value in module._buffers.items()]
    cache_lengths=[(module,module.max_seq_len_cached) for module in model.modules()
        if hasattr(module,'max_seq_len_cached')]
    default_dtype=torch.get_default_dtype()
    try:
        # Legacy mask factories omit dtype; do not inherit an FP16 default.
        torch.set_default_dtype(torch.float32)
        model.float()
        return call(),True
    finally:
        for name,param in model.named_parameters():
            param.data=param.data.to(dtype=parameter_dtypes[name])
        for module,name,value in buffers:module._buffers[name]=value
        for module,length in cache_lengths:module.max_seq_len_cached=length
        torch.set_default_dtype(default_dtype)
