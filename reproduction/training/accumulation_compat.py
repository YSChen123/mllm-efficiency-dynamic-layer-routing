"""Backport Accelerate's ZeRO gradient-partition no_sync exclusion."""
from contextlib import nullcontext

def compatible_no_sync(original,accelerator,model):
    partitioned=getattr(model,'zero_optimization_partition_gradients',None)
    if callable(partitioned) and partitioned():return nullcontext()
    return original(accelerator,model)
