"""Run via torchrun: small and model-sized collectives must preserve values."""
import os,torch
import torch.distributed as dist
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
for size in [16,4096,16777216]:
    value=torch.full((size,),dist.get_rank()+1.,device='cuda',dtype=torch.bfloat16)
    dist.all_reduce(value)
    torch.cuda.synchronize()
    assert bool((value==3).all())
    if dist.get_rank()==0:print('NCCL_ALL_REDUCE_PASS',size,flush=True)
dist.destroy_process_group()
