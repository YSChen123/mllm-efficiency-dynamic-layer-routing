"""Audited upstream AdaLLaVA training, measurements and resumable ZeRO-3 saves."""
import hashlib,json,math,os,random,sys,time,types
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import deepspeed
from accelerate import Accelerator
from deepspeed.utils import safe_get_full_fp32_param
from transformers import TrainerCallback,set_seed
from run_guard import data_ready,latest_complete,main_quota,has_capacity
from rng_compat import restore_rng_state
from optimizer_audit import optimizer_probe,scheduler_probe
from accumulation_compat import compatible_no_sync

old_no_sync=Accelerator.no_sync
Accelerator.no_sync=lambda self,model:compatible_no_sync(old_no_sync,self,model)

ROOT=Path('/home/SigConv/cys/AdaLLaVA')
sys.path.insert(0,str(ROOT/'training_repro/source/AdaLLaVA/src'))
import adallava.train.train as upstream
from llava.train.llava_trainer import LLaVATrainer

MODE=os.environ.get('ADA_RUN_MODE','smoke')
STOP=int(os.environ.get('ADA_STOP_AT','0'))
RESUME=os.environ.get('ADA_RESUME')=='1'
RUN=Path(os.environ['ADA_OUTPUT'])
RANK=int(os.environ.get('RANK','0'))
RUN.mkdir(parents=True,exist_ok=True)
if MODE=='formal':
    ready=json.loads((ROOT/'training_repro/data/DATA_READY.json').read_text())
    assert data_ready(ready),'Full dataset readiness gate failed'
    assert (ROOT/'training_repro/smoke/SMOKE_PASSED.json').is_file(),'Smoke gate failed'
assert str(RUN).startswith(str(ROOT/'training_repro/runs' if MODE=='formal' else ROOT/'training_repro/smoke'))

def emit(event,**fields):
    row={'time':time.time(),'event':event,'rank':RANK,**fields}
    with (RUN/f'metrics_rank{RANK}.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
    if RANK==0:print('ADA_EVENT',json.dumps(row,allow_nan=False),flush=True)

def digest(value):return hashlib.sha256(value).hexdigest()

class AuditCallback(TrainerCallback):
    def __init__(self,trainer):self.trainer=trainer;self.grad={};self.initial={};self.previous={};self.start=time.monotonic();self.loaded=None

    def snapshot(self):
        result={}
        for name,param in self.selected.items():
            value=safe_get_full_fp32_param(param)
            assert value is not None,name
            result[name]=value.detach().flatten()[:256].cpu().float().tolist()
        return result

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        self.start=time.monotonic();self.start_step=state.global_step;torch.cuda.reset_peak_memory_stats()
        self.selected={n:p for n,p in self.trainer.model.named_parameters() if n in ['scheduler.mlp_head.weight','model.norm.weight']}
        assert len(self.selected)==2,list(self.selected)
        self.initial=self.snapshot();self.previous=self.initial
        emit('TRAIN_BEGIN',step=state.global_step,max_steps=state.max_steps,epochs=args.num_train_epochs,
             world_size=args.world_size,micro_batch=args.per_device_train_batch_size,accumulation=args.gradient_accumulation_steps,
             global_batch=args.world_size*args.per_device_train_batch_size*args.gradient_accumulation_steps,
             parameter_counts=self.counts,initial_fp32=self.initial,optimizer_probe=optimizer_probe(self.trainer.model_wrapped),
             attention=self.trainer.model.config._attn_implementation,mode=MODE)
        if RESUME:
            cp=latest_complete(RUN);assert cp is not None
            expected=json.loads((cp/f'proof_rank{RANK}.json').read_text())
            assert state.global_step==expected['step']
            assert self.initial==expected['fp32'],'FP32 weights were not restored'
            assert optimizer_probe(self.trainer.model_wrapped)==expected['optimizer_probe'],'Optimizer state mismatch'
            assert scheduler_probe(self.trainer)==expected['lr_scheduler'],'LR scheduler state mismatch'
            self.loaded=expected
            emit('RESUME_WEIGHTS_OPTIMIZER_VERIFIED',step=state.global_step)
            if MODE=='smoke':
                # Exercise the corrected live audit callback against independently loaded disk states.
                self.on_save(args,state,control)
                emit('SAVE_CALLBACK_REVALIDATED',step=state.global_step)

    def on_step_end(self,args,state,control,**kwargs):
        current=self.snapshot() if MODE=='smoke' else {}
        deltas={n:max(abs(x-y) for x,y in zip(v,self.previous[n])) for n,v in current.items()}
        self.previous=current or self.previous
        emit('STEP',step=state.global_step,epoch=state.epoch,wall_seconds=time.monotonic()-self.start,
             effective_samples_per_second=(state.global_step-self.start_step)*args.world_size*args.per_device_train_batch_size*args.gradient_accumulation_steps/(time.monotonic()-self.start),
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
             parameter_updates=deltas,gradient_evidence=self.grad if MODE=='smoke' else {})
        self.grad={}
        if STOP and state.global_step>=STOP:control.should_training_stop=True
        if MODE=='formal' and state.global_step==state.max_steps:control.should_save=True

    def on_log(self,args,state,control,logs=None,**kwargs):
        if logs:
            for name in ['loss','grad_norm','learning_rate']:
                if name in logs:assert math.isfinite(float(logs[name])),(name,logs)
            emit('LOG',step=state.global_step,**logs)

    def on_save(self,args,state,control,**kwargs):
        cp=RUN/f'checkpoint-{state.global_step}'
        proof={'step':state.global_step,'fp32':self.snapshot(),'optimizer_probe':optimizer_probe(self.trainer.model_wrapped),
               'lr_scheduler':scheduler_probe(self.trainer)}
        (cp/f'proof_rank{RANK}.json').write_text(json.dumps(proof))
        dist.barrier()
        if RANK==0:
            optimizer_files=list(cp.rglob('*optim_states.pt'))
            rng_files=list(cp.glob('rng_state*.pth'))
            assert len(optimizer_files)==args.world_size and len(rng_files)==args.world_size
            total=sum(p.stat().st_size for p in cp.rglob('*') if p.is_file())
            (cp/'COMPLETE.json').write_text(json.dumps({'step':state.global_step,'bytes':total,'world_size':args.world_size}))
            emit('CHECKPOINT_COMPLETE',step=state.global_step,bytes=total,path=str(cp))
            # Rotate only after all partitions, RNG files and rank proofs are complete.
            original_rotate(self.trainer,use_mtime=True,output_dir=str(RUN))
        dist.barrier()

    def on_train_end(self,args,state,control,**kwargs):
        seconds=time.monotonic()-self.start
        emit('TRAIN_LOOP_END',step=state.global_step,max_steps=state.max_steps,epoch=state.epoch,
             wall_seconds=seconds,gpu_hours=seconds*args.world_size/3600,mode=MODE)

original_init=LLaVATrainer.__init__
def audited_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    audit=AuditCallback(self);self.add_callback(audit)
    parameters=list(self.model.named_parameters())
    audit.counts={'total':sum(p.numel() for _,p in parameters),'trainable':sum(p.numel() for _,p in parameters if p.requires_grad),
                  'scheduler':sum(p.numel() for n,p in parameters if 'scheduler' in n),'frozen_vision':sum(p.numel() for n,p in parameters if 'vision_tower' in n and not p.requires_grad)}
    assert all(not p.requires_grad for n,p in parameters if 'vision_tower' in n)
    assert any(p.requires_grad for n,p in parameters if 'model.layers.0.' in n)
    assert all(p.requires_grad for n,p in parameters if 'scheduler' in n)
    # ZeRO-3 has partitioned parameters during construction; ds_numel records original sizes.
    audit.counts={key:sum(int(getattr(p,'ds_numel',p.numel())) for n,p in parameters if predicate(n,p)) for key,predicate in {
        'total':lambda n,p:True,'trainable':lambda n,p:p.requires_grad,'scheduler':lambda n,p:'scheduler' in n,
        'frozen_vision':lambda n,p:'vision_tower' in n and not p.requires_grad}.items()}
    if MODE=='smoke':
        for name,param in parameters:
            if name in ['scheduler.mlp_head.weight','model.norm.weight']:
                def capture(gradient,name=name):
                    audit.grad[name]={'finite':bool(torch.isfinite(gradient).all()),'abs_sum':float(gradient.detach().float().abs().sum())}
                    assert audit.grad[name]['finite'],name
                param.register_hook(capture)
    # Verify routing without changing sampling: intercept the real scheduler.forward.
    scheduler=self.model.scheduler; original_forward=scheduler.forward
    def forward(x,latency):
        mask=original_forward(x,latency)
        if MODE=='smoke':
            reduced=mask.detach()[:,:,0,0]
            expected=torch.floor(latency.to(reduced.device)*32)
            assert torch.equal(reduced.sum(1).float(),expected.float())
            assert bool((reduced[:,:16]==1).all())
            emit('ROUTING',budgets=latency.detach().cpu().tolist(),masks=reduced.cpu().tolist())
        return mask
    scheduler.forward=forward
    emit('MODEL_READY',counts=audit.counts,base=self.model.config._name_or_path)
LLaVATrainer.__init__=audited_init

original_compute_loss=LLaVATrainer.compute_loss
def finite_loss(self,*args,**kwargs):
    result=original_compute_loss(self,*args,**kwargs)
    loss=result[0] if isinstance(result,tuple) else result
    assert bool(torch.isfinite(loss).all()),'Nonfinite loss'
    return result
LLaVATrainer.compute_loss=finite_loss

original_rng=LLaVATrainer._load_rng_state
def checked_rng(self,checkpoint):
    if checkpoint:
        rng=restore_rng_state(Path(checkpoint)/f'rng_state_{RANK}.pth')
        assert torch.equal(torch.random.get_rng_state(),rng['cpu'])
        assert random.getstate()==rng['python']
        a,b=np.random.get_state(),rng['numpy']; assert a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
        current=torch.cuda.random.get_rng_state_all();expected=rng['cuda']
        assert len(current)==len(expected) and all(torch.equal(a,b) for a,b in zip(current,expected))
        emit('RESUME_RNG_VERIFIED',checkpoint=str(checkpoint))
LLaVATrainer._load_rng_state=checked_rng

original_train=LLaVATrainer.train
def checked_train(self,*args,**kwargs):
    cp=latest_complete(RUN)
    if RESUME:
        assert cp is not None,'No complete checkpoint'
        kwargs['resume_from_checkpoint']=str(cp)
    else:
        assert not list(RUN.glob('checkpoint-*')),'Fresh start output contains old checkpoints'
        kwargs.pop('resume_from_checkpoint',None)
    return original_train(self,*args,**kwargs)
LLaVATrainer.train=checked_train

original_rotate=LLaVATrainer._rotate_checkpoints
LLaVATrainer._rotate_checkpoints=lambda self,*a,**kw:None
original_checkpoint=LLaVATrainer._save_checkpoint
def guarded_checkpoint(self,*args,**kwargs):
    # Upper bound from actual FP32 Adam master + two moments, frozen CLIP and metadata.
    params=sum(int(getattr(p,'ds_numel',p.numel())) for p in self.model.parameters() if p.requires_grad)
    incoming=params*12+2*1024**3
    quota=main_quota()
    assert has_capacity(quota['soft'],quota['used'],incoming,8*1024**3),('Checkpoint quota reserve',quota,incoming)
    emit('CHECKPOINT_CAPACITY',**quota,incoming=incoming,reserve=8*1024**3)
    return original_checkpoint(self,*args,**kwargs)
LLaVATrainer._save_checkpoint=guarded_checkpoint

def final_save(trainer,output_dir):
    engine=trainer.model_wrapped
    if MODE=='smoke':
        snapshot={}
        for name,param in trainer.model.named_parameters():
            if name in ['scheduler.mlp_head.weight','model.norm.weight']:
                snapshot[name]=safe_get_full_fp32_param(param).detach().flatten()[:256].cpu().float().tolist()
        emit('EXPORT_PROOF',fp32=snapshot,step=trainer.state.global_step)
    engine._config.zero_config.gather_16bit_weights_on_model_save=True
    torch.cuda.synchronize()
    trainer.save_model(output_dir)
    if RANK==0:
        marker='TRAINING_FINISHED.json' if MODE=='formal' else 'SMOKE_EXPORTED.json'
        (RUN/marker).write_text(json.dumps({'step':trainer.state.global_step,'epoch':trainer.state.epoch,'time':time.time()}))
upstream.safe_save_model_for_hf_trainer=final_save

# Upstream seeds Trainer after constructing the model; seed explicitly before fresh Scheduler initialization.
set_seed(42)
import ctypes
nccl_library=ctypes.CDLL(os.environ['LD_PRELOAD'].split(':')[0])
nccl_version=ctypes.c_int()
assert nccl_library.ncclGetVersion(ctypes.byref(nccl_version))==0
emit('ENTRY',mode=MODE,resume=RESUME,argv=sys.argv,torch_version=torch.__version__,torch_cuda=torch.version.cuda,
     torch_build_nccl_version=torch.cuda.nccl.version(),nccl_runtime_version=nccl_version.value,
     nccl_library=os.environ['LD_PRELOAD'],deepspeed_version=deepspeed.__version__)
entry_start=time.monotonic()
try:
    upstream.train(attn_implementation='sdpa')
    seconds=time.monotonic()-entry_start
    emit('TRAINING_TOTAL',wall_seconds=seconds,gpu_hours=seconds*int(os.environ.get('WORLD_SIZE','1'))/3600,mode=MODE)
except BaseException as exc:
    emit('FAILED',error_type=type(exc).__name__,error=str(exc))
    raise
finally:
    if dist.is_initialized():dist.destroy_process_group()
