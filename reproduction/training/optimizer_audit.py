import hashlib
import torch

def scheduler_probe(trainer):
    return trainer.lr_scheduler.state_dict()

def optimizer_probe(engine):
    # ZeRO temporarily clears param_groups after state_dict/save; raw state remains valid.
    values=list(engine.optimizer.optimizer.state.values())[:1]
    return [{str(key):hashlib.sha256(value.detach().flatten()[:4096].cpu().numpy().tobytes()).hexdigest()
             if torch.is_tensor(value) else str(value) for key,value in state.items()} for state in values]
