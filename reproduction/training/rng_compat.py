"""Restore trusted, locally written Trainer RNG states with Torch >=2.6."""
import random
import numpy as np
import torch

def restore_rng_state(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['cpu'])
    if 'cuda' in state:
        if isinstance(state['cuda'],list):torch.cuda.random.set_rng_state_all(state['cuda'])
        else:torch.cuda.random.set_rng_state(state['cuda'])
    return state
