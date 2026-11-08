"""Physical block removal in a process-local context; original weights are unchanged."""
from contextlib import contextmanager
import torch.nn as nn

@contextmanager
def skip_layers(backbone,dropped,candidate_start=16):
    original=backbone.model.layers
    dropped=list(dropped)
    if len(set(dropped))!=len(dropped) or any(type(i) is not int or not candidate_start<=i<len(original) for i in dropped):
        raise ValueError('Invalid candidate layer IDs')
    kept=[i for i in range(len(original)) if i not in set(dropped)]
    cache_ids=[block.self_attn.layer_idx for block in original]
    try:
        backbone.model.layers=nn.ModuleList([original[i] for i in kept])
        # HF DynamicCache needs dense indexes. These indexes are not router/original layer IDs.
        for index,block in enumerate(backbone.model.layers):block.self_attn.layer_idx=index
        yield kept
    finally:
        backbone.model.layers=original
        for block,index in zip(original,cache_ids):block.self_attn.layer_idx=index
