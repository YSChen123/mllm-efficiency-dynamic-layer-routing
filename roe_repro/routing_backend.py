"""Restore the public release hard-routing functions without reinitializing weights."""
import importlib.util
from pathlib import Path
import types

from .common import UPSTREAM

OFFICIAL = UPSTREAM / 'RoE/llava/train/roe_adapter_eval.py'
spec = importlib.util.spec_from_file_location('official_roe_hard_routing', OFFICIAL)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def cached_hard_forward(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        question_mask=None, weight_mask=None):
    outputs=release.forward_llama(self,hidden_states,attention_mask=attention_mask,
        position_ids=position_ids,past_key_value=past_key_value,
        output_attentions=output_attentions,use_cache=use_cache,
        question_mask=question_mask,weight_mask=weight_mask)
    if use_cache and not bool(self.weight.any()):
        # The public skip path returns an empty tensor, but LlamaModel reads
        # layer 0's cache length even when that layer was skipped. Shape-only
        # zero-stride views carry length; no attention or K/V projection runs.
        previous=0 if past_key_value is None else past_key_value[0].shape[2]
        shape=(hidden_states.shape[0],self.self_attn.num_key_value_heads,
               previous+hidden_states.shape[1],self.self_attn.head_dim)
        marker=hidden_states.new_zeros(()).expand(shape)
        cache_index=2 if output_attentions else 1
        outputs=outputs[:cache_index]+((marker,marker),)+outputs[cache_index+1:]
    return outputs


def install(model):
    for layer in model.model.layers:
        layer.adapter_MHA_router.forward = types.MethodType(release.Skip_Router.forward, layer.adapter_MHA_router)
        layer.forward = types.MethodType(cached_hard_forward, layer)
