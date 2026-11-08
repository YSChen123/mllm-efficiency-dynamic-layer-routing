"""39.39M reconstruction router. Dimensions/projection are explicit unverified choices."""
from dataclasses import dataclass
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaRMSNorm
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

@dataclass
class RouterConfig:
    input_size:int=4096
    hidden_size:int=1536
    intermediate_size:int=5120
    num_heads:int=12
    candidate_count:int=16
    init_std:float=0.02

class LayerRouter(nn.Module):
    def __init__(self,settings):
        super().__init__();self.settings=settings
        if settings.hidden_size%settings.num_heads:raise ValueError('Invalid attention dimensions')
        config=LlamaConfig(hidden_size=settings.hidden_size,intermediate_size=settings.intermediate_size,
            num_hidden_layers=1,num_attention_heads=settings.num_heads,num_key_value_heads=settings.num_heads,
            max_position_embeddings=4096,rms_norm_eps=1e-6,attention_dropout=0.)
        config._attn_implementation='eager'
        self.layer_embeddings=nn.Parameter(torch.zeros(settings.candidate_count,settings.input_size))
        self.project=nn.Linear(settings.input_size,settings.hidden_size,bias=False)
        self.block=LlamaDecoderLayer(config,layer_idx=0)
        self.norm=LlamaRMSNorm(settings.hidden_size,eps=config.rms_norm_eps)
        self.score=nn.Linear(settings.hidden_size,1,bias=False)
        def initialize(module):
            if isinstance(module,nn.Linear):nn.init.normal_(module.weight,mean=0.,std=settings.init_std)
        self.apply(initialize)

    def forward(self,visual,text,text_mask=None):
        if visual.ndim!=3 or text.ndim!=3 or visual.shape[0]!=text.shape[0]:raise ValueError('Invalid batched embeddings')
        if visual.shape[-1]!=self.settings.input_size or text.shape[-1]!=self.settings.input_size:raise ValueError('Invalid input width')
        batch=visual.shape[0];device=visual.device;dtype=self.project.weight.dtype
        if text_mask is None:text_mask=torch.ones(text.shape[:2],dtype=torch.bool,device=device)
        if text_mask.shape!=text.shape[:2]:raise ValueError('Invalid text mask')
        layers=self.layer_embeddings.unsqueeze(0).expand(batch,-1,-1)
        inputs=torch.cat([visual.to(dtype),text.to(dtype),layers],dim=1)
        hidden=self.project(inputs)
        mask=torch.cat([torch.ones(visual.shape[:2],dtype=torch.bool,device=device),text_mask.bool(),
            torch.ones((batch,self.settings.candidate_count),dtype=torch.bool,device=device)],dim=1)
        causal=_prepare_4d_causal_attention_mask(mask,hidden.shape[:2],hidden,0)
        positions=(mask.long().cumsum(dim=-1)-1).clamp_min(0)
        hidden=self.block(hidden,attention_mask=causal,position_ids=positions,use_cache=False)[0]
        return self.score(self.norm(hidden)[:,-self.settings.candidate_count:]).squeeze(-1)

    @staticmethod
    def discard_layers(logits,k=4):
        if logits.shape[-1]!=16 or not 0<=k<=16:raise ValueError('Invalid layer budget')
        return torch.topk(logits,k=k,dim=-1).indices.sort(dim=-1).values+16
