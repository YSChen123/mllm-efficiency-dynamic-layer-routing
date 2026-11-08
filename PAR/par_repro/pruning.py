import math,copy
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask,_prepare_4d_causal_attention_mask_for_sdpa
from par_repro.math_core import sample_layer_actions
from par_repro.layer_skip import skip_layers
from par_repro.positions import retained_positions

def noisy_actions(head_logits,image_start,seed):
    logits=head_logits.detach().cpu().float()
    if logits.ndim!=2 or not torch.isfinite(logits).all() or image_start<0 or image_start+576>logits.shape[1]:raise ValueError('Invalid QK ranking logits')
    generator=torch.Generator().manual_seed(seed);result=[]
    for layers in sample_layer_actions(seed):
        scores=(logits+math.sqrt(10)*torch.randn(logits.shape,generator=generator)).softmax(-1).mean(0)[image_start:image_start+576]
        tokens=torch.topk(scores,k=400,largest=False).indices.sort().values.tolist()
        result.append({'layers':layers,'tokens':tokens})
    return result

def balanced_indices(values,tolerance=1e-8):
    if len(values)!=5 or not all(math.isfinite(v) for v in values):raise ValueError('Require five finite KL scores')
    order=sorted(range(5),key=lambda i:(values[i],i))
    if values[order[3]]-values[order[1]]<=tolerance:return None
    return order[:2],order[-2:]

def action_mask(action,m):
    tokens=action['tokens'];layers=action['layers']
    if m not in [0,400] or len(tokens)!=m or len(set(tokens))!=m or any(type(i) is not int or not 0<=i<576 for i in tokens):raise ValueError('Invalid visual budget')
    if len(layers)!=4 or len(set(layers))!=4 or any(type(i) is not int or not 16<=i<32 for i in layers):raise ValueError('Invalid layer budget')
    mask=torch.zeros(592);mask[tokens]=1.;mask[torch.tensor(layers)-16+576]=1.;return mask

def balanced_loss(logits,positive,negative):
    if logits.ndim!=2 or positive.shape!=negative.shape or positive.shape!=(logits.shape[0],2,logits.shape[1]):raise ValueError('Require two positive and two negative actions per input')
    if not all(((a==0)|(a==1)).all() for a in [positive,negative]):raise ValueError('Nonbinary preference action')
    difference=positive[:,:,None,:]-negative[:,None,:,:]
    margins=(difference.to(logits.dtype)*logits[:,None,None,:]).sum(-1)
    return -torch.nn.functional.logsigmoid(margins).mean()

def stable_margin(logits,chosen,rejected):
    if logits.shape!=chosen.shape or logits.shape!=rejected.shape:raise ValueError('Action shape mismatch')
    if not (((chosen==0)|(chosen==1))&((rejected==0)|(rejected==1))).all():raise ValueError('Nonbinary action')
    return ((chosen-rejected).to(logits.dtype)*logits).sum(-1)

@torch.inference_mode()
def ranking_logits(model,prefix):
    if prefix.ndim!=3 or prefix.shape[0]!=1:raise ValueError('One unpadded prompt required')
    positions=torch.arange(prefix.shape[1],device=prefix.device).unsqueeze(0)
    prepare=_prepare_4d_causal_attention_mask_for_sdpa if model.model._use_sdpa else _prepare_4d_causal_attention_mask
    mask=prepare(None,prefix.shape[:2],prefix,0)
    hidden=model.model.layers[0](prefix,attention_mask=mask,position_ids=positions,use_cache=False)[0]
    block=model.model.layers[1];attn=block.self_attn;normalized=block.input_layernorm(hidden)
    batch,length,_=normalized.shape
    q=attn.q_proj(normalized).view(batch,length,attn.num_heads,attn.head_dim).transpose(1,2)
    k=attn.k_proj(normalized).view(batch,length,attn.num_key_value_heads,attn.head_dim).transpose(1,2)
    cos,sin=attn.rotary_emb(k,seq_len=length)
    q,k=apply_rotary_pos_emb(q,k,cos,sin,positions)
    k=repeat_kv(k,attn.num_key_value_groups)
    return (torch.matmul(q[:,:,-1:],k.transpose(2,3))/math.sqrt(attn.head_dim))[0,:,0].float()

@torch.inference_mode()
def forward_preserved(model,embeds,image_start,layers,tokens,past=None,use_cache=False,visual_tokens=576,candidate_start=16,mask_only=False):
    if embeds.ndim!=3 or embeds.shape[0]!=1 or (past is not None and not use_cache):raise ValueError('Invalid paper_v3 forward inputs')
    prefill=past is None
    keep=retained_positions(embeds.shape[1],image_start,tokens,visual_tokens) if prefill else None
    original_count=past.get_seq_length(0) if past is not None else 0
    positions=torch.arange(original_count,original_count+embeds.shape[1],device=embeds.device).unsqueeze(0)
    original_positions=positions
    cache=DynamicCache() if use_cache and past is None else past
    hidden=embeds;trace=[];modified=[]
    with skip_layers(model,layers,candidate_start=candidate_start) as retained:
        try:
            for dense,(original,block) in enumerate(zip(retained,model.model.layers)):
                count=cache.get_seq_length(dense) if cache is not None else 0
                keymask=None
                if mask_only and original>=2 and tokens:
                    if cache is not None:raise ValueError('Mask-only is diagnostic prefill only')
                    keymask=torch.ones(hidden.shape[:2],device=hidden.device,dtype=torch.bool)
                    keymask[:,[image_start+i for i in tokens]]=False
                prepare=_prepare_4d_causal_attention_mask_for_sdpa if model.model._use_sdpa else _prepare_4d_causal_attention_mask
                attention=prepare(keymask,hidden.shape[:2],hidden,count)
                rotary=block.self_attn.rotary_emb;previous=rotary.forward
                required=max(int(positions.max())+1,count+hidden.shape[1])
                def extended(x,seq_len=None,previous=previous,required=required):return previous(x,seq_len=max(seq_len or 0,required))
                rotary.forward=extended;modified.append((rotary,previous))
                trace.append((original,hidden.shape[1]))
                hidden=block(hidden,attention_mask=attention,position_ids=positions,past_key_value=cache,use_cache=use_cache)[0]
                if original==1 and prefill and tokens and not mask_only:
                    hidden=hidden[:,keep];positions=original_positions[:,keep]
            return model.lm_head(model.model.norm(hidden)).float(),cache,trace
        finally:
            for rotary,previous in modified:rotary.forward=previous

@torch.inference_mode()
def generate_preserved(model,prefix,image_start,layers,tokens,max_new_tokens,use_cache=True,visual_tokens=576,candidate_start=16):
    inputs=prefix;cache=None;generated=[]
    for step in range(max_new_tokens):
        logits,cache,_=forward_preserved(model,inputs,image_start,layers,tokens,past=cache if use_cache else None,
            use_cache=use_cache,visual_tokens=visual_tokens,candidate_start=candidate_start)
        token=logits[:,-1].argmax(-1).reshape(1,1);generated.append(token)
        if token.item()==model.config.eos_token_id:break
        value=model.get_input_embeddings()(token)
        inputs=value if use_cache else torch.cat([inputs,value],dim=1)
    return torch.cat(generated,dim=1)
