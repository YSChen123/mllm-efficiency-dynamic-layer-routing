"""Paper equations with reconstruction choices documented in configs and plan."""
import itertools
import math
import random
import torch
import torch.nn.functional as F

LAYER_ACTIONS=tuple(itertools.combinations(range(16,32),4))

def sample_layer_actions(seed):
    return [list(action) for action in random.Random(seed).sample(LAYER_ACTIONS,5)]

def kl_full_to_pruned(full_logits,pruned_logits):
    if full_logits.shape!=pruned_logits.shape or full_logits.ndim<2:
        raise ValueError('KL distributions must have identical token/vocabulary shapes')
    p=F.log_softmax(full_logits.float(),dim=-1)
    q=F.log_softmax(pruned_logits.float(),dim=-1)
    result=(p.exp()*(p-q)).sum(-1).mean()
    if not torch.isfinite(result) or result.item() < -1e-6:raise ValueError('Invalid KL')
    return result.clamp_min(0.)

def preference_extremes(kls,tolerance=1e-8):
    if len(kls)!=5 or not all(math.isfinite(x) and x>=0 for x in kls):
        raise ValueError('Expected five finite nonnegative KL values')
    chosen=min(range(5),key=lambda i:kls[i]);rejected=max(range(5),key=lambda i:kls[i])
    if kls[rejected]-kls[chosen]<=tolerance:return None
    return chosen,rejected

def action_log_prob(logits,discard_mask):
    if logits.shape!=discard_mask.shape:raise ValueError('Action/logit shape mismatch')
    if not ((discard_mask==0)|(discard_mask==1)).all():raise ValueError('Action bits must be binary')
    return -F.binary_cross_entropy_with_logits(logits,discard_mask.to(logits.dtype),reduction='none').sum(-1)

def dpo_loss(chosen_logp,rejected_logp,beta=1.):
    if chosen_logp.shape!=rejected_logp.shape or beta<=0:raise ValueError('Invalid DPO inputs')
    return -F.logsigmoid(beta*(chosen_logp-rejected_logp)).mean()
