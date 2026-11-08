import torch
from par_repro.pruning import generate_preserved


@torch.inference_mode()
def route(router, inputs):
    if router is None:
        return [], [], None
    logits = router(inputs['visual'], inputs['text'])
    if logits.shape != (1, 592) or not torch.isfinite(logits).all():
        raise ValueError('Invalid router output')
    tokens = torch.topk(logits[0, :576], router.prune_m).indices.sort().values.tolist()
    layers = (torch.topk(logits[0, 576:], 4).indices.sort().values + 16).tolist()
    return layers, tokens, logits


@torch.inference_mode()
def generate(model, tokenizer, router, inputs, generation, audit=False):
    layers, tokens, logits = route(router, inputs)
    original = model.model.layers
    events, hooks = [], []
    if audit:
        hooks = [block.register_forward_pre_hook(lambda module, args, i=i: events.append((i, args[0].shape[1])))
                 for i, block in enumerate(original)]
    try:
        ids = generate_preserved(model, inputs['prefix'], inputs['image_start'], layers, tokens,
                                 generation['max_new_tokens'], generation['use_cache'])
    finally:
        for hook in hooks:
            hook.remove()
    retained = [i for i in range(32) if i not in layers]
    if model.model.layers is not original or [b.self_attn.layer_idx for b in original] != list(range(32)):
        raise ValueError('Physical layer/cache state was not restored')
    if audit:
        for step in range(ids.shape[1]):
            expected = [(i, 1 if generation['use_cache'] and step else inputs['prefix'].shape[1] + step - (len(tokens) if i >= 2 else 0)) for i in retained]
            if events[step * len(retained):(step + 1) * len(retained)] != expected:
                raise ValueError('Physical execution trace differs from effective budget')
        if len(events) != len(retained) * ids.shape[1]:
            raise ValueError('Unexpected block execution count')
    text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if not text:
        raise ValueError('Empty generated text')
    return {'text': text, 'response_token_ids': ids[0].tolist(), 'dropped_original_layers': layers,
            'dropped_visual_tokens': tokens, 'retained_original_layers': retained, 'visual_tokens': 576 - len(tokens),
            'visual_tokens_first2': 576, 'retained_layer_count': len(retained), 'execution_audited': audit,
            'cache_and_original_layers_restored': True, 'router_logits': logits[0].cpu().tolist() if logits is not None else None}
