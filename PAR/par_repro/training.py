from pathlib import Path
import json
import math
from par_repro.io import atomic_json, file_sha, bind_identity, code_sha, object_sha, reconcile_step_log


def collate(items):
    import torch
    longest = max(x['text'].shape[0] for x in items)
    text = items[0]['text'].new_zeros((len(items), longest, items[0]['text'].shape[-1]))
    mask = torch.zeros((len(items), longest), dtype=torch.bool)
    for i, item in enumerate(items):
        length = item['text'].shape[0]
        text[i, :length] = item['text']
        mask[i, :length] = True
    return {'visual': torch.stack([x['visual'] for x in items]), 'text': text, 'text_mask': mask,
            **{name: torch.stack([x[name] for x in items]) for name in ['chosen', 'rejected', 'positive', 'negative']}}


def load_item(folder, index, manifest, m):
    import torch
    from par_repro.pruning import action_mask
    name = f'items/{index:05d}.json'
    if file_sha(folder / name) != manifest['files'][name]:
        raise ValueError('Metadata changed during training')
    item = json.loads((folder / name).read_text())
    if item['identity'] != manifest['identity'] or file_sha(folder / item['feature']) != item['feature_sha256']:
        raise ValueError('Features changed during training')
    pack = torch.load(folder / item['feature'], map_location='cpu', weights_only=True)
    masks = [action_mask(action, m) for action in item['actions']]
    return {'visual': pack['visual'], 'text': pack['text'], 'chosen': masks[item['chosen_index']], 'rejected': masks[item['rejected_index']],
            'positive': torch.stack([masks[i] for i in item['positive_indices']]), 'negative': torch.stack([masks[i] for i in item['negative_indices']])}


def train(workspace, config, name, resume=False, max_steps=None):
    import torch
    from par_repro.router import ControlledRouter
    from par_repro._paper_router import PaperRouterConfig
    from par_repro.batching import epoch_batches, accumulated_update
    from par_repro.pruning import stable_margin, balanced_loss
    from par_repro.preferences import checked_preferences
    workspace = Path(workspace)
    settings = config['training']
    folder, manifest = checked_preferences(workspace, settings['mode'], 'training')
    count = manifest['count']
    expected = 5000 if manifest['scope'] == 'FORMAL' else count
    if count != expected:
        raise ValueError('Wrong formal training count')
    if Path(name).name != name or name in ['.', '..', '']:
        raise ValueError('Trial name must be a single directory component')
    target = workspace / 'checkpoints' / name
    target.mkdir(parents=True, exist_ok=True)
    identity = {'config': config, 'code_sha256': code_sha(), 'preference_manifest_sha256': file_sha(folder / 'manifest.json'), 'scope': manifest['scope']}
    bind_identity(target / 'identity.json', identity)
    digest = object_sha(identity)
    torch.manual_seed(settings['seed'])
    model = ControlledRouter(PaperRouterConfig(**config['router']), settings['attention_mode']).cuda().float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], betas=tuple(settings['betas']),
                                 eps=settings['eps'], weight_decay=settings['weight_decay'])
    checkpoint = target / 'latest.pt'
    step = 0
    if checkpoint.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires --resume')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != digest or state['configuration'] != config:
            raise ValueError('Resume identity changed')
        model.load_state_dict(state['router'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng_cpu'])
        torch.cuda.set_rng_state(state['rng_cuda'])
        step = state['step']
        reconcile_step_log(target / 'steps.jsonl', step)
    elif (target / 'steps.jsonl').exists():
        raise ValueError('Unsaved previous attempt has no checkpoint; preserve it and use a new trial directory')
    batches = [b for e in range(settings['epochs']) for b in epoch_batches(e, settings['global_batch_size'], count)]
    total = len(batches)
    end = total if max_steps is None else min(total, max_steps)
    if not 0 <= step <= end or end <= 0:
        raise ValueError('Invalid resume/update boundary')
    per_epoch = math.ceil(count / settings['global_batch_size'])
    def loss_for(ids):
        items = [load_item(folder, i, manifest, config['public_constraints']['M']) for i in ids]
        batch = {k: v.cuda() for k, v in collate(items).items()}
        logits = model(batch['visual'], batch['text'], batch['text_mask'])
        if settings['pairing'] == 'balanced':
            return balanced_loss(logits, batch['positive'], batch['negative'])
        if settings['pairing'] == 'extremes':
            return -torch.nn.functional.logsigmoid(stable_margin(logits, batch['chosen'], batch['rejected'])).mean()
        raise ValueError('Unknown pairing rule')
    snapshots = {}
    with (target / 'steps.jsonl').open('a') as log:
        for at in range(step, end):
            loss = accumulated_update(model, optimizer, batches[at], loss_for, settings['micro_batch_size'])
            step = at + 1
            epoch = at // per_epoch + 1
            log.write(json.dumps({'step': step, 'epoch': epoch, 'loss': loss, 'effective_global_batch': len(batches[at])}) + '\n')
            log.flush()
            state = {'portable_format_version': 1, 'configuration': config, 'router': model.state_dict(), 'step': step, 'epoch': epoch,
                     'identity': digest, 'scope': manifest['scope'], 'optimizer': optimizer.state_dict(),
                     'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()}
            if step % 250 == 0 or step == end:
                temporary = checkpoint.with_suffix('.tmp')
                torch.save(state, temporary)
                temporary.replace(checkpoint)
            if step % per_epoch == 0 and epoch in settings['reported_epochs']:
                path = target / f'epoch{epoch}.pt'
                temporary = path.with_suffix('.tmp')
                torch.save({k: v for k, v in state.items() if k not in ['optimizer', 'rng_cpu', 'rng_cuda']}, temporary)
                temporary.replace(path)
                snapshots[str(epoch)] = file_sha(path)
            if step <= 3 or step % 100 == 0:
                print('TRAIN_PROGRESS', name, step, total, loss, flush=True)
    for epoch in settings['reported_epochs']:
        path = target / f'epoch{epoch}.pt'
        if path.exists():
            snapshots[str(epoch)] = file_sha(path)
    atomic_json(target / 'training_report.json', {'status': 'COMPLETED' if step == total else 'PARTIAL', 'step': step, 'total_steps': total,
        'configuration': config, 'identity': digest, 'scope': manifest['scope'], 'count': count, 'checkpoint_sha256': file_sha(checkpoint), 'snapshots': snapshots})
    print('TRAIN_FINISHED', name, step, total, flush=True)
