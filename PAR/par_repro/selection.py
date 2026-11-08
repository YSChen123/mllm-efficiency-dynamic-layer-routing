from pathlib import Path
import json
import math
from par_repro.io import atomic_json, file_sha, code_sha, model_resource_identity


def choose_candidate(rows, expected):
    if len(rows) != len(expected) or {(r['trial_id'], r['epoch']) for r in rows} != set(expected):
        raise ValueError('All unique preregistered candidates required')
    if not all(math.isfinite(r['mean_routed_KL']) and r['mean_routed_KL'] >= 0 for r in rows):
        raise ValueError('Invalid validation KL')
    minimum = min(r['mean_routed_KL'] for r in rows)
    return min((r for r in rows if r['mean_routed_KL'] <= minimum + 1e-12),
               key=lambda r: (r['epoch'], r['learning_rate'], r['trial_id']))


def select(workspace, config, search):
    import torch
    from par_repro.preferences import checked_preferences
    from par_repro.runtime import load_frozen
    from par_repro.checkpoints import load_router
    from par_repro.pruning import forward_preserved
    from par_repro.math_core import kl_full_to_pruned
    from par_repro.inference import route
    workspace = Path(workspace)
    folder, manifest = checked_preferences(workspace, config['training']['mode'], 'validation')
    if manifest['count'] != 256 or manifest['scope'] != 'FORMAL':
        raise ValueError('Formal selection needs all256 independent validation examples')
    expected = [(trial['id'], epoch) for trial in search['trials'] for epoch in search['reported_epochs']]
    rows, routers = [], []
    for trial in search['trials']:
        training = workspace / 'checkpoints' / trial['id']
        report = json.loads((training / 'training_report.json').read_text())
        if report['status'] != 'COMPLETED' or report['scope'] != 'FORMAL' or report['count'] != 5000:
            raise ValueError('All formal training trajectories required')
        for epoch in search['reported_epochs']:
            path = training / f'epoch{epoch}.pt'
            if file_sha(path) != report['snapshots'][str(epoch)]:
                raise ValueError('Candidate checksum mismatch')
            router, state = load_router(path)
            if state['model_resources'] != model_resource_identity(workspace):
                raise ValueError('Candidate model/source identity differs from validation resources')
            settings = state['configuration']['training']
            if settings['attention_mode'] != trial['attention_mode'] or settings['global_batch_size'] != trial['global_batch_size']:
                raise ValueError('Wrong effective candidate settings')
            routers.append(router)
            rows.append({'trial_id': trial['id'], 'epoch': epoch, 'learning_rate': settings['learning_rate'],
                'attention_mode': settings['attention_mode'], 'global_batch_size': settings['global_batch_size'],
                'checkpoint': str(path.relative_to(workspace)), 'checkpoint_sha256': file_sha(path), 'validation_KLs': []})
    model, _, _ = load_frozen(workspace, config)
    m = config['public_constraints']['M']
    for index in range(256):
        item = json.loads((folder / f'items/{index:05d}.json').read_text())
        pack = torch.load(folder / item['feature'], map_location='cpu', weights_only=True)
        with torch.inference_mode():
            visual, text = pack['visual'].unsqueeze(0).cuda(), pack['text'].unsqueeze(0).cuda()
            start = pack['image_start']
            prefix = torch.cat([text[:, :start], visual, text[:, start:]], dim=1)
            response = pack['response_token_ids'].unsqueeze(0).cuda()
            length = response.shape[1]
            begin = prefix.shape[1] - 1
            teacher = torch.cat([prefix, model.get_input_embeddings()(response)], dim=1)
            full, _, _ = forward_preserved(model, teacher, start, [], [])
            full = full[:, begin:begin + length].contiguous()
            cache = {}
            for row, router in zip(rows, routers):
                layers, tokens, _ = route(router, {'visual': visual, 'text': text})
                key = (tuple(layers), tuple(tokens))
                if key not in cache:
                    output, _, _ = forward_preserved(model, teacher, start, layers, tokens)
                    cache[key] = float(kl_full_to_pruned(full, output[:, begin - m:begin - m + length].contiguous()))
                row['validation_KLs'].append(cache[key])
        if index < 3 or (index + 1) % 16 == 0:
            print('VALIDATION_PROGRESS', index + 1, 256, flush=True)
    for row in rows:
        row['mean_routed_KL'] = sum(row['validation_KLs']) / 256
    selected = choose_candidate(rows, expected)
    result = {'status': 'SELECTED', 'selected': selected, 'candidates': rows, 'test_scores_used_for_selection': False,
              'criterion': 'minimum_actual_routed_Full_KL', 'validation_manifest_sha256': file_sha(folder / 'manifest.json'),
              'code_sha256': code_sha()}
    atomic_json(workspace / 'selection_result.json', result)
    print('SELECTED', selected['trial_id'], selected['epoch'], selected['mean_routed_KL'], flush=True)
    return workspace / selected['checkpoint']
