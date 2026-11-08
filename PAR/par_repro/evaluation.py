from pathlib import Path
import json
import hashlib
from par_repro.io import atomic_json, bind_identity, file_sha, object_sha, code_sha, check_prediction_prefix
from par_repro.config import resolve
from par_repro.data import read_annotations, resolve_image, index_seed_images, seed_prompt
from par_repro.metrics import pope_scores, seed_scores


def benchmark_cases(workspace, config, benchmark, limit=None):
    pope, seed = read_annotations(workspace)
    if benchmark == 'pope':
        cases = [dict(r, question=r['text']) for r in pope]
        if limit is not None:
            if limit < 3:
                raise ValueError('POPE smoke needs at least one row from each category')
            cases = [r for category in ['adversarial', 'popular', 'random'] for r in [c for c in cases if c['category'] == category][:limit // 3]]
    elif benchmark == 'seed':
        cases = [dict(r, category=str(r['question_type_id']), question=seed_prompt(r)) for r in seed]
        if limit is not None:
            cases = cases[:limit]
    else:
        raise ValueError('Unknown benchmark')
    return cases


def evaluate(workspace, config, benchmark, checkpoint=None, name='selected', limit=None):
    import torch
    from PIL import Image
    from par_repro.runtime import load_frozen, prepare_image
    from par_repro.checkpoints import load_router
    from par_repro.inference import generate
    workspace = Path(workspace)
    cases = benchmark_cases(workspace, config, benchmark, limit)
    full_count = 8910 if benchmark == 'pope' else 14233
    scope = 'FORMAL' if limit is None else 'SMOKE_ONLY'
    target = workspace / 'results' / name / (benchmark if limit is None else f'{benchmark}_smoke{len(cases)}')
    target.mkdir(parents=True, exist_ok=True)
    digest = file_sha(checkpoint) if checkpoint else None
    keys = [(str(r['category']), str(r['question_id'])) for r in cases]
    identity = {'config': config, 'checkpoint_sha256': digest, 'benchmark': benchmark, 'scope': scope,
                'count': len(cases), 'ordered_keys_sha256': object_sha(keys), 'code_sha256': code_sha()}
    bind_identity(target / 'identity.json', identity)
    path = target / 'predictions.jsonl'
    predictions = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    start = check_prediction_prefix(predictions, keys)
    if start == len(cases) and (target / 'metrics.json').exists():
        verify_evaluation(workspace, config, target)
        print('EVALUATION_ALREADY_VERIFIED', target, flush=True)
        return
    torch.manual_seed(0)
    model, tokenizer, vision = load_frozen(workspace, config)
    router, state = load_router(checkpoint) if checkpoint else (None, None)
    if state is not None:
        if state['configuration'] != config:
            raise ValueError('Evaluation config differs from checkpoint effective settings')
        if scope == 'FORMAL' and state['scope'] != 'FORMAL':
            raise ValueError('Smoke checkpoint cannot run formal benchmark')
    seed_index = index_seed_images(resolve(workspace, config['data']['seed_images'])) if benchmark == 'seed' else None
    with path.open('a') as handle:
        for i in range(start, len(cases)):
            case = cases[i]
            image_path = resolve_image(resolve(workspace, config['data']['coco_val']), case['image']) if benchmark == 'pope' else seed_index[Path(case['data_id']).stem]
            image = Image.open(image_path).convert('RGB')
            with torch.inference_mode():
                inputs = prepare_image(model, vision, tokenizer, image, case['question'])
                result = generate(model, tokenizer, router, inputs, config['generation'], audit=i < 3)
                if i == 0:
                    uncached = generate(model, tokenizer, router, inputs, dict(config['generation'], max_new_tokens=8, use_cache=False), audit=True)
                    cached = generate(model, tokenizer, router, inputs, dict(config['generation'], max_new_tokens=8), audit=True)
                    if cached['response_token_ids'] != uncached['response_token_ids']:
                        raise ValueError('Cached and uncached decoding differ')
            row = {'category': str(case['category']), 'question_id': case['question_id'], 'question': case['question'],
                   'full_prompt': inputs['prompt'], 'label': case.get('label'), 'answer': case.get('answer'),
                   'image_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(), **result}
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            predictions.append(row)
            if i < 3 or (i + 1) % 25 == 0:
                print('EVAL_PROGRESS', name, benchmark, i + 1, len(cases), flush=True)
    check_prediction_prefix(predictions, keys, complete=True)
    scores = pope_scores(predictions) if benchmark == 'pope' else seed_scores(predictions)
    metrics = {'status': 'COMPLETED', 'scope': scope, 'count': len(cases), 'checkpoint_sha256': digest,
               'predictions_sha256': file_sha(path), 'identity_sha256': object_sha(identity), **scores}
    atomic_json(target / 'metrics.json', metrics)
    verify_evaluation(workspace, config, target)
    print('EVALUATION_COMPLETE', name, benchmark, json.dumps(scores), flush=True)


def verify_evaluation(workspace, config, folder):
    folder = Path(folder)
    identity = json.loads((folder / 'identity.json').read_text())
    metrics = json.loads((folder / 'metrics.json').read_text())
    benchmark = identity['benchmark']
    limit = None if identity['scope'] == 'FORMAL' else identity['count']
    cases = benchmark_cases(workspace, config, benchmark, limit)
    rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
    check_prediction_prefix(rows, [(str(r['category']), str(r['question_id'])) for r in cases], complete=True)
    if metrics['count'] != len(cases) or metrics['checkpoint_sha256'] != identity['checkpoint_sha256'] or metrics['predictions_sha256'] != file_sha(folder / 'predictions.jsonl') or metrics['identity_sha256'] != object_sha(identity):
        raise ValueError('Metric identity/count/integrity mismatch')
    m = config['public_constraints']['M'] if identity['checkpoint_sha256'] else 0
    for row, case in zip(rows, cases):
        if row['question'] != case['question'] or row['label'] != case.get('label') or row['answer'] != case.get('answer'):
            raise ValueError('Prompt/label order changed')
        k = 4 if identity['checkpoint_sha256'] else 0
        layers, tokens = row['dropped_original_layers'], row['dropped_visual_tokens']
        if len(layers) != len(set(layers)) or len(layers) != k or any(i not in range(16, 32) for i in layers):
            raise ValueError('Invalid layer budget')
        if len(tokens) != len(set(tokens)) or len(tokens) != m or any(i not in range(576) for i in tokens):
            raise ValueError('Invalid visual budget')
        if row['retained_original_layers'] != [i for i in range(32) if i not in layers] or row['visual_tokens'] != 576 - m:
            raise ValueError('Wrong retained layers/tokens')
        logits = row['router_logits']
        if logits is not None:
            import math
            if len(logits) != 592 or not all(math.isfinite(x) for x in logits):
                raise ValueError('Invalid scores')
            if min(logits[576 + i - 16] for i in layers) < max(logits[576 + i - 16] for i in range(16, 32) if i not in layers):
                raise ValueError('Wrong layer top-k')
            if m and min(logits[i] for i in tokens) < max(logits[i] for i in range(576) if i not in tokens):
                raise ValueError('Wrong visual top-k')
    recomputed = pope_scores(rows) if benchmark == 'pope' else seed_scores(rows)
    if any(metrics[key] != value for key, value in recomputed.items()) or not all(r['execution_audited'] for r in rows[:3]):
        raise ValueError('Metric recomputation/execution audit failed')
    atomic_json(folder / 'verification.json', {'status': 'VERIFIED', 'scope': identity['scope'],
                'count': len(cases), 'metrics_sha256': file_sha(folder / 'metrics.json'), 'predictions_sha256': metrics['predictions_sha256']})
