from pathlib import Path
import json
import hashlib
from par_repro.io import atomic_json, bind_identity, file_sha, code_sha, object_sha, package_root, model_resource_identity
from par_repro.config import resolve
from par_repro.data import resolve_image


def preference_folder(workspace, mode, split):
    return Path(workspace) / 'preferences' / mode / split


def build_preferences(workspace, config, split='training', limit=None, shard=0, shards=1):
    import torch
    from PIL import Image
    from par_repro.runtime import load_frozen, prepare_image
    from par_repro.pruning import ranking_logits, noisy_actions, forward_preserved, balanced_indices
    from par_repro.math_core import kl_full_to_pruned, preference_extremes
    workspace = Path(workspace)
    selection = workspace / 'selection' / f'{split}.json'
    rows = json.loads(selection.read_text())
    count = len(rows) if limit is None else min(limit, len(rows))
    if not 0 <= shard < shards or count <= 0:
        raise ValueError('Invalid shard/count')
    mode = config['training']['mode']
    m = config['public_constraints']['M']
    folder = preference_folder(workspace, mode, split)
    (folder / 'items').mkdir(parents=True, exist_ok=True)
    (folder / 'features').mkdir(exist_ok=True)
    identity = {'mode': mode, 'split': split, 'count': count, 'scope': 'FORMAL' if count == len(rows) else 'SMOKE_ONLY',
                'selection_sha256': file_sha(selection), 'code_sha256': code_sha(), 'M': m,
                'generation': config['generation'], 'model_resources': model_resource_identity(workspace),
                'attention_backend': config['model']['attention_backend'], 'noise_variance': 10.,
                'likelihood': 'independent_Bernoulli', 'KL': 'Full||pruned_response_token_mean_including_EOS'}
    bind_identity(folder / 'identity.json', identity)
    digest = object_sha(identity)
    pending = []
    for index in range(shard, count, shards):
        path = folder / f'items/{index:05d}.json'
        if path.exists():
            item = json.loads(path.read_text())
            if item['identity'] != digest or file_sha(folder / item['feature']) != item['feature_sha256']:
                raise ValueError('Existing preference identity/features changed')
        else:
            pending.append(index)
    if not pending:
        print('PREFERENCES_SHARD_COMPLETE', mode, split, shard, count, flush=True)
        return
    torch.manual_seed(0)
    model, tokenizer, vision = load_frozen(workspace, config)
    image_root = resolve(workspace, config['data']['coco_train'])
    canonical_file = package_root() / 'configs/image_fingerprints.json'
    canonical = json.loads(canonical_file.read_text())['fingerprints'][split] if canonical_file.exists() else None
    for done, index in enumerate(pending):
        raw = rows[index]['record']
        question = next(t['value'] for t in raw['conversations'] if t['from'] == 'human').replace('<image>', '').strip()
        image_path = resolve_image(image_root, raw['image'])
        image = Image.open(image_path).convert('RGB')
        rgb_digest = hashlib.sha256(image.tobytes()).hexdigest()
        if canonical is not None:
            reference = canonical[index]
            if rows[index]['source_index'] != reference['source_index'] or list(image.size) != reference['size'] or rgb_digest != reference['rgb_sha256']:
                raise ValueError('Canonical decoded training/validation image differs; check dataset and Pillow version')
        with torch.inference_mode():
            inputs = prepare_image(model, vision, tokenizer, image, question)
            prefix, start = inputs['prefix'], inputs['image_start']
            generated = model.generate(inputs['ids'], images=inputs['pixels'], image_sizes=[image.size],
                **config['generation'], return_dict_in_generate=True, output_scores=True)
            length = len(generated.scores)
            if length <= 0:
                raise ValueError('Empty Full teacher response')
            response = generated.sequences[:, -length:]
            teacher = torch.cat([prefix, model.get_input_embeddings()(response)], dim=1)
            heads = ranking_logits(model, prefix).cpu()
            action_seed = index if split == 'training' else 20261108 + index
            actions = noisy_actions(heads, start, action_seed)
            if mode == 'layer_only':
                for action in actions:
                    action['tokens'] = []
            begin = prefix.shape[1] - 1
            full, _, _ = forward_preserved(model, teacher, start, [], [])
            full = full[:, begin:begin + length].contiguous()
            for action in actions:
                output, _, trace = forward_preserved(model, teacher, start, action['layers'], action['tokens'])
                action['kl'] = float(kl_full_to_pruned(full, output[:, begin - m:begin - m + length].contiguous()))
                action['trace'] = [list(pair) for pair in trace]
            pair = preference_extremes([a['kl'] for a in actions])
            sets = balanced_indices([a['kl'] for a in actions])
            if pair is None or sets is None:
                atomic_json(folder / f'unresolved_{index:05d}.json', {'index': index, 'actions': actions})
                raise ValueError('Unresolved preferences; no sample may be silently removed')
            feature = folder / f'features/{index:05d}.pt'
            pack = {'visual': inputs['visual'][0].cpu(), 'text': inputs['text'][0].cpu(), 'head_logits': heads,
                    'response_token_ids': response[0].cpu(), 'image_start': start, 'prefix_length': prefix.shape[1]}
            temporary = feature.with_suffix('.tmp')
            torch.save(pack, temporary)
            temporary.replace(feature)
        item = {'identity': digest, 'index': index, 'source_index': rows[index]['source_index'], 'source_id': str(raw['id']),
                'image_name': raw['image'], 'image_file_sha256': file_sha(image_path),
                'image_rgb_sha256': rgb_digest, 'image_size': list(image.size),
                'question': question, 'full_prompt': inputs['prompt'], 'response_token_ids': response[0].cpu().tolist(),
                'image_start': start, 'prefix_length': prefix.shape[1], 'action_seed': action_seed, 'actions': actions,
                'chosen_index': pair[0], 'rejected_index': pair[1], 'positive_indices': sets[0], 'negative_indices': sets[1],
                'feature': str(feature.relative_to(folder)), 'feature_sha256': file_sha(feature), 'backbone_frozen': True,
                'router_text_excludes_response': True, 'actual_attention_backend': model.model.layers[0].self_attn.__class__.__name__}
        atomic_json(folder / f'items/{index:05d}.json', item)
        if done < 3 or (done + 1) % 20 == 0:
            print('PREFERENCE_PROGRESS', mode, split, shard, index, done + 1, len(pending), flush=True)
        del teacher, full, output, inputs, prefix, response
    print('PREFERENCES_SHARD_COMPLETE', mode, split, shard, count, flush=True)


def finalize_preferences(workspace, mode, split):
    folder = preference_folder(workspace, mode, split)
    identity = json.loads((folder / 'identity.json').read_text())
    digest = object_sha(identity)
    files = {}
    for index in range(identity['count']):
        path = folder / f'items/{index:05d}.json'
        item = json.loads(path.read_text())
        feature = (folder / item['feature']).resolve()
        if item['identity'] != digest or not feature.is_relative_to(folder.resolve()) or file_sha(feature) != item['feature_sha256']:
            raise ValueError('Preference identity or integrity mismatch')
        files[str(path.relative_to(folder))] = file_sha(path)
        files[item['feature']] = item['feature_sha256']
    manifest = {'status': 'VERIFIED', 'identity': digest, 'count': identity['count'], 'scope': identity['scope'], 'files': files}
    atomic_json(folder / 'manifest.json', manifest)
    print('PREFERENCES_VERIFIED', mode, split, identity['count'], flush=True)
    return manifest


def checked_preferences(workspace, mode, split):
    folder = preference_folder(workspace, mode, split)
    manifest = json.loads((folder / 'manifest.json').read_text())
    identity = json.loads((folder / 'identity.json').read_text())
    if manifest['status'] != 'VERIFIED' or manifest['identity'] != object_sha(identity):
        raise ValueError('Invalid preference manifest')
    if identity['model_resources'] != model_resource_identity(workspace):
        raise ValueError('Preference model/source resource identity differs')
    for name, digest in manifest['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or file_sha(path) != digest:
            raise ValueError('Preference changed: ' + name)
    return folder, manifest
