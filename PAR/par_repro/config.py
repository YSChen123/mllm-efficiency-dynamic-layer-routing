from pathlib import Path
import json
from par_repro.io import package_root


def load_config(path=None):
    path = Path(path) if path else package_root() / 'configs/joint.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    constraints = cfg['public_constraints']
    mode = cfg['training']['mode']
    if mode not in ['joint', 'layer_only'] or constraints['K'] != 4 or constraints['M'] != (400 if mode == 'joint' else 0):
        raise ValueError('Supported public budgets: Joint K4/M400 or Layer-only K4/M0')
    if constraints['training_count'] != 5000 or constraints['actions_per_input'] != 5 or cfg['training']['beta'] != 1:
        raise ValueError('Public training constraints changed')
    if cfg['generation'] != {'max_new_tokens': 128, 'do_sample': False, 'num_beams': 1, 'use_cache': True}:
        raise ValueError('Unsupported reference decoding settings')
    return cfg


def resolve(workspace, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(workspace) / path).resolve()
