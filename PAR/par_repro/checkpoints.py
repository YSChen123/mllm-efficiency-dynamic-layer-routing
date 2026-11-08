from pathlib import Path
from par_repro.io import atomic_json, file_sha, object_sha, model_resource_identity


def validate_legacy_configuration(effective, identity, config):
    if ':scope=formal:count=5000:' not in identity and ':count=5000:scope=formal:' not in identity:
        raise ValueError('Only authenticated formal5000 historical checkpoints may be converted')
    if 'base' in effective:
        if config['training']['mode'] != 'joint' or effective['base']['router'] != config['router']:
            raise ValueError('Conversion mode/router mismatch')
        trial = effective['trial']
        old_settings = effective['base'].get('training', {})
        search = effective.get('search', {})
        normalized = {'attention_mode': trial.get('attention_mode', 'bidirectional'),
            'global_batch_size': trial.get('global_batch_size', old_settings.get('global_batch_size', 4)),
            'learning_rate': trial.get('learning_rate', search.get('learning_rate', old_settings.get('learning_rate', .0001))),
            'pairing': trial.get('pairing', search.get('pairing', 'extremes'))}
        for field, value in normalized.items():
            if value != config['training'][field]:
                raise ValueError('Conversion effective setting mismatch: ' + field)
    else:
        if effective['router'] != config['router']:
            raise ValueError('Conversion router mismatch')
        variant = effective.get('experiment_variant')
        if variant not in ['a', 'b'] or (variant == 'b') != (config['training']['mode'] == 'layer_only'):
            raise ValueError('Conversion variant/mode mismatch')
        if config['training']['attention_mode'] != 'bidirectional' or config['training']['pairing'] != 'balanced':
            raise ValueError('Historical v3 router uses bidirectional mask and balanced pairs')
        for field in ['global_batch_size', 'learning_rate', 'epochs']:
            if effective.get('training', {}).get(field) != config['training'][field]:
                raise ValueError('Historical v3 training setting mismatch: ' + field)


def load_router(path):
    import torch
    from par_repro.router import ControlledRouter
    from par_repro._paper_router import PaperRouterConfig
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('portable_format_version') != 1:
        raise ValueError('Convert historical checkpoint explicitly before portable inference')
    config = state['configuration']
    router = ControlledRouter(PaperRouterConfig(**config['router']), config['training']['attention_mode'])
    router.load_state_dict(state['router'], strict=True)
    router.cuda().float().eval().requires_grad_(False)
    router.prune_m = config['public_constraints']['M']
    return router, state


def convert_checkpoint(source, destination, config, workspace):
    import torch
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Converted checkpoint already exists')
    old = torch.load(source, map_location='cpu', weights_only=True)
    effective = old['configuration']
    validate_legacy_configuration(effective, old['dataset_identity'], config)
    state = {'portable_format_version': 1, 'configuration': config, 'router': old['router'], 'step': old['step'],
             'model_resources': model_resource_identity(workspace), 'epoch': old.get('epoch'),
             'identity': old['dataset_identity'], 'scope': 'FORMAL', 'converted_from_sha256': file_sha(source)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(destination)
    atomic_json(destination.with_suffix('.provenance.json'), {k: v for k, v in state.items() if k != 'router'})
    print('CONVERTED_CHECKPOINT', destination.name, file_sha(destination), flush=True)
