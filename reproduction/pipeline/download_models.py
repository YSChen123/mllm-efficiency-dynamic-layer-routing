import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/SigConv/cys/AdaLLaVA')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = str(ROOT / 'cache/model_hub')
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
from huggingface_hub import hf_hub_download

metadata = json.loads((ROOT / 'configs/checkpoint_metadata.json').read_text())
checkpoint = ROOT / 'checkpoints/ada-llava-L-v1.5-7b'
for item in metadata['siblings']:
    name = item['rfilename']
    if not (name.endswith(('.json', '.model', '.safetensors'))):
        continue
    path = Path(hf_hub_download(metadata['id'], name, revision=metadata['sha'], local_dir=checkpoint))
    assert path.stat().st_size == item['size'], name
    if item.get('lfs'):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        assert digest.hexdigest() == item['lfs']['sha256'], f'SHA256 mismatch: {name}'
    print('VERIFIED', name, path.stat().st_size, flush=True)

# Per-file official Hub downloads avoid fetching TensorFlow duplicate weights.
vision = ROOT / 'checkpoints/clip-vit-large-patch14-336'
vision_files = ['config.json', 'preprocessor_config.json', 'pytorch_model.bin']
for name in vision_files:
    path = hf_hub_download('openai/clip-vit-large-patch14-336', name, local_dir=vision)
    print('VISION_DOWNLOADED', name, Path(path).stat().st_size, flush=True)

# Resolve the required vision model locally without modifying downloaded config.
alias = ROOT / 'models/ada-llava-L-v1.5-7b'
alias.mkdir(exist_ok=True)
for path in checkpoint.iterdir():
    if path.is_file() and path.name != 'config.json' and not (alias / path.name).exists():
        (alias / path.name).symlink_to(path)
config = json.loads((checkpoint / 'config.json').read_text())
config['mm_vision_tower'] = str(vision)
(alias / 'config.json').write_text(json.dumps(config, indent=2))
vision_alias = ROOT / 'models/clip-vit-large-patch14-336'
if not vision_alias.exists():
    vision_alias.symlink_to(vision, target_is_directory=True)
index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
assert any('scheduler' in key for key in index['weight_map']), 'Scheduler weights missing'
for shard in set(index['weight_map'].values()):
    assert (checkpoint / shard).is_file(), shard
json.dump({'checkpoint_revision': metadata['sha'], 'checkpoint': str(checkpoint),
           'vision': str(vision), 'vision_files': vision_files},
          (ROOT / 'configs/models_ready.json').open('w'), indent=2)
print('ALL_MODELS_READY', flush=True)
