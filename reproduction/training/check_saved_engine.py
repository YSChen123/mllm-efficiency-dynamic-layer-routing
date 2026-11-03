"""Read metadata from a trusted own ZeRO checkpoint using CPU mmap, no model load."""
import datetime
import inspect
import json
import sys
from pathlib import Path
import torch
from accelerate import Accelerator
from deepspeed.runtime.config import get_gradient_clipping
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

root = Path('/home/SigConv/cys/AdaLLaVA/training_repro')
sys.path.insert(0, str(root))
from run_guard import latest_complete
cp = latest_complete(root / 'runs/adallava-L-20261027')
assert cp is not None
model_files = list(cp.rglob('*model_states.pt'))
assert model_files
metadata = torch.load(model_files[0], map_location='cpu', weights_only=False, mmap=True)
keys = ['global_steps', 'global_samples', 'skipped_steps', 'dp_world_size', 'mp_world_size', 'ds_version']
result = {'checkpoint': str(cp), 'file': str(model_files[0]),
          'metadata': {key: metadata.get(key) for key in keys},
          'engine_config': metadata.get('ds_config'),
          'effective_gradient_clipping': get_gradient_clipping(metadata.get('ds_config') or {})}
assert result['metadata']['global_steps'] == int(cp.name.split('-')[1])
assert result['metadata']['skipped_steps'] == 0
assert result['metadata']['global_samples'] == result['metadata']['global_steps'] * 128
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
folder = root / 'audit' / ('engine_metadata_' + stamp)
folder.mkdir()
(folder / 'metadata.json').write_text(json.dumps(result, indent=2, default=str))
functions = [Accelerator.clip_grad_norm_, HfTrainerDeepSpeedConfig.fill_match,
             DeepSpeedEngine._backward_prologue, DeepSpeedEngine._scale_loss_by_gas,
             DeepSpeedEngine.is_gradient_accumulation_boundary, get_gradient_clipping]
(folder / 'effective_gradient_sources.txt').write_text('\n\n'.join(inspect.getsource(f) for f in functions))
print(json.dumps({'folder': str(folder), 'result': result}, default=str))
