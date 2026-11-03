"""Read-only audit of current formal training; never initializes a process group/model."""
import dataclasses
import datetime
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/SigConv/cys/AdaLLaVA')
TR = ROOT / 'training_repro'
sys.path.insert(0, str(TR))
sys.path.insert(0, str(TR / 'source/AdaLLaVA/src'))
sys.path.insert(0, str(ROOT / 'LLaVA'))
import torch
from run_guard import latest_complete
from adallava.train.train import TrainingArguments
from llava.train.llava_trainer import LLaVATrainer, LengthGroupedSampler
from transformers import Trainer
from accelerate import Accelerator
from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
from deepspeed.runtime.engine import DeepSpeedEngine

run = TR / 'runs/adallava-L-20261027'
cp = latest_complete(run)
assert cp is not None
args = torch.load(cp / 'training_args.bin', map_location='cpu', weights_only=False)
keys = ['num_train_epochs', 'max_steps', 'per_device_train_batch_size', 'gradient_accumulation_steps',
        'learning_rate', 'weight_decay', 'lr_scheduler_type', 'warmup_ratio', 'warmup_steps',
        'optim', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm', 'seed', 'data_seed',
        'bf16', 'fp16', 'tf32', 'gradient_checkpointing', 'gradient_checkpointing_kwargs',
        'model_max_length', 'group_by_modality_length', 'dataloader_num_workers', 'dataloader_pin_memory',
        'dataloader_drop_last', 'save_strategy', 'save_steps', 'save_total_limit', 'bits', 'lora_enable',
        'tune_mm_mlp_adapter', 'freeze_mm_mlp_adapter', 'mm_projector_lr', 'ddp_find_unused_parameters',
        'label_smoothing_factor', 'remove_unused_columns', 'deepspeed']
out = {'captured_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
       'checkpoint': str(cp), 'actual_training_args': {key: getattr(args, key, '<absent>') for key in keys}}
ds = getattr(args, 'hf_deepspeed_config', None)
out['resolved_deepspeed_config'] = ds.config if ds is not None else None
state = json.loads((cp / 'trainer_state.json').read_text())
out['saved_trainer'] = {key: state.get(key) for key in ['global_step', 'max_steps', 'num_train_epochs', 'epoch', 'train_batch_size']}
out['saved_complete'] = json.loads((cp / 'COMPLETE.json').read_text())
out['checkpoint_config'] = json.loads((cp / 'config.json').read_text())
out['sources'] = {}
for repo in [TR / 'source/AdaLLaVA', ROOT / 'LLaVA']:
    out['sources'][str(repo)] = {
        'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'tracked_diff': subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--stat'], text=True),
    }
out['entry_source_files'] = {name: hashlib.sha256((TR / name).read_bytes()).hexdigest()
    for name in ['train_entry.py', 'run_train.sh', 'zero3.json', 'rng_compat.py', 'optimizer_audit.py', 'accumulation_compat.py']}
proof = json.loads((cp / 'proof_rank0.json').read_text())
out['lr_scheduler_state'] = proof['lr_scheduler']
out['optimizer_probe'] = proof['optimizer_probe']
out['ranks'] = []
for rank in [0, 1]:
    rows = [json.loads(line) for line in (run / f'metrics_rank{rank}.jsonl').read_text().splitlines()]
    start = max(i for i, row in enumerate(rows) if row['event'] == 'ENTRY')
    rows = rows[start:]
    begin = next(row for row in rows if row['event'] == 'TRAIN_BEGIN')
    entry = rows[0]
    steps = [row for row in rows if row['event'] == 'STEP']
    losses = [row for row in rows if row['event'] == 'LOG' and 'loss' in row]
    out['ranks'].append({'rank': rank, 'entry': entry,
        'begin': {key: value for key, value in begin.items() if key not in ['initial_fp32', 'optimizer_probe']},
        'latest_step': steps[-1], 'recent_losses': losses[-3:],
        'all_logged_losses_finite': all(math.isfinite(row['loss']) for row in losses),
        'failures': [row for row in rows if row['event'] == 'FAILED'],
        'complete_saves': [row for row in rows if row['event'] == 'CHECKPOINT_COMPLETE']})
out['data_ready'] = json.loads((TR / 'data/DATA_READY.json').read_text())
out['import_paths'] = {'Trainer': inspect.getfile(Trainer), 'LLaVATrainer': inspect.getfile(LLaVATrainer),
                       'TrainingArguments': inspect.getfile(TrainingArguments)}
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
dest = TR / 'audit' / ('strict_settings_' + stamp)
dest.mkdir()
code = []
for method in [Trainer.training_step, Accelerator.backward, Accelerator.accumulate,
               DeepSpeedEngineWrapper.backward, DeepSpeedEngine.backward,
               LLaVATrainer.create_optimizer, LLaVATrainer._get_train_sampler, LengthGroupedSampler.__iter__]:
    code.append('\n### ' + method.__qualname__ + '\n' + inspect.getsource(method))
(dest / 'gradient_and_optimizer_sources.txt').write_text('\n'.join(code))
(dest / 'actual_settings.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({'folder': str(dest), 'checkpoint': str(cp), 'args': out['actual_training_args'],
                  'saved_step': state['global_step'], 'source_status': out['sources'],
                  'rank_steps': [row['latest_step']['step'] for row in out['ranks']],
                  'resolved_ds': out['resolved_deepspeed_config']}, default=str))
