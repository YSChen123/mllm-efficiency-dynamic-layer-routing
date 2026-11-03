import json
import os
import sys
from pathlib import Path
import time

from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfApi

from pipeline import ROOT, TASKS, atomic_json, storage_guard
from recovery_helpers import retry_hub

task = sys.argv[1]
repo, name, split, metric = TASKS[task]
cache = ROOT / 'datasets' / task / 'arrow'
manifest = ROOT / 'configs/datasets' / f'{task}.json'
if manifest.exists():
    recorded = json.loads(manifest.read_text())
    assert (recorded['repo'], recorded['config'], recorded['split']) == (repo, name, split), 'Dataset identity changed'
    revision = recorded['revision']
else:
    revision = retry_hub(lambda: HfApi().dataset_info(repo).sha)
kwargs = dict(name=name, cache_dir=str(cache), token=False, revision=revision)
if task == 'textvqa_val':
    # Zero-shot evaluation uses only validation; retain the exact pinned source.
    kwargs['data_files'] = {split: f'hf://datasets/{repo}@{revision}/data/{split}-*.parquet'}
builder = retry_hub(lambda: load_dataset_builder(repo, **kwargs))
download_size = builder.info.download_size
dataset_size = builder.info.dataset_size
if download_size is not None and dataset_size is not None:
    storage_guard(max(2, (download_size + dataset_size) / 1024**3))
if task == 'textvqa_val':
    # The Hub card describes all three splits, even with data_files restricted.
    # Keep normal integrity verification, scoped to the requested split.
    from datasets import SplitDict
    builder.info.splits = SplitDict({split: builder.info.splits[split]}, dataset_name=builder.info.dataset_name)
    retry_hub(lambda: builder.download_and_prepare())
    data = builder.as_dataset()
else:
    data = retry_hub(lambda: load_dataset(repo, **kwargs))
assert split in data, f'Missing split: {split}'
selected = data[split]
assert len(selected) > 0
first = selected[0]
if 'image' in first and first['image'] is not None:
    first['image'].convert('RGB').load()
info = {'task': task, 'repo': repo, 'config': name, 'split': split,
        'revision': revision, 'count': len(selected), 'features': str(selected.features),
        'fingerprint': selected._fingerprint,
        'download_size': download_size, 'dataset_size': dataset_size,
        'info_version': str(selected.info.version), 'cache': str(cache),
        'splits': {key: len(value) for key, value in data.items()},
        'arrow_files': {key: [item['filename'] for item in value.cache_files] for key, value in data.items()},
        'prepared_unix_time': time.time()}
if task == 'textvqa_val':
    assert len(selected) == 5000, 'Official validation count changed'
    info['preparation_scope'] = 'validation-only at unchanged pinned revision'
    baseline_state = json.loads((ROOT / 'configs/states/textvqa_val_1.0.json').read_text())
    baseline = [json.loads(line) for line in Path(baseline_state['result']['samples_file']).read_text().splitlines() if line.strip()]
    columns = selected.select_columns(['question_id', 'question', 'ocr_tokens'])
    assert len(baseline) == len(columns)
    for old, current in zip(sorted(baseline, key=lambda row: row['doc_id']), columns):
        for key in ['question_id', 'question', 'ocr_tokens']:
            assert old['doc'][key] == current[key], f'Baseline validation mismatch at {old["doc_id"]}: {key}'
    info['baseline_identity_verified_samples'] = len(baseline)
atomic_json(ROOT / 'configs/datasets' / f'{task}.json', info)
# Make evaluation resolve the exact same revision that was prepared offline.
task_files = {'scienceqa_img': 'scienceqa/scienceqa_img.yaml', 'pope': 'pope/pope.yaml',
              'mme': 'mme/mme.yaml', 'mmbench_en_dev': 'mmbench/_default_template_mmbench_en_yaml',
              'textvqa_val': 'textvqa/_default_template_textvqa_yaml'}
path = ROOT / 'lmms-eval/lmms_eval/tasks' / task_files[task]
text = path.read_text()
import re
text = re.sub(r'^  revision:.*\n', '', text, flags=re.MULTILINE)
if 'dataset_kwargs:\n' in text:
    text = text.replace('dataset_kwargs:\n', 'dataset_kwargs:\n  revision: ' + revision + '\n', 1)
else:
    text = 'dataset_kwargs:\n  token: False\n  revision: ' + revision + '\n' + text
path.write_text(text)
print(json.dumps(info), flush=True)
