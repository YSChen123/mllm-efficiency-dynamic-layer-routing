"""Resumable sequential official AdaLLaVA evaluation with durable evidence."""
import argparse
import csv
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = ['1.0', '0.85', '0.70', '0.55']
TASKS = {
    'scienceqa_img': ('lmms-lab/ScienceQA', 'ScienceQA-IMG', 'test', 'exact_match'),
    'pope': ('lmms-lab/POPE', None, 'test', 'pope_accuracy'),
    'mme': ('lmms-lab/MME', None, 'test', 'mme_perception_score'),
    'mmbench_en_dev': ('lmms-lab/MMBench', 'en', 'dev', 'gpt_eval_score'),
    'textvqa_val': ('lmms-lab/textvqa', None, 'validation', 'exact_match'),
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def validate_result(output, task, expected_count, smoke=False):
    output = Path(output)
    files = list(output.rglob('*results.json'))
    if len(files) != 1:
        raise ValueError(f'Expected one aggregate result, found {len(files)}')
    data = json.loads(files[0].read_text())
    if not smoke and data.get('config', {}).get('limit') is not None:
        raise ValueError('A limited run is not a formal result')
    metrics = data.get('results', {}).get(task, {})
    primary = TASKS[task][3]
    scores = [v for k, v in metrics.items() if k.split(',')[0] == primary]
    if not scores or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in scores):
        raise ValueError('Missing or nonfinite benchmark score')
    if task == 'mme':
        cognition = [v for k, v in metrics.items() if k.startswith('mme_cognition_score,')]
        if not cognition or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in cognition):
            raise ValueError('Missing or nonfinite MME cognition score')
    count = data.get('n-samples', {}).get(task, {}).get('effective')
    if count != expected_count:
        raise ValueError(f'Expected {expected_count} samples, aggregate reports {count}')
    sample_files = list(output.rglob(f'*samples_{task}.jsonl'))
    if len(sample_files) != 1:
        raise ValueError('Missing unique sample log')
    with sample_files[0].open() as stream:
        samples = [json.loads(line) for line in stream if line.strip()]
    if len(samples) != expected_count or len({s['doc_id'] for s in samples}) != expected_count:
        raise ValueError('Sample count mismatch or duplicate doc IDs')
    return {'result_file': str(files[0]), 'samples_file': str(sample_files[0]),
            'metrics': metrics, 'sample_count': expected_count}


def validate_measurements(output, expected_count, budget):
    records = [json.loads(line) for line in (Path(output) / 'measurements.jsonl').read_text().splitlines()]
    if len(records) != expected_count:
        raise ValueError('Measurement count differs from completed sample count')
    for row in records:
        if abs(row['target_budget'] - float(budget)) > 1e-5 or abs(row['scheduler_observed_budget'] - float(budget)) > 1e-3:
            raise ValueError('Measured requested/observed scheduler budget mismatch')
        for key in ['target_budget', 'scheduler_observed_budget', 'generation_wall_seconds', 'generation_cuda_event_ms', 'executed_layers', 'total_layers', 'layer_skipping_ratio']:
            if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f'Invalid measurement: {key}')
        if row['total_layers'] <= 0 or row['executed_layers'] > row['total_layers']:
            raise ValueError('Invalid executed layer count')
        if abs(row['layer_skipping_ratio'] - (1-row['executed_layers']/row['total_layers'])) > 1e-6:
            raise ValueError('Inconsistent layer skipping ratio')
    return records


def current_revision(task):
    manifest = ROOT / 'configs/datasets' / f'{task}.json'
    return json.loads(manifest.read_text())['revision'] if manifest.exists() else None


def completed_state(path, task, expected_revision=None):
    path = Path(path)
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if state.get('status') != 'COMPLETED' or state.get('exit_code') != 0:
        return False
    if expected_revision is not None and state.get('dataset_revision') != expected_revision:
        return False
    validate_result(state['output'], task, state['count'])
    validate_measurements(state['output'], state['count'], state['budget'])
    return True


def assert_cleanup_safe(data, datasets_root, states):
    data, datasets_root = Path(data), Path(datasets_root)
    if data.is_symlink() or data.resolve().parent != datasets_root.resolve():
        raise ValueError('Cleanup target is not a direct owned dataset directory')
    if not (data / '.adallava-owned').is_file():
        raise ValueError('Missing project ownership marker')
    if len(states) != 4 or {s.get('budget') for s in states} != set(BUDGETS):
        raise ValueError('All four budgets must be complete')
    if any(s.get('status') != 'COMPLETED' or s.get('exit_code') != 0 or not s.get('validated') for s in states):
        raise ValueError('Unvalidated or failed stage cannot be cleaned')


def environment(task):
    env = os.environ.copy()
    data = ROOT / 'datasets' / task
    env.update(CUDA_VISIBLE_DEVICES=os.getenv('ADA_GPU', '2'),
               HF_ENDPOINT='https://hf-mirror.com',
               HF_HOME=str(data / 'hub_home'),
               HF_HUB_CACHE=str(data / 'hub_home/hub'),
               HF_DATASETS_CACHE=str(data / 'arrow'),
               HF_HUB_DOWNLOAD_TIMEOUT='120', HF_HUB_ETAG_TIMEOUT='30',
               TMPDIR=str(ROOT / 'tmp'), PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', WANDB_MODE='disabled',
               PYTHONPATH=str(ROOT / 'AdaLLaVA') + ':' + str(ROOT / 'AdaLLaVA/src'))
    return env


def storage_guard(required_gib=8):
    quota = subprocess.run(['quota', '-w', '-u', 'SigConv'], capture_output=True, text=True, timeout=30)
    row = next((line.split() for line in quota.stdout.splitlines()
                if line.startswith('/dev/mapper/vg_nvme_merged_251218-lv_merged_251218')), None)
    if not row or len(row) < 4:
        raise RuntimeError('Cannot verify home quota; refusing a large download')
    used, soft = int(row[1].strip('*')), int(row[2])
    remaining = (soft - used) * 1024
    if remaining < (required_gib + 15) * 1024 ** 3:
        raise RuntimeError(f'Insufficient quota reserve: remaining={remaining / 1024**3:.2f} GiB')
    if shutil.disk_usage(ROOT).free < (required_gib + 15) * 1024 ** 3:
        raise RuntimeError('Insufficient filesystem space')
    return {'remaining_home_quota_gib': remaining / 1024 ** 3, 'reserve_gib': 15}


def update_status(status, **extra):
    previous = {}
    if status == 'FAILED' and (ROOT / 'configs/pipeline_status.json').exists():
        previous = json.loads((ROOT / 'configs/pipeline_status.json').read_text())
    obj = {**previous, 'status': status, 'updated_utc': now(), 'pid': os.getpid(), **extra}
    atomic_json(ROOT / 'configs/pipeline_status.json', obj)
    (ROOT / 'docs/RUN_STATUS.md').write_text('# Live pipeline status\n\n```json\n' + json.dumps(obj, indent=2) + '\n```\n')
    print(json.dumps(obj), flush=True)


def prepare(task):
    data = ROOT / 'datasets' / task
    data.mkdir(parents=True, exist_ok=True)
    (data / '.adallava-owned').write_text(task)
    state_path = ROOT / 'configs/states' / f'{task}_data.json'
    atomic_json(state_path, {'status': 'DOWNLOADING', 'updated_utc': now()})
    storage_guard(12)
    log = ROOT / 'logs' / f'prepare_{task}.log'
    with log.open('a') as stream:
        process = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_dataset.py'), task],
                                 env=environment(task), stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'Dataset preparation failed ({process.returncode}); see {log}')
    metadata = json.loads((ROOT / 'configs/datasets' / f'{task}.json').read_text())
    assert metadata['count'] > 0
    atomic_json(state_path, {'status': 'READY', 'updated_utc': now(), **metadata})
    return metadata['count']


def check_gpu():
    gpu = os.getenv('ADA_GPU', '2')
    out = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, utilization = map(int, out.strip().split(','))
    applications = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if memory > 1024 or applications:
        raise RuntimeError(f'GPU {gpu} is occupied; refusing to overlap another job: {out.strip()}')


def run_eval(task, budget, count, smoke=False):
    state_path = ROOT / 'configs/states' / f'{task}_{budget}.json'
    if not smoke and completed_state(state_path, task, current_revision(task)):
        print('RESUME_SKIP', task, budget, flush=True)
        return
    check_gpu()
    attempt = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / ('outputs/smoke' if smoke else 'outputs/formal') / task / budget / attempt
    out.mkdir(parents=True)
    log = ROOT / 'logs' / f'{"smoke" if smoke else "eval"}_{task}_{budget}_{attempt}.log'
    state = {'status': 'RUNNING', 'task': task, 'budget': budget, 'dataset_revision': current_revision(task), 'output': str(out),
             'log': str(log), 'count': count, 'started_utc': now()}
    if not smoke:
        atomic_json(state_path, state)
    update_status('SMOKE_RUNNING' if smoke else 'RUNNING', task=task, budget=budget, log=str(log), output=str(out))
    args = [sys.executable, str(ROOT / 'scripts/eval_entry.py'), '--model', 'adallava',
            '--model_args', f'pretrained={ROOT}/models/ada-llava-L-v1.5-7b,latency={budget},attn_implementation=sdpa',
            '--tasks', task, '--batch_size', '1', '--log_samples', '--output_path', str(out),
            '--verbosity', 'INFO']
    if smoke:
        args.extend(['--limit', str(count)])
    atomic_json(out / 'invocation.json', {'argv': args, 'smoke': smoke, 'budget': budget,
                                       'dataset': json.loads((ROOT / 'configs/datasets' / f'{task}.json').read_text())})
    env = environment(task)
    env['ADA_MEASUREMENTS'] = str(out / 'measurements.jsonl')
    env['ADA_DATASET_MANIFEST'] = str(ROOT / 'configs/datasets' / f'{task}.json')
    env['HF_HUB_OFFLINE'] = '1'
    env['HF_DATASETS_OFFLINE'] = '1'
    with log.open('w') as stream:
        start = time.monotonic()
        process = subprocess.run(args, env=env, cwd=ROOT / 'AdaLLaVA', stdout=stream, stderr=subprocess.STDOUT)
    state.update(exit_code=process.returncode, elapsed_wall_seconds=time.monotonic()-start, ended_utc=now())
    try:
        if process.returncode:
            raise RuntimeError(f'Evaluation exit code {process.returncode}; see {log}')
        validated = validate_result(out, task, count, smoke=smoke)
        validate_measurements(out, count, budget)
        state.update(status='COMPLETED', validated=True, result=validated)
    except Exception as error:
        state.update(status='FAILED', error=str(error))
        atomic_json(out / 'attempt_status.json', state)
        if not smoke:
            atomic_json(state_path, state)
        raise
    atomic_json(out / 'attempt_status.json', state)
    if not smoke:
        atomic_json(state_path, state)
    print('VALIDATED', task, budget, count, flush=True)


def collect():
    rows = []
    for task in TASKS:
        for budget in BUDGETS:
            path = ROOT / 'configs/states' / f'{task}_{budget}.json'
            failure = ROOT / 'configs/states' / f'{task}_failure.json'
            try:
                state = (json.loads(path.read_text()) if path.exists() else
                         json.loads(failure.read_text()) if failure.exists() else {'status': 'PENDING'})
            except Exception as error:
                state = {'status': 'INVALID_STATE', 'error': f'{path}: {error}'}
            if state.get('status') == 'RESOLVED':
                state = {'status': 'PENDING'}
            row = {'task': task, 'target_budget': budget, 'status': state['status'],
                   'exit_code': state.get('exit_code', 'UNAVAILABLE'), 'output': state.get('output', 'UNAVAILABLE'),
                   'error': state.get('error', '')}
            if state['status'] == 'COMPLETED':
                try:
                    result = validate_result(state['output'], task, state['count'])
                    row.update(result['metrics'])
                    row = {key: ('UNAVAILABLE' if value is None else value) for key, value in row.items()}
                    records = validate_measurements(state['output'], state['count'], budget)
                    for key in ['generation_wall_seconds', 'generation_cuda_event_ms', 'executed_layers', 'layer_skipping_ratio']:
                        row['mean_' + key] = sum(r[key] for r in records) / len(records)
                    row['measured_samples'] = len(records)
                except Exception as e:
                    row.update(status='INVALID_RESULT', error=str(e))
            rows.append(row)
    destination = ROOT / 'outputs/summary'
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / 'summary.json', rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (destination / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, restval='UNAVAILABLE')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Experiment summary', '', '| Task | Budget | Status |', '|---|---|---|']
    lines += [f"| {r['task']} | {r['target_budget']} | {r['status']} |" for r in rows]
    lines += ['', 'Target budget is an input. FLOPs and prefill_time from LLM-Viewer are analytical estimates; generation_wall_seconds and generation_cuda_event_ms are measured. Missing numeric fields are UNAVAILABLE, never zero-filled.', '', 'Full metrics: summary.csv and summary.json. Preserve outputs/formal, configs, docs, and logs for analysis.']
    (destination / 'summary.md').write_text('\n'.join(lines))
    print(destination)


def pipeline(smoke=False):
    (ROOT / 'configs').mkdir(exist_ok=True)
    with (ROOT / 'configs/pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (ROOT / 'configs/models_ready.json').is_file():
            raise RuntimeError('Model verification has not completed')
        if not smoke and not (ROOT / 'configs/smoke_passed.json').is_file():
            raise RuntimeError('Smoke tests have not passed')
        failures = []
        for task in (['scienceqa_img'] if smoke else TASKS):
            try:
                if not smoke and all(completed_state(ROOT / 'configs/states' / f'{task}_{b}.json', task, current_revision(task)) for b in BUDGETS):
                    continue
                if task == 'mmbench_en_dev' and not (os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_API_KEY')):
                    raise RuntimeError('Official MMBench GPT judge requires an API key; no key supplied. Other benchmarks continue.')
                update_status('DOWNLOADING', task=task)
                count = prepare(task)
                for budget in (['1.0', '0.85'] if smoke else BUDGETS):
                    run_eval(task, budget, min(2, count) if smoke else count, smoke)
                if smoke:
                    atomic_json(ROOT / 'configs/smoke_passed.json', {'status': 'PASSED', 'budgets': ['1.0', '0.85'], 'updated_utc': now()})
                else:
                    states = [json.loads((ROOT / 'configs/states' / f'{task}_{b}.json').read_text()) for b in BUDGETS]
                    for b in BUDGETS:
                        assert completed_state(ROOT / 'configs/states' / f'{task}_{b}.json', task, current_revision(task))
                    data = ROOT / 'datasets' / task
                    # Reclaim only if needed for subsequent data; retain adequate-space caches.
                    if os.getenv('ADA_CLEAN_COMPLETED_DATA', '1') == '1':
                        assert_cleanup_safe(data, ROOT / 'datasets', states)
                        shutil.rmtree(data)
                        atomic_json(ROOT / 'configs/states' / f'{task}_data.json', {'status': 'CLEANED_AFTER_VALIDATION', 'updated_utc': now()})
                    failure_path = ROOT / 'configs/states' / f'{task}_failure.json'
                    if failure_path.exists():
                        atomic_json(failure_path, {'status': 'RESOLVED', 'updated_utc': now()})
            except Exception as error:
                traceback.print_exc()
                failures.append({'task': task, 'error': str(error)})
                failure_name = f'{task}_{"smoke_" if smoke else ""}failure.json'
                atomic_json(ROOT / 'configs/states' / failure_name, {'status': 'FAILED', 'phase': 'SMOKE' if smoke else 'FORMAL', 'error': str(error), 'updated_utc': now()})
                if smoke:
                    raise
            finally:
                if not smoke:
                    try:
                        collect()
                    except Exception:
                        traceback.print_exc()
                        print('SUMMARY_FAILED: preserved evaluation states; continuing other tasks', flush=True)
        update_status('SMOKE_PASSED' if smoke else ('FINISHED_WITH_FAILURES' if failures else 'COMPLETED'), failures=failures)
        return 1 if failures else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'smoke', 'collect', 'storage'])
    args = parser.parse_args()
    if args.mode == 'collect':
        collect()
    elif args.mode == 'storage':
        print(json.dumps(storage_guard(0), indent=2))
    else:
        try:
            raise SystemExit(pipeline(args.mode == 'smoke'))
        except BlockingIOError:
            print('ALREADY_RUNNING: another pipeline holds the exclusive lock', file=sys.stderr)
            raise SystemExit(73)
        except Exception as error:
            update_status('FAILED', error=str(error))
            raise
