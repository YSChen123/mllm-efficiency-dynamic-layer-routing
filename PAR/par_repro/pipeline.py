from pathlib import Path
import os
import sys
import json
import subprocess
import copy
from par_repro.io import atomic_json, package_root


def run(workspace, config, gpus, smoke=False):
    workspace = Path(workspace).resolve()
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('Unique GPU IDs required')
    cfg = copy.deepcopy(config)
    if smoke:
        cfg['training'].update(epochs=4, reported_epochs=[1, 4], global_batch_size=4)
    actual_config = workspace / 'configs/run.json'
    atomic_json(actual_config, cfg)
    def stage(value):
        atomic_json(workspace / 'pipeline_state.json', {'stage': value, 'scope': 'SMOKE_ONLY' if smoke else 'FORMAL'})
    def command(script_args, gpu, config_path=actual_config, log=None):
        argv = [sys.executable, '-u', '-m', 'par_repro', '--work-dir', str(workspace), '--config', str(config_path), *script_args]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
        return subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT if log else None)
    def checked(args, gpu=gpus[0], config_path=actual_config):
        p = command(args, gpu, config_path)
        if p.wait():
            raise RuntimeError('Pipeline command failed: ' + repr(args))
    def parallel(jobs):
        pending = list(jobs)
        while pending:
            current = []
            for gpu in gpus:
                if not pending:
                    break
                args, conf, label = pending.pop(0)
                log_path = workspace / 'logs' / f'{label}.log'
                log_path.parent.mkdir(exist_ok=True)
                log = log_path.open('a')
                current.append((command(args, gpu, conf, log), log, label))
            failures = []
            for p, log, label in current:
                result = p.wait()
                log.close()
                if result:
                    failures.append(label)
            if failures:
                raise RuntimeError('Pipeline workers failed: ' + ', '.join(failures))
    try:
        for split in ['training', 'validation']:
            stage('PREFERENCES_' + split.upper())
            jobs = []
            for shard in range(len(gpus)):
                args = ['preferences', '--split', split, '--shard', str(shard), '--shards', str(len(gpus))]
                if smoke:
                    args += ['--limit', '4']
                jobs.append((args, actual_config, f'preferences_{split}_{shard}'))
            parallel(jobs)
            checked(['finalize-preferences', '--split', split])
        stage('TRAINING')
        if smoke or cfg['training']['mode'] == 'layer_only':
            name = 'smoke' if smoke else 'layer_only'
            args = ['train', '--name', name]
            if (workspace / 'checkpoints' / name / 'latest.pt').exists():
                args += ['--resume']
            checked(args)
            checkpoint = workspace / 'checkpoints' / name / ('latest.pt' if smoke else f"epoch{cfg['training']['epochs']}.pt")
            selected_config = actual_config
        else:
            search_path = package_root() / 'configs/search.json'
            search = json.loads(search_path.read_text())
            jobs = []
            for trial in search['trials']:
                trial_config = copy.deepcopy(cfg)
                trial_config['training'].update({key: trial[key] for key in ['attention_mode', 'global_batch_size']})
                conf = workspace / 'configs' / (trial['id'] + '.json')
                atomic_json(conf, trial_config)
                args = ['train', '--name', trial['id']]
                if (workspace / 'checkpoints' / trial['id'] / 'latest.pt').exists():
                    args += ['--resume']
                jobs.append((args, conf, 'train_' + trial['id']))
            parallel(jobs)
            stage('VALIDATION_SELECT')
            checked(['select', '--search', str(search_path)])
            selected = json.loads((workspace / 'selection_result.json').read_text())['selected']
            checkpoint = workspace / selected['checkpoint']
            selected_config = workspace / 'configs' / (selected['trial_id'] + '.json')
        for benchmark in ['pope', 'seed']:
            stage('TEST_' + benchmark.upper())
            args = ['evaluate', '--benchmark', benchmark, '--checkpoint', str(checkpoint), '--name', 'selected']
            if smoke:
                args += ['--limit', '6' if benchmark == 'pope' else '4']
            checked(args, config_path=selected_config)
        stage('COMPLETED')
    except BaseException:
        stage('FAILED')
        raise
