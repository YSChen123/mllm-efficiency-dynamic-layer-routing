"""Snapshot small reproducibility evidence; never copy model/optimizer tensors."""
import datetime
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

root = Path('/home/SigConv/cys/AdaLLaVA/training_repro')
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
dest = root / 'handoff' / stamp
dest.mkdir(parents=True, exist_ok=False)
run = root / 'runs/adallava-L-20261027'
checks = []
for rank in [0, 1]:
    rows = [json.loads(line) for line in (run / f'metrics_rank{rank}.jsonl').read_text().splitlines()]
    start = max(i for i, row in enumerate(rows) if row['event'] == 'ENTRY')
    rows = rows[start:]
    assert not any(row['event'] == 'FAILED' for row in rows), 'Latest formal attempt failed'
    steps = [row for row in rows if row['event'] == 'STEP']
    losses = [row for row in rows if row['event'] == 'LOG' and 'loss' in row]
    assert steps[-1]['step'] >= 5 and len(losses) >= 5, 'Formal stability window not complete'
    assert all(math.isfinite(row['loss']) for row in losses)
    checks.append({'rank': rank, 'last_step': steps[-1], 'losses': losses})
status = subprocess.check_output([sys.executable, str(root / 'training_status.py')], text=True)
assert json.loads(status)['formal_status'] == 'RUNNING'
(dest / 'formal_status.json').write_text(status)
(dest / 'stability_checks.json').write_text(json.dumps(checks, indent=2))
sources = list((root / 'audit').glob('*'))
sources += list((root / 'logs').glob('*.log')) + list((root / 'logs').glob('*.exit'))
sources += list((root / 'logs').glob('gpu_*.csv'))
sources += list((root / 'data').glob('*verified.json'))
sources += [root / 'data/DATA_READY.json', root / 'data/instruction_manifest.json']
sources += list((root / 'smoke').glob('*PASSED.json'))
sources += list((root / 'smoke/evidence').rglob('*'))
sources += list((run / 'provenance').rglob('*'))
sources += list(run.glob('metrics_rank*.jsonl'))
inventory = []
for source in sorted(set(sources)):
    if not source.is_file():
        continue
    assert source.stat().st_size < 64 * 1024**2, ('Unexpected large evidence', source)
    target = dest / source.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    inventory.append({'path': str(source.relative_to(root)), 'bytes': target.stat().st_size,
                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(dest / 'inventory.json').write_text(json.dumps(inventory, indent=2))
archive = dest.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(dest, arcname=stamp)
print(json.dumps({'path': str(archive), 'files': len(inventory), 'bytes': archive.stat().st_size,
                  'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
