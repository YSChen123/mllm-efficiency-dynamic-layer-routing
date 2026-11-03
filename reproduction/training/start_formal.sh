#!/usr/bin/env bash
set -euo pipefail
cd /home/SigConv/cys/AdaLLaVA
action=${1:-fresh}
envs/adallava-train/bin/python - "$action" <<'PY'
import datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,'training_repro')
from run_guard import data_ready,latest_complete,main_quota,has_capacity
root=Path('training_repro')
ready=json.loads((root/'data/DATA_READY.json').read_text());assert data_ready(ready)
proof=json.loads((root/'smoke/SMOKE_PASSED.json').read_text());assert proof['passed']
quota=main_quota()
needed=proof['checkpoint_bytes']*(1 if sys.argv[1]=='resume' else 2)
assert has_capacity(quota['soft'],quota['used'],needed,10*1024**3),('Insufficient two-checkpoint peak reserve',quota)
run=root/'runs/adallava-L-20261027'
if sys.argv[1]=='resume':assert latest_complete(run) is not None,'No complete formal checkpoint'
else:assert not run.exists() or not list(run.glob('checkpoint-*')),'Use resume for existing checkpoints'
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
for line in gpus.splitlines():
 index,uuid,free=[p.strip() for p in line.split(',')]
 if index in ['0','1']:assert int(free)>90000 and uuid not in apps,('Selected GPU became occupied',index)
snapshot=run/'provenance'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
snapshot.mkdir(parents=True,exist_ok=False)
hashes={}
for name in ['train_entry.py','run_train.sh','zero3.json','run_guard.py','rng_compat.py','optimizer_audit.py','accumulation_compat.py','managed_train.sh']:
 source=root/name;shutil.copyfile(source,snapshot/name);hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(snapshot/'sha256.json').write_text(json.dumps(hashes,indent=2))
(snapshot/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
(snapshot/'preflight.json').write_text(json.dumps({'quota':quota,'data_ready':ready,'smoke':proof,'gpu':gpus,'action':sys.argv[1]},indent=2))
print('FORMAL_PREFLIGHT_PASSED',quota)
PY
if tmux has-session -t ada_train_formal 2>/dev/null;then echo FORMAL_SESSION_EXISTS;exit 1;fi
stamp=$(date +%Y%m%d_%H%M%S)
log=/home/SigConv/cys/AdaLLaVA/training_repro/logs/formal_${stamp}.log
ln -sfn "$(basename "$log")" training_repro/logs/formal.log
rm -f -- training_repro/logs/formal.exit
tmux new-session -d -s ada_train_formal "cd /home/SigConv/cys/AdaLLaVA; bash training_repro/managed_train.sh $action > $log 2>&1"
echo "FORMAL_STARTED $log"
