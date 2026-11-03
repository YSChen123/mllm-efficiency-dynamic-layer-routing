set -euo pipefail
cd /home/SigConv/cys/AdaLLaVA
action=${1:-fresh}
envs/adallava/bin/python - <<'PY'
import json,subprocess
from pathlib import Path
root=Path.cwd();base=root/'checkpoints/llava-v1.5-7b';view=root/'training_repro/models/llava-v1.5-7b';view.mkdir(parents=True,exist_ok=True)
for path in base.iterdir():
 if path.name.startswith('.') or path.name=='config.json':continue
 dest=view/path.name
 if not dest.exists():dest.symlink_to(path)
config=json.load(open(base/'config.json'));config['mm_vision_tower']=str(root/'checkpoints/clip-vit-large-patch14-336')
(view/'config.json').write_text(json.dumps(config,indent=2))
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
for line in gpus.splitlines():
 index,uuid,free=[p.strip() for p in line.split(',')]
 if index in ['0','1']:
  assert int(free)>90000,(index,free)
  assert uuid not in apps,(index,apps)
print('CLEAN_GPU_PREFLIGHT_PASSED')
PY
if tmux has-session -t ada_train_smoke 2>/dev/null;then echo ALREADY_RUNNING;exit 0;fi
stamp=$(date +%Y%m%d_%H%M%S)
log=/home/SigConv/cys/AdaLLaVA/training_repro/logs/smoke_${action}_${stamp}.log
old=training_repro/logs/smoke_${action}.log
if [[ -f $old && ! -L $old ]];then mv -- "$old" "${old%.log}_nccl2262.log";fi
ln -sfn "$(basename "$log")" "$old"
rm -f -- "training_repro/logs/smoke_${action}.exit"
tmux new-session -d -s ada_train_smoke "cd /home/SigConv/cys/AdaLLaVA; bash training_repro/run_train.sh smoke $action > $log 2>&1; echo \$? > training_repro/logs/smoke_${action}.exit"
echo SMOKE_STARTED
