#!/usr/bin/env bash
set -u
ROOT=/home/SigConv/cys/AdaLLaVA
printf '\nTMUX\n'
tmux list-sessions 2>/dev/null || true
printf '\nPIPELINE\n'
cat "$ROOT/configs/pipeline_status.json" 2>/dev/null || echo NOT_STARTED
printf '\nSTAGES\n'
python3 - "$ROOT" <<'PY'
import json,sys,pathlib,os
root=pathlib.Path(sys.argv[1])
for path in sorted((root/'configs/states').glob('*.json')):
 data=json.loads(path.read_text())
 print(path.stem, data.get('status'), 'exit=',data.get('exit_code','N/A'),data.get('error',''))
path=root/'configs/pipeline_status.json'
if path.exists():
 data=json.loads(path.read_text())
 pid=data.get('pid')
 print('RECORDED_PID',pid,'ALIVE',bool(pid and pathlib.Path('/proc',str(pid)).exists()))
 print('LATEST_LOG',data.get('log','N/A'))
 print('OUTPUT',data.get('output','N/A'))
PY
printf '\nSTORAGE\n'
df -hT "$ROOT" /home/SigConv/runs
quota -w -u SigConv 2>/dev/null || true
printf '\nGPU\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
