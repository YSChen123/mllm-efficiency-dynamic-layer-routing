"""One-shot status; no polling and no credentials."""
import json,subprocess
from pathlib import Path
from run_guard import latest_complete,main_quota
ROOT=Path('/home/SigConv/cys/AdaLLaVA/training_repro')
out={'data_ready':(ROOT/'data/DATA_READY.json').is_file(),'smoke_passed':(ROOT/'smoke/SMOKE_PASSED.json').is_file()}
run=ROOT/'runs/adallava-L-20261027'
out['finished']=(run/'TRAINING_FINISHED.json').is_file()
if run.exists():
    cp=latest_complete(run);out['latest_complete_checkpoint']=str(cp) if cp else None
    file=run/'metrics_rank0.jsonl'
    if file.exists():
        lines=file.read_text().splitlines()
        out['recent_events']=[json.loads(line) for line in lines[-6:] if 'initial_fp32' not in line]
        begin=next((json.loads(line) for line in reversed(lines) if '"event": "TRAIN_BEGIN"' in line),None)
        if begin:out['run_config']={key:begin[key] for key in ['max_steps','world_size','micro_batch','accumulation','global_batch']}
for name in ['download_inputs','prepare_images','prefetch_images','smoke_fresh','smoke_resume','formal']:
    path=ROOT/'logs'/(name+'.exit')
    out[name+'_exit']=path.read_text().strip() if path.exists() else None
out['main_quota']=main_quota()
out['tmux']=subprocess.run(['tmux','list-sessions'],capture_output=True,text=True).stdout.strip()
active=subprocess.run(['tmux','has-session','-t','ada_train_formal'],capture_output=True).returncode==0
out['formal_status']='RUNNING' if active else 'FINISHED' if out['finished'] else 'FAILED' if out.get('formal_exit') not in [None,'0'] else 'NOT_STARTED'
if out['smoke_passed']:
    out['historical_smoke_exit_codes']={'fresh':out.pop('smoke_fresh_exit'),'resume':out.pop('smoke_resume_exit'),
      'recovered_and_verified':True,'note':json.loads((ROOT/'smoke/SMOKE_PASSED.json').read_text())['note']}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps(out,indent=2,ensure_ascii=False))
