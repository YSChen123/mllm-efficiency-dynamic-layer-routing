"""Reuse the verified evaluator, keeping all new model results in separate roots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path('/home/SigConv/cys/AdaLLaVA')
EVAL=ROOT/'evaluation_repro/20261031'
sys.path.insert(0,str(ROOT/'scripts'))
import pipeline as p
from eval_policy import validate_telemetry, best_effort_collect

BUDGETS=['1.0','0.85','0.70','0.60','0.55','0.90','0.80','0.50']
EXTRA=['0.90','0.80','0.50']
TASKS=['scienceqa_img','pope','mme','textvqa_val']
original_environment=p.environment
original_validate=p.validate_measurements
ROLE=None
LANE=None

def environment(task):
    env=original_environment(task)
    env.update(ADA_ROLE=ROLE,ADA_MODEL_MANIFEST=str(p.ROOT/'model_manifest.json'))
    env.pop('ADA_SHARED_GPU',None);env.pop('ADA_CUDA_MEMORY_GIB',None)
    return env

def validate_measurements(output,count,budget):
    rows=original_validate(output,count,budget)
    return validate_telemetry(rows,count,float(budget),ROLE)

def update_status(status,**extra):
    row={'status':status,'role':ROLE,'lane':LANE,'pid':os.getpid(),'updated_utc':p.now(),**extra}
    p.atomic_json(EVAL/f'status_{LANE}.json',row)
    print(json.dumps(row),flush=True)

def select_role(role):
    global ROLE
    ROLE=role;p.ROOT=EVAL/role
    p.TASKS={task:p.TASKS[task] for task in TASKS}
    p.BUDGETS=BUDGETS if role=='reproduced' else ['1.0'] if role=='dense' else EXTRA
    data=(p.ROOT/'model_manifest.json').read_bytes()
    identity={'role':role,'manifest_sha256':hashlib.sha256(data).hexdigest()}
    path=p.ROOT/'RUN_IDENTITY.json'
    if path.exists():assert json.loads(path.read_text())==identity,'Model identity changed; use a new result root'
    else:p.atomic_json(path,identity)

def status():
    obj={'mmbench':json.loads((EVAL/'mmbench_status.json').read_text()),'lanes':{},'models':{}}
    for lane in ['own','reference']:
        path=EVAL/f'status_{lane}.json'
        if path.exists():
            row=json.loads(path.read_text());row['pid_alive']=Path('/proc',str(row['pid'])).exists();obj['lanes'][lane]=row
    for role in ['reproduced','dense','official']:
        rows=[json.loads(path.read_text()) for path in (EVAL/role/'configs/states').glob('*.json')]
        obj['models'][role]={'completed':sum(row.get('status')=='COMPLETED' for row in rows),
                             'failed':sum(row.get('status')=='FAILED' for row in rows),
                             'running':[row for row in rows if row.get('status')=='RUNNING']}
    print(json.dumps(obj,indent=2))

def main(mode,lane,gpu):
    global LANE
    LANE=lane
    if mode=='status':status();return 0
    os.environ['ADA_GPU']=gpu
    p.environment=environment;p.validate_measurements=validate_measurements;p.update_status=update_status
    if mode=='smoke':
        jobs=([('reproduced','1.0'),('reproduced','0.50')] if lane=='own'
              else [('dense','1.0'),('official','0.85')])
        for role,budget in jobs:
            select_role(role);p.run_eval('scienceqa_img',budget,3,smoke=True)
        p.atomic_json(EVAL/f'SMOKE_{lane}_PASSED.json',{'passed':True,'jobs':jobs,'updated_utc':p.now()})
        update_status('SMOKE_PASSED');return 0
    assert (EVAL/'SMOKE_own_PASSED.json').is_file() and (EVAL/'SMOKE_reference_PASSED.json').is_file()
    assert (EVAL/'REVIEW_PASSED.json').is_file(),'Fresh review not complete'
    if lane=='own':jobs=[('reproduced',task,budget) for task in TASKS for budget in BUDGETS]
    else:jobs=[('dense',task,'1.0') for task in TASKS]+[('official',task,budget) for task in TASKS for budget in EXTRA]
    failures=[]
    for role,task,budget in jobs:
        select_role(role)
        count=json.loads((p.ROOT/'configs/datasets'/f'{task}.json').read_text())['count']
        try:p.run_eval(task,budget,count)
        except Exception as error:
            traceback.print_exc();failures.append({'role':role,'task':task,'budget':budget,'error':str(error)})
        finally:best_effort_collect(p.collect,lambda error:print('SUMMARY_FAILED_PRESERVED_STATES_CONTINUING',str(error),flush=True))
    update_status('FINISHED_WITH_FAILURES' if failures else 'COMPLETED',failures=failures)
    return int(bool(failures))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','run','status'])
    parser.add_argument('--lane',choices=['own','reference'],default='own')
    parser.add_argument('--gpu',default='2')
    args=parser.parse_args()
    raise SystemExit(main(args.mode,args.lane,args.gpu))
