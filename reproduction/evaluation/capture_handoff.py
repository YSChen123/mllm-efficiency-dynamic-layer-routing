"""Capture small launch evidence and validate reusable old official scores."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path('/home/SigConv/cys/AdaLLaVA')
EVAL=ROOT/'evaluation_repro/20261031'
sys.path.insert(0,str(ROOT/'scripts'))
import pipeline as p
baseline=[]
for task in ['scienceqa_img','pope','mme','textvqa_val']:
    metadata=json.loads((ROOT/'configs/datasets'/f'{task}.json').read_text())
    for budget in ['1.0','0.85','0.70','0.60','0.55']:
        path=ROOT/'configs/states'/f'{task}_{budget}.json'
        state=json.loads(path.read_text())
        assert p.completed_state(path,task,metadata['revision'])
        baseline.append({'task':task,'budget':budget,'state_path':str(path),
            'state_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'sample_count':state['count'],'metrics':state['result']['metrics'],
            'output':state['output'],'reuse_scope':'accuracy; old analytical prefill is not a new measured latency'})
(EVAL/'official_reused_20.json').write_text(json.dumps(baseline,indent=2))
evidence=EVAL/'launch_evidence';evidence.mkdir(exist_ok=True)
for name in ['training_summary.json','reproduced_model.json','SMOKE_own_PASSED.json','SMOKE_reference_PASSED.json',
             'REVIEW_PASSED.json','environment_freeze.txt','mmbench_status.json','official_reused_20.json','unit_tests.log']:
    shutil.copyfile(EVAL/name,evidence/name)
for lane in ['own','reference']:
    state=json.loads((EVAL/f'status_{lane}.json').read_text())
    assert state['status']=='RUNNING' and Path('/proc',str(state['pid'])).exists()
    measure=Path(state['output'])/'measurements.jsonl'
    rows=[json.loads(line) for line in measure.read_text().splitlines()]
    assert len(rows)>=20,'Formal generation has not progressed far enough'
    state['completed_generations']=len(rows)
    state['last_measurement']=rows[-1]
    state['measurement_file']=str(measure)
    (evidence/f'status_{lane}.json').write_text(json.dumps(state,indent=2))
    print(lane,state['role'],state['task'],'progress',len(rows),'/',2017,'last_doc',rows[-1]['doc_id'])
(evidence/'gpu_snapshot.csv').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True))
archive=shutil.make_archive(str(EVAL/'launch_evidence'),'gztar',evidence)
print(json.dumps({'archive':archive,'bytes':Path(archive).stat().st_size,'reused_official_results':len(baseline)}))
