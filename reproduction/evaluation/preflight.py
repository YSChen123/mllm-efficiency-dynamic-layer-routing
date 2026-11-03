"""Verify completed training and prepare independent, immutable evaluation roots."""
import csv
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from eval_policy import validate_training_rows

ROOT=Path('/home/SigConv/cys/AdaLLaVA')
EVAL=ROOT/'evaluation_repro/20261031'
RUN=ROOT/'training_repro/runs/adallava-L-20261027'
EVAL.mkdir(parents=True,exist_ok=True)
assert (ROOT/'training_repro/logs/formal.exit').read_text().strip()=='0'
finish=json.loads((RUN/'TRAINING_FINISHED.json').read_text());assert finish['step']==5197
cp=RUN/'checkpoint-5197'
complete=json.loads((cp/'COMPLETE.json').read_text());assert complete['step']==5197
proof=json.loads((cp/'proof_rank0.json').read_text())
cost={'finished':finish,'checkpoint':str(cp),'checkpoint_bytes':complete['bytes'],'ranks':[]}
for rank in [0,1]:
    rows=[json.loads(line) for line in (RUN/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
    validate_training_rows(rows)
    start=max(i for i,row in enumerate(rows) if row['event']=='ENTRY' and row.get('mode')=='formal')
    rows=rows[start:]
    steps=[row for row in rows if row['event']=='STEP']
    logs=[row for row in rows if row['event']=='LOG' and 'loss' in row]
    totals=rows[-1]
    cost['ranks'].append({'rank':rank,'steps':len(steps),'total':totals,
        'peak_allocated_bytes':max(row['peak_allocated_bytes'] for row in steps),
        'peak_reserved_bytes':max(row['peak_reserved_bytes'] for row in steps),
        'last_loss':logs[-1]['loss'],'mean_step_loss':sum(row['loss'] for row in logs)/len(logs)})
    if rank==0:
        for filename in ['metrics_rank0.jsonl','metrics_rank1.jsonl']:
            shutil.copyfile(RUN/filename,EVAL/filename)
        (EVAL/'training_loss.json').write_text(json.dumps(logs))
cost['training_flops']='NOT_MEASURED'
(EVAL/'training_summary.json').write_text(json.dumps(cost,indent=2))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()

idx=json.loads((RUN/'model.safetensors.index.json').read_text())
shards=sorted(set(idx['weight_map'].values()))
assert len(shards)==3
file_hashes={}
for name in shards:
    path=RUN/name;assert path.is_file() and path.stat().st_size>10**9
    file_hashes[name]={'bytes':path.stat().st_size,'sha256':sha(path)}
samples={}
for name,values in proof['fp32'].items():
    with safe_open(RUN/idx['weight_map'][name],framework='pt',device='cpu') as file:
        actual=file.get_tensor(name).flatten()[:len(values)]
        wanted=torch.tensor(values,dtype=torch.bfloat16)
        assert torch.equal(actual,wanted),(name,'Export/master mismatch')
        samples[name]=values
manifest={'role':'reproduced','model_path':str(RUN),'step':5197,'files':file_hashes,
          'config_sha256':sha(RUN/'config.json'),'master_samples':samples,
          'verified_export_matches_final_checkpoint':True}
(EVAL/'reproduced_model.json').write_text(json.dumps(manifest,indent=2))

tasks=['scienceqa_img','pope','mme','textvqa_val']
datasets={}
sys.path.insert(0,str(ROOT/'scripts'))
from local_dataset import load_prepared_dataset
for task in tasks:
    data=json.loads((ROOT/'configs/datasets'/f'{task}.json').read_text())
    loaded=load_prepared_dataset(data)
    assert len(loaded[data['split']])==data['count']
    loaded[data['split']][0]['image'].convert('RGB').load()
    datasets[task]=data
roles={'reproduced':RUN,'dense':ROOT/'training_repro/models/llava-v1.5-7b',
       'official':ROOT/'models/ada-llava-L-v1.5-7b'}
for role,model in roles.items():
    root=EVAL/role
    for dirname in ['scripts','models','configs/datasets','configs/states','logs','tmp','docs','outputs']:
        (root/dirname).mkdir(parents=True,exist_ok=True)
    for name,target in [('AdaLLaVA',ROOT/'AdaLLaVA'),('datasets',ROOT/'datasets')]:
        link=root/name
        if not link.exists():link.symlink_to(target,target_is_directory=True)
    alias=root/'models/ada-llava-L-v1.5-7b'
    if not alias.exists():alias.symlink_to(model,target_is_directory=True)
    for task,metadata in datasets.items():
        (root/'configs/datasets'/f'{task}.json').write_text(json.dumps(metadata,indent=2))
    shutil.copyfile(ROOT/'scripts/local_dataset.py',root/'scripts/local_dataset.py')
    detail=manifest if role=='reproduced' else {'role':role,'model_path':str(model.resolve()),
        'config_sha256':sha(model/'config.json')}
    (root/'model_manifest.json').write_text(json.dumps(detail,indent=2))
    for source in ['eval_entry.py','eval_policy.py']:
        shutil.copyfile(Path(__file__).parent/source,root/'scripts'/source)

judge=bool(os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_API_KEY'))
(EVAL/'mmbench_status.json').write_text(json.dumps({'status':'JUDGE_AVAILABLE' if judge else 'BLOCKED_JUDGE_CREDENTIAL',
    'reason':'Official MMBench-EN Dev GPT grading requires configured credentials; no fabricated score.',
    'other_benchmarks_continue':True},indent=2))
print(json.dumps({'passed':True,'root':str(EVAL),'shards':file_hashes,'training_total':cost['ranks'][0]['total'],
                  'dataset_counts':{task:d['count'] for task,d in datasets.items()},'judge_configured':judge}))
