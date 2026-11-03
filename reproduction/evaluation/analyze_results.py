"""Revalidate complete experiments and compare accuracy/observed latency faithfully."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path('/home/SigConv/cys/AdaLLaVA')
EVAL=ROOT/'evaluation_repro/20261031'
DEST=EVAL/'analysis';DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
import pipeline as p
from eval_policy import validate_telemetry

TASKS=['scienceqa_img','pope','mme','textvqa_val']
BUDGETS=['1.0','0.90','0.85','0.80','0.70','0.60','0.55','0.50']
METRICS={'scienceqa_img':'exact_match,none','pope':'pope_f1_score,none',
         'mme':'mme_perception_score,none','textvqa_val':'exact_match,none'}
all_rows=[];measurements={}

def add(role,task,budget,path,new):
    state=json.loads(path.read_text())
    assert state['status']=='COMPLETED' and state['exit_code']==0
    data=json.loads((ROOT/'configs/datasets'/f'{task}.json').read_text())
    assert state['dataset_revision']==data['revision'] and state['count']==data['count']
    result=p.validate_result(state['output'],task,state['count'])
    records=p.validate_measurements(state['output'],state['count'],budget)
    assert {r['doc_id'] for r in records}==set(range(state['count'])) if new else True
    if new:validate_telemetry(records,state['count'],float(budget),role)
    metrics=result['metrics']
    value=metrics[METRICS[task]]*(1 if task=='mme' else 100)
    assert math.isfinite(value)
    row={'role':role,'task':task,'budget':float(budget),'count':state['count'],
         'primary_metric':METRICS[task],'score':value,'all_metrics':metrics,'new_eval':new,
         'state_path':str(path),'result_file':result['result_file'],'samples_file':result['samples_file'],
         'result_sha256':hashlib.sha256(Path(result['result_file']).read_bytes()).hexdigest(),
         'elapsed_wall_seconds':state['elapsed_wall_seconds']}
    if task=='pope':row['accuracy_percent']=metrics['pope_accuracy,none']*100
    if task=='mme':row['cognition_score']=metrics['mme_cognition_score,none']
    if new:
        for r in records:assert r['role']==role and r['task']==task
        warm=[r for r in records if not r['warmup_exclusion_candidate']]
        assert len(warm)==state['count']-3
        row['observed']={'warm_samples':len(warm),'gpu_indices':sorted({r['gpu_index'] for r in warm}),
            'mean_generate_ms':statistics.mean(r['generation_wall_seconds']*1000 for r in warm),
            'median_generate_ms':statistics.median(r['generation_wall_seconds']*1000 for r in warm),
            'mean_lm_prefill_ms':statistics.mean(r['lm_prefill_cuda_ms'] for r in warm),
            'mean_vision_plus_prefill_ms':statistics.mean(r['vision_plus_lm_prefill_cuda_ms'] for r in warm),
            'mean_output_tokens':statistics.mean(r['generated_len'] for r in warm),
            'mean_prompt_tokens':statistics.mean(r['expanded_prompt_tokens'] for r in warm),
            'mean_executed_layers':statistics.mean(r['executed_layers'] for r in records),
            'budget_satisfaction_rate':sum(r['budget_satisfied'] for r in records)/len(records),
            'maximal_budget_utilization_rate':sum(r['maximal_quantized_budget_utilization'] for r in records)/len(records),
            'peak_allocated_mib':max(r['cuda_peak_allocated_mib'] for r in records),
            'peak_reserved_mib':max(r['cuda_peak_reserved_mib'] for r in records)}
        for field in ['scheduler_forward_cuda_ms','scheduler_latency_encoding_cuda_ms']:
            if all(field in r for r in warm):row['observed']['mean_'+field]=statistics.mean(r[field] for r in warm)
        measurements[(role,task,float(budget))]={r['doc_id']:r for r in warm}
    all_rows.append(row)

for role in ['reproduced','dense','official']:
    files=sorted((EVAL/role/'configs/states').glob('*.json'))
    expected=32 if role=='reproduced' else 4 if role=='dense' else 12
    assert len(files)==expected,(role,len(files))
    for path in files:
        state=json.loads(path.read_text());add(role,state['task'],state['budget'],path,True)
for task in TASKS:
    for budget in ['1.0','0.85','0.70','0.60','0.55']:
        add('official',task,budget,ROOT/'configs/states'/f'{task}_{budget}.json',False)
assert len(all_rows)==68
indexed={(r['role'],r['task'],r['budget']):r for r in all_rows}
comparison=[]
for task in TASKS:
    dense=indexed['dense',task,1.]
    for budget in map(float,BUDGETS):
        own=indexed['reproduced',task,budget];official=indexed['official',task,budget]
        own_records=measurements['reproduced',task,budget]
        dense_records=measurements['dense',task,1.]
        ids=sorted(own_records.keys() & dense_records.keys())
        assert len(ids)>=own['count']-6
        pair={}
        for field,label in [('generation_wall_seconds','generate'),('lm_prefill_cuda_ms','lm_prefill'),
                            ('vision_plus_lm_prefill_cuda_ms','vision_plus_prefill')]:
            a=statistics.mean(own_records[i][field] for i in ids)
            b=statistics.mean(dense_records[i][field] for i in ids)
            pair[label+'_speedup_vs_dense']=b/a
        comparison.append({'task':task,'budget':budget,'dense':dense['score'],'official':official['score'],
            'reproduced':own['score'],'delta_vs_official':own['score']-official['score'],
            'delta_vs_dense':own['score']-dense['score'],'paired_timing_samples':len(ids),**pair})

out={'captured_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
     'validated_new_experiments':48,'validated_reused_experiments':20,'rows':all_rows,
     'comparison':comparison,'training':json.loads((EVAL/'training_summary.json').read_text()),
     'timing_notes':['Exclude first3 generation samples from every job; pair doc_id with Dense.',
         'Observed on same GPU model, different physical GPUs; not locked-clock controlled profiling.',
         'Full generate includes vision/prefill/decode, excludes CPU preprocessing/tokenization.',
         'Output lengths differ. Whole-generate speedup is observed, not fixed-token throughput.',
         'Old official20 scores reused for accuracy only; no new measured prefill for those points.']}
(DEST/'validated_results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'validated':68,'score_order':TASKS,
    'dense':[indexed['dense',t,1.]['score'] for t in TASKS],
    'by_budget':{b:{role:[indexed[role,t,float(b)]['score'] for t in TASKS] for role in ['official','reproduced']} for b in BUDGETS},
    'timing_comparison':[r for r in comparison if r['budget'] in [1.,.85,.6,.5]]},indent=2))
