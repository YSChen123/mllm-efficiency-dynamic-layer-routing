"""Gate formal launch on observed gradients, updates, complete saves and actual resume."""
import json
from pathlib import Path
from run_guard import latest_complete
ROOT=Path('/home/SigConv/cys/AdaLLaVA/training_repro')
run=ROOT/'smoke/run'
proofs=[]
for rank in [0,1]:
    events=[json.loads(line) for line in (run/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
    start=max(i for i,row in enumerate(events) if row['event']=='ENTRY' and not row['resume'])
    events=events[start:]
    resumed=max(i for i,row in enumerate(events) if row['event']=='ENTRY' and row['resume'])
    assert not any(row['event']=='FAILED' for row in events[resumed:]),'Latest resume sequence contains a failure'
    for row in events[:resumed]:
        if row['event']=='FAILED':
            assert row['error_type']=='AttributeError' and row['error']=="'NoneType' object has no attribute 'state_dict'",row
    for name in ['scheduler.mlp_head.weight','model.norm.weight']:
        grads=[row['gradient_evidence'][name] for row in events if row['event']=='STEP' and name in row.get('gradient_evidence',{})]
        assert grads and all(row['finite'] for row in grads) and any(row['abs_sum']>0 for row in grads),(rank,name,'gradient')
        assert any(row.get('parameter_updates',{}).get(name,0)>0 for row in events),(rank,name,'update')
    for name in ['RESUME_WEIGHTS_OPTIMIZER_VERIFIED','RESUME_RNG_VERIFIED','SAVE_CALLBACK_REVALIDATED']:
        assert any(row['event']==name for row in events),(rank,name)
    assert any(row['event']=='TRAIN_LOOP_END' and row['step']==3 for row in events),rank
    proofs.append({'rank':rank,'verified':True})
cp=latest_complete(run);assert cp is not None
assert json.loads((ROOT/'smoke/EXPORT_RELOAD_PASSED.json').read_text())['passed'],'Portable export/reload failed'
saved=json.loads((cp/'COMPLETE.json').read_text())
had_failure=any(row['event']=='FAILED' for row in events)
begin=next(row for row in events if row['event']=='TRAIN_BEGIN')
result={'passed':True,'checkpoint':str(cp),'checkpoint_bytes':saved['bytes'],'rank_proofs':proofs,
        'accumulation':begin['accumulation'],'global_batch':begin['global_batch'],
        'note':'Recovered prior audit error; live save/resume verified.' if had_failure else 'Clean gradient-accumulation smoke: save, actual resume and portable export/reload verified.'}
(ROOT/'smoke/SMOKE_PASSED.json').write_text(json.dumps(result,indent=2))
print('SMOKE_PASSED',json.dumps(result))
