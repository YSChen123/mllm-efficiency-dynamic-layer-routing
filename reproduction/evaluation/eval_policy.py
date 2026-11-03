"""Pure validation and sample tracking shared by the isolated evaluation run."""
from contextlib import contextmanager
import math


def evaluation_module(role):
    return 'lmms_eval' if role=='dense' else 'adallava.eval.run_lmms_eval'


def best_effort_collect(collect, on_error):
    try:
        collect()
        return True
    except Exception as error:
        on_error(error)
        return False


def expected_layers(budget):
    if not math.isfinite(budget) or not .5 <= budget <= 1:
        raise ValueError('AdaLLaVA-L budget must be finite and within [.5,1]')
    return math.floor(32 * budget + 1e-7)


def validate_training_rows(rows, total_steps=5197):
    starts = [i for i,row in enumerate(rows) if row['event']=='ENTRY' and row.get('mode')=='formal']
    if not starts: raise ValueError('No formal training entry')
    rows = rows[starts[-1]:]
    if any(row['event']=='FAILED' for row in rows):raise ValueError('Latest attempt failed')
    steps = [row['step'] for row in rows if row['event']=='STEP']
    if steps != list(range(1,total_steps+1)):raise ValueError('Incomplete or duplicate update sequence')
    losses = [row for row in rows if row['event']=='LOG' and 'loss' in row]
    if len(losses)!=total_steps or not all(math.isfinite(row['loss']) for row in losses):
        raise ValueError('Missing or nonfinite training losses')
    if not any(row['event']=='TRAIN_LOOP_END' and row['step']==total_steps and row['max_steps']==total_steps for row in rows):
        raise ValueError('Training loop did not reach total steps')
    if rows[-1]['event']!='TRAINING_TOTAL':raise ValueError('Export did not finish normally')
    return True


def validate_telemetry(rows, count, budget, role):
    if len(rows)!=count or len({row['doc_id'] for row in rows})!=count:
        raise ValueError('Incomplete or duplicate telemetry samples')
    target = 32 if role=='dense' else expected_layers(float(budget))
    for row in rows:
        if abs(row['target_budget']-float(budget))>1e-6:raise ValueError('Wrong requested budget')
        if abs(row['scheduler_observed_budget']-float(budget))>1e-3:raise ValueError('Wrong observed budget')
        for key in ['generation_wall_seconds','generation_cuda_event_ms','lm_prefill_cuda_ms','vision_plus_lm_prefill_cuda_ms']:
            if not math.isfinite(row[key]) or row[key]<=0:raise ValueError('Missing or invalid timing: '+key)
        mask=row['layer_execution_mask']
        if len(mask)!=32 or any(value not in [0,1] for value in mask) or mask[:16]!=[1]*16:
            raise ValueError('Invalid layer execution mask')
        if row['total_layers']!=32 or row['executed_layers']!=target or sum(mask)!=target:
            raise ValueError('Routing violates quantized budget')
    return rows


@contextmanager
def track_requests(requests, on_doc):
    saved=[]
    try:
        for request in requests:
            original=request.arguments
            context,kwargs,visual,doc_id,task,split=original
            def tracked(doc,visual=visual,doc_id=doc_id,task=task,split=split):
                on_doc({'doc_id':doc_id,'task':task,'split':split})
                return visual(doc)
            saved.append((request,original))
            request.arguments=(context,kwargs,tracked,doc_id,task,split)
        yield
    finally:
        for request,original in saved:request.arguments=original
