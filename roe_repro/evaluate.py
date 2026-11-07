"""Single-GPU full/dynamic evaluation with explicit provenance and completion checks."""
import argparse
import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from .common import ROOT,COMMIT,load_config,verify_upstream,environment,require_idle_gpus,write_json,checkpoint_view
from .tasks import TASKS,materialize


def run(c,args,limit):
    task=TASKS[args.task];alias=task['alias']
    phase='smoke' if limit is not None else 'formal'
    runroot=c['runtime']/'evaluation'/args.run_name/args.mode/args.task/phase
    if runroot.exists():raise FileExistsError(f'Output already exists: {runroot}; choose another --run-name')
    runroot.mkdir(parents=True)
    taskdir=materialize(runroot/'tasks')
    model={'full':'llava','dynamic':'llava_roe_protected','dynamic-fp16':'llava_roe_dynamic'}[args.mode]
    modelargs='pretrained='+str(c['checkpoint'])+',model_name=llava-v1.5-7b'
    if args.mode!='full':modelargs+=',roe_stats_dir='+str(runroot/'efficiency')
    if ',' in str(c['checkpoint']) or ',' in str(runroot):raise ValueError('Model argument paths cannot contain commas')
    command=[sys.executable,'-m','roe_repro.eval_entry','--model',model,'--model_args',modelargs,
      '--tasks',alias,'--include_path',str(taskdir),'--batch_size','1','--log_samples',
      '--output_path',str(runroot/'results')]
    if limit is not None:command+=['--limit',str(limit)]
    env=environment(c,'eval',[args.gpu]);
    if args.sync_debug:env['CUDA_LAUNCH_BLOCKING']='1'
    manifest={'state':'dry_run' if args.dry_run else 'evaluating','source_commit':COMMIT,
      'created_at':datetime.datetime.now().astimezone().isoformat(),'task':args.task,
      'task_alias':alias,'dataset_revision':task['revision'],'mode':args.mode,
      'precision':'FP16 with selective FP32 recovery' if args.mode=='dynamic' else 'FP16',
      'target_skip_ratio':0.3,'command':command,'limit':limit,'sync_debug':args.sync_debug}
    write_json(runroot/'status.json',manifest)
    print('COMMAND',command,flush=True);print('RUN_DIR',runroot,flush=True)
    if args.dry_run:return
    try:
        if not c['checkpoint'].is_dir():raise FileNotFoundError(c['checkpoint'])
        view=checkpoint_view(c,runroot)
        command[command.index('--model_args')+1]=modelargs.replace('pretrained='+str(c['checkpoint']),
            'pretrained='+str(view),1)
        manifest['gpu_selection']=require_idle_gpus([args.gpu]);write_json(runroot/'status.json',manifest)
        with (runroot/'run.log').open('w',encoding='utf-8') as log:
            completed=subprocess.run(command,env=env,cwd=runroot,stdout=log,stderr=subprocess.STDOUT)
        files=list((runroot/'results').rglob('*_results.json'))
        if completed.returncode or len(files)!=1:
            raise RuntimeError(f'Evaluation did not produce a unique successful result; inspect {runroot/"run.log"}')
        data=json.loads(files[0].read_text());expected=limit if limit is not None else task['count']
        assert data['n-samples'][alias]['effective']==expected
        assert data['config']['limit']==(float(limit) if limit is not None else None)
        metrics=data['results'][alias]
        assert all(math.isfinite(v) for v in metrics.values() if isinstance(v,(int,float)))
        final=dict(manifest,state='completed_smoke' if limit is not None else 'completed',
            samples=expected,metrics=metrics,result=str(files[0]),
            completed_at=datetime.datetime.now().astimezone().isoformat())
        if args.mode!='full':
            stats=json.loads((runroot/'efficiency/summary.json').read_text())
            assert stats['state']=='generated' and stats['samples']==expected
            assert stats['router_calls']==32*expected and stats['total']==32*expected
            assert stats['skipped']+stats['full']==stats['total'] and stats['skipped_samples']==0
            assert stats['skipped']>0,'No actual skipping; do not call this dynamic validation'
            if limit is not None:assert stats['empty_answers']==0,'Empty smoke answer; inspect before formal evaluation'
            final.update(actual_skip_rate=stats['actual_skip_rate'],precision_retries=stats['precision_retries'],
                empty_answers=stats['empty_answers'])
        write_json(runroot/'status.json',final);print('COMPLETED',json.dumps(final),flush=True)
    except Exception as error:
        write_json(runroot/'status.json',dict(manifest,state='failed',error=repr(error)))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.local.json')
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--mode',choices=['full','dynamic','dynamic-fp16'],default='dynamic')
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--run-name',default='reproduction')
    p.add_argument('--limit',type=int);p.add_argument('--smoke-first',action='store_true')
    p.add_argument('--sync-debug',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();c=load_config(args.config);verify_upstream()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.run_name) or args.run_name in ['.','..']:
        raise ValueError('Invalid run name')
    if args.limit is not None and args.limit<=0:raise ValueError('--limit must be positive')
    if args.limit is not None and args.smoke_first:raise ValueError('Use --limit or --smoke-first, not both')
    if args.smoke_first:run(c,args,16 if args.task=='scienceqa_img' else 8)
    run(c,args,args.limit)


if __name__=='__main__':main()
