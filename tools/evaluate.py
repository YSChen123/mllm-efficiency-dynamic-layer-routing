"""Run the preserved evaluator against a freshly downloaded or retrained model."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()

def prepare_role(root,out,role,model,tasks):
    root=Path(root).resolve();out=Path(out).resolve();model=Path(model).resolve()
    if out==root or out==model or out in model.parents:
        raise ValueError('Evaluation output must be separate from runtime and model')
    idx=model/'model.safetensors.index.json'
    if not idx.exists():idx=model/'pytorch_model.bin.index.json'
    files=sorted(set(json.loads(idx.read_text())['weight_map'].values()))
    manifest={'role':role,'model_path':str(model),'config_sha256':sha(model/'config.json'),
        'files':{name:{'bytes':(model/name).stat().st_size,'sha256':sha(model/name)} for name in files}}
    if role=='reproduced':
        finish=json.loads((model/'TRAINING_FINISHED.json').read_text());step=finish['step']
        cp=model/f'checkpoint-{step}'
        complete=json.loads((cp/'COMPLETE.json').read_text());assert complete['step']==step
        manifest['master_samples']=json.loads((cp/'proof_rank0.json').read_text())['fp32']
        manifest['step']=step
    identity=out/'RUN_IDENTITY.json'
    if identity.exists() and json.loads(identity.read_text())!=manifest:
        raise ValueError('Model identity differs; select a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    for directory in ['scripts','models','configs/datasets','configs/states','logs','tmp','docs','outputs']:
        (out/directory).mkdir(parents=True,exist_ok=True)
    for name,target in [('AdaLLaVA',root/'AdaLLaVA'),('datasets',root/'datasets'),('models/ada-llava-L-v1.5-7b',model)]:
        link=out/name
        if link.is_symlink():
            if link.resolve()!=target.resolve():raise ValueError(f'Unexpected existing model/source link: {link}')
        elif link.exists():raise ValueError(f'Refusing to replace existing path: {link}')
        else:link.symlink_to(target,target_is_directory=True)
    for task in tasks:
        data=json.loads((root/'configs/datasets'/f'{task}.json').read_text())
        for files in data['arrow_files'].values():
            for file in files:
                if not Path(file).is_file():raise FileNotFoundError(f'Prepare pinned dataset first: {file}')
        (out/'configs/datasets'/f'{task}.json').write_text(json.dumps(data,indent=2))
    for name in ['eval_entry.py','eval_policy.py']:
        source=root/'evaluation_repro/scripts'/name
        if source.exists():shutil.copyfile(source,out/'scripts'/name)
    source=root/'scripts/local_dataset.py'
    if source.exists():shutil.copyfile(source,out/'scripts/local_dataset.py')
    identity.write_text(json.dumps(manifest,indent=2))
    (out/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--role',choices=['reproduced','official','dense'],required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--tasks',nargs='+',default=['scienceqa_img','pope','mme','textvqa_val'])
    parser.add_argument('--budgets',nargs='+',default=['1.0','0.9','0.85','0.8','0.7','0.6','0.55','0.5'])
    parser.add_argument('--smoke',action='store_true',help='Only three samples per task/budget; cannot mark a formal test complete')
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    if args.role=='dense':args.budgets=['1.0']
    if any(not .5<=float(x)<=1 for x in args.budgets):parser.error('Budget must be in [0.5,1.0]')
    prepare_role(root,out,args.role,args.model,args.tasks)
    sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(root/'evaluation_repro/scripts'))
    import pipeline as p
    from eval_policy import validate_telemetry,best_effort_collect
    p.ROOT=out;p.TASKS={task:p.TASKS[task] for task in args.tasks};p.BUDGETS=args.budgets
    original_env=p.environment;original_validate=p.validate_measurements
    def environment(task):
        env=original_env(task);env.update(ADA_ROLE=args.role,ADA_GPU=args.gpu,CUDA_VISIBLE_DEVICES=args.gpu,
            ADA_MODEL_MANIFEST=str(out/'model_manifest.json'))
        return env
    p.environment=environment
    def validate(output,count,budget):
        rows=original_validate(output,count,budget)
        return validate_telemetry(rows,count,float(budget),args.role)
    p.validate_measurements=validate
    os.environ['ADA_GPU']=args.gpu
    failures=[]
    for task in args.tasks:
        count=json.loads((out/'configs/datasets'/f'{task}.json').read_text())['count']
        for budget in args.budgets:
            try:p.run_eval(task,budget,min(3,count) if args.smoke else count,smoke=args.smoke)
            except Exception as error:failures.append({'task':task,'budget':budget,'error':repr(error)})
    best_effort_collect(p.collect,lambda error:print('SUMMARY_COLLECTION_FAILED',repr(error),file=sys.stderr))
    p.atomic_json(out/'EVALUATION_STATUS.json',{'status':'FAILED' if failures else 'SMOKE_PASSED' if args.smoke else 'COMPLETED','failures':failures})
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
