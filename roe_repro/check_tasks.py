"""Check pinned aliases against original tasks without decoding image payloads."""
import argparse
import hashlib
import os
import sys
from .common import UPSTREAM,load_config,verify_upstream,environment,write_json
from .tasks import TASKS,materialize


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.local.json');args=p.parse_args()
    c=load_config(args.config);verify_upstream();os.environ.update(environment(c,'eval'))
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(UPSTREAM/'lmms-eval'))
    from lmms_eval.tasks import TaskManager,get_task_dict
    from lmms_eval.utils import load_yaml_config
    directory=materialize(c['runtime']/'task_validation/tasks')
    aliases=get_task_dict([t['alias'] for t in TASKS.values()],TaskManager(include_path=str(directory)))
    references=[]
    for spec in TASKS.values():
        config=load_yaml_config(str(UPSTREAM/'lmms-eval/lmms_eval/tasks'/spec['yaml']))
        config['dataset_kwargs']=dict(config.get('dataset_kwargs') or {},token=False,revision=spec['revision'])
        references.append(config)
    originals=get_task_dict(references,TaskManager())
    reports=[]
    for name,spec in TASKS.items():
        task=aliases[spec['alias']];original=originals['seedbench' if name=='seedbench_image' else name]
        split=spec['split'];a=task.dataset_no_image[split];b=original.dataset_no_image[split]
        if name=='seedbench_image':b=b.select([i for i,k in enumerate(b['data_type']) if k=='image'])
        assert len(a)==len(b)==spec['count']
        digest=hashlib.sha256()
        for row_a,row_b in zip(a,b):
            text=task.fewshot_context(row_a,0);target=str(task.doc_to_target(row_a))
            assert text==original.fewshot_context(row_b,0)
            assert target==str(original.doc_to_target(row_b))
            digest.update((text+'\0'+target+'\0').encode())
        reports.append({'task':name,'samples':len(a),'revision':spec['revision'],
            'all_text_contexts_and_targets_match_original':True,'text_target_sha256':digest.hexdigest()})
    write_json(c['runtime']/'task_validation/results.json',reports)
    print(reports)


if __name__=='__main__':main()
