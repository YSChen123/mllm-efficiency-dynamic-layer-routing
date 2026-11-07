"""Pin public dataset revisions while inheriting every official prompt/scorer."""
from pathlib import Path
import shutil
from .common import UPSTREAM,ROOT

TASKS={
 'scienceqa_img':{'alias':'scienceqa_img_release','count':2017,'repo':'lmms-lab/ScienceQA',
   'config':'ScienceQA-IMG','revision':'69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50',
   'yaml':'scienceqa/scienceqa_img.yaml','split':'test'},
 'gqa':{'alias':'gqa_release','count':12578,'repo':'lmms-lab/GQA',
   'config':'testdev_balanced_instructions','revision':'a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8',
   'yaml':'gqa/gqa.yaml','split':'testdev'},
 'seedbench_image':{'alias':'seedbench_image_release','count':14233,'repo':'lmms-lab/SEED-Bench',
   'config':'default','revision':'74c4ea0cee96786739e4ccb97d227818a05ae752',
   'yaml':'seedbench/seedbench.yaml','split':'test'},
}


def materialize(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'roe_repro/seed_image_utils.py',directory/'seed_image_utils.py')
    for name,task in TASKS.items():
        original=UPSTREAM/'lmms-eval/lmms_eval/tasks'/task['yaml']
        lines=[f'include: {str(original)!r}',f'task: {task["alias"]}',
          'dataset_kwargs:','  token: false',f'  revision: {task["revision"]!r}']
        if name=='seedbench_image':
            lines+=['process_docs: !function seed_image_utils.select_image_docs',
              'process_results: !function seed_image_utils.image_result',
              'metric_list:','  - metric: seed_image',
              '    aggregation: !function seed_image_utils.image_aggregation','    higher_is_better: true']
        (directory/(task['alias']+'.yaml')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return directory
