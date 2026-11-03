"""Use official MME aggregation on saved predictions, without new inference."""
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path('/home/SigConv/cys/AdaLLaVA')
DEST=ROOT/'evaluation_repro/20261031/analysis'
sys.path.insert(0,str(ROOT/'lmms-eval'))
from lmms_eval.tasks.mme.utils import mme_aggregate_results, parse_pred_ans
data=json.loads((DEST/'validated_results.json').read_text())
rows={(r['role'],r['task'],r['budget']):r for r in data['rows']}
by_role={};by_doc={}

def response(value):
    if isinstance(value,str):return value
    if isinstance(value,list):return response(value[0])
    if isinstance(value,dict):return response(value['resp'])
    raise TypeError(type(value))

for role in ['dense','official','reproduced']:
    row=rows[role,'mme',1.]
    samples=[json.loads(line) for line in Path(row['samples_file']).read_text().splitlines()]
    categories=defaultdict(list);mapped={};unparsed=0
    for sample in samples:
        filtered=sample['filtered_resps']
        pred=response(filtered['none'] if isinstance(filtered,dict) and 'none' in filtered else filtered)
        unparsed+=parse_pred_ans(pred)=='other'
        if 'mme_perception_score' not in sample:continue
        score=sample['mme_perception_score'];categories[score['category']].append(score)
        mapped[sample['doc_id']]={'score':score['score'],'category':score['category'],
            'question_id':score['question_id'],'question':sample['doc']['question'],
            'answer':sample['doc']['answer'],'prediction':pred}
    scores={cat:mme_aggregate_results(items) for cat,items in categories.items()}
    assert abs(sum(scores.values())-row['score'])<1e-6
    by_role[role]={'categories':scores,'category_question_counts':{cat:len(items) for cat,items in categories.items()},
                   'unparsed_answers_all_2374':unparsed,'perception_questions':len(mapped)}
    by_doc[role]=mapped
assert by_doc['official'].keys()==by_doc['reproduced'].keys()
changed=[]
for doc_id,own in by_doc['reproduced'].items():
    official=by_doc['official'][doc_id]
    assert all(own[key]==official[key] for key in ['category','question_id','question','answer'])
    if own['score']!=official['score']:changed.append({'doc_id':doc_id,'own':own,'official':official})
comparison=[{'category':cat,'official':by_role['official']['categories'][cat],
             'reproduced':value,'delta':value-by_role['official']['categories'][cat]}
            for cat,value in by_role['reproduced']['categories'].items()]
comparison.sort(key=lambda row:row['delta'])
out={'full_budget':by_role,'category_comparison':comparison,
     'official_correct_own_wrong':sum(row['official']['score']>row['own']['score'] for row in changed),
     'own_correct_official_wrong':sum(row['own']['score']>row['official']['score'] for row in changed),
     'changed_predictions':changed}
(DEST/'mme_diagnosis.json').write_text(json.dumps(out,indent=2))
print(json.dumps({key:value for key,value in out.items() if key!='changed_predictions'},indent=2))
