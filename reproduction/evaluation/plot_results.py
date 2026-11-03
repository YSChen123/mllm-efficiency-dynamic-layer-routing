"""Standalone scientific plots from validated observed results, without inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/home/SigConv/cys/AdaLLaVA/evaluation_repro/20261031/analysis')
data=json.loads((root/'validated_results.json').read_text())
rows=data['rows']
tasks=['scienceqa_img','pope','mme','textvqa_val']
labels={'scienceqa_img':'ScienceQA-IMG accuracy (%)','pope':'POPE F1 (%)',
        'mme':'MME Perception score','textvqa_val':'TextVQA accuracy (%)'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':180,'font.family':'DejaVu Sans'})
colors={'reproduced':'#cc6237','official':'#2d7095','dense':'#526052'}
fig,axes=plt.subplots(2,2,figsize=(10.6,7.1),layout='constrained')
for ax,task in zip(axes.flat,tasks):
    for role,label in [('official','Official checkpoint'),('reproduced','Our trained checkpoint')]:
        values=sorted([r for r in rows if r['task']==task and r['role']==role],key=lambda r:r['budget'])
        ax.plot([r['budget'] for r in values],[r['score'] for r in values],'-o',ms=4,lw=1.6,color=colors[role],label=label)
    dense=next(r['score'] for r in rows if r['task']==task and r['role']=='dense')
    ax.axhline(dense,color=colors['dense'],ls='--',lw=1.3,label='Measured Dense baseline')
    ax.set(xlabel='Requested budget (quantized by layer count)',ylabel=labels[task])
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
fig.suptitle('AdaLLaVA-L reproduction: full-dataset scores',fontsize=14)
fig.savefig(root/'accuracy_budget.png');fig.savefig(root/'accuracy_budget.svg');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10.6,7.1),layout='constrained')
for ax,task in zip(axes.flat,tasks):
    for role,label,marker in [('reproduced','Our trained checkpoint','o'),('official','Official: new timing points','s')]:
        values=sorted([r for r in rows if r['task']==task and r['role']==role and 'observed' in r],key=lambda r:r['budget'])
        ax.plot([r['observed']['mean_lm_prefill_ms'] for r in values],[r['score'] for r in values],
                '-'+marker,ms=4,lw=1.4,color=colors[role],label=label)
    dense=next(r for r in rows if r['task']==task and r['role']=='dense')
    ax.scatter([dense['observed']['mean_lm_prefill_ms']],[dense['score']],marker='*',s=100,color=colors['dense'],label='Dense')
    ax.set(xlabel='Observed mean LM prefill (CUDA ms; first 3 excluded)',ylabel=labels[task])
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
fig.suptitle('Observed accuracy-latency tradeoff (same GPU model, different cards)',fontsize=13)
fig.savefig(root/'accuracy_latency.png');fig.savefig(root/'accuracy_latency.svg');plt.close(fig)
print('PLOTS_SAVED',root)
