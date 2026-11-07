"""Initialize the pinned submodule without checking out upstream CUDA logs on Windows."""
from pathlib import Path
import os
import subprocess

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'upstream/RoE'
PIN='4f529e468d396246dd96f44fb7e13bd3cba61261'


def run(*args):subprocess.run(list(args),cwd=ROOT,check=True)


new_clone=not (TARGET/'.git').exists()
if new_clone:
    run('git','submodule','init')
    run('git','clone','--no-checkout','https://github.com/DoubtedSteam/RoE.git',str(TARGET))
    run('git','submodule','absorbgitdirs','upstream/RoE')
else:
    for command in [('diff','--quiet'),('diff','--cached','--quiet')]:
        if subprocess.run(['git','-C',str(TARGET),*command]).returncode:
            raise RuntimeError('Upstream contains tracked changes; not replacing or resetting them')
if os.name=='nt':
    run('git','-C',str(TARGET),'config','core.protectNTFS','false')
run('git','-C',str(TARGET),'sparse-checkout','set','--no-cone','/*',
    '!/lmms-eval/skip_result/','!**/__pycache__/','!**/*.pyc')
run('git','-C',str(TARGET),'read-tree','-mu','HEAD')
run('git','-C',str(TARGET),'checkout',PIN)
print('Pinned upstream ready:',PIN)
