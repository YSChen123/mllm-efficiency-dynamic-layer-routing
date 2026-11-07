"""Import/version checks and real decoder regressions; no dataset or weight download."""
import argparse
import subprocess
import sys
from .common import ROOT,UPSTREAM,load_config,verify_upstream,environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.local.json')
    p.add_argument('--phase',choices=['train','eval'],required=True);p.add_argument('--tests',action='store_true')
    args=p.parse_args();c=load_config(args.config);verify_upstream()
    if sys.version_info[:2]!=(3,10):raise RuntimeError('Recorded training/evaluation used Python 3.10')
    expected='4.31.0' if args.phase=='train' else '4.39.2'
    code=f"""import importlib,importlib.metadata as m,json,sys
import torch,transformers,llava
assert transformers.__version__=={expected!r},transformers.__version__
assert torch.__version__=='2.7.1+cu128',torch.__version__
assert str(llava.__file__).startswith({str(UPSTREAM/('RoE' if args.phase=='train' else 'lmms-eval'))!r})
importlib.import_module({'llava.train.roe_trainer' if args.phase=='train' else 'lmms_eval.__main__'!r})
print(json.dumps({{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'llava':llava.__file__}},indent=2))
"""
    env=environment(c,args.phase)
    subprocess.run([sys.executable,'-c',code],env=env,cwd=ROOT,check=True)
    if args.tests:
        if args.phase!='eval':raise ValueError('Run the decoder test suite in the evaluation environment')
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],env=env,cwd=ROOT,check=True)


if __name__=='__main__':main()
