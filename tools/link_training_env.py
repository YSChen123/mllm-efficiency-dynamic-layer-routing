"""Make the training venv inherit the eval venv, including editable registrations."""
import argparse,json,subprocess
from pathlib import Path

def site_path(python):
    return Path(subprocess.check_output([str(python),'-c','import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip())

def link_training_site(evaluation_python,training_python):
    source=site_path(evaluation_python);target=site_path(training_python)
    if source==target:raise ValueError('Evaluation and training environments must be separate')
    bridge=target/'zz_adallava_evaluation_site.pth'
    # addsitedir processes the inherited editable .pth files; a plain path line does not.
    # Training site-packages has already been added, so its own NCCL keeps precedence.
    bridge.write_text('import site; site.addsitedir('+repr(str(source))+')\n',encoding='utf-8')
    return bridge

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation_python',type=Path);parser.add_argument('training_python',type=Path)
    args=parser.parse_args();print(link_training_site(args.evaluation_python,args.training_python))
