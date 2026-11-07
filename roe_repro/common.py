import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
COMMIT='4f529e468d396246dd96f44fb7e13bd3cba61261'
UPSTREAM=Path(os.environ.get('ROE_UPSTREAM_DIR',ROOT/'upstream/RoE')).resolve()
PATH_KEYS=['runtime','base_model','vision_model','train_source','train_subset','images_dir','checkpoint','cache']
DEFAULTS={'runtime':'runtime/roe','base_model':'runtime/assets/llava-v1.5-7b',
 'vision_model':'runtime/assets/clip-vit-large-patch14-336',
 'train_source':'runtime/data/llava_v1_5_mix665k.json',
 'train_subset':'runtime/data/llava_v1_5_mix665k_50.json',
 'images_dir':'runtime/data/images','checkpoint':'runtime/checkpoints/llava-v1.5-7b-roe30',
 'cache':'runtime/cache/huggingface','train_gpus':[0,1]}


def checkpoint_view(config,runroot):
    source=config['checkpoint']
    metadata=json.loads((source/'config.json').read_text(encoding='utf-8'))
    vision=str(config['vision_model'].resolve())
    if metadata.get('mm_vision_tower')==vision:return source
    if not config['vision_model'].is_dir():raise FileNotFoundError(config['vision_model'])
    target=Path(runroot)/'model_view/llava-v1.5-7b-roe30'
    target.mkdir(parents=True)
    for file in source.iterdir():
        if not file.is_file() or file.name=='config.json':continue
        if os.name=='nt':os.link(file,target/file.name)
        else:(target/file.name).symlink_to(file.resolve())
    metadata['mm_vision_tower']=vision
    write_json(target/'config.json',metadata)
    write_json(Path(runroot)/'checkpoint_path_override.json',{
        'source_checkpoint':source,'view':target,'changed_config_fields':['mm_vision_tower'],
        'weights_linked_without_copy_or_update':True})
    return target


def load_config(path):
    path=Path(path).resolve()
    values=dict(DEFAULTS,**json.loads(path.read_text(encoding='utf-8')))
    for key in PATH_KEYS:
        p=Path(os.path.expandvars(os.path.expanduser(values[key])))
        values[key]=(path.parent/p).resolve() if not p.is_absolute() else p.resolve()
    return values


def select_training_rows(rows):
    return [row for i,row in enumerate(rows) if i%10<6]


def render_training_script(source,config):
    replacements={'liuhaotian/llava-v1.5-7b':'base_model',
      'openai/clip-vit-large-patch14-336':'vision_model','/path/to/data/json':'train_subset',
      '/data/qiong_code/data/':'images_dir','/path/to/checkpoint':'checkpoint'}
    for old,key in replacements.items():
        if source.count(old)!=1:raise ValueError(f'Expected exactly one upstream placeholder: {old}')
        source=source.replace(old,shlex.quote(str(config[key])))
    return source


def verify_upstream():
    head=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT:raise RuntimeError(f'Wrong upstream commit: {head}; expected {COMMIT}')
    for args in [['diff','--quiet'],['diff','--cached','--quiet']]:
        if subprocess.run(['git','-C',str(UPSTREAM),*args]).returncode:
            raise RuntimeError('Upstream tracked files modified; refusing an ambiguous reproduction')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
    tmp.replace(path)


def environment(config,phase,gpus=(),online=False):
    env=dict(os.environ)
    for key in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:
        env.pop(key,None)
    env.pop('CUDA_LAUNCH_BLOCKING',None)
    cache=str(config['cache'])
    env.update(PYTHONPATH=os.pathsep.join([str(UPSTREAM/('RoE' if phase=='train' else 'lmms-eval')),str(ROOT)]),
      PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HOME=cache,
      HF_HUB_CACHE=str(config['cache']/'hub'),HF_DATASETS_CACHE=str(config['cache']/'datasets'),
      HF_HUB_OFFLINE='0' if online else '1',HF_DATASETS_OFFLINE='0' if online else '1',
      TRANSFORMERS_OFFLINE='0' if online else '1',WANDB_MODE='offline',TOKENIZERS_PARALLELISM='false',
      OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='1',
      ROE_UPSTREAM_DIR=str(UPSTREAM),GIT_DIR=str(UPSTREAM/'.git'),GIT_WORK_TREE=str(UPSTREAM))
    # A submodule has a .git FILE, whereas GIT_DIR must point at its actual directory.
    env['GIT_DIR']=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','--absolute-git-dir'],text=True).strip()
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
    return env


def require_idle_gpus(gpus):
    if len(set(gpus))!=len(gpus):raise ValueError('Duplicate physical GPU IDs')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'],text=True)
    selected=[]
    for gpu in gpus:
        row=next((r.split(',') for r in rows if int(r.split(',')[0])==gpu),None)
        if row is None or int(row[2])>=512 or row[1].strip() in apps:
            raise RuntimeError(f'GPU {gpu} is occupied or unavailable; no processes were stopped')
        selected.append({'index':gpu,'uuid':row[1].strip(),'memory_before_mib':int(row[2])})
    return selected
