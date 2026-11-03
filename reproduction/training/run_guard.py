import json
import subprocess
from pathlib import Path

def latest_complete(root):
    candidates=[]
    for path in Path(root).glob('checkpoint-*'):
        try:
            step=int(path.name.split('-')[-1])
            proof=json.loads((path/'COMPLETE.json').read_text())
            trainer=json.loads((path/'trainer_state.json').read_text())
            if proof['step']==trainer['global_step']==step:candidates.append((step,path))
        except (OSError,ValueError,KeyError):continue
    return max(candidates,default=(None,None))[1]

def has_capacity(soft_limit,used,incoming,reserve):
    return soft_limit-used>=incoming+reserve

def data_ready(ready):
    return ready.get('status')=='READY' and ready.get('total_samples')==665298 and ready.get('unique_images')==349034

def main_quota():
    result=subprocess.run(['quota','-w','-p'],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for line in result.stdout.splitlines():
        if line.startswith('/dev/mapper/vg_nvme_merged_251218-lv_merged_251218'):
            fields=line.split()
            return {'used':int(fields[1].rstrip('*'))*1024,'soft':int(fields[2])*1024}
    raise RuntimeError('Cannot read actual home filesystem quota')
