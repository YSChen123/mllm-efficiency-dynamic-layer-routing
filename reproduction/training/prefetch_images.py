"""Two bounded transfer workers; same local_dir/locks as prepare_images, no copies."""
import json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
ROOT=Path('/home/SigConv/cys/AdaLLaVA')
sys.path.insert(0,str(ROOT/'scripts'))
from recovery_helpers import retry_hub
repo='kaiyuyue/llava-1.5-665k-instructions'
info=json.loads((ROOT/'training_repro/audit/image_mirrors.json').read_text())[repo]
def download(name):
    path=retry_hub(lambda:hf_hub_download(repo,'train_split/'+name+'.tar.gz',repo_type='dataset',revision=info['revision'],local_dir=ROOT/'training_repro/data/archives'))
    print('PREFETCH_COMPLETE',name,Path(path).stat().st_size,flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,['ocr_vqa','textvqa','vg','gqa']))
