import json,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root=Path('/home/SigConv/cys/AdaLLaVA')
os.environ.update(HF_ENDPOINT='https://hf-mirror.com', HF_HOME=str(root/'cache/model_hub'),
                  HF_HUB_DOWNLOAD_TIMEOUT='120', HF_HUB_ENABLE_HF_TRANSFER='1')
from huggingface_hub import hf_hub_download
d=json.loads((root/'configs/checkpoint_metadata.json').read_text())
names=['model-00002-of-00003.safetensors','model-00003-of-00003.safetensors']
def fetch(name):
 print('START',name,flush=True)
 path=hf_hub_download(d['id'], name, revision=d['sha'], local_dir=root/'checkpoints/ada-llava-L-v1.5-7b')
 print('DONE',name,Path(path).stat().st_size,flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 list(pool.map(fetch,names))
