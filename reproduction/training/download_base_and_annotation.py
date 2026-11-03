"""Fetch only missing official training inputs at audited fixed revisions."""
import hashlib,json,sys
from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT=Path('/home/SigConv/cys/AdaLLaVA')
sys.path.insert(0,str(ROOT/'scripts'))
from recovery_helpers import retry_hub
metadata=json.loads((ROOT/'training_repro/audit/hf_source_metadata.json').read_text())
for repo,kind,folder,names in [
 ('liuhaotian/LLaVA-Instruct-150K','dataset',ROOT/'training_repro/data',['llava_v1_5_mix665k.json']),
 ('liuhaotian/llava-v1.5-7b','model',ROOT/'checkpoints/llava-v1.5-7b',None)]:
 info=metadata[repo]
 selected=[f for f in info['files'] if (names is None and f['path']!='mm_projector.bin') or (names and f['path'] in names)]
 for f in selected:
  path=retry_hub(lambda:hf_hub_download(repo,f['path'],repo_type=kind,revision=info['revision'],local_dir=folder))
  digest=hashlib.sha256(Path(path).read_bytes()).hexdigest() if f['size']<2048**3 else None
  if digest is None:
   sha=hashlib.sha256()
   with open(path,'rb') as stream:
    for chunk in iter(lambda:stream.read(16*1024**2),b''):sha.update(chunk)
   digest=sha.hexdigest()
  if f['lfs']:assert digest==f['lfs']['sha256'],f['path']
  print('VERIFIED',repo,f['path'],Path(path).stat().st_size,digest,flush=True)
 if kind=='dataset':
  import collections
  data=json.loads((folder/names[0]).read_text())
  unique=set(row['image'] for row in data if 'image' in row)
  counts=collections.Counter(row['image'].split('/')[0] if 'image' in row else 'text_only' for row in data)
  image_counts=collections.Counter(name.split('/')[0] for name in unique)
  manifest={'source_repo':repo,'revision':info['revision'],'sha256':selected[0]['lfs']['sha256'],
   'total_samples':len(data),'sample_counts':dict(counts),'unique_image_counts':dict(image_counts),'unique_images':sorted(unique)}
  (ROOT/'training_repro/data/instruction_manifest.json').write_text(json.dumps(manifest,indent=2))
  print('FULL_DATASET_COUNTS',len(data),dict(counts),dict(image_counts),flush=True)
print('OFFICIAL_BASE_AND_ANNOTATION_READY',flush=True)
