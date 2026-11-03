"""Resume fixed-revision raw archives, hash-check, extract and read every image."""
import hashlib,json,os,sys,time
from pathlib import Path
from huggingface_hub import hf_hub_download
from data_gate import extract_referenced,validate_images
ROOT=Path('/home/SigConv/cys/AdaLLaVA')
sys.path.insert(0,str(ROOT/'scripts'))
from recovery_helpers import retry_hub
folder=ROOT/'training_repro/data'
manifest=json.loads((folder/'instruction_manifest.json').read_text())
meta=json.loads((ROOT/'training_repro/audit/image_mirrors.json').read_text())
repo='kaiyuyue/llava-1.5-665k-instructions'; info=meta[repo]
images=folder/'images'; images.mkdir(exist_ok=True)
secondary=Path('/home/SigConv/backup2/adallava-training-20261027/images')
secondary.mkdir(parents=True,exist_ok=True)
records=[]
for name in ['coco','ocr_vqa','textvqa','vg','gqa']:
    if name!='coco':
        dest=secondary/name; dest.mkdir(exist_ok=True)
        link=images/name
        if not link.exists():link.symlink_to(dest,target_is_directory=True)
        assert link.resolve()==dest.resolve()
    refs=[p for p in manifest['unique_images'] if p.split('/')[0]==name]
    done=folder/(name+'_verified.json')
    if done.exists():
        record=json.loads(done.read_text())
        assert record['revision']==info['revision'] and record['found']==len(refs)
        assert all((images/p).is_file() for p in refs)
        records.append(record); print('REUSED_VERIFIED_SOURCE',name,flush=True); continue
    file='train_split/'+name+'.tar.gz'
    entry=next(f for f in info['files'] if f['path']==file)
    path=Path(retry_hub(lambda:hf_hub_download(repo,file,repo_type='dataset',revision=info['revision'],local_dir=folder/'archives')))
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024**2),b''):sha.update(chunk)
    assert sha.hexdigest()==entry['lfs']['sha256'],file
    print('ARCHIVE_SHA_VERIFIED',file,sha.hexdigest(),flush=True)
    record={'repo':repo,'revision':info['revision'],'file':file,'archive_sha256':sha.hexdigest()}
    record.update(extract_referenced(path,images,refs))
    record.update(validate_images(images,refs))
    done.write_text(json.dumps(record,indent=2)); records.append(record)
    # Delete only our freshly downloaded, verified archive after full image decoding succeeds.
    path.unlink()
    print('SOURCE_READY',name,json.dumps(record),flush=True)
ready={'status':'READY','annotation_sha256':manifest['sha256'],'total_samples':manifest['total_samples'],
       'unique_images':sum(r['found'] for r in records),'sources':records,'image_root':str(images),'completed_time':time.time()}
assert ready['total_samples']==665298 and ready['unique_images']==len(manifest['unique_images'])
(folder/'DATA_READY.json').write_text(json.dumps(ready,indent=2))
print('FULL_IMAGES_READY',json.dumps(ready),flush=True)
