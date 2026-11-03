"""Full-image readiness gate; copies original bytes without re-encoding."""
from pathlib import Path, PurePosixPath
import hashlib,shutil,tarfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def canonical_name(name):
    parts=PurePosixPath(name).parts
    if not parts or '..' in parts or PurePosixPath(name).is_absolute():return None
    parts=list(parts)
    if parts[0]=='ocr-vqa':parts[0]='ocr_vqa'
    if parts[:2]==['coco','images']:parts.pop(1)
    return '/'.join(parts)

def extract_referenced(archive,root,refs):
    root=Path(root); wanted=set(refs); found=set()
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            name=canonical_name(member.name)
            if name not in wanted or not member.isfile():continue
            dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
            partial=dest.with_suffix(dest.suffix+'.partial')
            with stream.extractfile(member) as source,partial.open('wb') as target:shutil.copyfileobj(source,target,1024**2)
            if partial.stat().st_size!=member.size:raise ValueError('truncated '+name)
            if dest.is_file() and hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(partial.read_bytes()).digest():
                raise ValueError('mirror differs from existing original image: '+name)
            partial.replace(dest); found.add(name)
            if len(found)%10000==0:print('EXTRACTED',archive.name,len(found),flush=True)
    missing=wanted-found
    if missing:raise ValueError('missing archive images: '+str(sorted(missing)[:10]))
    return {'found':len(found),'bytes':sum((root/name).stat().st_size for name in found)}

def validate_images(root,refs):
    root=Path(root)
    def check(name):
        path=root/name
        if not path.is_file():return ('missing',name)
        try:
            with Image.open(path) as image:image.verify()
            with Image.open(path) as image:image.convert('RGB').load()
        except Exception:return ('unreadable',name)
    problems=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,result in enumerate(pool.map(check,refs),1):
            if result:problems.append(result)
            if i%10000==0:print('VALIDATED',i,flush=True)
    if problems:raise ValueError(str(problems[:20])+' total='+str(len(problems)))
    return {'validated':len(refs),'bytes':sum((root/name).stat().st_size for name in refs)}
