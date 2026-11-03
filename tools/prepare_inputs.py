"""Download pinned training inputs and regenerate readiness evidence on a new server."""
import argparse,hashlib,json,os,runpy,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['base','images','smoke','all'],default='all');args=p.parse_args()
    root=args.root.resolve();os.chdir(root)
    sys.path.insert(0,str(root/'training_repro'));sys.path.insert(0,str(root/'scripts'))
    from portable_storage import require_space
    require_space(root,100 if args.stage in ['all','images'] else 20)
    if args.stage in ['base','all']:
        runpy.run_path(str(root/'training_repro/download_base_and_annotation.py'),run_name='__main__')
        from huggingface_hub import snapshot_download
        vision=Path(snapshot_download('openai/clip-vit-large-patch14-336',revision='ce19dc912ca5cd21c8a653c79e251e808ccabcd1',local_dir=root/'checkpoints/clip-vit-large-patch14-336'))
        # Record actual downloaded files rather than borrowing an old readiness marker.
        result={}
        for file in vision.iterdir():
            if file.is_file():
                h=hashlib.sha256()
                with file.open('rb') as f:
                    for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
                result[file.name]=h.hexdigest()
        (root/'training_repro/audit/downloaded-clip-sha256.json').write_text(json.dumps(result,indent=2))
        view=root/'training_repro/models/llava-v1.5-7b';view.mkdir(parents=True,exist_ok=True)
        for file in (root/'checkpoints/llava-v1.5-7b').iterdir():
            if file.is_file() and not file.name.startswith('.') and file.name!='config.json':
                dest=view/file.name
                if not dest.exists():dest.symlink_to(file)
        config=json.loads((root/'checkpoints/llava-v1.5-7b/config.json').read_text())
        config['mm_vision_tower']=str(vision)
        (view/'config.json').write_text(json.dumps(config,indent=2))
    if args.stage in ['images','all']:
        runpy.run_path(str(root/'training_repro/prepare_images.py'),run_name='__main__')
    if args.stage in ['smoke','all']:
        data=json.loads((root/'training_repro/data/llava_v1_5_mix665k.json').read_text())
        (root/'training_repro/data/smoke32.json').write_text(json.dumps(data[:32]))
        print('SMOKE_DATA_READY',32)

if __name__=='__main__':main()
