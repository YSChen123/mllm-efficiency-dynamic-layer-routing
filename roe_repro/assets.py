"""Download public, pinned assets on explicit request; never package their data."""
import argparse
import json
import os
from pathlib import Path
import shutil
from .common import ROOT,load_config,sha256
from .tasks import TASKS


def validate(folder,files):
    for name,expected in files.items():
        p=folder/name
        if not p.is_file() or p.stat().st_size!=expected['bytes'] or sha256(p)!=expected['sha256']:
            raise ValueError(f'Asset differs from the recorded bytes: {p}')


def validate_canonical_clip(config):
    from huggingface_hub import try_to_load_from_cache
    entries=json.loads((ROOT/'results/assets.json').read_text())
    entry=next(e for e in entries if e['name']=='clip-vit-large-patch14-336')
    for name,expected in entry['files'].items():
        cached=try_to_load_from_cache(entry['repo_id'],name,revision='main',cache_dir=str(config['cache']/'hub'))
        if not isinstance(cached,str) or sha256(cached)!=expected['sha256']:
            raise RuntimeError('Canonical offline CLIP cache is absent/different. Run: python -m roe_repro.assets models')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config.local.json')
    parser.add_argument('action',choices=['models','annotation','eval-data','verify-models'])
    args=parser.parse_args();c=load_config(args.config)
    manifest=json.loads((ROOT/'results/assets.json').read_text())
    if args.action=='verify-models':
        for entry in manifest:validate(c['base_model' if entry['name']=='llava-v1.5-7b' else 'vision_model'],entry['files'])
        print('All model files match recorded SHA256 values');return
    # The imported Hub/datasets packages read these variables at import time.
    for key in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE']:os.environ[key]='0'
    os.environ.update(HF_HOME=str(c['cache']),HF_HUB_CACHE=str(c['cache']/'hub'),
        HF_DATASETS_CACHE=str(c['cache']/'datasets'))
    if args.action=='models':
        from huggingface_hub import snapshot_download
        for entry in manifest:
            path=c['base_model' if entry['name']=='llava-v1.5-7b' else 'vision_model']
            snapshot=Path(snapshot_download(repo_id=entry['repo_id'],revision=entry['revision'],
                allow_patterns=list(entry['files']),cache_dir=str(c['cache']/'hub'),max_workers=2))
            validate(snapshot,entry['files'])
            # Upstream constructs CLIP from the base config's original Hub ID
            # before applying --vision_tower. Bind that lookup deterministically
            # inside this dedicated cache, without altering the base model config.
            ref=c['cache']/'hub'/('models--'+entry['repo_id'].replace('/','--'))/'refs/main'
            if ref.exists() and ref.read_text().strip()!=entry['revision']:
                raise FileExistsError('Dedicated cache main ref points elsewhere; use a separate cache directory')
            ref.parent.mkdir(parents=True,exist_ok=True);ref.write_text(entry['revision'])
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                if os.name=='nt':
                    path.mkdir()
                    for name in entry['files']:os.link(snapshot/name,path/name)
                else:path.symlink_to(snapshot,target_is_directory=True)
            validate(path,entry['files'])
    elif args.action=='annotation':
        from huggingface_hub import hf_hub_download
        expected=json.loads((ROOT/'results/training_data_hashes.json').read_text())['llava_v1_5_mix665k.json']
        if c['train_source'].exists():
            if sha256(c['train_source'])!=expected['sha256']:raise FileExistsError('Existing source annotation differs; refusing overwrite')
            print('Source annotation already matches');return
        path=hf_hub_download(repo_id='liuhaotian/LLaVA-Instruct-150K',repo_type='dataset',
            filename='llava_v1_5_mix665k.json',revision='main',cache_dir=str(c['cache']/'hub'))
        if sha256(path)!=expected['sha256']:raise ValueError('Downloaded annotation changed upstream; expected SHA256 is recorded')
        c['train_source'].parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,c['train_source'])
    else:
        from datasets import load_dataset
        for task in TASKS.values():
            data=load_dataset(task['repo'],task['config'],revision=task['revision'],token=False,
                cache_dir=str(c['cache']/'datasets'))
            print(task['repo'],task['revision'],{split:len(rows) for split,rows in data.items()})


if __name__=='__main__':main()
