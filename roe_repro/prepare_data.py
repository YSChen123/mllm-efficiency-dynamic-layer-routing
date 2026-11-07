"""Reproduce the release's deterministic subset and check image references."""
import argparse
import json
from pathlib import Path
from .common import ROOT,load_config,select_training_rows,sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.local.json')
    p.add_argument('--check-images',action='store_true');args=p.parse_args();c=load_config(args.config)
    expected=json.loads((ROOT/'results/training_data_hashes.json').read_text())
    if sha256(c['train_source'])!=expected['llava_v1_5_mix665k.json']['sha256']:
        raise ValueError('Source annotation SHA256 differs from this experiment; do not silently train another corpus')
    rows=json.loads(c['train_source'].read_text(encoding='utf-8'))
    if len(rows)!=665298:raise ValueError('Expected 665,298 source conversations')
    selected=select_training_rows(rows)
    if len(selected)!=399180:raise ValueError('Expected 399,180 selected conversations')
    if c['train_subset']==c['train_source']:raise ValueError('Source and subset paths must differ')
    c['train_subset'].parent.mkdir(parents=True,exist_ok=True)
    if c['train_subset'].exists():
        if sha256(c['train_subset'])!=expected['llava_v1_5_mix665k_50.json']['sha256']:
            raise FileExistsError('Existing subset differs; refusing to overwrite annotations')
    else:
        with c['train_subset'].open('w',encoding='utf-8',newline='\n') as f:json.dump(selected,f,indent=4)
    if sha256(c['train_subset'])!=expected['llava_v1_5_mix665k_50.json']['sha256']:
        raise ValueError('Generated subset bytes differ from recorded release selection')
    referenced=sorted({row['image'] for row in selected if row.get('image')})
    missing=[]
    if args.check_images:
        root=c['images_dir'].resolve()
        for name in referenced:
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Annotation image path must be relative without parent traversal')
            # Dataset folders intentionally may be symlinks to existing storage.
            file=root/relative
            if not file.is_file():missing.append(name)
    report={'source_rows':len(rows),'selected_rows':len(selected),'unique_image_paths':len(referenced),
        'subset_sha256':sha256(c['train_subset']),'image_paths_checked':args.check_images,
        'missing_images':len(missing),'first_missing':missing[:10]}
    write_json(c['runtime']/'data_validation.json',report)
    print(json.dumps(report,indent=2))
    if missing:raise FileNotFoundError('Referenced images missing; see data_validation.json')


if __name__=='__main__':main()
