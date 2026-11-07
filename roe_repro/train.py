"""Render the author's training script with only its five asset paths replaced."""
import argparse
import datetime
from pathlib import Path
import subprocess
import sys

from .common import UPSTREAM,COMMIT,load_config,verify_upstream,render_training_script,environment,require_idle_gpus,write_json,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='config.local.json');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();c=load_config(args.config);verify_upstream()
    source=UPSTREAM/'RoE/scripts/v1_5/finetune_RoE.sh'
    script=c['runtime']/'commands/finetune_RoE.paths_only.sh'
    script.parent.mkdir(parents=True,exist_ok=True)
    script.write_text(render_training_script(source.read_text(encoding='utf-8'),c),encoding='utf-8',newline='\n')
    if not args.dry_run:
        if sys.version_info[:2]!=(3,10):raise RuntimeError('Use the documented Python 3.10 training environment')
        if len(c['train_gpus'])!=2:raise ValueError('Recorded run used 2 GPUs, batch 4 x accumulation 4 = global batch 32')
        if c['checkpoint'].exists():raise FileExistsError('Checkpoint directory exists; refusing automatic overwrite/resume')
        for key in ['base_model','vision_model','train_subset','images_dir']:
            if not c[key].exists():raise FileNotFoundError(f'Missing {key}: {c[key]}')
        from .assets import validate_canonical_clip
        validate_canonical_clip(c)
        selected=require_idle_gpus(c['train_gpus'])
    else:selected=[]
    write_json(c['runtime']/'commands/training_manifest.json',{
        'created_at':datetime.datetime.now().astimezone().isoformat(),'source_commit':COMMIT,
        'source_script_sha256':sha256(source),'rendered_script_sha256':sha256(script),
        'path_replacements_only':True,'training_target_skip_ratio':0.3,'physical_gpus':selected,
        'dry_run':args.dry_run,'configuration':c})
    print('Rendered paths-only official training script:',script,flush=True)
    if not args.dry_run:
        subprocess.run(['bash',str(script)],cwd=UPSTREAM/'RoE',
            env=environment(c,'train',c['train_gpus']),check=True)


if __name__=='__main__':main()
