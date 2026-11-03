"""Materialize an isolated Linux runtime from immutable executed source copies."""
import argparse,json,re,shutil
from pathlib import Path

OLD='/home/SigConv/cys/AdaLLaVA'
SECONDARY='/home/SigConv/backup2/adallava-training-20261027/images'

def materialize(repo:Path,root:Path):
    repo=Path(repo).resolve();root=Path(root).resolve()
    if root.exists():raise FileExistsError(f'Refusing to overwrite existing runtime: {root}')
    if re.fullmatch(r'[A-Za-z0-9_./:-]+',root.as_posix()) is None:
        raise ValueError('Legacy shell scripts require a runtime path without whitespace or shell metacharacters')
    root.mkdir(parents=True)
    for name in ['AdaLLaVA','LLaVA','lmms-eval']:
        shutil.copytree(repo/'vendor'/name,root/name)
    for source,target in [('pipeline','scripts'),('training','training_repro'),('evaluation','evaluation_repro/scripts')]:
        shutil.copytree(repo/'reproduction'/source,root/target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','README.md'))
    shutil.copytree(repo/'vendor/AdaLLaVA',root/'training_repro/source/AdaLLaVA')
    for directory in ['training_repro/audit','training_repro/data','training_repro/data/images','training_repro/smoke/run',
        'training_repro/runs','training_repro/logs','training_repro/models','configs/datasets','configs/states',
        'logs','docs','tmp','cache','checkpoints','models','datasets']:
        (root/directory).mkdir(parents=True,exist_ok=True)
    for name in ['hf_source_metadata.json','image_mirrors.json','nccl_wheel.json']:
        shutil.copyfile(repo/'provenance'/name,root/'training_repro/audit'/name)
    for p in (repo/'provenance/dataset-manifests').glob('*.json'):
        shutil.copyfile(p,root/'configs/datasets'/p.name)
    # Relocate generated runtime copies only. Published evidence and executed source remain byte-identical.
    for directory in ['scripts','training_repro','evaluation_repro/scripts','configs/datasets']:
        for p in (root/directory).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.sh','.json'}:
                text=p.read_text(encoding='utf-8')
                text=text.replace(SECONDARY,(root/'training_repro/data/images').as_posix()).replace(OLD,root.as_posix())
                text=text.replace('/home/SigConv/runs',root.as_posix()).replace('quota -w -u SigConv','quota -w')
                # Normalize the original one-server quota check through the portable explicit policy.
                if p==root/'scripts/pipeline.py':
                    text=re.sub(r'def storage_guard\(required_gib=8\):.*?(?=\n\ndef update_status)',
                        'def storage_guard(required_gib=8):\n    import sys\n    sys.path.insert(0,str(ROOT / "training_repro"))\n    from portable_storage import require_space\n    return require_space(ROOT, required_gib + 15)\n',text,flags=re.S)
                p.write_text(text,encoding='utf-8',newline='\n')
                if p.suffix=='.sh':p.chmod(0o755)
    shutil.copyfile(repo/'tools/portable_storage.py',root/'training_repro/portable_storage.py')
    with (root/'training_repro/run_guard.py').open('a',encoding='utf-8') as f:
        f.write('\n# Migration-only storage policy; historical source is preserved in the Git archive.\nfrom portable_storage import main_quota\n')
    # Fresh dataset preparation must not require the previous server's TextVQA result log.
    p=root/'scripts/prepare_dataset.py';text=p.read_text()
    text=re.sub(r"    baseline_state =.*?(?=atomic_json\(ROOT)",
        "    info['preparation_scope'] = 'validation-only at unchanged pinned revision; fresh server'\n",text,flags=re.S)
    p.write_text(text,encoding='utf-8',newline='\n')
    (root/'MIGRATION.json').write_text(json.dumps({'repository':str(repo),'runtime':str(root),
        'note':'Path-relocated source copies; new data verification and smoke/resume are required.'},indent=2))
    return root

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='New, nonexistent Linux runtime directory')
    args=parser.parse_args();print(materialize(Path(__file__).resolve().parents[1],args.root))

if __name__=='__main__':main()
