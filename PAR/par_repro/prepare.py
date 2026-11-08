from pathlib import Path
import json
import hashlib
import shutil
import subprocess
import urllib.request
from par_repro.io import package_root, file_sha, atomic_json, safe_extract_zip
from par_repro.config import resolve
from par_repro.data import prepare_selection


def verify_resource(path, entry):
    path = Path(path)
    if entry.get('size') is not None and path.stat().st_size != entry['size']:
        raise ValueError('Resource size mismatch: ' + path.name)
    digest = entry.get('sha256') or entry.get('lfs_sha256')
    if digest and file_sha(path) != digest:
        raise ValueError('Resource SHA256 mismatch: ' + path.name)
    blob = entry.get('git_blob_sha1')
    if blob and not digest:
        h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as handle:
            for part in iter(lambda: handle.read(4 * 1024 ** 2), b''):
                h.update(part)
        if h.hexdigest() != blob:
            raise ValueError('Resource Git blob mismatch: ' + path.name)


def hf_file(repo, revision, filename, destination, entry, dataset=False):
    destination = Path(destination)
    if not destination.exists():
        from huggingface_hub import hf_hub_download
        location = hf_hub_download(repo_id=repo, repo_type='dataset' if dataset else 'model', revision=revision,
                                   filename=filename, token=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(location, destination)
    verify_resource(destination, entry)
    return destination


def prepare(workspace, config, download_models=False, download_seed_images=False):
    workspace = Path(workspace).resolve()
    annotations = workspace / 'annotations'
    annotations.mkdir(parents=True, exist_ok=True)
    resources = json.loads((package_root() / 'configs/resources.json').read_text())
    protocol = json.loads((package_root() / 'configs/annotations.json').read_text())
    train = resources['training_annotation']
    entry = train['files'][0]
    hf_file(train['repo_id'], train['revision'], entry['name'], annotations / entry['name'], entry, True)
    seed = protocol['seed']
    hf_file(seed['repo_id'], seed['revision'], seed['filename'], annotations / seed['filename'], seed, True)
    for category, digest in protocol['pope']['sha256'].items():
        path = annotations / f'coco_pope_{category}.json'
        if not path.exists():
            url = f"https://raw.githubusercontent.com/AoiDragon/POPE/{protocol['pope']['commit']}/output/coco/{path.name}"
            request = urllib.request.Request(url, headers={'User-Agent': 'PAR-reproduction'})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read(4 * 1024 ** 2)
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(data)
            verify_resource(temporary, {'sha256': digest})
            temporary.replace(path)
        verify_resource(path, {'sha256': digest})
    source = resolve(workspace, config['model']['llava_source'])
    checkout = source.parent
    baseline = protocol['baseline']
    if not source.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', baseline['repo'], str(checkout)], check=True)
        subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', baseline['commit']], check=True)
    commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != baseline['commit']:
        raise ValueError('Wrong baseline source commit')
    if subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--', 'LLaVA'], text=True).strip():
        raise ValueError('Pinned baseline source modified')
    model_manifest = {}
    for name in ['backbone', 'vision']:
        resource = resources[name]
        destination = resolve(workspace, config['model'][name])
        for file in resource['files']:
            path = destination / file['name']
            if download_models:
                hf_file(resource['repo_id'], resource['revision'], file['name'], path, file)
            else:
                verify_resource(path, file)
        model_manifest[name] = {'repo_id': resource['repo_id'], 'revision': resource['revision'],
                                'files': {r['name']: file_sha(destination / r['name']) for r in resource['files']},
                                'stats': {r['name']: {'size': (destination / r['name']).stat().st_size,
                                                    'mtime_ns': (destination / r['name']).stat().st_mtime_ns} for r in resource['files']}}
    if download_seed_images:
        item = protocol['seed_images']
        archive = hf_file(item['repo_id'], item['revision'], item['filename'], workspace / 'downloads' / item['filename'], item, True)
        safe_extract_zip(archive, resolve(workspace, config['data']['seed_images']))
    prepare_selection(workspace, config)
    atomic_json(workspace / 'prepared.json', {'status': 'PREPARED', 'models': model_manifest, 'model_paths': config['model'],
        'source_commit': commit, 'annotations': {p.name: file_sha(p) for p in sorted(annotations.glob('*.json'))},
        'data_paths': config['data'], 'seed_images_note': 'Use pinned official image ZIP or supplied official directory; no old cache dependency'})
    print('PREPARE_COMPLETE', workspace, flush=True)
