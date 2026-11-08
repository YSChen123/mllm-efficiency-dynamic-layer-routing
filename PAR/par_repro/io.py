from pathlib import Path
import hashlib
import json
import os
import zipfile
import tempfile
import time


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        temporary = Path(handle.name)
    try:
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if os.name != 'nt' or attempt == 7:
                    raise
                time.sleep(.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def bind_identity(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Identity changed; use a new output directory: {path}')
    finally:
        temporary.unlink(missing_ok=True)


def check_prediction_prefix(rows, keys, complete=False):
    if len(rows) > len(keys):
        raise ValueError('Too many predictions')
    actual = [(str(row['category']), str(row['question_id'])) for row in rows]
    if actual != [(str(a), str(b)) for a, b in keys[:len(rows)]] or any(not row['text'] for row in rows):
        raise ValueError('Prediction order/IDs/text invalid')
    if complete and len(rows) != len(keys):
        raise ValueError('Incomplete predictions cannot produce formal metrics')
    return len(rows)


def safe_extract_zip(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe archive member')
        archive.extractall(destination)


def package_root():
    return Path(__file__).resolve().parents[1]


def code_sha():
    root = Path(__file__).resolve().parent
    return object_sha({str(path.relative_to(root)): hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() for path in sorted(root.glob('*.py'))})


def reconcile_step_log(path, step):
    path = Path(path)
    if not path.exists():
        if step:
            raise ValueError('Saved checkpoint has no matching step log')
        return
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    if len(lines) < step or [json.loads(line)['step'] for line in lines[:step]] != list(range(1, step + 1)):
        raise ValueError('Checkpoint/log prefix mismatch')
    if len(lines) > step:
        archive = path.with_name('steps.unsaved.' + hashlib.sha256(original).hexdigest()[:12] + '.jsonl')
        if not archive.exists():
            archive.write_bytes(original)
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(b''.join(lines[:step]))
        temporary.replace(path)


def require_resume_checkpoint(path, resume):
    exists = Path(path).exists()
    if resume and not exists:
        raise FileNotFoundError('Explicit --resume requires an existing checkpoint')
    if exists and not resume:
        raise ValueError('Existing checkpoint requires --resume')


def model_resource_identity(workspace):
    prepared = json.loads((Path(workspace) / 'prepared.json').read_text())
    if prepared['status'] != 'PREPARED':
        raise ValueError('Resources are not prepared')
    return {'source_commit': prepared['source_commit'],
            'models': {name: {key: value for key, value in row.items() if key != 'stats'}
                       for name, row in prepared['models'].items()}}
