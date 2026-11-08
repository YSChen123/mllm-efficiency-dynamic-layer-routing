from pathlib import Path
import hashlib
import json
import os
import zipfile


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
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def bind_identity(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Identity changed; use a new output directory: {path}')
    else:
        atomic_json(path, value)


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
    return object_sha({str(path.relative_to(root)): file_sha(path) for path in sorted(root.glob('*.py'))})


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
