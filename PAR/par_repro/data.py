from pathlib import Path
import json
import random
import re
from par_repro.io import atomic_json, bind_identity, file_sha, package_root, object_sha


def image_id(value):
    match = re.search(r'(\d+)(?=\.[^.]+$)', Path(str(value)).name)
    return int(match.group(1)) if match else None


def choose_training(records, count=5000, seed=0):
    return [{'source_index': i, 'record': records[i]} for i in random.Random(seed).sample(range(len(records)), count)]


def choose_validation(records, training_images, benchmark_images, count=256, seed=20261108):
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    seen = set(training_images) | set(benchmark_images)
    selected = []
    for i in order:
        record = records[i]
        key = image_id(record.get('image', ''))
        if key is None or key in seen or not any(t.get('from') == 'human' for t in record.get('conversations', [])):
            continue
        seen.add(key)
        selected.append({'source_index': i, 'record': record})
        if len(selected) == count:
            return selected
    raise ValueError('Not enough disjoint validation images')


def seed_prompt(case):
    return (case['question'] + '\n' + f"A. {case['choice_a']}\nB. {case['choice_b']}\nC. {case['choice_c']}\nD. {case['choice_d']}\n"
            + "Answer with the option's letter from the given choices directly.")


def read_annotations(workspace):
    from par_repro.prepare import verify_resource
    folder = Path(workspace) / 'annotations'
    protocol = json.loads((package_root() / 'configs/annotations.json').read_text())
    for category, digest in protocol['pope']['sha256'].items():
        verify_resource(folder / f'coco_pope_{category}.json', {'sha256': digest})
    verify_resource(folder / 'SEED-Bench.json', protocol['seed'])
    pope = []
    for category in ['adversarial', 'popular', 'random']:
        pope.extend(dict(json.loads(line), category=category) for line in (folder / f'coco_pope_{category}.json').read_text().splitlines())
    seed = [r for r in json.loads((folder / 'SEED-Bench.json').read_text())['questions'] if r['data_type'] == 'image']
    if len(pope) != 8910 or len(seed) != 14233:
        raise ValueError('Wrong annotation release/count')
    return pope, seed


def prepare_selection(workspace, config):
    workspace = Path(workspace)
    records = json.loads((workspace / 'annotations/llava_instruct_80k.json').read_text())
    training = choose_training(records)
    pope, seed = read_annotations(workspace)
    train_images = {image_id(r['record']['image']) for r in training}
    benchmark_images = {image_id(r['image']) for r in pope} | {image_id(r['data_id']) for r in seed}
    validation = choose_validation(records, train_images, benchmark_images)
    for name, rows in [('training', training), ('validation', validation)]:
        canonical = [{'source_index': r['source_index'], 'id': str(r['record']['id']), 'image': r['record']['image']} for r in rows]
        expected = package_root() / 'configs' / f'{name}_ids.json'
        if expected.exists() and canonical != json.loads(expected.read_text()):
            raise ValueError(f'Canonical {name} IDs differ from recorded experiment')
        bind_identity(workspace / 'selection' / f'{name}.json', rows)
    bind_identity(workspace / 'selection/config.json', {'training_count': 5000, 'training_seed': 0,
        'validation_count': 256, 'validation_seed': 20261108, 'annotation_sha256': file_sha(workspace / 'annotations/llava_instruct_80k.json')})


def resolve_image(root, name):
    root = Path(root).expanduser().resolve()
    direct = (root / name).resolve()
    if direct.is_relative_to(root) and direct.is_file():
        return direct
    base = root / Path(name).name
    if base.is_file():
        return base
    raise FileNotFoundError(f'Image unavailable under configured directory: {name}')


def index_seed_images(root):
    index = {}
    for path in Path(root).rglob('*'):
        if path.is_file() and not path.name.startswith('.') and path.suffix.lower() in ['', '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff']:
            if path.stem in index:
                raise ValueError('Duplicate SEED image stem: ' + path.stem)
            index[path.stem] = path
    return index
