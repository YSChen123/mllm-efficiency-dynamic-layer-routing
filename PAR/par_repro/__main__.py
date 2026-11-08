import argparse
import json
from pathlib import Path
from par_repro.config import load_config
from par_repro.io import package_root


def main():
    parser = argparse.ArgumentParser(description='PAR public-settings reconstruction; no claim of complete author-settings equivalence')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--download-models', action='store_true')
    p.add_argument('--download-seed-images', action='store_true')
    p = sub.add_parser('preferences')
    p.add_argument('--split', choices=['training', 'validation'], required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p = sub.add_parser('finalize-preferences')
    p.add_argument('--split', choices=['training', 'validation'], required=True)
    p = sub.add_parser('train')
    p.add_argument('--name', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-steps', type=int)
    p = sub.add_parser('select')
    p.add_argument('--search', type=Path, default=package_root() / 'configs/search.json')
    p = sub.add_parser('evaluate')
    p.add_argument('--benchmark', choices=['pope', 'seed'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--name', default='selected')
    p.add_argument('--limit', type=int)
    p = sub.add_parser('verify')
    p.add_argument('--result-dir', type=Path, required=True)
    p = sub.add_parser('convert-checkpoint')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--gpus', default='0')
    p.add_argument('--smoke', action='store_true')
    sub.add_parser('status')
    args = parser.parse_args()
    config = load_config(args.config)
    workspace = args.work_dir.expanduser().resolve()
    if args.command in ['preferences', 'train', 'select', 'evaluate']:
        from par_repro.runtime import require_idle_gpu
        require_idle_gpu()
    if args.command == 'prepare':
        from par_repro.prepare import prepare
        prepare(workspace, config, args.download_models, args.download_seed_images)
    elif args.command == 'preferences':
        from par_repro.preferences import build_preferences
        build_preferences(workspace, config, args.split, args.limit, args.shard, args.shards)
    elif args.command == 'finalize-preferences':
        from par_repro.preferences import finalize_preferences
        finalize_preferences(workspace, config['training']['mode'], args.split)
    elif args.command == 'train':
        from par_repro.training import train
        train(workspace, config, args.name, args.resume, args.max_steps)
    elif args.command == 'select':
        from par_repro.selection import select
        select(workspace, config, json.loads(args.search.read_text()))
    elif args.command == 'evaluate':
        from par_repro.evaluation import evaluate
        evaluate(workspace, config, args.benchmark, args.checkpoint, args.name, args.limit)
    elif args.command == 'verify':
        from par_repro.evaluation import verify_evaluation
        verify_evaluation(workspace, config, args.result_dir)
    elif args.command == 'convert-checkpoint':
        from par_repro.checkpoints import convert_checkpoint
        convert_checkpoint(args.source, args.destination, config, workspace)
    elif args.command == 'run':
        from par_repro.pipeline import run
        run(workspace, config, args.gpus.split(','), args.smoke)
    else:
        for name in ['pipeline_state.json', 'selection_result.json']:
            path = workspace / name
            if path.exists():
                value = json.loads(path.read_text())
                print(name, json.dumps(value.get('selected', value), ensure_ascii=False))
        for path in sorted((workspace / 'checkpoints').glob('*/training_report.json')):
            value = json.loads(path.read_text())
            print(path.parent.name, value['status'], f"{value['step']}/{value['total_steps']}")


if __name__ == '__main__':
    main()
