"""Run only the four approved 0.60-budget supplements using the verified pipeline."""
import argparse
import os
import sys
import subprocess
import time

import pipeline as p
from recovery_helpers import select_gpu

BASELINE_BUDGETS = ['1.0', '0.85', '0.70', '0.55']
SUPPLEMENT_BUDGET = '0.60'
APPROVED_TASKS = ['scienceqa_img', 'pope', 'mme', 'textvqa_val']
original_collect = p.collect


def check_available_gpu():
    # The user explicitly permits sharing a card if its free VRAM is sufficient.
    deadline = time.monotonic() + 86400
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
        try:
            gpu = select_gpu(rows, minimum_mib=26 * 1024)
            os.environ['ADA_GPU'] = gpu['index']
            # Bound only our PyTorch allocator, leaving >=4 GiB startup headroom.
            os.environ['ADA_CUDA_MEMORY_GIB'] = '22'
            os.environ['ADA_SHARED_GPU'] = '1'
            p.atomic_json(p.ROOT / 'configs/supplement_060_gpu.json',
                          {**gpu, 'updated_utc': p.now(), 'allocator_cap_gib': 22,
                           'selection_policy': 'user-authorized available-VRAM sharing'})
            print('SELECTED_GPU', gpu, flush=True)
            return
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError('No GPU met the free-VRAM requirement within 24 hours') from error
            p.update_status('WAITING_FOR_GPU_MEMORY', task='textvqa_val', budget=SUPPLEMENT_BUDGET, error=str(error))
            time.sleep(60)


def collect_combined():
    saved_budgets, saved_tasks = p.BUDGETS, p.TASKS
    try:
        p.BUDGETS = BASELINE_BUDGETS + [SUPPLEMENT_BUDGET]
        p.TASKS = {name: saved_tasks[name] for name in APPROVED_TASKS}
        original_collect()
    finally:
        p.BUDGETS, p.TASKS = saved_budgets, saved_tasks


def main(mode):
    p.TASKS = {name: p.TASKS[name] for name in APPROVED_TASKS}
    p.BUDGETS = [SUPPLEMENT_BUDGET]
    p.collect = collect_combined
    if mode == 'collect':
        collect_combined()
        return 0
    # The previous four-budget cleanup gate is intentionally kept unchanged.
    # There is enough quota for all four isolated caches plus the 15-GiB reserve.
    # Keep supplemental data on success and on failure; never delete old evidence.
    os.environ['ADA_CLEAN_COMPLETED_DATA'] = '0'
    os.environ.setdefault('ADA_GPU', '2')
    p.check_gpu = check_available_gpu
    try:
        return p.pipeline()
    except BlockingIOError:
        print('ALREADY_RUNNING: pipeline lock is held', file=sys.stderr)
        return 73
    except Exception as error:
        p.update_status('FAILED', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'collect'])
    raise SystemExit(main(parser.parse_args().mode))
