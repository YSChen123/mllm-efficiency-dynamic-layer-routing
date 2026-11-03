"""Runtime-only measurement hooks around the unmodified official generation."""
import json
import os
from pathlib import Path
import runpy
import time

import torch
import datasets
from local_dataset import load_prepared_dataset

if os.getenv('ADA_CUDA_MEMORY_GIB'):
    cap = float(os.environ['ADA_CUDA_MEMORY_GIB']) * 1024 ** 3
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / torch.cuda.get_device_properties(0).total_memory))
    print('ADA_CUDA_ALLOCATOR_CAP_GIB', os.environ['ADA_CUDA_MEMORY_GIB'], flush=True)

# datasets 2.16 may resolve Hub metadata even when all Arrow data is cached.
# Route only this explicitly selected public benchmark to its verified local data.
manifest = json.loads(Path(os.environ['ADA_DATASET_MANIFEST']).read_text())
original_load_dataset = datasets.load_dataset


def local_load_dataset(path, name=None, *args, **kwargs):
    if path != manifest['repo']:
        return original_load_dataset(path, name, *args, **kwargs)
    if name != manifest['config'] or kwargs.get('revision') != manifest['revision']:
        raise ValueError('Evaluation dataset configuration/revision differs from prepared manifest')
    data = load_prepared_dataset(manifest)
    split = kwargs.get('split')
    return data[split] if split else data


datasets.load_dataset = local_load_dataset
from src.adallava.eval.adallava_wrapper import AdaLlava

original_init = AdaLlava.__init__


def measured_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    original_generate = self.model.generate
    observed = {}

    # Upstream invokes scheduler=self.scheduler.forward directly, bypassing
    # nn.Module.__call__ hooks. Wrap the actual callable without changing it.
    original_scheduler_forward = self.model.scheduler.forward

    def observed_scheduler_forward(hidden_states, latency):
        observed['scheduler_observed_budget'] = float(latency.detach().float().item())
        return original_scheduler_forward(hidden_states, latency)

    self.model.scheduler.forward = observed_scheduler_forward

    def generate(*generation_args, **generation_kwargs):
        observed.clear()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_time = time.perf_counter()
        start_event.record()
        output = original_generate(*generation_args, **generation_kwargs)
        end_event.record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - start_time
        plan = [int(x) for x in output.execution_plan]
        executed = sum(x >= 0 for x in plan)
        row = {'target_budget': float(generation_kwargs['latency']), **observed,
               'generation_wall_seconds': wall,
               'generation_cuda_event_ms': start_event.elapsed_time(end_event),
               'executed_layers': executed, 'total_layers': len(plan),
               'layer_skipping_ratio': 1 - executed / len(plan),
               'execution_plan_heads_per_layer': plan,
               'prompt_len': int(output.prompt_len), 'generated_len': int(output.gen_len),
               'phase': 'full_generate_including_vision_prefill_decode',
               'cold_first_sample': not getattr(self, '_measurement_has_sample', False)}
        row['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 1024 ** 2
        row['cuda_peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 1024 ** 2
        row['shared_gpu'] = os.getenv('ADA_SHARED_GPU') == '1'
        if 'scheduler_observed_budget' not in row:
            raise RuntimeError('Scheduler did not execute during generation')
        if abs(row['target_budget'] - row['scheduler_observed_budget']) > 1e-3:
            raise RuntimeError('Observed scheduler input differs from requested budget')
        self._measurement_has_sample = True
        with Path(os.environ['ADA_MEASUREMENTS']).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        print('ADA_INFERENCE ' + json.dumps(row), flush=True)
        return output

    self.model.generate = generate


AdaLlava.__init__ = measured_init
runpy.run_module('adallava.eval.run_lmms_eval', run_name='__main__')
