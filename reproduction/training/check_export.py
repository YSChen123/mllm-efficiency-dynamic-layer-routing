"""Reload the real gathered portable checkpoint and match saved master weights."""
import json,sys,torch
from pathlib import Path
ROOT=Path('/home/SigConv/cys/AdaLLaVA')
sys.path.insert(0,str(ROOT/'training_repro/source/AdaLLaVA/src'))
from adallava.model.ada_llava_llama import AdaLlavaLlamaForCausalLM
run=ROOT/'training_repro/smoke/run'
assert (run/'SMOKE_EXPORTED.json').is_file()
model,info=AdaLlavaLlamaForCausalLM.from_pretrained(run,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
assert not info['missing_keys'],info['missing_keys']
assert all('vision_tower' in key for key in info['unexpected_keys']),info['unexpected_keys']
assert not info['mismatched_keys'],info['mismatched_keys']
events=[json.loads(line) for line in (run/'metrics_rank0.jsonl').read_text().splitlines()]
# STEP snapshots in the last checkpoint proof refer to step 2; exported-after-resume model is step 3.
# Capture exact current values from the export event written by the trainer before gathering.
proof=next(row for row in reversed(events) if row['event']=='EXPORT_PROOF')
params=dict(model.named_parameters())
for name,expected in proof['fp32'].items():
    wanted=torch.tensor(expected,dtype=torch.bfloat16)
    actual=params[name].detach().flatten()[:len(expected)]
    assert torch.equal(actual,wanted),(name,(actual.float()-wanted.float()).abs().max().item())
report={'passed':True,'weights_matched':list(proof['fp32']),'missing_keys':info['missing_keys'],
        'unexpected_vision_keys':len(info['unexpected_keys']),'checkpoint':str(run)}
(ROOT/'training_repro/smoke/EXPORT_RELOAD_PASSED.json').write_text(json.dumps(report,indent=2))
print('PORTABLE_EXPORT_RELOAD_PASSED',json.dumps(report))
