set -euo pipefail
cd /home/SigConv/cys/AdaLLaVA
envs/adallava/bin/python - <<'PY'
from pathlib import Path
import json,shutil
root=Path('/home/SigConv/cys/AdaLLaVA/training_repro')
proof=json.loads((root/'smoke/SMOKE_PASSED.json').read_text());assert proof['passed']
assert json.loads((root/'smoke/EXPORT_RELOAD_PASSED.json').read_text())['passed']
run=root/'smoke/run';cp=Path(proof['checkpoint']);assert cp.resolve()==run/'checkpoint-2'
assert not cp.is_symlink() and (cp/'COMPLETE.json').is_file()
evidence=root/'smoke/evidence';evidence.mkdir(exist_ok=True)
inventory={str(p.relative_to(run)):p.stat().st_size for p in run.rglob('*') if p.is_file()}
(evidence/'all_file_sizes.json').write_text(json.dumps(inventory,indent=2))
for path in run.rglob('*'):
 if not path.is_file() or path.stat().st_size>1024**2:continue
 target=evidence/path.relative_to(run);target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(path,target)
shutil.rmtree(cp)
removed=[]
for path in list(run.glob('model*.safetensors'))+list(run.glob('pytorch_model*.bin')):
 assert path.resolve().parent==run and not path.is_symlink()
 removed.append((path.name,path.stat().st_size));path.unlink()
proof['debug_weights_retained']=False
proof['evidence_path']=str(evidence)
(root/'smoke/SMOKE_PASSED.json').write_text(json.dumps(proof,indent=2))
print('SMOKE_EVIDENCE_ARCHIVED_DEBUG_WEIGHTS_REMOVED',removed)
PY
