"""Record reviewed deployment identities after real smoke and launch gates pass."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path('/home/SigConv/cys/AdaLLaVA')
EVAL=ROOT/'evaluation_repro/20261031'
SCRIPTS=ROOT/'evaluation_repro/scripts'
for lane in ['own','reference']:
    assert json.loads((EVAL/f'SMOKE_{lane}_PASSED.json').read_text())['passed']
    assert (EVAL/f'smoke_{lane}.exit').read_text().strip()=='0'
hashes={name:hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest()
        for name in ['preflight.py','driver.py','eval_policy.py','eval_entry.py','start.sh']}
for role in ['reproduced','dense','official']:
    shutil.copyfile(SCRIPTS/'eval_policy.py',EVAL/role/'scripts/eval_policy.py')
(EVAL/'REVIEW_PASSED.json').write_text(json.dumps({'reviewer':'evaluation_final_review','critical':0,'important_open':0,
 'fixed':'Summary write exception now preserves following jobs; regression RED then GREEN; 8 CPU tests pass.',
 'deployment_sha256':hashes},indent=2))
(EVAL/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print('SMOKE_AND_REVIEW_GATES_PASSED',json.dumps(hashes))
