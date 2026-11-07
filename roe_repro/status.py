import argparse
import json
from .common import load_config

p=argparse.ArgumentParser();p.add_argument('--config',default='config.local.json');args=p.parse_args()
c=load_config(args.config)
for path in sorted((c['runtime']/'evaluation').glob('*/*/*/*/status.json')):
    d=json.loads(path.read_text())
    print(path.parent.relative_to(c['runtime']),d['state'])
    if 'metrics' in d:print(' metrics:',d['metrics'],'actual skip:',d.get('actual_skip_rate'),'FP32 retries:',d.get('precision_retries'))
    elif 'error' in d:print(' error:',d['error'])
    else:
        log=path.parent/'run.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-12000));tail=f.read().decode(errors='replace')
            print(next((s for s in reversed(tail.replace('\r','\n').splitlines()) if 'Model Responding:' in s),'loading / preparing contexts'))
