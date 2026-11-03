"""Fail closed unless storage policy is explicit on a newly deployed server."""
import os,shutil
from pathlib import Path

def main_quota():
    root=Path(__file__).resolve().parents[1]
    disk=shutil.disk_usage(root)
    limit=os.getenv('ADA_STORAGE_LIMIT_GIB')
    if limit:
        # Administrator/user supplies this ACCOUNT'S available quota, not the whole disk size.
        available=int(float(limit)*1024**3)
        if available<=0:raise RuntimeError('ADA_STORAGE_LIMIT_GIB must be positive')
        # This is an available-space cap, refreshed by callers between stages as documented.
        initial_file=root/'.storage-policy.json'
        import json
        used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        if initial_file.exists():initial=json.loads(initial_file.read_text())
        else:
            initial={'initial_used':used,'available_bytes':available}
            initial_file.write_text(json.dumps(initial))
        remaining=min(disk.free,max(0,initial['available_bytes']-(used-initial['initial_used'])))
    elif os.getenv('ADA_NO_QUOTA')=='1':remaining=disk.free
    else:
        raise RuntimeError('Set ADA_STORAGE_LIMIT_GIB to current account quota headroom, or ADA_NO_QUOTA=1 only for a filesystem without user quotas. See README.')
    return {'used':0,'soft':remaining}

def require_space(root,gib):
    quota=main_quota();physical=shutil.disk_usage(root).free
    remaining=min(quota['soft']-quota['used'],physical)
    if remaining<gib*1024**3:raise RuntimeError(f'Insufficient storage: {remaining/1024**3:.1f} GiB available; {gib:.1f} GiB required')
    return {'remaining_home_quota_gib':remaining/1024**3,'reserve_gib':gib}
