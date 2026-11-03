import hashlib
import json
from pathlib import Path

root=Path('/home/SigConv/cys/AdaLLaVA')
checkpoint=root/'checkpoints/ada-llava-L-v1.5-7b'
vision=root/'checkpoints/clip-vit-large-patch14-336'
alias=root/'models/ada-llava-L-v1.5-7b'
if alias.is_symlink():
    alias.unlink()  # Remove only this project's link; all weights remain in checkpoints/.
alias.mkdir(exist_ok=True)
for path in checkpoint.iterdir():
    if path.is_file() and path.name!='config.json' and not (alias/path.name).exists():
        (alias/path.name).symlink_to(path)
config=json.loads((checkpoint/'config.json').read_text())
config['mm_vision_tower']=str(vision)
(alias/'config.json').write_text(json.dumps(config,indent=2))
index=json.loads((checkpoint/'model.safetensors.index.json').read_text())
assert any('scheduler' in key for key in index['weight_map'])
for name in set(index['weight_map'].values()):
    assert (checkpoint/name).is_file()
revisions=set()
for name in ['config.json','preprocessor_config.json','pytorch_model.bin']:
    fields=(vision/'.cache/huggingface/download'/f'{name}.metadata').read_text().splitlines()
    revisions.add(fields[0])
    if name.endswith('.bin'):
        digest=hashlib.sha256()
        with (vision/name).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(chunk)
        assert digest.hexdigest()==fields[1].strip('"'), 'Vision SHA256 mismatch'
assert len(revisions)==1, 'Vision files are from inconsistent revisions'
metadata=json.loads((root/'configs/models_ready.json').read_text())
metadata.update(vision_revision=revisions.pop(),vision_sha256=digest.hexdigest(),loader_path=str(alias))
(root/'configs/models_ready.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
