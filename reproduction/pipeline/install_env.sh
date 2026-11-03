set -euo pipefail
ROOT=/home/SigConv/cys/AdaLLaVA
export CONDA_PKGS_DIRS="$ROOT/cache/conda:/opt/anaconda3/pkgs"
export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/cache/pip"
if [ ! -x "$ROOT/envs/adallava/bin/python" ]; then
 /opt/anaconda3/bin/conda create -y -p "$ROOT/envs/adallava" python=3.10 pip --override-channels -c https://repo.anaconda.com/pkgs/main
fi
PY="$ROOT/envs/adallava/bin/python"
"$PY" -m pip install --no-cache-dir 'pip<26' 'setuptools<81' wheel
"$PY" -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-cache-dir 'transformers==4.37.2' 'tokenizers==0.15.1' 'peft==0.13.2' 'accelerate==0.34.2' 'huggingface-hub==0.26.5' 'numpy==1.26.4' 'datasets==2.16.1' 'pyarrow==17.0.0' 'evaluate==0.4.3' 'sentencepiece==0.1.99' 'protobuf==3.20.3' 'einops==0.6.1' 'einops-exts==0.0.4' 'timm==0.6.13' 'scikit-learn==1.2.2' 'Pillow==10.4.0' shortuuid safetensors loguru jsonlines numexpr pytablewriter sacrebleu sqlitedict ftfy 'opencv-python-headless==4.10.0.84' nltk tqdm-multiprocess 'httpx==0.23.3' 'tenacity==8.3.0' openpyxl wandb tiktoken zstandard pycocotools 'openai<2' 'pydantic<3' pytest
"$PY" -m pip install --no-deps -e "$ROOT/LLaVA" -e "$ROOT/lmms-eval" -e "$ROOT/AdaLLaVA"
"$PY" -m pip freeze > "$ROOT/docs/pip_freeze.txt"
CUDA_VISIBLE_DEVICES=2 "$PY" - <<'PY'
import torch,transformers,peft
print('VERSIONS',torch.__version__,torch.version.cuda,transformers.__version__,peft.__version__)
print('CUDA',torch.cuda.get_device_name(),torch.cuda.get_device_capability())
a=torch.randn(256,256,device='cuda',dtype=torch.float16)
b=a@a
torch.cuda.synchronize()
assert torch.isfinite(b).all()
print('CUDA_MATMUL_PASS')
PY
touch "$ROOT/logs/install.ok"
