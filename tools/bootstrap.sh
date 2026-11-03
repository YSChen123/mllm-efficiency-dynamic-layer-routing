#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?Usage: bash tools/bootstrap.sh /absolute/new/runtime}
REPO=$(cd "$(dirname "$0")/.." && pwd)
[[ -f "$ROOT/MIGRATION.json" ]] || { echo 'Run tools/materialize.py first'; exit 1; }
PYTHON=${PYTHON:-python3.10}
"$PYTHON" -c 'import sys; assert sys.version_info[:2]==(3,10), sys.version'
"$PYTHON" -m venv "$ROOT/envs/adallava"
PY="$ROOT/envs/adallava/bin/python"
"$PY" -m pip install 'pip==25.3' 'setuptools==80.10.2' 'wheel==0.48.0'
"$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-deps -r "$REPO/requirements/evaluation.lock"
"$PY" -m pip install --no-deps --no-build-isolation -e "$ROOT/LLaVA" -e "$ROOT/lmms-eval" -e "$ROOT/AdaLLaVA"
"$PY" -m venv --system-site-packages "$ROOT/envs/adallava-train"
"$PY" "$REPO/tools/link_training_env.py" "$PY" "$ROOT/envs/adallava-train/bin/python"
DS_BUILD_OPS=0 "$ROOT/envs/adallava-train/bin/python" -m pip install --no-deps -r "$REPO/requirements/training-extra.lock"
# Keep the eval environment's bundled NCCL intact; only the train venv preloads 2.31.2.
"$ROOT/envs/adallava-train/bin/python" -m pip install --no-deps nvidia-nccl-cu12==2.31.2
"$ROOT/envs/adallava-train/bin/python" - <<'PY'
import torch,transformers,accelerate,deepspeed,adallava,llava,lmms_eval
print('TRAIN_IMPORTS_VERIFIED',torch.__version__,transformers.__version__,accelerate.__version__,deepspeed.__version__)
PY
"$PY" -m pip freeze > "$ROOT/docs/evaluation-pip-freeze.txt"
"$ROOT/envs/adallava-train/bin/python" -m pip freeze > "$ROOT/docs/training-pip-freeze.txt"
echo "Environment installed. Run README input preparation, new smoke/resume and GPU checks before formal training."
