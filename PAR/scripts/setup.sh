#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
venv_path=${1:-.venv}
python_bin=${PAR_PYTHON:-python3.10}
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python3.10"'
if [ ! -e "$venv_path/bin/python" ]; then
  "$python_bin" -m venv "$venv_path"
fi
py="$venv_path/bin/python"
export PYTHONNOUSERSITE=1
"$py" -m pip install 'pip==25.2' 'setuptools>=68,<81' wheel
"$py" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$py" -m pip install -r requirements.txt
"$py" -m pip install --no-deps -e .
"$py" -m pip check
"$py" -c 'import torch,transformers,PIL; print(torch.__version__,transformers.__version__,PIL.__version__); assert transformers.__version__=="4.37.2"'
"$py" -m unittest discover -s tests -v
