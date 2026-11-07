#!/usr/bin/env bash
set -euo pipefail
phase="${1:?Usage: bash scripts/setup_env.sh train|eval [venv-directory]}"
case "$phase" in train|eval) ;; *) echo 'phase must be train or eval' >&2; exit 2;; esac
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${2:-$repo_root/.venvs/$phase}"
python_bin="${ROE_PYTHON:-python3.10}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
if [[ -e "$venv_path" ]]; then echo "Refusing to overwrite existing environment: $venv_path" >&2; exit 2; fi
"$python_bin" -m venv "$venv_path"
"$venv_path/bin/python" -m pip install 'pip<26' 'setuptools<70' wheel
"$venv_path/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if [[ "$phase" == train ]]; then
  # DeepSpeed 0.9.5 imports build/runtime helpers during metadata generation.
  "$venv_path/bin/python" -m pip install packaging==23.2 ninja==1.11.1.1 pydantic==1.10.13 numpy==1.26.4 hjson py-cpuinfo psutil==7.2.2
fi
DS_BUILD_OPS=0 "$venv_path/bin/python" -m pip install --no-build-isolation -r "$repo_root/requirements/$phase.txt"
"$venv_path/bin/python" -m pip check
echo "Activate with: source $venv_path/bin/activate"
