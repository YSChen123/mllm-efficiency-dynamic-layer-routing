set -euo pipefail
ROOT=/home/SigConv/cys/AdaLLaVA
cd "$ROOT"
export CONDA_PKGS_DIRS="$ROOT/cache/conda:/opt/anaconda3/pkgs"
if [ ! -x "$ROOT/envs/adallava/bin/python" ]; then
  /opt/anaconda3/bin/conda create -y -p "$ROOT/envs/adallava" python=3.10 pip --offline
fi
"$ROOT/envs/adallava/bin/python" --version
touch "$ROOT/logs/env_created.ok"
