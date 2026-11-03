#!/usr/bin/env bash
set -euo pipefail
exec "/home/SigConv/cys/AdaLLaVA/envs/adallava/bin/python" "/home/SigConv/cys/AdaLLaVA/scripts/pipeline.py" "run" "$@"
