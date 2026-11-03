#!/usr/bin/env bash
set -euo pipefail
export ADA_GPU=2
exec "/home/SigConv/cys/AdaLLaVA/envs/adallava/bin/python" "/home/SigConv/cys/AdaLLaVA/scripts/supplement_060.py" run
