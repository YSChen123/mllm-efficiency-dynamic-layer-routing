#!/usr/bin/env bash
set -euo pipefail
cd "/home/SigConv/cys/AdaLLaVA/AdaLLaVA"
exec "/home/SigConv/cys/AdaLLaVA/envs/adallava/bin/python" "/home/SigConv/cys/AdaLLaVA/scripts/eval_entry.py" "$@"
