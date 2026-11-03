#!/usr/bin/env bash
set -uo pipefail
cd /home/SigConv/cys/AdaLLaVA
stamp=$(date +%Y%m%d_%H%M%S)
started=$(date +%s)
nvidia-smi --id=0,1 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv -l 30 > "training_repro/logs/gpu_${stamp}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
printf 'MANAGED_TRAIN_START %s %s\n' "$started" "${1:-fresh}"
bash training_repro/run_train.sh formal "${1:-fresh}"
code=$?
ended=$(date +%s)
printf 'MANAGED_TRAIN_END start=%s end=%s exit=%s\n' "$started" "$ended" "$code"
printf '%s\n' "$code" > training_repro/logs/formal.exit
exit "$code"
