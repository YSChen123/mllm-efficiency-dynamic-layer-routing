#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/SigConv/cys/AdaLLaVA
cd "$ROOT"
export ADA_GPU=2
printf 'Bootstrap started %s\n' "$(date -Is)"
# Waiting is done on the server. Keep interactive checks infrequent.
for step in $(seq 1 240); do
 if [ -f logs/download_models.exit ]; then
  if [ "$(cat logs/download_models.exit)" != 0 ]; then
   printf 'Model preparation failed; see logs/download_models.log\n'
   exit 91
  fi
  break
 fi
 sleep 30
done
test -f configs/models_ready.json || { echo MODEL_NOT_READY; exit 92; }
"$ROOT/envs/adallava/bin/python" "$ROOT/scripts/verify_models.py"
for step in $(seq 1 120); do
 if [ -f logs/prepare_bootstrap.exit ]; then
  if [ "$(cat logs/prepare_bootstrap.exit)" != 0 ]; then
   printf 'ScienceQA preparation failed; see logs/prepare_scienceqa_img.log\n'
   exit 93
  fi
  break
 fi
 sleep 30
done
test -f configs/datasets/scienceqa_img.json || { echo DATASET_NOT_READY; exit 94; }
"$ROOT/envs/adallava/bin/python" "$ROOT/scripts/pipeline.py" smoke
printf 'Smoke tests passed; starting formal evaluation %s\n' "$(date -Is)"
exec bash "$ROOT/scripts/run_pipeline.sh"
