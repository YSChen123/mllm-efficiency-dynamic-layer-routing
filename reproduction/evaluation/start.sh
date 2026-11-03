#!/usr/bin/env bash
set -euo pipefail
cd /home/SigConv/cys/AdaLLaVA
mode=${1:-smoke}
if [[ "$mode" != smoke && "$mode" != run ]];then exit 2;fi
for pair in 'own 0' 'reference 3';do
 read -r lane gpu <<< "$pair"
 session="ada_eval_${mode}_${lane}"
 if tmux has-session -t "$session" 2>/dev/null;then echo "SESSION_EXISTS $session";exit 1;fi
 stamp=$(date +%Y%m%d_%H%M%S)
 log="evaluation_repro/20261031/${mode}_${lane}_${stamp}.log"
 tmux new-session -d -s "$session" "cd /home/SigConv/cys/AdaLLaVA; envs/adallava/bin/python -u evaluation_repro/scripts/driver.py $mode --lane $lane --gpu $gpu > $log 2>&1; code=\$?; printf '%s\\n' \"\$code\" > evaluation_repro/20261031/${mode}_${lane}.exit"
 ln -sfn "$(basename "$log")" "evaluation_repro/20261031/${mode}_${lane}.log"
 printf 'STARTED %s GPU=%s LOG=%s\n' "$session" "$gpu" "$log"
done
