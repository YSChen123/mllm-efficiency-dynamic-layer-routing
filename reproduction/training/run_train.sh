#!/usr/bin/env bash
set -euo pipefail
cd /home/SigConv/cys/AdaLLaVA
mode=${1:-smoke}
action=${2:-fresh}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export LD_PRELOAD=/home/SigConv/cys/AdaLLaVA/envs/adallava-train/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export PYTHONPATH=/home/SigConv/cys/AdaLLaVA/training_repro/source/AdaLLaVA/src:/home/SigConv/cys/AdaLLaVA/LLaVA
export ADA_RUN_MODE=$mode
export ADA_RESUME=0
if [[ $action == resume ]];then export ADA_RESUME=1;fi
export ADA_OUTPUT=/home/SigConv/cys/AdaLLaVA/training_repro/smoke/run
data=training_repro/data/smoke32.json
extra=(--max_steps 3)
accum=4
micro=2
save=2
export ADA_STOP_AT=2
if [[ $mode == formal ]];then
 export ADA_OUTPUT=/home/SigConv/cys/AdaLLaVA/training_repro/runs/adallava-L-20261027
 data=training_repro/data/llava_v1_5_mix665k.json
 extra=()
 accum=32
 micro=2
 save=500
 export ADA_STOP_AT=0
fi
mkdir -p "$ADA_OUTPUT"
exec envs/adallava-train/bin/deepspeed --include localhost:0,1 --master_port 29727 training_repro/train_entry.py \
 --deepspeed training_repro/zero3.json \
 --model_name_or_path training_repro/models/llava-v1.5-7b --version v1 \
 --data_path "$data" --image_folder training_repro/data/images \
 --vision_tower /home/SigConv/cys/AdaLLaVA/checkpoints/clip-vit-large-patch14-336 \
 --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 --mm_use_im_start_end False --mm_use_im_patch_token False \
 --image_aspect_ratio pad --group_by_modality_length True --freeze_backbone False \
 --bf16 True --num_prefix_layers 16 --token_selecting none --scheduler_type L \
 --output_dir "$ADA_OUTPUT" --num_train_epochs 1 --per_device_train_batch_size "$micro" --gradient_accumulation_steps "$accum" \
 --evaluation_strategy no --save_strategy steps --save_steps "$save" --save_total_limit 1 \
 --learning_rate 1e-5 --weight_decay 0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
 --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 4 \
 --lazy_preprocess True --report_to none --run_name adallava-L-20261027 --seed 42 "${extra[@]}"
