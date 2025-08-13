#!/bin/bash
export PYTHONPATH=$(pwd)

LLM_VERSION="./checkpoints/llava-clip-qwen7b-victor"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
export WANDB_MODE=disabled

export RANK=0
export NUM_GPUS=8  
export NNODES=1    
export ADDR="localhost"  
export PORT=29502  
PROMPT_VERSION="qwen_1_5"

RUN_NAME="llava-clip-qwen7b-victor-740ksft"
echo "RUN_NAME: ${RUN_NAME}"
mkdir -p "./checkpoints/${RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path llava_next_raw_format/llava_next_raw_format_processed.json \
    --image_folder llava_next_raw_format \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --mm_patch_merge_type flat \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir "./checkpoints/${RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 321120 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --victor_stage False \
    --dataloader_drop_last True | tee "./checkpoints/${RUN_NAME}/$(date +%Y%m%d_%H%M%S).log"
