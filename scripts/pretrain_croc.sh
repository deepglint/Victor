#!/bin/bash
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="localhost"
export PORT="29503"
export PYTHONPATH=$(pwd)

LLM_VERSION="lmsys/vicuna-7b-v1.5"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
DATA_ROOT="data_root"

############### Pretrain Croc ################

PROMPT_VERSION=v1

BASE_NAME="croc-pretrain-stage1.5"
SAVE_PATH="./checkpoints/${BASE_NAME}"

echo "SAVE_PATH: ${SAVE_PATH}"

mkdir -p ${SAVE_PATH}

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    croc/train/train_croc.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version v1 \
    --data_path ${DATA_ROOT}/croc_mix1.5m.json \
    --image_folder ${DATA_ROOT}/train_images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --pretrain_mm_mlp_adapter ./checkpoints/croc-pretrain-stage1/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --tune_llm_from_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir  ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --tune_mm_im_head False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --croc_stage True \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing True 2>&1 | tee ${SAVE_PATH}/train_log
