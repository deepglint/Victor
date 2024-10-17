#!/bin/bash
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="localhost"
export PORT="29500"
export PYTHONPATH=$(pwd)

LLM_VERSION="lmsys/vicuna-7b-v1.5"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
DATA_ROOT="data_root"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_NAME="croc-pretrain-stage1"
SAVE_PATH="./checkpoints/${BASE_NAME}"

echo "SAVE_PATH: ${SAVE_PATH}"

mkdir -p ${SAVE_PATH}

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    croc/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version plain \
    --data_path ${DATA_ROOT}/blip_laion_cc_sbu_558k.json \
    --image_folder ${DATA_ROOT}/LLaVA-Pretrain/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True 2>&1 | tee ${SAVE_PATH}/train_log
