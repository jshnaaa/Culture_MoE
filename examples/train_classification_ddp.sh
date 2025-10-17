#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1
export NCCL_DEBUG=INFO  # 调试信息（可选）

# 训练参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/normad_ed.json"
OUTPUT_DIR="/root/autodl-fs/output/classi_$(date +%Y%m%d_%H%M%S)"

# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    examples/train_classification.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --fp16 \
    --gradient_checkpointing \
    --freeze_llama \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 500 \
    --max_length 512 \
    --val_split 0.1 \
    --dataloader_num_workers 4