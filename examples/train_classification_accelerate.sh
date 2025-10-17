#!/bin/bash

# 首先配置 accelerate（只需运行一次）
# accelerate config

# 使用 accelerate 启动训练
accelerate launch \
    --mixed_precision fp16 \
    --num_processes 2 \
    examples/train_classification.py \
    --model_name_or_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct \
    --train_file /root/autodl-fs/normad_ed.json \
    --output_dir /root/autodl-fs/output/classi_$(date +%Y%m%d_%H%M%S) \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --fp16 \
    --gradient_checkpointing \
    --freeze_llama