#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1
export NCCL_DEBUG=INFO  # 调试信息（可选）

# 训练参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"  # 使用新的双路输入数据集
OUTPUT_DIR="/root/autodl-fs/output/CultureMoE/culturemoe_$(date +%Y%m%d_%H%M)"

# ✅ 使用 torchrun 启动分布式训练（双路输入 + LLaMA LoRA 微调）
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    examples/train_classification.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --fp16 \
    --gradient_checkpointing \
    --use_dual_input True \
    --freeze_llama False \
    --use_llama_lora True \
    --llama_lora_rank 8 \
    --llama_lora_alpha 16 \
    --llama_lora_dropout 0.05 \
    --llama_lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --max_length 512 \
    --val_split 0.1 \
    --dataloader_num_workers 4

