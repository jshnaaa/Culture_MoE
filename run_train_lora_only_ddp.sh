#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1
export NCCL_DEBUG=INFO  # 调试信息（可选）

# 训练参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
OUTPUT_DIR="/root/autodl-fs/output/lora_only/lora_only_$(date +%Y%m%d_%H%M)"

# ✅ 使用 torchrun 启动分布式训练（LoRA 微调，无 MoE）
torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    train_and_eval_lora_only.py \
    --model_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-6 \
    --lora_rank 8 \
    --max_length 512 \
    --val_split 0.1

echo ""
echo "============================================================"
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "   Model saved to: $OUTPUT_DIR"
else
    echo "❌ Training failed!"
fi
echo "============================================================"

