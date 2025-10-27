#!/bin/bash

# 训练 LoRA only 模型（固定标签版本）
# 支持 LLaMA 3.1 和 Qwen 2.5

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1
export NCCL_DEBUG=INFO  # 调试信息（可选）

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama，可以通过第一个参数指定 qwen
TRAIN_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    OUTPUT_DIR="/root/autodl-fs/output/lora_only_qwen/lora_only_qwen_$(date +%Y%m%d_%H%M)"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_DIR="/root/autodl-fs/output/lora_only_llama/lora_only_llama_$(date +%Y%m%d_%H%M)"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

echo "============================================================"
echo "Training LoRA Only Model - $MODEL_NAME"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Model: $MODEL_PATH"
echo "Train file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# ✅ 使用 torchrun 启动分布式训练（LoRA 微调，无 MoE）
torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    train_and_eval_lora_only.py \
    --model_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --backbone $BACKBONE \
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

