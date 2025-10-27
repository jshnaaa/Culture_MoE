#!/bin/bash

# 训练 CultureMoE 模型（灵活标签版本）
# 适用于 WVS 等标签不统一的数据集

# ✅ 配置参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge.json"
OUTPUT_DIR="/root/autodl-fs/output/culturemoe_flexible_$(date +%Y%m%d_%H%M)"

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=4
EVAL_BATCH_SIZE=8
LEARNING_RATE=2e-5

# MoE 参数
NUM_EXPERTS=6
LORA_RANK=16

# LLaMA 参数
FREEZE_LLAMA=true  # 是否冻结 LLaMA
USE_LLAMA_LORA=false  # 是否对 LLaMA 使用 LoRA
LLAMA_LORA_RANK=8

# 其他参数
MAX_LENGTH=512
VAL_SPLIT=0.1

echo "============================================================"
echo "Training CultureMoE Model (Flexible Labels)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Train file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Num experts: $NUM_EXPERTS"
echo "LoRA rank: $LORA_RANK"
echo "Freeze LLaMA: $FREEZE_LLAMA"
echo "Use LLaMA LoRA: $USE_LLAMA_LORA"
echo "Note: Supports variable number of classes and label names"
echo "============================================================"
echo ""

# 构建命令
CMD="python train_culturemoe_flexible.py \
    --model_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_experts $NUM_EXPERTS \
    --lora_rank $LORA_RANK \
    --max_length $MAX_LENGTH \
    --val_split $VAL_SPLIT \
    --llama_lora_rank $LLAMA_LORA_RANK"

# 添加可选参数
if [ "$FREEZE_LLAMA" = "true" ]; then
    CMD="$CMD --freeze_llama"
fi

if [ "$USE_LLAMA_LORA" = "true" ]; then
    CMD="$CMD --use_llama_lora"
fi

# ✅ 运行训练
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    echo "Model saved to: $OUTPUT_DIR"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

