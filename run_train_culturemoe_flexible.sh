#!/bin/bash

# 训练 CultureMoE 模型（支持不同类别数）
# 支持 LLaMA 3.1 和 Qwen 2.5
# 适用于 WVS 等多分类数据集

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama，可以通过第一个参数指定 qwen
NUM_CLASSES="${2:-4}"   # 默认 4 分类，可以通过第二个参数指定其他值（如 5）
TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge.json"

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    OUTPUT_DIR="/root/autodl-fs/output/culturemoe_qwen_${NUM_CLASSES}class_$(date +%Y%m%d_%H%M)"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_DIR="/root/autodl-fs/output/culturemoe_llama_${NUM_CLASSES}class_$(date +%Y%m%d_%H%M)"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

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
echo "Training CultureMoE Model - $MODEL_NAME (${NUM_CLASSES}-class)"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Model: $MODEL_PATH"
echo "Train file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "Num classes: $NUM_CLASSES"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Num experts: $NUM_EXPERTS"
echo "LoRA rank: $LORA_RANK"
echo "Freeze LLaMA: $FREEZE_LLAMA"
echo "Use LLaMA LoRA: $USE_LLAMA_LORA"
echo "============================================================"
echo ""

# 构建命令
CMD="python train_culturemoe_flexible.py \
    --model_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --backbone $BACKBONE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_experts $NUM_EXPERTS \
    --lora_rank $LORA_RANK \
    --max_length $MAX_LENGTH \
    --val_split $VAL_SPLIT \
    --llama_lora_rank $LLAMA_LORA_RANK \
    --num_classes $NUM_CLASSES"

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

