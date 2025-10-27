#!/bin/bash

# 训练 LoRA only 模型（灵活标签版本）
# 适用于 WVS 等标签不统一的数据集

# ✅ 配置参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge.json"
OUTPUT_DIR="/root/autodl-fs/output/lora_only_flexible_$(date +%Y%m%d_%H%M)"

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=4
EVAL_BATCH_SIZE=8
LEARNING_RATE=5e-6
LORA_RANK=8
MAX_LENGTH=512
VAL_SPLIT=0.1
SAVE_MODEL=false  # 是否保存模型（true/false）。如果为 false，只保存评估结果

echo "============================================================"
echo "Training LoRA Only Model (Flexible Labels)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Train file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "LoRA rank: $LORA_RANK"
echo "Save model: $SAVE_MODEL"
echo "Note: Supports variable number of classes and label names"
echo "============================================================"
echo ""

# 构建命令
CMD="python train_and_eval_lora_only_flexible.py \
    --model_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_rank $LORA_RANK \
    --max_length $MAX_LENGTH \
    --val_split $VAL_SPLIT"

# 添加可选参数
if [ "$SAVE_MODEL" = "true" ]; then
    CMD="$CMD --save_model"
fi

# ✅ 运行训练
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    if [ "$SAVE_MODEL" = "true" ]; then
        echo "Model saved to: $OUTPUT_DIR"
    fi
    echo "Eval results saved to: $OUTPUT_DIR/eval_results.json"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

