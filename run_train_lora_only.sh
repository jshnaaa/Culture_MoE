#!/bin/bash

# 训练 LoRA only 模型（支持 2/3/4/5 分类）
# 支持 LLaMA 3.1 和 Qwen 2.5

# ✅ 限制只使用一个 GPU（避免 DataParallel 问题）
export CUDA_VISIBLE_DEVICES=0

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama，可以通过第一个参数指定 qwen
NUM_CLASSES="${2:-2}"   # 默认 2 分类，可以通过第二个参数指定其他值（2/3/4/5）
SAVE_MODEL="${3:-false}"  # 默认不保存模型，可以通过第三个参数指定 true/false

# 根据 num_classes 选择数据集
case $NUM_CLASSES in
    2)
        TRAIN_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
        ;;
    3)
        TRAIN_FILE="/root/autodl-fs/normad_ed_merge.json"
        ;;
    4)
        TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge_4.json"
        ;;
    5)
        TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge_5.json"
        ;;
    *)
        echo "❌ Error: Invalid num_classes=$NUM_CLASSES. Must be 2, 3, 4, or 5."
        exit 1
        ;;
esac

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    OUTPUT_DIR="/root/autodl-fs/output/lora_only_qwen_${NUM_CLASSES}class_$(date +%Y%m%d_%H%M)"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_DIR="/root/autodl-fs/output/lora_only_llama_${NUM_CLASSES}class_$(date +%Y%m%d_%H%M)"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 训练参数
NUM_EPOCHS=1
BATCH_SIZE=4
EVAL_BATCH_SIZE=8
LEARNING_RATE=5e-6
LORA_RANK=8
MAX_LENGTH=512
VAL_SPLIT=0.1

echo "============================================================"
echo "Training LoRA Only Model - $MODEL_NAME (${NUM_CLASSES}-class)"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Model: $MODEL_PATH"
echo "Train file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "Num classes: $NUM_CLASSES"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "LoRA rank: $LORA_RANK"
echo "Save model: $SAVE_MODEL"
echo "============================================================"
echo ""

# 构建命令
CMD="python train_and_eval_lora_only.py \
    --model_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --backbone $BACKBONE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_rank $LORA_RANK \
    --max_length $MAX_LENGTH \
    --val_split $VAL_SPLIT \
    --num_classes $NUM_CLASSES"

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

