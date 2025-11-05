#!/bin/bash

# ============================================================
# 训练 LoRA Only 模型（生成式版本）
#
# 使用方法：
#   sh run_train_lora_only_gen.sh <BACKBONE>
#
# 示例：
#   sh run_train_lora_only_gen.sh llama
#   sh run_train_lora_only_gen.sh qwen
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"      # 默认使用 llama

# 根据 backbone 选择 base 模型路径和数据路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

DATA_PATH="/root/autodl-fs/cultureLLM_merge_gen.json"  # 生成式验证数据
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Training LoRA Only Model (Generative Version)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Base model: $BASE_MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found: $DATA_PATH"
    exit 1
fi

# 检测可用 GPU 数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

# 运行训练脚本（支持多卡）
if [ $NUM_GPUS -gt 1 ]; then
    echo "Using multi-GPU training with $NUM_GPUS GPUs"
    echo ""

    # 使用 torchrun 进行多卡训练
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_lora_only_gen.py \
        --model_name_or_path $BASE_MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_length 512 \
        --save_steps 500
else
    echo "Using single-GPU training"
    echo ""

    # 单卡训练
    python train_lora_only_gen.py \
        --model_name_or_path $BASE_MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_length 512 \
        --save_steps 500
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Merge LoRA weights: sh run_merge_lora_gen.sh $BACKBONE"
    echo "  2. Evaluate model: sh run_eval_lora_only_gen.sh $BACKBONE"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

