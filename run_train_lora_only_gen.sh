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
    DATA_PATH="/root/autodl-fs/wvs_gen_merged"  # 生成式数据集
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    DATA_PATH="/root/autodl-fs/wvs_gen_merged"  # 生成式数据集
fi

OUTPUT_DIR="/root/autodl-fs/output/lora_only_gen_${BACKBONE}_$(date +%Y%m%d_%H%M)"

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

# 运行训练脚本
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

