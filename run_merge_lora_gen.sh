#!/bin/bash

# ============================================================
# 合并 LoRA 权重到 base 模型（生成式版本）
#
# 使用方法：
#   sh run_merge_lora_gen.sh <BACKBONE> <LORA_MODEL_DIR>
#
# 示例：
#   sh run_merge_lora_gen.sh llama /root/autodl-fs/output/lora_only_gen_llama_20241102_1234
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"
LORA_MODEL_DIR="${2}"

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 如果没有指定 LoRA 模型目录，查找最新的
if [ -z "$LORA_MODEL_DIR" ]; then
    LORA_MODEL_DIR=$(find /root/autodl-fs/output -name "lora_only_gen_${BACKBONE}_*" -type d | sort -r | head -n 1)
    if [ -z "$LORA_MODEL_DIR" ]; then
        echo "❌ Error: No LoRA model found for $BACKBONE"
        exit 1
    fi
    echo "Using latest LoRA model: $LORA_MODEL_DIR"
fi

OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/${BACKBONE}_merge_gen"

echo "============================================================"
echo "Merging LoRA Weights (Generative Version)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Base model: $BASE_MODEL_PATH"
echo "LoRA model: $LORA_MODEL_DIR"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -d "$LORA_MODEL_DIR" ]; then
    echo "❌ Error: LoRA model not found: $LORA_MODEL_DIR"
    exit 1
fi

# 运行合并脚本
python merge_lora_weights_gen.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_model_path $LORA_MODEL_DIR \
    --output_path $OUTPUT_DIR \
    --device auto

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Merging completed successfully!"
    echo "============================================================"
    echo "Merged model saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate model: sh run_eval_lora_only_gen.sh $BACKBONE"
    echo "  2. Train CultureMoE: sh run_train_culturemoe_gen.sh $BACKBONE"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Merging failed!"
    echo "============================================================"
    exit 1
fi

