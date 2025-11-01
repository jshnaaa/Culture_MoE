#!/bin/bash

# ============================================================
# 合并 LoRA 权重到 base 模型
# 用法：sh run_merge_lora.sh [backbone] [lora_model_dir] [output_dir]
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama
NUM_CLASSES="${2:-2}"

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_MODEL_DIR="/root/autodl-fs/model/qwen_lora_only_${NUM_CLASSES}"
    OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/qwen_merge_${NUM_CLASSES}"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_MODEL_DIR="/root/autodl-fs/model/llama_lora_only_${NUM_CLASSES}"
    OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/llama_merge_${NUM_CLASSES}"
fi

echo "============================================================"
echo "Merging LoRA Weights to Base Model"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Base model: $BASE_MODEL_PATH"
echo "LoRA model: $LORA_MODEL_DIR"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 运行合并脚本
python merge_lora_weights.py \
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
    echo "  1. Use this model for inference"
    echo "  2. Use this model for CultureMoE training (as frozen LLM)"
    echo ""
    echo "Example: Train CultureMoE with merged model"
    echo "  python train_culturemoe.py \\"
    echo "    --model_path $OUTPUT_DIR \\"
    echo "    --train_file /path/to/data.json \\"
    echo "    --output_dir /path/to/output \\"
    echo "    --num_classes 2"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Merging failed!"
    echo "============================================================"
    exit 1
fi

