#!/bin/bash

# ============================================================
# 合并 LoRA 权重到 base 模型
# 用法：sh run_merge_lora.sh [backbone] [lora_model_dir] [output_dir]
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama
#="${2}"   # LoRA 模型目录（必须提供）
#OUTPUT_DIR="${3}"       # 输出目录（必须提供）

# 检查参数
if [ -z "$LORA_MODEL_DIR" ]; then
    echo "❌ Error: LoRA model directory not provided"
    echo ""
    echo "Usage: sh run_merge_lora.sh [backbone] [lora_model_dir] [output_dir]"
    echo ""
    echo "Example:"
    echo "  sh run_merge_lora.sh llama /path/to/lora_model /path/to/output"
    echo "  sh run_merge_lora.sh qwen /path/to/lora_model /path/to/output"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "❌ Error: Output directory not provided"
    echo ""
    echo "Usage: sh run_merge_lora.sh [backbone] [lora_model_dir] [output_dir]"
    exit 1
fi

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_MODEL_DIR = "/root/autodl-tmp/CultureMoE/Culture_Alignment/qwen_lora_only"
    OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/qwen_merge"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_MODEL_DIR = "/root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only"
    OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/llama_merge"
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

