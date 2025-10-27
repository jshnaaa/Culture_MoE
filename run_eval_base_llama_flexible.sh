#!/bin/bash

# 评估 Base LLaMA 3.1 模型（灵活标签版本）
# 适用于 WVS 等标签不统一的数据集

# ✅ 配置参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TEST_FILE="/root/autodl-fs/wvs_all_llama_merge.json"
OUTPUT_FILE="/root/autodl-fs/output/base_llama/eval_results_base_llama_flexible_$(date +%Y%m%d_%H%M).json"

echo "============================================================"
echo "Evaluating Base LLaMA 3.1 Model (Flexible Labels)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output: $OUTPUT_FILE"
echo "Note: Supports variable number of classes and label names"
echo "============================================================"
echo ""

# ✅ 运行评估（灵活标签）
python eval_base_llama_flexible.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --batch_size 8 \
    --max_length 512 \
    --output_file $OUTPUT_FILE \
    --device cuda:0

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_FILE"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

