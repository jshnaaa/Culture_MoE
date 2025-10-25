#!/bin/bash

# 评估 Base LLaMA 3.1 模型（无训练，无 MoE）
# 全部数据作为测试集

# ✅ 配置参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TEST_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
OUTPUT_FILE="/root/autodl-fs/eval_results_base_llama_$(date +%Y%m%d_%H%M).json"

echo "============================================================"
echo "Evaluating Base LLaMA 3.1 Model (No Training)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output: $OUTPUT_FILE"
echo "============================================================"
echo ""

# ✅ 运行评估
python eval_base_llama.py \
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

