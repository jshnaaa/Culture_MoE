#!/bin/bash

# 评估 Base LLaMA 3.1 模型（无训练，无 MoE）
# 四分类任务

# ✅ 配置参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TEST_FILE="/root/autodl-fs/wvs_all_llama_merge.json"  # 四分类数据集
OUTPUT_FILE="/root/autodl-fs/output/base_llama/eval_results_base_llama_wvs_$(date +%Y%m%d_%H%M).json"

echo "============================================================"
echo "Evaluating Base LLaMA 3.1 Model (4-Class Classification)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output: $OUTPUT_FILE"
echo "Num classes: 4"
echo "============================================================"
echo ""

# ✅ 运行评估（四分类）
python eval_base_llama.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --batch_size 8 \
    --max_length 512 \
    --output_file $OUTPUT_FILE \
    --device cuda:0 \
    --num_classes 5

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

