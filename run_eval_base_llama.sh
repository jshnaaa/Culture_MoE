#!/bin/bash

# 评估 Base 模型（无训练，无 MoE）
# 支持 LLaMA 3.1 和 Qwen 2.5

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama，可以通过第一个参数指定 qwen
TEST_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    OUTPUT_FILE="/root/autodl-fs/output/base_qwen/eval_results_base_qwen_$(date +%Y%m%d_%H%M).json"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_FILE="/root/autodl-fs/output/base_llama/eval_results_base_llama_$(date +%Y%m%d_%H%M).json"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

echo "============================================================"
echo "Evaluating Base $MODEL_NAME Model (No Training)"
echo "============================================================"
echo "Backbone: $BACKBONE"
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
    --device cuda:0 \
    --num_classes 2 \
    --backbone $BACKBONE

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

