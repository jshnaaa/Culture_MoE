#!/bin/bash

# 评估双路输入模型的脚本

# ✅ 配置参数
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_XXXXXX"  # 修改为你的模型路径
TEST_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"  # CultureBench 测试集
OUTPUT_FILE="/root/autodl-fs/eval_results_$(date +%Y%m%d_%H%M%S).json"

# ✅ 运行评估
python examples/eval_classification_dual.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --batch_size 8 \
    --max_length 512 \
    --output_file $OUTPUT_FILE \
    --use_lora

echo ""
echo "============================================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================================"

