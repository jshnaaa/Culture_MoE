#!/bin/bash

# 评估双路输入模型的脚本（三分类 → 二分类）

# ✅ 配置参数
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_131545"  # 训练输出的 checkpoint 目录
BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"  # 原始 LLaMA 模型路径
TEST_FILE="/root/autodl-fs/CulturalBench_Hard_binary.json"  # 二分类测试集
OUTPUT_FILE="/root/autodl-fs/eval_results_binary_$(date +%Y%m%d_%H%M%S).json"

# ✅ 映射策略选择
# - "merge_neutral_to_no": neutral → no (保守策略)
# - "merge_neutral_to_yes": neutral → yes (激进策略)
MAPPING_STRATEGY="merge_neutral_to_no"

# ✅ 运行评估
python examples/eval_classification_dual_binary.py \
    --model_path $MODEL_PATH \
    --base_model_path $BASE_MODEL_PATH \
    --test_file $TEST_FILE \
    --batch_size 8 \
    --max_length 512 \
    --output_file $OUTPUT_FILE \
    --mapping_strategy $MAPPING_STRATEGY \
    --use_lora

echo ""
echo "============================================================"
echo "Evaluation completed!"
echo "Mapping strategy: $MAPPING_STRATEGY"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================================"

