#!/bin/bash

# 评估双路输入模型的脚本（三分类 → 二分类）
# 使用合并后的模型进行评估

# ✅ 配置参数

# 方式 1：使用已经合并好的模型（推荐）
# 如果你已经运行了 bash run_merge_lora.sh，使用这个路径
MERGED_MODEL_PATH="/root/autodl-fs/merge/llama_lora_merged_20251019"  # 合并后的 LLaMA 模型路径

# MoE 权重的 checkpoint 路径
MOE_CHECKPOINT_PATH="/root/autodl-fs/output/classi_dual_20251019_1822"

# 测试数据和输出
TEST_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"  # 二分类测试集
OUTPUT_FILE="/root/autodl-fs/eval_results_merged_$(date +%Y%m%d_%H%M%S).json"

# ✅ 映射策略选择
# - "merge_neutral_to_no": neutral → no (保守策略)
# - "merge_neutral_to_yes": neutral → yes (激进策略)
MAPPING_STRATEGY="merge_neutral_to_no"

echo "============================================================"
echo "Evaluating Merged Model"
echo "============================================================"
echo "Merged model: $MERGED_MODEL_PATH"
echo "MoE checkpoint: $MOE_CHECKPOINT_PATH"
echo "Test file: $TEST_FILE"
echo "Mapping strategy: $MAPPING_STRATEGY"
echo "============================================================"
echo ""

# ✅ 运行评估（使用合并后的模型）
python examples/eval_classification_dual_merged.py \
    --merged_model_path $MERGED_MODEL_PATH \
    --moe_checkpoint_path $MOE_CHECKPOINT_PATH \
    --test_file $TEST_FILE \
    --batch_size 8 \
    --max_length 512 \
    --output_file $OUTPUT_FILE \
    --mapping_strategy $MAPPING_STRATEGY

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo "Mapping strategy: $MAPPING_STRATEGY"
    echo "Results saved to: $OUTPUT_FILE"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    echo ""
    echo "Possible issues:"
    echo "  1. Merged model not found at: $MERGED_MODEL_PATH"
    echo "     → Run: bash run_merge_lora.sh first"
    echo ""
    echo "  2. MoE checkpoint not found at: $MOE_CHECKPOINT_PATH"
    echo "     → Check the path is correct"
    echo ""
    echo "  3. Test file not found at: $TEST_FILE"
    echo "     → Check the path is correct"
    echo "============================================================"
    exit 1
fi

