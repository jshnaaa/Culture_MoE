#!/bin/bash

# ============================================================
# 加载 CultureMoE 模型并在测试集上评估
#
# 使用方法：
#   sh run_load_culturemoe.sh <BACKBONE> <NUM_CLASSES>
#
# 示例：
#   sh run_load_culturemoe.sh llama 2
#   sh run_load_culturemoe.sh qwen 4
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"      # 默认使用 llama
NUM_CLASSES="${2:-2}"       # 默认 2 分类

# 根据 backbone 和 num_classes 构建路径
MERGED_LLM_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/${BACKBONE}_merge_${NUM_CLASSES}"
MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_${BACKBONE}_${NUM_CLASSES}"
TEST_FILE="/root/autodl-fs/wvs_${NUM_CLASSES}_merge"
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/eval_results/${BACKBONE}_${NUM_CLASSES}class_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "CultureMoE Model Evaluation"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Num classes: $NUM_CLASSES"
echo "Merged LLM: $MERGED_LLM_PATH"
echo "MoE weights: $MOE_WEIGHTS_PATH"
echo "Test file: $TEST_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径是否存在
if [ ! -d "$MERGED_LLM_PATH" ]; then
    echo "❌ Error: Merged LLM not found: $MERGED_LLM_PATH"
    echo ""
    echo "Please ensure you have completed the following steps:"
    echo "  1. Train LoRA Only: sh run_train_lora_only.sh $BACKBONE $NUM_CLASSES true"
    echo "  2. Merge weights: sh run_merge_lora.sh $BACKBONE $NUM_CLASSES"
    exit 1
fi

if [ ! -d "$MOE_WEIGHTS_PATH" ]; then
    echo "❌ Error: MoE weights not found: $MOE_WEIGHTS_PATH"
    echo ""
    echo "Please ensure you have completed the following step:"
    echo "  Train CultureMoE: sh run_train_culturemoe_from_merged.sh $BACKBONE $NUM_CLASSES True 6 true"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    echo ""
    echo "Please ensure the test file exists at: $TEST_FILE"
    exit 1
fi

# 运行评估脚本
python load_culturemoe.py \
    --moe_weights_path $MOE_WEIGHTS_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    --batch_size 8 \
    --max_length 512

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE"
    echo "  Num classes: $NUM_CLASSES"
    echo "  Merged LLM: $MERGED_LLM_PATH"
    echo "  MoE weights: $MOE_WEIGHTS_PATH"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果)"
    echo "  - evaluation_summary.json (摘要)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

