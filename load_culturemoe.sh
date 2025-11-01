#!/bin/bash

# ============================================================
# 加载 CultureMoE 模型（从 MoE 权重 + 合并后的 LLM）
#
# 使用方法：
#   sh load_culturemoe.sh <BACKBONE> <NUM_CLASSES> [TEST_INPUT]
#
# 示例：
#   sh load_culturemoe.sh llama 2
#   sh load_culturemoe.sh qwen 4 "Your test input here"
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"      # 默认使用 llama
NUM_CLASSES="${2:-2}"       # 默认 2 分类
TEST_INPUT="${3:-}"         # 可选的测试输入

# 根据 backbone 和 num_classes 构建路径
MERGED_LLM_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/${BACKBONE}_merge_${NUM_CLASSES}"
MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_${BACKBONE}_${NUM_CLASSES}"

echo "============================================================"
echo "Loading CultureMoE Model"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Num classes: $NUM_CLASSES"
echo "Merged LLM path: $MERGED_LLM_PATH"
echo "MoE weights path: $MOE_WEIGHTS_PATH"
if [ -n "$TEST_INPUT" ]; then
    echo "Test input: $TEST_INPUT"
fi
echo "============================================================"
echo ""

# 检查路径是否存在
if [ ! -d "$MERGED_LLM_PATH" ]; then
    echo "❌ Error: Merged LLM not found: $MERGED_LLM_PATH"
    echo ""
    echo "Please ensure you have completed the following steps:"
    echo "  1. Train LoRA Only: sh run_train_lora_only.sh $BACKBONE $NUM_CLASSES true"
    echo "  2. Merge weights: sh run_merge_lora.sh $BACKBONE"
    exit 1
fi

if [ ! -d "$MOE_WEIGHTS_PATH" ]; then
    echo "❌ Error: MoE weights not found: $MOE_WEIGHTS_PATH"
    echo ""
    echo "Please ensure you have completed the following step:"
    echo "  Train CultureMoE: sh run_train_culturemoe_from_merged.sh $BACKBONE $NUM_CLASSES True 6 true"
    exit 1
fi

# 构建加载命令
LOAD_CMD="python load_culturemoe.py \
    --moe_weights_path $MOE_WEIGHTS_PATH \
    --device cuda"

# 添加测试输入（如果提供）
if [ -n "$TEST_INPUT" ]; then
    LOAD_CMD="$LOAD_CMD --test_input \"$TEST_INPUT\""
fi

# 运行加载脚本
eval $LOAD_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Model loaded successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE"
    echo "  Num classes: $NUM_CLASSES"
    echo "  Merged LLM: $MERGED_LLM_PATH"
    echo "  MoE weights: $MOE_WEIGHTS_PATH"
    echo ""
    echo "💡 To test with custom input:"
    echo "   sh run_load_culturemoe.sh $BACKBONE $NUM_CLASSES \"Your test input here\""
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Failed to load model!"
    echo "============================================================"
    exit 1
fi

