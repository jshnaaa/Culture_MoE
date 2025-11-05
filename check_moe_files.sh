#!/bin/bash

# 检查 MoE 模型文件

BACKBONE="llama"
NUM_CLASSES="2"

MERGED_LLM_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/${BACKBONE}_merge_${NUM_CLASSES}"
MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_${BACKBONE}_${NUM_CLASSES}"

echo "============================================================"
echo "Checking MoE Model Files"
echo "============================================================"
echo ""

echo "1. Checking Merged LLM: $MERGED_LLM_PATH"
if [ -d "$MERGED_LLM_PATH" ]; then
    echo "   ✅ Directory exists"
    echo "   Files:"
    ls -lh "$MERGED_LLM_PATH" | head -20
else
    echo "   ❌ Directory not found"
fi

echo ""
echo "2. Checking MoE Weights: $MOE_WEIGHTS_PATH"
if [ -d "$MOE_WEIGHTS_PATH" ]; then
    echo "   ✅ Directory exists"
    echo "   Files:"
    ls -lh "$MOE_WEIGHTS_PATH"

    echo ""
    echo "   Checking moe_config.json:"
    if [ -f "$MOE_WEIGHTS_PATH/moe_config.json" ]; then
        echo "   ✅ moe_config.json exists"
        echo "   Content:"
        cat "$MOE_WEIGHTS_PATH/moe_config.json" | jq '.'
    else
        echo "   ❌ moe_config.json not found"
    fi

    echo ""
    echo "   Checking moe_weights.bin:"
    if [ -f "$MOE_WEIGHTS_PATH/moe_weights.bin" ]; then
        echo "   ✅ moe_weights.bin exists"
        ls -lh "$MOE_WEIGHTS_PATH/moe_weights.bin"
    else
        echo "   ❌ moe_weights.bin not found"
    fi
else
    echo "   ❌ Directory not found"
fi

echo ""
echo "============================================================"
echo "Check Complete"
echo "============================================================"

