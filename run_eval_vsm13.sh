#!/bin/bash

# ============================================================
# VSM13 文化一致性测试
#
# 使用方法：
#   sh run_eval_vsm13.sh <MODEL> <BACKBONE> [NUM_CLASSES]
#
# 示例：
#   sh run_eval_vsm13.sh base llama
#   sh run_eval_vsm13.sh lora qwen 2
#   sh run_eval_vsm13.sh moe llama 4
# ============================================================

# ✅ 配置参数
MODEL="${1:-base}"          # 模型类型：base/lora/moe
BACKBONE="${2:-llama}"      # 骨干模型：llama/qwen
NUM_CLASSES="${3:-2}"       # 分类数量（仅用于 lora 和 moe）

echo "============================================================"
echo "VSM13 Cultural Consistency Test"
echo "============================================================"
echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
if [ "$MODEL" != "base" ]; then
    echo "Num classes: $NUM_CLASSES"
fi
echo "============================================================"
echo ""

# 检查模型类型
if [ "$MODEL" != "base" ] && [ "$MODEL" != "lora" ] && [ "$MODEL" != "moe" ]; then
    echo "❌ Error: Invalid model type: $MODEL"
    echo "   Valid options: base, lora, moe"
    exit 1
fi

# 检查骨干模型
if [ "$BACKBONE" != "llama" ] && [ "$BACKBONE" != "qwen" ]; then
    echo "❌ Error: Invalid backbone: $BACKBONE"
    echo "   Valid options: llama, qwen"
    exit 1
fi

# 检查测试文件
TEST_FILE="/root/autodl-fs/vsm13_test.json"
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

# 根据模型类型检查必要的文件
if [ "$MODEL" = "base" ]; then
    if [ "$BACKBONE" = "llama" ]; then
        MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    else
        MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    fi

    if [ ! -d "$MODEL_PATH" ]; then
        echo "❌ Error: Base model not found: $MODEL_PATH"
        exit 1
    fi
elif [ "$MODEL" = "lora" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/${BACKBONE}_merge_${NUM_CLASSES}"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "❌ Error: LoRA model not found: $MODEL_PATH"
        echo ""
        echo "Please ensure you have completed the following steps:"
        echo "  1. Train LoRA Only: sh run_train_lora_only.sh $BACKBONE $NUM_CLASSES true"
        echo "  2. Merge weights: sh run_merge_lora.sh $BACKBONE $NUM_CLASSES"
        exit 1
    fi
elif [ "$MODEL" = "moe" ]; then
    MERGED_LLM_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/${BACKBONE}_merge_${NUM_CLASSES}"
    MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_${BACKBONE}_${NUM_CLASSES}"

    if [ ! -d "$MERGED_LLM_PATH" ]; then
        echo "❌ Error: Merged LLM not found: $MERGED_LLM_PATH"
        exit 1
    fi

    if [ ! -d "$MOE_WEIGHTS_PATH" ]; then
        echo "❌ Error: MoE weights not found: $MOE_WEIGHTS_PATH"
        exit 1
    fi
fi

# 运行评估脚本
if [ "$MODEL" = "base" ]; then
    python eval_vsm13.py \
        --model $MODEL \
        --backbone $BACKBONE \
        --device cuda
else
    python eval_vsm13.py \
        --model $MODEL \
        --backbone $BACKBONE \
        --num_classes $NUM_CLASSES \
        --device cuda
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ VSM13 test completed successfully!"
    echo "============================================================"
    echo ""
    echo "Results saved to: /root/autodl-fs/vsm13_output/${MODEL}_${BACKBONE}/"
    echo ""
    echo "💡 To view results:"
    echo "   cat /root/autodl-fs/vsm13_output/${MODEL}_${BACKBONE}/vsm13_test_results_${MODEL}_${BACKBONE}.json | jq"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ VSM13 test failed!"
    echo "============================================================"
    exit 1
fi

