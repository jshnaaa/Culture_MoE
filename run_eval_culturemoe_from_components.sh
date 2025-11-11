#!/bin/bash

# ============================================================
# 从 Base 模型 + LoRA 权重 + MoE 权重还原 CultureMoE 模型并评估（生成式版本）
#
# 使用方法：
#   sh run_eval_culturemoe_from_components.sh <BACKBONE>
#
# 示例：
#   sh run_eval_culturemoe_from_components.sh llama
#   sh run_eval_culturemoe_from_components.sh qwen
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"           # 默认使用 llama
USE_CULTURE_LOSS="${2:-True}"       # 默认使用文化损失
NUM_EXPERTS="${3:-6}"               # 默认 6 个专家
NUM_GPUS="${4:-2}"                  # 默认使用 2 个 GPU
CULTURE_LOSS_WEIGHT="${5:-0.5}"     # 默认文化损失权重 0.5

if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_qwen_20251111_1421/best_lora"
    MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_moe_gen_unified_all_datasets_qwen_experts${NUM_EXPERTS}_CULTURE_LOSS_WEIGHT${CULTURE_LOSS_WEIGHT}_20251112_/best_moe"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_llama_20251112_2135/best_lora"
    MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_moe_gen_unified_all_datasets_llama_experts${NUM_EXPERTS}_CULTURE_LOSS_WEIGHT${CULTURE_LOSS_WEIGHT}_20251112_/best_moe"
fi

# 测试数据集（WVS 生成式数据集，标签 1-10）
TEST_FILE="/root/autodl-fs/wvs_merge_gen.json"
DATASET_NAME="WVS_Gen"
NUM_CLASSES=10  # 1-10 共 10 个类别

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft_test_results/ft_culturemoe_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "CultureMoE Evaluation (From Components)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Num classes: $NUM_CLASSES"
echo "Dataset: $DATASET_NAME"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo "  MoE weights: $MOE_WEIGHTS_PATH"
echo ""
echo "Test file: $TEST_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

# 检查 MoE 权重
MOE_PATH=$(ls -d $MOE_WEIGHTS_PATH 2>/dev/null | head -1)
if [ -z "$MOE_PATH" ]; then
    echo "❌ Error: MoE weights not found: $MOE_WEIGHTS_PATH"
    echo ""
    echo "Please first run:"
    echo "  sh run_ft_culturemoe_gen.sh $BACKBONE 4 True 6 2 0.5"
    exit 1
fi

echo "Found MoE weights: $MOE_PATH"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检测可用 GPU 数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

# 运行评估
if [ $NUM_GPUS -gt 1 ]; then
    echo "Using multi-GPU evaluation with $NUM_GPUS GPUs"
    echo ""

    # 使用 DataParallel 进行多卡评估
    python eval_culturemoe_from_components.py \
        --base_model_path $BASE_MODEL_PATH \
        --lora_weights_path $LORA_WEIGHTS_PATH \
        --moe_weights_path $MOE_PATH \
        --test_file $TEST_FILE \
        --output_dir $OUTPUT_DIR \
        --num_classes $NUM_CLASSES \
        --device cuda \
        --use_multi_gpu
else
    echo "Using single-GPU evaluation"
    echo ""

    # 单卡评估
    python eval_culturemoe_from_components.py \
        --base_model_path $BASE_MODEL_PATH \
        --lora_weights_path $LORA_WEIGHTS_PATH \
        --moe_weights_path $MOE_PATH \
        --test_file $TEST_FILE \
        --output_dir $OUTPUT_DIR \
        --num_classes $NUM_CLASSES \
        --device cuda
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Num classes: $NUM_CLASSES"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo "  MoE weights: $MOE_PATH"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果)"
    echo "  - evaluation_summary.json (摘要)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "💡 To view detailed answers:"
    echo "   cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -100"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

