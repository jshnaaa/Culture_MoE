#!/bin/bash

# ============================================================
# 🧪 SIMPLE MOE EVALUATION FROM COMPONENTS
# 从 Base 模型 + LoRA 权重 + Simple MoE 权重还原 Simple MoE 模型并评估
# ============================================================
#
# 功能：
#   1. 从基础模型、LoRA权重、Simple MoE权重三部分还原完整模型
#   2. 在测试集上进行评估，输出详细的评估结果
#   3. 保存生成答案、评估指标、模型配置等信息
#   4. 支持多数据集评估（unified dataset训练的模型在其他数据集上评估）
#
# 使用方法：
#   sh run_eval_simple_moe_from_components.sh <BACKBONE> <DATA_ID> <NUM_EXPERTS> <TOP_K>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 测试数据集ID (6=moral_stories, 7=cultureAtlas, 8=social_bias, 默认 6)
#   NUM_EXPERTS: 专家数量 (默认 12)
#   TOP_K: Top-K路由 (默认 2)
#
# 注意：此脚本使用在unified_all_datasets上训练的Simple MoE模型
#
# 示例：
#   # 在moral stories数据集上评估
#   sh run_eval_simple_moe_from_components.sh llama 6 12 2
#
#   # 在cultureAtlas数据集上评估
#   sh run_eval_simple_moe_from_components.sh llama 7 12 2
#
#   # 在social bias数据集上评估
#   sh run_eval_simple_moe_from_components.sh qwen 8 12 2
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-6}"                   # 默认使用 moral stories (6)
NUM_EXPERTS="${3:-12}"              # 默认 12 个专家
TOP_K="${4:-2}"                     # 默认 top-2 路由

# 根据 backbone 选择 base 模型路径和 unified dataset 的 LoRA 权重路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    # 使用 unified dataset 训练的 LoRA 权重
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_qwen_20251111_1421/best_lora"
    # Simple MoE 权重路径模式（需要根据实际训练结果调整时间戳）
    MOE_WEIGHTS_PATTERN="/root/autodl-fs/data/ft/ft_simple_moe_unified_all_datasets_qwen_experts${NUM_EXPERTS}_top${TOP_K}_*"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    # 使用 unified dataset 训练的 LoRA 权重
    LORA_WEIGHTS_PATH="/autodl-fs/data/data/ft/ft_lora_only_gen_unified_all_datasets_llama_20251117_1218/best_lora"
    # Simple MoE 权重路径模式（需要根据实际训练结果调整时间戳）
    MOE_WEIGHTS_PATTERN="/root/autodl-fs/data/ft/ft_simple_moe_unified_all_datasets_llama_experts${NUM_EXPERTS}_top${TOP_K}_*"
fi

# 查找最新的 Simple MoE 训练目录
MOE_WEIGHTS_BASE=$(ls -td $MOE_WEIGHTS_PATTERN 2>/dev/null | head -1)

if [ -z "$MOE_WEIGHTS_BASE" ] || [ ! -d "$MOE_WEIGHTS_BASE" ]; then
    echo "❌ Error: Simple MoE training directory not found"
    echo "Expected pattern: $MOE_WEIGHTS_PATTERN"
    echo ""
    echo "Please first run Simple MoE training on unified_all_datasets:"
    echo "  sh run_ft_simple_moe_gen.sh $BACKBONE 1 $NUM_EXPERTS $TOP_K"
    exit 1
fi

# 检查 MoE 权重文件
MOE_WEIGHTS_PATH="$MOE_WEIGHTS_BASE/final_moe_weights"
if [ ! -d "$MOE_WEIGHTS_PATH" ] || [ ! -f "$MOE_WEIGHTS_PATH/moe_weights.pth" ]; then
    echo "❌ Error: Simple MoE weights not found in $MOE_WEIGHTS_PATH"
    echo "Expected file: $MOE_WEIGHTS_PATH/moe_weights.pth"
    exit 1
fi

# 根据 DATA_ID 选择测试数据集
case $DATA_ID in
    6)
        TEST_FILE="/autodl-fs/data/moral_stories_merge_gen.json"
        DATASET_NAME="moral_stories"
        NUM_CLASSES=2
        ;;
    7)
        TEST_FILE="/autodl-fs/data/cultureAtlas_merge_gen.json"
        DATASET_NAME="cultureAtlas"
        NUM_CLASSES=3
        ;;
    8)
        TEST_FILE="/autodl-fs/data/bbq_merge_gen.json"
        DATASET_NAME="social_bias"
        NUM_CLASSES=10
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 6, 7, or 8."
        echo ""
        echo "DATA_ID options:"
        echo "  6 - moral_stories"
        echo "  7 - cultureAtlas"
        echo "  8 - social_bias"
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-fs/data/simple_moe_eval_results/eval_simple_moe_${DATASET_NAME}_${BACKBONE}_experts${NUM_EXPERTS}_top${TOP_K}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Simple MoE Model Evaluation (From Components)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Test Dataset: $DATASET_NAME (Classes: $NUM_CLASSES)"
echo "Model Config: $NUM_EXPERTS experts, Top-$TOP_K routing"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo "  Simple MoE weights: $MOE_WEIGHTS_PATH"
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

if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 强制单GPU评估（避免DataParallel兼容性问题）
echo "  Using: Single-GPU evaluation (forced for stability)"
USE_MULTI_GPU=""

echo "Starting Simple MoE evaluation..."
echo ""

# 运行评估
python eval_simple_moe_from_components.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_weights_path $LORA_WEIGHTS_PATH \
    --moe_weights_path $MOE_WEIGHTS_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --backbone $BACKBONE \
    --num_experts $NUM_EXPERTS \
    --top_k $TOP_K \
    --device cuda \
    $USE_MULTI_GPU

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Simple MoE Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Config: $NUM_EXPERTS experts, Top-$TOP_K routing"
    echo "  Test Dataset: $DATASET_NAME ($NUM_CLASSES classes)"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo "  Simple MoE weights: $MOE_WEIGHTS_PATH"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果)"
    echo "  - evaluation_summary.json (摘要)"
    echo "  - generated_answers.json (生成答案)"
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
    echo "❌ Simple MoE Evaluation failed!"
    echo "============================================================"
    exit 1
fi