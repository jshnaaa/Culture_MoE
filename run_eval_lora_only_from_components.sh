#!/bin/bash

# ============================================================
# 从 Base 模型 + LoRA 权重还原模型并评估（生成式版本）
#
# 使用方法：
#   sh run_eval_lora_only_from_components.sh <BACKBONE>
#
# 示例：
#   sh run_eval_lora_only_from_components.sh llama
#   sh run_eval_lora_only_from_components.sh qwen
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"           # 默认使用 llama
DATA_ID="${2:-1}"           # 默认使用1

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_qwen_20251111_1421/best_lora"
#    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_all_qwen_$(date +%Y%m%d_%H%M)/best_lora"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    # LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_llama_20251112_2135/best_lora"
   LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_20251112_1551/best_lora"
fi
# 根据 DATA_ID 选择数据集
case $DATA_ID in
    11)
        TEST_FILE="/root/autodl-fs/wvs_merge_gen.json"
        DATASET_NAME="WVS_Gen"
        OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    12)
        TEST_FILE="/root/autodl-fs/wvs_merge_gen_id.json"
        DATASET_NAME="WVS_Gen_ID"
        OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_id_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    13)
        TEST_FILE="/root/autodl-fs/wvs_merge_gen_ood.json"
        DATASET_NAME="WVS_Gen_OOD"
        OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_ood_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    21)
        TEST_FILE="/autodl-fs/data/moral_stories_merge_gen.json"
        DATASET_NAME="moral_Gen"
        OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_moral_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
esac

NUM_CLASSES=10  # 1-10 共 10 个类别

echo "============================================================"
echo "LoRA Only Model Evaluation (From Components)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Num classes: $NUM_CLASSES"
echo "Dataset: $DATASET_NAME"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
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
    python eval_lora_only_from_components.py \
        --base_model_path $BASE_MODEL_PATH \
        --lora_weights_path $LORA_WEIGHTS_PATH \
        --test_file $TEST_FILE \
        --output_dir $OUTPUT_DIR \
        --num_classes $NUM_CLASSES \
        --device cuda \
        --use_multi_gpu
else
    echo "Using single-GPU evaluation"
    echo ""

    # 单卡评估
    python eval_lora_only_from_components.py \
        --base_model_path $BASE_MODEL_PATH \
        --lora_weights_path $LORA_WEIGHTS_PATH \
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
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果)"
    echo "  - evaluation_summary.json (摘要)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "💡 To view detailed answers:"
    echo "   cat $OUTPUT_DIR/generated_answer.json | python -m json.tool | head -100"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

