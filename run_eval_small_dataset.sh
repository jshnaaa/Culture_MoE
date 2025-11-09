#!/bin/bash

# ============================================================
# 使用小规模数据集评估 LoRA Only 模型
#
# 使用方法：
#   sh run_eval_small_dataset.sh <BACKBONE> <NUM_SAMPLES>
#
# 参数说明：
#   BACKBONE: llama 或 qwen
#   NUM_SAMPLES: 测试样本数量（默认 100）
#
# 示例：
#   sh run_eval_small_dataset.sh llama 100
#   sh run_eval_small_dataset.sh qwen 50
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"           # 默认使用 llama
NUM_SAMPLES="${2:-50}"          # 默认 100 个样本

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_all_qwen_$(date +%Y%m%d_%H%M)/best_lora"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_all_llama_20251107_2321/best_lora"
fi

# 原始测试数据集（WVS 生成式数据集，标签 1-10）
ORIGINAL_TEST_FILE="/root/autodl-fs/wvs_merge_gen.json"
DATASET_NAME="WVS_Gen"
NUM_CLASSES=10  # 1-10 共 10 个类别

# 小规模测试数据集
SMALL_TEST_FILE="/root/autodl-fs//wvs_merge_gen_${NUM_SAMPLES}_samples.json"
SMALL_TEST_DIR=$(dirname "$SMALL_TEST_FILE")

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/lora_only_${BACKBONE}_small_${NUM_SAMPLES}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "LoRA Only Model Evaluation (Small Dataset)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Num classes: $NUM_CLASSES"
echo "Dataset: $DATASET_NAME"
echo "Test samples: $NUM_SAMPLES"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo ""
echo "Test file: $SMALL_TEST_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$ORIGINAL_TEST_FILE" ]; then
    echo "❌ Error: Original test file not found: $ORIGINAL_TEST_FILE"
    exit 1
fi

# 创建小规模数据集（如果不存在）
if [ ! -f "$SMALL_TEST_FILE" ]; then
    echo "Creating small test dataset..."
    mkdir -p "$SMALL_TEST_DIR"

    python create_small_test_dataset.py \
        --input_file "$ORIGINAL_TEST_FILE" \
        --output_file "$SMALL_TEST_FILE" \
        --num_samples "$NUM_SAMPLES" \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to create small test dataset"
        exit 1
    fi
    echo ""
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
        --test_file $SMALL_TEST_FILE \
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
        --test_file $SMALL_TEST_FILE \
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
    echo "  Test samples: $NUM_SAMPLES"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果)"
    echo "  - evaluation_summary.json (摘要)"
    echo "  - generated_answers.json (生成的答案)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "💡 To view detailed answers:"
    echo "   cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -50"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

