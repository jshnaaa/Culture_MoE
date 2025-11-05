#!/bin/bash

# ============================================================
# 从 Base 模型 + LoRA 权重还原模型并评估
#
# 使用方法：
#   sh run_eval_lora_only_from_components.sh <BACKBONE> <NUM_CLASSES> <LORA_DIR>
#
# 示例：
#   sh run_eval_lora_only_from_components.sh llama 2 /path/to/lora_output
#   sh run_eval_lora_only_from_components.sh qwen 4 /path/to/lora_output
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"           # 默认使用 llama
NUM_CLASSES="${2:-4}"            # 默认 2 分类


# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# LoRA 权重路径（best_lora 目录）
LORA_WEIGHTS_PATH="/root/autodl-fs/model/llama_lora_only_${NUM_CLASSES}"

# 根据 num_classes 选择测试数据集
case $NUM_CLASSES in
    4)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_4.json"
        DATASET_NAME="WVS_4class"
        ;;
    5)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_5.json"
        DATASET_NAME="WVS_5class"
        ;;
    *)
        echo "❌ Error: Invalid num_classes=$NUM_CLASSES. Must be 2, 3, 4, or 5."
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/lora_only_test_results/${BACKBONE}_${NUM_CLASSES}class_from_components_$(date +%Y%m%d_%H%M)"

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

# 运行评估
python eval_lora_only_from_components.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_weights_path $LORA_WEIGHTS_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --num_classes $NUM_CLASSES \
    --batch_size 8 \
    --max_length 512 \
    --device cuda

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
    echo "💡 To compare with merged model evaluation:"
    echo "   sh run_eval_lora_only.sh $BACKBONE $NUM_CLASSES"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

