#!/bin/bash

# ============================================================
# 🧪 LORA-ONLY EVALUATION FROM COMPONENTS (IMPROVED VERSION)
# 从 Base 模型 + LoRA 权重还原完整模型并评估 - 改进版本（约束答案范围）
# ============================================================
#
# 功能：
#   1. 从基础模型和LoRA权重两部分还原完整模型
#   2. 在测试集上进行评估，输出详细的评估结果
#   3. 使用约束prompt明确指定答案范围和格式
#   4. 解决了标签超出范围和无关词汇输出问题
#   5. 保存生成答案、评估指标、模型配置等信息
#
# 使用方法：
#   sh run_eval_lora_only_from_components_improved.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 测试数据集ID (6=moral_stories, 7=cultureAtlas, 8=social_bias, 默认 6)
#
# 示例：
#   # 在moral stories数据集上评估
#   sh run_eval_lora_only_from_components_improved.sh llama 6
#
#   # 在cultureAtlas数据集上评估
#   sh run_eval_lora_only_from_components_improved.sh llama 7
#
#   # 在social bias数据集上评估
#   sh run_eval_lora_only_from_components_improved.sh qwen 8
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-6}"                   # 默认使用 moral stories (6)

# 根据 backbone 选择 base 模型路径和 unified dataset 的 LoRA 权重路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    # 使用 unified dataset 训练的 LoRA 权重
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_qwen_20251111_1421/best_lora"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    # 使用 unified dataset 训练的 LoRA 权重
    LORA_WEIGHTS_PATH="/autodl-fs/data/data/ft/ft_lora_only_gen_unified_all_datasets_llama_20251117_1218/best_lora"
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
        echo "  6 - moral_stories (2 classes)"
        echo "  7 - cultureAtlas (3 classes)"
        echo "  8 - social_bias (10 classes)"
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-fs/data/lora_eval_results_improved/eval_lora_only_${DATASET_NAME}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "LoRA-Only Model Evaluation (Improved Version - Constrained Answers)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Test Dataset: $DATASET_NAME (Classes: $NUM_CLASSES)"
echo ""
echo "🔧 Improvements Applied:"
echo "  ✅ Constrained prompt with explicit answer range"
echo "  ✅ Clear instruction: 'only answer a number, no explanation'"
echo "  ✅ Range-specific prompts (1-2, 1-3, 1-10)"
echo "  ✅ Reduced max_new_tokens to 5 (only enough for one number)"
echo "  ✅ Enhanced out-of-range detection and clipping"
echo "  ✅ Improved invalid text answer handling"
echo "  ✅ Detailed answer quality statistics"
echo ""
echo "📋 Prompt Examples:"
if [ "$NUM_CLASSES" = "2" ]; then
    echo "  '请从以下选项中选择一个数字：1 或 2'"
elif [ "$NUM_CLASSES" = "3" ]; then
    echo "  '请从以下选项中选择一个数字：1、2 或 3'"
elif [ "$NUM_CLASSES" = "10" ]; then
    echo "  '请从以下选项中选择一个数字：1、2、3、4、5、6、7、8、9 或 10'"
fi
echo "  '只回答一个数字，不要解释：'"
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

echo "Starting LoRA-only evaluation (improved version)..."
echo ""

# 运行评估 - 使用改进版本的脚本
python eval_lora_only_from_components_improved.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_weights_path $LORA_WEIGHTS_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --num_classes $NUM_CLASSES \
    --device cuda \
    $USE_MULTI_GPU

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ LoRA-Only Evaluation (Improved) completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Test Dataset: $DATASET_NAME ($NUM_CLASSES classes)"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo ""
    echo "🔧 Applied Improvements:"
    echo "  - Constrained prompt with explicit answer range (1-$NUM_CLASSES)"
    echo "  - Clear instruction to only answer numbers"
    echo "  - Reduced generation length (max_new_tokens=5)"
    echo "  - Enhanced out-of-range detection and automatic clipping"
    echo "  - Better invalid text answer detection and mapping"
    echo "  - Comprehensive answer quality statistics"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果)"
    echo "  - evaluation_summary.json (摘要)"
    echo "  - generated_answers.json (生成答案)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "💡 To check answer quality:"
    echo "   python -c \"import json; data = json.load(open('$OUTPUT_DIR/evaluation_summary.json')); print(f'Valid answers: {data[\\\"valid_rate\\\"]:.1%}'); print(f'Out-of-range: {data[\\\"out_of_range_answers\\\"]} samples'); print(f'Invalid text: {data[\\\"invalid_text_answers\\\"]} samples')\""
    echo ""
    echo "💡 To view accuracy:"
    echo "   python -c \"import json; data = json.load(open('$OUTPUT_DIR/evaluation_summary.json')); print(f'Accuracy: {data[\\\"accuracy\\\"]:.4f}')\""
    echo ""
    echo "💡 To view sample answers:"
    echo "   cat $OUTPUT_DIR/generated_answers.json | python -c \"import json,sys; data=json.load(sys.stdin); [print(f'Q: {item[\\\"instruction\\\"][:50]}...\\\\nTrue: {item[\\\"true_label\\\"]+1} | Predicted: {item[\\\"predicted_label\\\"]+1} | Raw: {item[\\\"raw_answer\\\"]} | Correct: {item[\\\"correct\\\"]}\\\\n---') for item in data[:5]]\""
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ LoRA-Only Evaluation (Improved) failed!"
    echo "============================================================"
    exit 1
fi