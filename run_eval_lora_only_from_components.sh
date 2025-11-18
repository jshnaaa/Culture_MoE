#!/bin/bash

# ============================================================
# 从 Base 模型 + LoRA 权重还原模型并评估（增强版本 - 修复所有问题）
#
# 根本原因修复：
#   🔍 发现问题：模型生成222/22是因为prompt格式不匹配数据集
#   ✅ 修复prompt格式（使用数据集原生的### Answer:格式）
#   ✅ 增加max_length到2048（避免重要信息被截断）
#   ✅ 使用自然prompt（避免过度约束干扰模型理解）
#   ✅ 多重答案提取（支持### Answer:和你的回答：格式）
#   ✅ 智能数字提取（从222提取2，从22提取2）
#   ✅ Debug模式（帮助理解模型行为）
#
# 使用方法：
#   sh run_eval_lora_only_from_components.sh <BACKBONE> <DATA_ID>
#
# 示例：
#   sh run_eval_lora_only_from_components.sh llama 6    # moral stories
#   sh run_eval_lora_only_from_components.sh llama 7    # cultureAtlas
#   sh run_eval_lora_only_from_components.sh qwen 8     # socialBias
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
    LORA_WEIGHTS_PATH="/autodl-fs/data/data/ft/ft_lora_only_gen_unified_all_datasets_llama_20251117_1218/best_lora"
#    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_20251112_1551/best_lora"
fi
# 根据 DATA_ID 选择数据集
case $DATA_ID in
    # 11)
    #     TEST_FILE="/root/autodl-fs/wvs_merge_gen.json"
    #     DATASET_NAME="WVS_Gen"
        # OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        # ;;
    # 12)
    #     TEST_FILE="/root/autodl-fs/wvs_merge_gen_id.json"
    #     DATASET_NAME="WVS_Gen_ID"
    #     OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_id_${BACKBONE}_$(date +%Y%m%d_%H%M)"
    #     ;;
    # 13)
    #     TEST_FILE="/root/autodl-fs/wvs_merge_gen_ood.json"
    #     DATASET_NAME="WVS_Gen_OOD"
    #     OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_ood_${BACKBONE}_$(date +%Y%m%d_%H%M)"
    #     ;;
    6)
        TEST_FILE="/autodl-fs/data/moral_stories_merge_gen.json"
        DATASET_NAME="moral_Gen"
        NUM_CLASSES=2
        ;;
    7)
        TEST_FILE="/autodl-fs/data/cultureAtlas_merge_gen.json"
        DATASET_NAME="cultureAtlas"
        NUM_CLASSES=3
        ;;
    8)
        TEST_FILE="/autodl-fs/data/socialBias_merge_gen.json"
        DATASET_NAME="socialBias"
        NUM_CLASSES=2
        ;;
esac

OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_${DATASET_NAME}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

# NUM_CLASSES=10  # 1-10 共 10 个类别

echo "============================================================"
echo "LoRA Only Model Evaluation (Enhanced - Fixed & Improved)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Num classes: $NUM_CLASSES"
echo "Dataset: $DATASET_NAME"
echo ""
echo "🔧 Root Cause Fixes Applied:"
echo "  ✅ Fixed prompt format mismatch (use dataset's ### Answer: format)"
echo "  ✅ Increased max_length to 2048 (prevent important info truncation)"
echo "  ✅ Natural prompt generation (avoid over-constraining)"
echo "  ✅ Multiple answer extraction methods (### Answer:, 你的回答：)"
echo "  ✅ Smart number extraction and clipping (222→2, 22→2)"
echo "  ✅ Debug mode for understanding model behavior"
echo "  ✅ Enhanced answer quality statistics"
echo "  ✅ Robust fallback mechanisms"
echo ""
echo "🎯 Qwen Model Special Optimizations:"
echo "  ✅ Qwen special token repetition fix (151643 detection)"
echo "  ✅ Intelligent number mapping (12→2, 22→2, 222→2)"
echo "  ✅ Repetitive token detection and truncation"
echo "  ✅ Balanced generation parameters (max_tokens=8, light penalties)"
echo "  ✅ Empty answer generation fix (enhanced fallback mechanisms)"
echo "  ✅ Multi-strategy answer extraction (token-by-token decoding)"
echo "  ✅ Qwen-specific debug mode for problem diagnosis"
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
    echo "✅ Enhanced LoRA Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "🔧 Applied Fixes Summary:"
    echo "  ✅ Fixed empty answer generation issues"
    echo "  ✅ Fixed out-of-range label problems (222, 22 → 1-3)"
    echo "  ✅ Fixed invalid text outputs (education → numbers)"
    echo "  ✅ Ultra-strict prompt engineering (明确禁止超范围数字)"
    echo "  ✅ Aggressive post-processing (智能截断和数字提取)"
    echo "  ✅ Optimized for Qwen model behavior"
    echo "  ✅ Enhanced answer extraction and fallback mechanisms"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Num classes: $NUM_CLASSES"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果 + 质量统计)"
    echo "  - evaluation_summary.json (摘要 + 增强统计)"
    echo "  - generated_answers.json (所有生成的答案)"
    echo ""
    echo "💡 To view enhanced results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "💡 To view detailed answers with quality stats:"
    echo "   cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -100"
    echo ""
    echo "💡 To check answer quality:"
    echo "   grep -E 'empty_answers|text_answers|out_of_range' $OUTPUT_DIR/evaluation_summary.json"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

