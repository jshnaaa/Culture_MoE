#!/bin/bash

# ============================================================
# 在新格式 CultureLLM 数据集上评估 Base 模型（LLaMA 或 Qwen）
#
# 使用方法：
#   sh run_ft_base_new_format.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
#
# 示例：
#   # 评估 LLaMA Base 模型 + CultureLLM 数据集（新格式）
#   sh run_ft_base_new_format.sh llama 4
#
#   # 评估 Qwen Base 模型 + CultureLLM 数据集（新格式）
#   sh run_ft_base_new_format.sh qwen 4
#
#   # 评估 LLaMA Base 模型 + CulturalBench 数据集（新格式）
#   sh run_ft_base_new_format.sh llama 2
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-4}"                   # 默认 CultureLLM (4)

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 根据 DATA_ID 选择数据集
case $DATA_ID in
    2)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen_small.json"
        DATASET_TAG="CulturalBench"
        echo "Using CulturalBench dataset (new format)"
        ;;
    3)
        # NormAD
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        echo "Using NormAD dataset (new format)"
        ;;
    4)
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        echo "Using CultureLLM dataset (new format)"
        ;;
    5)
        # wvs
        DATASET_NAME="wvs"
        TRAIN_FILE="/root/autodl-fs/wvs_merge_rp_gen.json"
        DATASET_TAG="wvs"
        echo "Using wvs dataset (new format)"
        ;;
    22)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_rp_gen.json" # /autodl-fs/data/CulturalBench_merge_gen_small_samples.json
        DATASET_TAG="CulturalBench_rp"
        echo "Using CulturalBench dataset (new format)"
        ;;
    32)
        # NormAD
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_rp_gen.json"
        DATASET_TAG="normad_rp"
        echo "Using NormAD dataset (new format)"
        ;;
    42)
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_rp_gen.json"
        DATASET_TAG="cultureLLM_rp"
        echo "Using CultureLLM dataset (new format)"
        ;;
    52)
        # wvs
        DATASET_NAME="wvs"
        TRAIN_FILE="/root/autodl-fs/wvs_merge_rp_gen.json"
        DATASET_TAG="wvs_rp"
        echo "Using wvs dataset (new format)"
        ;;
    23)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/autodl-fs/data/CulturalBench_merge_gen_samples.json" #
        DATASET_TAG="CulturalBench_icl"
        echo "Using CulturalBench dataset (new format)"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 2, 3, 4, or 5."
        echo ""
        echo "DATA_ID options:"
        echo "  2 - CulturalBench"
        echo "  3 - NormAD"
        echo "  4 - CultureLLM (default)"
        echo "  5 - wvs"
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-fs/data/ft/ft_base_${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Evaluating Base Model on CultureLLM Dataset (New Format)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo ""
echo "Train file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Train file not found: $TRAIN_FILE"
    echo ""
    echo "Please ensure the data file exists in new format:"
    echo "  {\"instruction\": ..., \"instruction_mask\": ..., \"input\": ..., \"output\": ..., \"label\": ...}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行评估
echo "Starting evaluation..."
echo ""

python ft_base_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 512 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - eval_results.json (评估结果)"
    echo "  - generated_answers.json (生成的答案)"
    echo "  - config.json (配置)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/eval_results.json | python -m json.tool"
    echo ""
    echo "💡 To view generated answers:"
    echo "   cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -50"
    echo ""
    echo "💡 To view accuracy:"
    echo "   python -c \"import json; data = json.load(open('$OUTPUT_DIR/eval_results.json')); print(f'Accuracy: {data[\\\"accuracy\\\"]:.4f}')\""
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

