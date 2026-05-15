#!/bin/bash

# ============================================================
# 使用新数据格式微调 LoRA Only 模型
#
# 使用方法：
#   sh run_ft_lora_only_gen.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM, 5=CultureAtlas, 24=CulturalBench+CultureLLM (默认 24)
#
# 示例：
#   # 使用 LLaMA + CulturalBench+CultureLLM 合并数据集（默认）
#   sh run_ft_lora_only_gen.sh llama 24
#
#   # 使用 Qwen + CulturalBench+CultureLLM 合并数据集
#   sh run_ft_lora_only_gen.sh qwen 24
#
#   # 使用 LLaMA + 单独的CultureLLM 数据集
#   sh run_ft_lora_only_gen.sh llama 4
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-24}"                  # 默认 CulturalBench+CultureLLM (24)

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/base/Qwen2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 根据 DATA_ID 选择数据集
case $DATA_ID in
    1)
        # CultureLLM (默认)
        DATASET_NAME="unified_all_datasets"
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_TAG="unified_all_datasets"
        echo "Using unified_all_datasets dataset (new format)"
        ;;
    2)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
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
        # CultureLLM
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        echo "Using CultureLLM dataset (new format)"
        ;;
    5)
        # CultureAtlas
        DATASET_NAME="CultureAtlas"
        TRAIN_FILE="/autodl-fs/data/cultureAtlas_merge_gen.json"
        DATASET_TAG="cultureAtlas"
        echo "Using CultureAtlas dataset (new format)"
        ;;
    24)
        # CulturalBench + CultureLLM 合并数据集（默认）
        DATASET_NAME="CulturalBench+CultureLLM"
        TRAIN_FILE="MERGED:CulturalBench+CultureLLM"  # 特殊标识，由Python脚本处理
        DATASET_TAG="CulturalBench_CultureLLM"
        echo "Using CulturalBench+CultureLLM merged dataset (new format)"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 2, 3, 4, 5, or 24."
        echo ""
        echo "DATA_ID options:"
        echo "  2  - CulturalBench"
        echo "  3  - NormAD"
        echo "  4  - CultureLLM"
        echo "  5  - CultureAtlas"
        echo "  24 - CulturalBench+CultureLLM (default)"
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-fs/data/ft/ft_lora_only_gen_${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Fine-tuning LoRA Only Model with New Data Format"
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

# 🔧 修改：对于MERGED:开头的特殊标识，跳过文件检查
if [[ "$TRAIN_FILE" == MERGED:* ]]; then
    echo "✅ 检测到合并数据集标识: $TRAIN_FILE"
    echo "将由Python脚本处理具体的数据文件..."
else
    # 对于普通文件，进行存在性检查
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "❌ Error: Train file not found: $TRAIN_FILE"
        echo ""
        echo "Please ensure the data file exists in new format:"
        echo "  {\"instruction\": ..., \"instruction_mask\": ..., \"input\": ..., \"output\": ..., \"label\": ...}"
        exit 1
    fi
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行训练
echo "Starting training..."
echo ""

python ft_lora_only_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 12 \
    --batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-4 \
    --weight_decay 0.001 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --eval_interval 3 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - best_lora/ (Best LoRA weights)"
    echo "  - epoch_eval_results.json (Epoch-by-epoch results)"
    echo "  - generated_answers.json (Generated answers on validation set)"
    echo "  - config.json (Training configuration)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/epoch_eval_results.json | python -m json.tool"
    echo ""
    echo "💡 To view generated answers:"
    echo "   cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -50"
    echo ""
    echo "💡 To view accuracy:"
    echo "   python -c \"import json; data = json.load(open('$OUTPUT_DIR/epoch_eval_results.json')); print(f'Final Accuracy: {data[-1][\\\"eval_accuracy\\\"]:.4f}')\""
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

