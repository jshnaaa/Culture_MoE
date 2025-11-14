#!/bin/bash

# ============================================================
# 使用新数据格式微调 LoRA Only 模型
#
# 使用方法：
#   sh run_ft_lora_only_gen.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
#
# 示例：
#   # 使用 LLaMA + CultureLLM 数据集（新格式）
#   sh run_ft_lora_only_gen.sh llama 4
#
#   # 使用 Qwen + CultureLLM 数据集（新格式）
#   sh run_ft_lora_only_gen.sh qwen 4
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
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        echo "Using CultureLLM dataset (new format)"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 2, 3, or 4."
        echo ""
        echo "DATA_ID options:"
        echo "  2 - CulturalBench"
        echo "  3 - NormAD"
        echo "  4 - CultureLLM (default)"
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

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Train file not found: $TRAIN_FILE"
    echo ""
    echo "Please ensure the data file exists in new format:"
    echo "  {\"instruction\": ..., \"instruction_mask\": ..., \"input\": ..., \"output\": ..., \"label\": ...}"
    exit 1
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

