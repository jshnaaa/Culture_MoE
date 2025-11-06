#!/bin/bash

# ============================================================
# 训练 LoRA Only 模型（生成式版本）
#
# 使用方法：
#   sh run_train_lora_only_gen.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen
#   DATA_ID: 数据集标识符
#     0 - 合并所有数据集 (CulturalBench + NormAD + CultureLLM)
#     2 - CulturalBench
#     3 - NormAD
#     4 - CultureLLM (默认)
#
# 示例：
#   sh run_train_lora_only_gen.sh llama 4    # CultureLLM
#   sh run_train_lora_only_gen.sh llama 0    # 所有数据集
#   sh run_train_lora_only_gen.sh qwen 2     # CulturalBench
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"      # 默认使用 llama
DATA_ID="${2:-4}"           # 默认使用 CultureLLM (4)

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
    0)
        # 合并所有数据集
        DATASET_NAME="All_Datasets"
        DATA_PATH="/root/autodl-fs/CulturalBench_merge_gen.json,/root/autodl-fs/normad_merge_gen.json,/root/autodl-fs/cultureLLM_merge_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_all_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        echo "Using all datasets (CulturalBench + NormAD + CultureLLM)"
        ;;
    2)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        DATA_PATH="/root/autodl-fs/CulturalBench_merge_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_CulturalBench_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        echo "Using CulturalBench dataset"
        ;;
    3)
        # NormAD
        DATASET_NAME="NormAD"
        DATA_PATH="/root/autodl-fs/normad_merge_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_normad_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        echo "Using NormAD dataset"
        ;;
    4)
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        DATA_PATH="/root/autodl-fs/cultureLLM_merge_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        echo "Using CultureLLM dataset"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 0, 2, 3, or 4."
        echo ""
        echo "DATA_ID options:"
        echo "  0 - All datasets (CulturalBench + NormAD + CultureLLM)"
        echo "  2 - CulturalBench"
        echo "  3 - NormAD"
        echo "  4 - CultureLLM (default)"
        exit 1
        ;;
esac

echo "============================================================"
echo "Training LoRA Only Model (Generative Version)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME (DATA_ID=$DATA_ID)"
echo "Base model: $BASE_MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

# 检查数据文件（支持多个文件，用逗号分隔）
IFS=',' read -ra DATA_FILES <<< "$DATA_PATH"
for file in "${DATA_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Data file not found: $file"
        exit 1
    fi
done
echo "✅ All data files found"
echo ""

# 检测可用 GPU 数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

# 运行训练脚本（支持多卡）
if [ $NUM_GPUS -gt 1 ]; then
    echo "Using multi-GPU training with $NUM_GPUS GPUs"
    echo ""

    # 使用 torchrun 进行多卡训练
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_lora_only_gen.py \
        --model_name_or_path $BASE_MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_length 512 \
        --save_steps 500
else
    echo "Using single-GPU training"
    echo ""

    # 单卡训练
    python train_lora_only_gen.py \
        --model_name_or_path $BASE_MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_length 512 \
        --save_steps 500
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Merge LoRA weights: sh run_merge_lora_gen.sh $BACKBONE"
    echo "  2. Evaluate model: sh run_eval_lora_only_gen.sh $BACKBONE"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

