#!/bin/bash

# ============================================================
# 使用新数据格式微调 CultureMoE 模型
#
# 使用方法：
#   sh run_ft_culturemoe_gen.sh <BACKBONE> <DATA_ID> <USE_CULTURE_LOSS> <NUM_EXPERTS> <NUM_GPUS> <CULTURE_LOSS_WEIGHT>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
#   USE_CULTURE_LOSS: True 或 False (默认 True)
#   NUM_EXPERTS: 专家数量 (默认 6)
#   NUM_GPUS: GPU 数量 (默认 2)
#   CULTURE_LOSS_WEIGHT: 文化损失权重 (默认 0.5)
#
# 示例：
#   # 基础训练
#   sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5
#
#   # 消融实验：低文化损失权重
#   sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.1
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-4}"                   # 默认 CultureLLM (4)
USE_CULTURE_LOSS="${3:-True}"       # 默认使用文化损失
NUM_EXPERTS="${4:-6}"               # 默认 6 个专家
NUM_GPUS="${5:-2}"                  # 默认使用 2 个 GPU
CULTURE_LOSS_WEIGHT="${6:-0.5}"     # 默认文化损失权重 0.5

# 根据 backbone 选择 base 模型路径和 LoRA 权重路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_*/best_lora"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_*/best_lora"
fi

# 根据 DATA_ID 选择数据集
case $DATA_ID in
    2)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen_small.json"
        DATASET_TAG="CulturalBench"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_qwen_*/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_llama_*/best_lora"
        fi
        echo "Using CulturalBench dataset (new format)"
        ;;
    3)
        # NormAD
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_qwen_*/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_llama_*/best_lora"
        fi
        echo "Using NormAD dataset (new format)"
        ;;
    4)
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen_small.json"
        DATASET_TAG="cultureLLM"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_*/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_*/best_lora"
        fi
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
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_moe_gen_${DATASET_TAG}_${BACKBONE}_experts${NUM_EXPERTS}_CULTURE_LOSS_WEIGHT${CULTURE_LOSS_WEIGHT}_$(date +%Y%m%d_%H%M)"

# 设置 GPU
if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
    GPU_INFO="Single GPU (GPU 0)"
elif [ "$NUM_GPUS" = "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    GPU_INFO="Dual GPUs (GPU 0,1)"
else
    export CUDA_VISIBLE_DEVICES=0,1
    GPU_INFO="Dual GPUs (GPU 0,1)"
fi

echo "============================================================"
echo "Fine-tuning CultureMoE Model with New Data Format"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo "Use culture loss: $USE_CULTURE_LOSS"
echo "Culture loss weight: $CULTURE_LOSS_WEIGHT"
echo "Num experts: $NUM_EXPERTS"
echo "GPUs: $GPU_INFO"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
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
    exit 1
fi

# 检查 LoRA 权重
LORA_PATH=$(ls -d $LORA_WEIGHTS_PATH 2>/dev/null | head -1)
if [ -z "$LORA_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    echo ""
    echo "Please first run:"
    echo "  sh run_ft_lora_only_gen.sh $BACKBONE $DATA_ID"
    exit 1
fi

echo "Found LoRA weights: $LORA_PATH"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行训练
echo "Starting training..."
echo ""

python ft_culturemoe_from_base_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --lora_weights_path "$LORA_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --use_culture_loss "$USE_CULTURE_LOSS" \
    --culture_loss_lambda "$CULTURE_LOSS_WEIGHT" \
    --num_epochs 30 \
    --num_experts "$NUM_EXPERTS" \
    --shared_hidden_dim 4096 \
    --router_hidden_dim 2048 \
    --experts_hidden_dim 4096 \
    --moe_lora_rank 32 \
    --classification_hidden_dim 512 \
    --dropout 0.05 \
    --num_heads 8 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - best_moe/ (Best MOE weights)"
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

