#!/bin/bash

# ============================================================
# 从 Base 模型 + LoRA 权重训练 CultureMoE（生成式版本）
#
# 使用方法：
#   sh run_train_culturemoe_from_base_gen.sh <BACKBONE> <NUM_CLASSES> <USE_CULTURE_LOSS> <NUM_EXPERTS> <NUM_GPUS> <MASK_USE>
#
# 示例：
#   sh run_train_culturemoe_from_base_gen.sh llama 5 True 6 1 true
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-4}"                   # 默认 CultureLLM (4)
USE_CULTURE_LOSS="${3:-True}"       # 默认使用文化损失
NUM_EXPERTS="${4:-6}"               # 默认 6 个专家
NUM_GPUS="${5:-2}"                  # 默认使用 2 个 GPU
MASK_USE="${6:-true}"               # 默认使用 instruction_mask

# 根据 backbone 选择 base 模型路径和 LoRA 权重路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251107_2124/best_lora"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora"
fi

# 根据 DATA_ID 选择数据集
case $DATA_ID in
    2)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_CulturalBench_qwen_20251107_2124/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_CulturalBench_llama_20251108_2126/best_lora"
        fi
        echo "Using CulturalBench dataset"
        ;;
    3)
        # NormAD
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_normad_qwen_20251107_2124/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_normad_llama_20251107_2245/best_lora"
        fi
        echo "Using NormAD dataset"
        ;;
    4)
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251107_2124/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora"
        fi
        echo "Using CultureLLM dataset"
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
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_${DATASET_TAG}_${BACKBONE}_experts${NUM_EXPERTS}_USE_CULTURE_LOSS${USE_CULTURE_LOSS}_MASK_USE${MASK_USE}_$(date +%Y%m%d_%H%M)"

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
echo "CultureMoE Training (From Base + LoRA, Gen Version)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo "Use culture loss: $USE_CULTURE_LOSS"
echo "Num experts: $NUM_EXPERTS"
echo "Use instruction_mask: $MASK_USE"
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

if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    echo ""
    echo "Please ensure you have completed:"
    echo "  sh run_train_lora_only_gen.sh $BACKBONE <DATA_ID>"
    echo ""
    echo "The training should create a 'best_lora' directory containing:"
    echo "  - adapter_config.json"
    echo "  - adapter_model.safetensors"
    echo "  - tokenizer files"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Train file not found: $TRAIN_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建训练命令
TRAIN_CMD="python train_culturemoe_from_base_gen.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_weights_path $LORA_WEIGHTS_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --use_culture_loss $USE_CULTURE_LOSS \
    --culture_loss_lambda 0.5 \
    --use_instruction_mask $MASK_USE \
    \
    --num_epochs 30 \
    --num_experts $NUM_EXPERTS \
    --shared_hidden_dim 2048 \
    --router_hidden_dim 1024 \
    --experts_hidden_dim 2048 \
    --moe_lora_rank 32 \
    --classification_hidden_dim 512 \
    --dropout 0.05 \
    --num_heads 8 \
    \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2"

# 运行训练
eval $TRAIN_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - best_moe/ (Best MoE weights based on accuracy)"
    echo "    - moe_config.json"
    echo "    - moe_state_dict.pt"
    echo "  - epoch_eval_results.json (Epoch-by-epoch results)"
    echo "  - final_eval_results.json (Final evaluation results)"
    echo "  - config.json (Training configuration)"
    echo ""
    echo "💡 Next steps:"
    echo "  1. Evaluate on test set:"
    echo "     python eval_culturemoe_from_components_gen.py \\"
    echo "       --base_model_path $BASE_MODEL_PATH \\"
    echo "       --lora_weights_path $LORA_WEIGHTS_PATH \\"
    echo "       --moe_weights_path $OUTPUT_DIR/best_moe \\"
    echo "       --test_file /path/to/test.json \\"
    echo "       --output_dir /path/to/eval_output"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

