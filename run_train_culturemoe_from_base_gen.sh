#!/bin/bash

# ============================================================
# 从 Base 模型 + LoRA 权重训练 CultureMoE（生成式版本）
#
# 使用方法：
#   sh run_train_culturemoe_from_base_gen.sh <BACKBONE> <NUM_CLASSES> <USE_CULTURE_LOSS> <NUM_EXPERTS> <SAVE_MODEL> <NUM_GPUS> <MASK_USE> <LORA_DIR>
#
# 示例：
#   sh run_train_culturemoe_from_base_gen.sh llama 5 True 6 false 1 true /path/to/lora_output
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
NUM_CLASSES="${2:-5}"               # 默认 5 分类
USE_CULTURE_LOSS="${3:-True}"       # 默认使用文化损失
NUM_EXPERTS="${4:-6}"               # 默认 6 个专家
SAVE_MODEL="${5:-false}"            # 默认不保存完整模型
NUM_GPUS="${6:-1}"                  # 默认使用 1 个 GPU
MASK_USE="${7:-true}"               # 默认使用 instruction_mask
LORA_OUTPUT_DIR="${8}"              # LoRA 训练输出目录

# 检查必需参数
if [ -z "$LORA_OUTPUT_DIR" ]; then
    echo "❌ Error: LORA_OUTPUT_DIR is required"
    echo ""
    echo "Usage: sh run_train_culturemoe_from_base_gen.sh <BACKBONE> <NUM_CLASSES> <USE_CULTURE_LOSS> <NUM_EXPERTS> <SAVE_MODEL> <NUM_GPUS> <MASK_USE> <LORA_DIR>"
    echo ""
    echo "Example:"
    echo "  sh run_train_culturemoe_from_base_gen.sh llama 5 True 6 false 1 true \\"
    echo "    /root/autodl-tmp/.../lora_only_gen_cultureLLM_llama_20251105_1234"
    exit 1
fi

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# LoRA 权重路径（best_lora 目录）
LORA_WEIGHTS_PATH="${LORA_OUTPUT_DIR}/best_lora"

# 根据 num_classes 选择数据集
case $NUM_CLASSES in
    2)
        TRAIN_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
        DATASET_NAME="CulturalBench_Hard"
        ;;
    3)
        TRAIN_FILE="/root/autodl-fs/normad_ed_merge.json"
        DATASET_NAME="NormAD_ED"
        ;;
    4)
        TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge_4.json"
        DATASET_NAME="WVS_4class"
        ;;
    5)
        TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge_5.json"
        DATASET_NAME="WVS_5class"
        ;;
    *)
        echo "❌ Error: Invalid num_classes=$NUM_CLASSES. Must be 2, 3, 4, or 5."
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_gen_output/culturemoe_gen_${BACKBONE}_${NUM_CLASSES}class_experts${NUM_EXPERTS}_USE_CULTURE_LOSS${USE_CULTURE_LOSS}_MASK_USE${MASK_USE}_$(date +%Y%m%d_%H%M)"

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
echo "Num classes: $NUM_CLASSES"
echo "Dataset: $DATASET_NAME"
echo "Use culture loss: $USE_CULTURE_LOSS"
echo "Num experts: $NUM_EXPERTS"
echo "Save model: $SAVE_MODEL"
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
    echo "Expected path: ${LORA_OUTPUT_DIR}/best_lora"
    echo ""
    echo "Please ensure you have completed:"
    echo "  sh run_train_lora_only_gen.sh $BACKBONE"
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
    --num_classes $NUM_CLASSES \
    --use_culture_loss $USE_CULTURE_LOSS \
    --culture_loss_lambda 0.5 \
    --use_instruction_mask $MASK_USE \
    \
    --num_epochs 10 \
    --num_experts $NUM_EXPERTS \
    --shared_hidden_dim 2048 \
    --router_hidden_dim 1024 \
    --experts_hidden_dim 2048 \
    --moe_lora_rank 16 \
    --classification_hidden_dim 512 \
    --dropout 0.1 \
    --num_heads 8 \
    \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2"

# 添加 save_model 参数
if [ "$SAVE_MODEL" = "true" ]; then
    MODEL_SAVE_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_gen_${BACKBONE}_${NUM_CLASSES}"
    TRAIN_CMD="$TRAIN_CMD --save_model --model_save_path $MODEL_SAVE_PATH"
    echo "Model will be saved to: $MODEL_SAVE_PATH"
    echo ""
fi

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
    echo "  - config.json (Training configuration)"
    echo ""
    echo "💡 Next steps:"
    echo "  1. Evaluate the model:"
    echo "     sh run_eval_culturemoe_from_components.sh $BACKBONE $NUM_CLASSES \\"
    echo "       $LORA_OUTPUT_DIR \\"
    echo "       $OUTPUT_DIR/best_moe"
    echo ""
    echo "  2. Or load from components:"
    echo "     python load_culturemoe_gen.py \\"
    echo "       --base_model_path $BASE_MODEL_PATH \\"
    echo "       --lora_weights_path $LORA_WEIGHTS_PATH \\"
    echo "       --moe_weights_path $OUTPUT_DIR/best_moe \\"
    echo "       --test_file /path/to/test.json \\"
    echo "       --output_file /path/to/results.json"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

