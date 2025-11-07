#!/bin/bash

# ============================================================
# 从 Base 模型 + 统一 LoRA 权重训练通用 CultureMoE（生成式版本）
#
# 使用方法：
#   sh run_train_culturemoe_from_base_gen_unified.sh <BACKBONE> <USE_CULTURE_LOSS> <NUM_EXPERTS> <NUM_GPUS> <MASK_USE>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   USE_CULTURE_LOSS: True 或 False (默认 True)
#   NUM_EXPERTS: 专家数量 (默认 6)
#   NUM_GPUS: GPU 数量 (默认 1)
#   MASK_USE: true 或 false (默认 true)
#
# 示例：
#   sh run_train_culturemoe_from_base_gen_unified.sh llama True 6 1 true
#   sh run_train_culturemoe_from_base_gen_unified.sh qwen True 8 2 true
#
# 说明：
#   此脚本使用统一标签的数据集（标签范围 1-15）训练通用 CultureMoE 模型
#   - CultureLLM:    1-10
#   - NormAD:        11 (yes), 12 (no), 13 (neutral)
#   - CulturalBench: 14 (TRUE), 15 (FALSE)
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
USE_CULTURE_LOSS="${2:-True}"       # 默认使用文化损失
NUM_EXPERTS="${3:-6}"               # 默认 6 个专家
NUM_GPUS="${4:-1}"                  # 默认使用 1 个 GPU
MASK_USE="${5:-true}"               # 默认使用 instruction_mask

# 固定参数
NUM_CLASSES=10  # 统一标签范围：1-15

# 根据 backbone 选择 base 模型路径和 LoRA 权重路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    # 使用最新的统一 LoRA 模型
    LORA_WEIGHTS_PATH=$(ls -td /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_unified_qwen_*/best_lora 2>/dev/null | head -1)
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    # 使用最新的统一 LoRA 模型
    LORA_WEIGHTS_PATH=$(ls -td /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_unified_llama_*/best_lora 2>/dev/null | head -1)
fi

# 检查 LoRA 权重是否存在
if [ -z "$LORA_WEIGHTS_PATH" ] || [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: No unified LoRA model found for $BACKBONE"
    echo ""
    echo "Please train the unified LoRA model first:"
    echo "  sh run_train_lora_only_gen_unified.sh $BACKBONE"
    echo ""
    exit 1
fi

# 统一数据集路径
TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
DATASET_NAME="Unified (CultureLLM + NormAD + CulturalBench)"

# 检查统一数据集是否存在
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Unified dataset not found: $TRAIN_FILE"
    echo ""
    echo "Please prepare the unified dataset first:"
    echo "  python prepare_unified_dataset.py \\"
    echo "    --culturellm /root/autodl-fs/cultureLLM_merge_gen.json \\"
    echo "    --normad /root/autodl-fs/normad_merge_gen.json \\"
    echo "    --culturalbench /root/autodl-fs/CulturalBench_merge_gen.json \\"
    echo "    --output /root/autodl-fs/unified_all_datasets.json \\"
    echo "    --shuffle"
    echo ""
    exit 1
fi

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/moe_unified_${BACKBONE}_experts${NUM_EXPERTS}_USE_CULTURE_LOSS${USE_CULTURE_LOSS}_MASK_USE${MASK_USE}_$(date +%Y%m%d_%H%M)"

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
echo "Unified CultureMoE Training (From Base + Unified LoRA)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Num classes: $NUM_CLASSES (Unified: 1-15)"
echo "  - CultureLLM:    1-10"
echo "  - NormAD:        11 (yes), 12 (no), 13 (neutral)"
echo "  - CulturalBench: 14 (TRUE), 15 (FALSE)"
echo "Dataset: $DATASET_NAME"
echo "Use culture loss: $USE_CULTURE_LOSS"
echo "Num experts: $NUM_EXPERTS"
echo "Use instruction_mask: $MASK_USE"
echo "GPUs: $GPU_INFO"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  Unified LoRA weights: $LORA_WEIGHTS_PATH"
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
    echo "❌ Error: Unified LoRA weights not found: $LORA_WEIGHTS_PATH"
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

echo "Starting training..."
echo ""

# 运行训练
eval $TRAIN_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Num classes: $NUM_CLASSES (Unified: 1-15)"
    echo "  Num experts: $NUM_EXPERTS"
    echo "  Use culture loss: $USE_CULTURE_LOSS"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - best_moe/ (Best MoE weights based on accuracy)"
    echo "    - moe_config.json"
    echo "    - moe_state_dict.pt (MoE only)"
    echo "  - generated_answers.json (Last epoch's generated answers)"
    echo "  - eval_metrics.json (Last epoch's evaluation metrics)"
    echo "  - epoch_eval_results.json (Epoch-by-epoch results)"
    echo "  - final_eval_results.json (Final evaluation results)"
    echo "  - config.json (Training configuration)"
    echo ""
    echo "💡 Next steps:"
    echo "  1. Evaluate on CultureLLM test set:"
    echo "     sh run_eval_unified_culturemoe.sh $BACKBONE culturellm"
    echo ""
    echo "  2. Evaluate on NormAD test set:"
    echo "     sh run_eval_unified_culturemoe.sh $BACKBONE normad"
    echo ""
    echo "  3. Evaluate on CulturalBench test set:"
    echo "     sh run_eval_unified_culturemoe.sh $BACKBONE culturalbench"
    echo ""
    echo "  4. Evaluate on all test sets:"
    echo "     sh run_eval_unified_culturemoe.sh $BACKBONE all"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

