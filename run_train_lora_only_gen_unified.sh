#!/bin/bash

# ============================================================
# 训练统一标签的通用 LoRA 模型（生成式版本）
#
# 使用方法：
#   sh run_train_lora_only_gen_unified.sh <BACKBONE>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#
# 示例：
#   sh run_train_lora_only_gen_unified.sh llama
#   sh run_train_lora_only_gen_unified.sh qwen
#
# 说明：
#   此脚本会自动：
#   1. 检查统一数据集是否存在
#   2. 如果不存在，自动运行预处理脚本
#   3. 训练通用模型
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"      # 默认使用 llama

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 数据集路径
CULTURELLM_DATA="/root/autodl-fs/cultureLLM_merge_gen.json"
NORMAD_DATA="/root/autodl-fs/normad_merge_gen.json"
CULTURALBENCH_DATA="/root/autodl-fs/CulturalBench_merge_gen.json"
UNIFIED_DATA="/root/autodl-fs/unified_all_datasets.json"

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_all_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Training Unified LoRA Model (Generative Version)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: Unified (CultureLLM + NormAD + CulturalBench)"
echo "Base model: $BASE_MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查 base 模型
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

# 检查统一数据集是否存在
if [ ! -f "$UNIFIED_DATA" ]; then
    echo "⚠️  Unified dataset not found: $UNIFIED_DATA"
    echo ""
    echo "Preparing unified dataset..."
    echo "============================================================"

    # 检查原始数据集
    if [ ! -f "$CULTURELLM_DATA" ]; then
        echo "❌ Error: CultureLLM data not found: $CULTURELLM_DATA"
        exit 1
    fi

    if [ ! -f "$NORMAD_DATA" ]; then
        echo "❌ Error: NormAD data not found: $NORMAD_DATA"
        exit 1
    fi

    if [ ! -f "$CULTURALBENCH_DATA" ]; then
        echo "❌ Error: CulturalBench data not found: $CULTURALBENCH_DATA"
        exit 1
    fi

    # 运行预处理脚本
    python prepare_unified_dataset.py \
        --culturellm "$CULTURELLM_DATA" \
        --normad "$NORMAD_DATA" \
        --culturalbench "$CULTURALBENCH_DATA" \
        --output "$UNIFIED_DATA" \
        --shuffle

    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to prepare unified dataset"
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo "✅ Unified dataset prepared successfully!"
    echo "============================================================"
    echo ""
else
    echo "✅ Unified dataset found: $UNIFIED_DATA"
    echo ""
fi

# 检测可用 GPU 数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

# 运行训练脚本（支持多卡）
echo "============================================================"
echo "Starting training..."
echo "============================================================"
echo ""

if [ $NUM_GPUS -gt 1 ]; then
    echo "Using multi-GPU training with $NUM_GPUS GPUs"
    echo ""

    # 使用 torchrun 进行多卡训练
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_lora_only_gen.py \
        --model_name_or_path $BASE_MODEL_PATH \
        --data_path $UNIFIED_DATA \
        --output_dir $OUTPUT_DIR \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --learning_rate 1e-4 \
        --num_train_epochs 12 \
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
        --data_path $UNIFIED_DATA \
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
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Dataset: Unified (CultureLLM + NormAD + CulturalBench)"
    echo "  Label range: 1-15"
    echo "    - CultureLLM:    1-10"
    echo "    - NormAD:        11 (yes), 12 (no), 13 (neutral)"
    echo "    - CulturalBench: 14 (TRUE), 15 (FALSE)"
    echo ""
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate on CultureLLM:    sh run_eval_unified_model.sh $BACKBONE culturellm"
    echo "  2. Evaluate on NormAD:        sh run_eval_unified_model.sh $BACKBONE normad"
    echo "  3. Evaluate on CulturalBench: sh run_eval_unified_model.sh $BACKBONE culturalbench"
    echo "  4. Evaluate on all:           sh run_eval_unified_model.sh $BACKBONE all"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

# 训练完成后自动评估
if [ -f "${OUTPUT_DIR}/generated_answers.json" ]; then
    echo ""
    echo "Running post-evaluation on validation set..."
    python eval_from_generated_answers.py \
        --input "${OUTPUT_DIR}/generated_answers.json" \
        --output "${OUTPUT_DIR}/eval_metrics.json"
fi

