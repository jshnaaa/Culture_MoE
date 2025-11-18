#!/bin/bash

# ============================================================
# Simple MoE 模型训练脚本 - 支持多数据集和8:1:1数据划分
# 在 LoRA 微调后的完整模型基础上添加简单的 MoE 结构
#
# 功能：
# 1. 支持 CulturalBench、NormAD、CultureLLM、unified_all_datasets 数据集
# 2. 8:1:1数据划分（训练:验证:测试）
# 3. 验证集选择最佳模型，测试集最终评估
# 4. 保存最佳MoE权重和生成答案
#
# 使用方法：
#   sh run_ft_simple_moe_gen.sh <BACKBONE> <DATA_ID> <NUM_EXPERTS> <TOP_K>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 1=unified_all_datasets, 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 1)
#   NUM_EXPERTS: 专家数量 (默认 12)
#   TOP_K: Top-K路由 (默认 2)
#
# 示例：
#   sh run_ft_simple_moe_gen.sh llama 1 12 2  # unified_all_datasets
#   sh run_ft_simple_moe_gen.sh llama 2 12 2  # CulturalBench
#   sh run_ft_simple_moe_gen.sh qwen 3 8 2    # NormAD
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"           # 默认使用 llama
DATA_ID="${2:-1}"               # 默认使用数据集1 (unified_all_datasets)
NUM_EXPERTS="${3:-12}"          # 默认12个专家
TOP_K="${4:-2}"                 # 默认top-2路由

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 根据 DATA_ID 选择数据集和对应的LoRA权重
case $DATA_ID in
    1)
        # unified_all_datasets
        DATASET_NAME="unified_all_datasets"
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_TAG="unified_all_datasets"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_qwen_20251111_1421/best_lora"
        else
            LORA_WEIGHTS_PATH="/autodl-fs/data/data/ft/ft_lora_only_gen_unified_all_datasets_llama_20251117_1218/best_lora"
        fi
        echo "Using unified_all_datasets dataset"
        ;;
    2)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_qwen_20251112_1228/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_llama_20251112_1141/best_lora"
        fi
        echo "Using CulturalBench dataset"
        ;;
    3)
        # NormAD
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_qwen_20251111_1204/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_llama_20251112_1335/best_lora"
        fi
        echo "Using NormAD dataset"
        ;;
    4)
        # CultureLLM
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-fs/data/ft/ft_lora_only_gen_cultureLLM_qwen_20251114_1301/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_20251112_1551/best_lora"
        fi
        echo "Using CultureLLM dataset"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 1, 2, 3, or 4."
        echo ""
        echo "DATA_ID options:"
        echo "  1 - unified_all_datasets"
        echo "  2 - CulturalBench"
        echo "  3 - NormAD"
        echo "  4 - CultureLLM"
        exit 1
        ;;
esac

# 动态 batch size 配置
if [ "$DATA_ID" = "2" ] || [ "$DATA_ID" = "3" ]; then
    # CultureBench 或 NormAD_ICL
    if [ "$BACKBONE" = "llama" ]; then
        BATCH_SIZE=3
    else
        BATCH_SIZE=8
    fi
else
    # WVS_Gen 或 CultureLLM
    if [ "$BACKBONE" = "llama" ]; then
        BATCH_SIZE=2
    else
        BATCH_SIZE=4
    fi
fi

# 输出目录
OUTPUT_DIR="/root/autodl-fs/data/ft/ft_simple_moe_${DATASET_TAG}_${BACKBONE}_experts${NUM_EXPERTS}_top${TOP_K}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Simple MoE Training"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo "Experts: $NUM_EXPERTS (Top-$TOP_K routing)"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo ""
echo "Training file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# GPU 配置
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# 根据环境变量决定是否使用多卡
if [ "${NUM_GPUS:-1}" = "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "🔧 GPU Configuration: Dual GPU training with DataParallel"
    echo ""
    USE_MULTI_GPU="--use_multi_gpu"
else
    export CUDA_VISIBLE_DEVICES=0
    echo "🔧 GPU Configuration: Single GPU training"
    echo ""
    USE_MULTI_GPU=""
fi

echo "Starting Simple MoE training..."
echo ""

# 运行训练
python ft_simple_moe_gen.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_weights_path $LORA_WEIGHTS_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_experts $NUM_EXPERTS \
    --top_k $TOP_K \
    --expert_hidden_dim 4096 \
    --router_hidden_dim 512 \
    --dropout 0.1 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --num_epochs 12 \
    --batch_size $BATCH_SIZE \
    --max_length 512 \
    --num_workers 2 \
    --save_interval 3 \
    --device cuda \
    $USE_MULTI_GPU

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Simple MoE Training completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Experts: $NUM_EXPERTS (Top-$TOP_K routing)"
    echo "  Dataset: $DATASET_NAME"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo ""
    echo "Training results saved to: $OUTPUT_DIR"
    echo "  - final_model/ (最终模型权重)"
    echo "  - training_results.json (训练结果)"
    echo "  - config.json (训练配置)"
    echo ""
    echo "💡 To evaluate the trained model:"
    echo "   Use the saved MoE weights with a custom evaluation script"
    echo ""
    echo "💡 Model structure:"
    echo "   Base Model + LoRA (frozen) + Simple MoE Layer (trainable)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Simple MoE Training failed!"
    echo "============================================================"
    exit 1
fi