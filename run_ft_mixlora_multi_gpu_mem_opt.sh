#!/bin/bash

# ============================================================
# 使用MixLoRA方法进行双卡并行微调 - 极限内存优化版本
#
# 这个脚本专门为内存受限的环境设计，使用最小的参数配置
#
# 使用方法：
#   sh run_ft_mixlora_multi_gpu_mem_opt.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
#
# 示例：
#   # 使用 LLaMA + CultureLLM 数据集，双卡训练（极限内存优化）
#   sh run_ft_mixlora_multi_gpu_mem_opt.sh llama 4
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
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        echo "Using CulturalBench dataset"
        ;;
    3)
        # NormAD
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        echo "Using NormAD dataset"
        ;;
    4)
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        echo "Using CultureLLM dataset"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 2, 3, or 4."
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_mixlora_multigpu_memopt_${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Fine-tuning Model with MixLoRA (Multi-GPU Extreme Memory Optimization)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo ""
echo "Multi-GPU Configuration:"
echo "  - Number of GPUs: 2"
echo "  - Distributed backend: NCCL"
echo "  - Batch size per GPU: 1"
echo "  - Effective batch size: 2"
echo "  - Gradient accumulation: 8 steps"
echo ""
echo "MixLoRA Configuration (Extreme Memory Optimization):"
echo "  - Number of experts: 2 (minimal for MoE)"
echo "  - Top-K routing: 1 (minimal activation)"
echo "  - LoRA rank: 16 (minimal rank)"
echo "  - LoRA alpha: 8"
echo "  - Max sequence length: 256 (minimal length)"
echo "  - FFN modules: gate_proj, up_proj, down_proj (MixLoRA)"
echo "  - Attention modules: None (避免维度问题)"
echo "  - Auxiliary loss coefficient: 0.001"
echo "  - Memory optimizations: maximum"
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
    exit 1
fi

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "❌ Error: At least 2 GPUs are required for multi-GPU training"
    echo "Available GPUs: $GPU_COUNT"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行分布式训练
echo "Starting MixLoRA extreme memory optimized multi-GPU training..."
echo ""
echo "🚀 Extreme Memory Optimization Setup:"
echo "  - Minimal expert count (2 experts)"
echo "  - Minimal activation (Top-1 routing)"
echo "  - Minimal LoRA rank (16)"
echo "  - Minimal sequence length (256)"
echo "  - Maximum gradient accumulation (8 steps)"
echo "  - Aggressive memory cleanup enabled"
echo ""

# 设置极限内存优化环境变量（多GPU兼容配置）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    ft_mixlora.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 12 \
    --batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.001 \
    --max_length 256 \
    --val_split 0.1 \
    --num_workers 0 \
    --lora_r 16 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --num_experts 2 \
    --top_k 1 \
    --aux_loss_coef 0.001 \
    --eval_interval 4 \
    --gradient_accumulation_steps 8 \
    --memory_efficient

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ MixLoRA Extreme Memory Optimized Multi-GPU Training completed!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "💡 Extreme memory optimization summary:"
    echo "   - Used minimal MixLoRA configuration for maximum memory efficiency"
    echo "   - 2 experts with Top-1 routing (minimal MoE overhead)"
    echo "   - LoRA rank 16 (minimal parameter count)"
    echo "   - Sequence length 256 (minimal memory per sample)"
    echo "   - Batch size 1 per GPU with 8x gradient accumulation"
    echo "   - Aggressive memory cleanup and optimization enabled"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ MixLoRA Extreme Memory Optimized Training failed!"
    echo "============================================================"
    echo "If this configuration still fails with OOM, consider:"
    echo "  1. Using a smaller model (7B instead of 8B)"
    echo "  2. Further reducing sequence length to 128"
    echo "  3. Using CPU offloading techniques"
    echo "  4. Using model sharding across more GPUs"
    echo "============================================================"
    exit 1
fi