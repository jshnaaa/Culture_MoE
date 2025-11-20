#!/bin/bash

# ============================================================
# 使用MixLoRA方法进行双卡并行微调
#
# 这个脚本支持多GPU训练，可以有效解决OOM问题
#
# 使用方法：
#   sh run_ft_mixlora_multi_gpu.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
#
# 示例：
#   # 使用 LLaMA + CultureLLM 数据集，双卡训练
#   sh run_ft_mixlora_multi_gpu.sh llama 4
#
#   # 使用 Qwen + CulturalBench 数据集，双卡训练
#   sh run_ft_mixlora_multi_gpu.sh qwen 2
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
        # 统一数据集
        DATASET_NAME="unified_all_datasets"
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_TAG="unified_all_datasets"
        echo "Using unified_all_datasets dataset"
        ;;
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
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 1, 2, 3, or 4."
        echo ""
        echo "DATA_ID options:"
        echo "  1 - unified_all_datasets"
        echo "  2 - CulturalBench"
        echo "  3 - NormAD"
        echo "  4 - CultureLLM (default)"
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_mixlora_multigpu_${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Fine-tuning Model with MixLoRA (Multi-GPU)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo ""
echo "Multi-GPU Configuration:"
echo "  - Number of GPUs: 2"
echo "  - Distributed backend: NCCL"
echo "  - Batch size per GPU: 4"
echo "  - Effective batch size: 8"
echo ""
echo "MixLoRA Configuration:"
echo "  - Number of experts: 6"
echo "  - Top-K routing: 2"
echo "  - LoRA rank: 64"
echo "  - LoRA alpha: 16"
echo "  - FFN modules: gate_proj, up_proj, down_proj (MixLoRA)"
echo "  - Attention modules: None (避免维度问题)"
echo "  - Auxiliary loss coefficient: 0.01"
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
    echo "Please ensure the data file exists in correct format:"
    echo "  {\"instruction\": ..., \"instruction_mask\": ..., \"input\": ..., \"output\": ..., \"label\": ...}"
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
echo "Starting MixLoRA multi-GPU training..."
echo ""
echo "🚀 Multi-GPU Setup:"
echo "  - Using torch.distributed.launch for process management"
echo "  - Backend: NCCL (optimized for GPU communication)"
echo "  - Memory optimization: Reduced batch size per GPU"
echo "  - Gradient synchronization across GPUs"
echo ""

# 使用 torchrun (推荐) 或 python -m torch.distributed.launch
export CUDA_VISIBLE_DEVICES=0,1

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    ft_mixlora.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 12 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-4 \
    --weight_decay 0.001 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_experts 6 \
    --top_k 2 \
    --aux_loss_coef 0.01 \
    --eval_interval 3 \
    --gradient_accumulation_steps 2

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ MixLoRA Multi-GPU Training completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - best_mixlora/ (Best MixLoRA weights and config)"
    echo "    ├── mixlora_config.json (MixLoRA configuration)"
    echo "    ├── mixlora_weights.pt (MixLoRA expert weights)"
    echo "    └── tokenizer files"
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
    echo ""
    echo "💡 Multi-GPU training summary:"
    echo "   - Used 2 GPUs with distributed data parallel (DDP)"
    echo "   - Effective batch size: 8 (4 per GPU)"
    echo "   - Memory usage optimized across GPUs"
    echo "   - Gradient synchronization ensured model consistency"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ MixLoRA Multi-GPU Training failed!"
    echo "============================================================"
    echo "Please check the error messages above and verify:"
    echo "  1. Model path is correct and accessible"
    echo "  2. Data file exists and is in the correct format"
    echo "  3. Both GPUs have sufficient memory"
    echo "  4. All dependencies are properly installed"
    echo "  5. CUDA and NCCL are properly configured"
    echo ""
    echo "Common multi-GPU issues:"
    echo "  - CUDA out of memory: Further reduce batch_size"
    echo "  - NCCL initialization error: Check GPU connectivity"
    echo "  - Port conflict: Change --master_port to different value"
    echo "  - Process hanging: Check firewall/network settings"
    echo "============================================================"
    exit 1
fi