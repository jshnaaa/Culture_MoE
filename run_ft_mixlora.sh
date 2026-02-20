#!/bin/bash

# ============================================================
# 使用MixLoRA方法微调模型
#
# MixLoRA是一种基于LoRA的参数高效混合专家方法，将LoRA的参数高效性
# 与MoE模型的强大性能结合起来。
#
# 核心特性：
# - 在FFN层使用多个LoRA专家 + 注意力层使用普通LoRA
# - Top-K路由：每次只激活K个专家，提高计算效率
# - 负载均衡：使用辅助损失确保专家使用的均衡分布
# - 计算优化：共享FFN计算，减少重复计算
# - 支持合并数据集训练（CulturalBench + CultureLLM）
#
# 使用方法：
#   sh run_ft_mixlora.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM, 24=CulturalBench+CultureLLM (默认 24)
#
# 示例：
#   # 使用 LLaMA + CulturalBench+CultureLLM 合并数据集（默认）
#   sh run_ft_mixlora.sh llama 24
#
#   # 使用 Qwen + CulturalBench+CultureLLM 合并数据集
#   sh run_ft_mixlora.sh qwen 24
#
#   # 使用 LLaMA + 单独的CultureLLM 数据集
#   sh run_ft_mixlora.sh llama 4
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-24}"                  # 默认 CulturalBench+CultureLLM (24)

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
        # CultureLLM
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        echo "Using CultureLLM dataset"
        ;;
    24)
        # CulturalBench + CultureLLM 合并数据集（默认）
        DATASET_NAME="CulturalBench+CultureLLM"
        TRAIN_FILE="MERGED:CulturalBench+CultureLLM"  # 特殊标识，由Python脚本处理
        DATASET_TAG="CulturalBench_CultureLLM"
        echo "Using CulturalBench+CultureLLM merged dataset"
        echo "  - 将分别对两个数据集进行8:1:1划分"
        echo "  - 生成独立的pkl文件: data_split_8_1_1_CulturalBench.pkl, data_split_8_1_1_CultureLLM.pkl"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 1, 2, 3, 4, or 24."
        echo ""
        echo "DATA_ID options:"
        echo "  1  - unified_all_datasets"
        echo "  2  - CulturalBench"
        echo "  3  - NormAD"
        echo "  4  - CultureLLM"
        echo "  24 - CulturalBench+CultureLLM (default)"
        exit 1
        ;;
esac

# 设置时间戳和输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/mixlora/${MODEL_NAME}_${DATASET_TAG}_${TIMESTAMP}"

echo "============================================================"
echo "Fine-tuning Model with MixLoRA"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo ""
echo "MixLoRA Configuration:"
echo "  - Number of experts: 4"
echo "  - Top-K routing: 2"
echo "  - LoRA rank: 64"
echo "  - LoRA alpha: 16"
echo "  - FFN modules: gate_proj, up_proj, down_proj (MixLoRA)"
echo "  - Attention modules: q_proj, v_proj (LoRA)"
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

# 🔧 修改：对于MERGED:开头的特殊标识，跳过文件检查
if [[ "$TRAIN_FILE" == MERGED:* ]]; then
    echo "✅ 检测到合并数据集标识: $TRAIN_FILE"
    echo "将由Python脚本处理具体的数据文件..."
else
    # 对于普通文件，进行存在性检查
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "❌ Error: Train file not found: $TRAIN_FILE"
        echo ""
        echo "Please ensure the data file exists in correct format:"
        echo "  {\"instruction\": ..., \"instruction_mask\": ..., \"input\": ..., \"output\": ..., \"label\": ...}"
        exit 1
    fi
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行训练
echo "Starting MixLoRA training..."
echo ""

python ft_mixlora.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 6 \
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
    --num_experts 4 \
    --top_k 2 \
    --aux_loss_coef 0.01 \
    --eval_interval 3 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ MixLoRA Training completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - best_mixlora/ (Best MixLoRA weights and config - 仅保存训练参数)"
    echo "    ├── mixlora_config.json (MixLoRA configuration)"
    echo "    └── mixlora_weights.pt (MixLoRA expert weights only)"
    echo "  - epoch_eval_results.json (Epoch-by-epoch results, 包含合并和分离评估)"
    echo "  - generated_answers.json (Generated answers on merged validation set)"
    echo "  - individual_dataset_results.json (分离数据集评估结果)"
    echo "  - data_split_8_1_1_*.pkl (各数据集划分信息)"
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
    echo "💡 Model architecture summary:"
    echo "   - Base model: $MODEL_NAME (frozen)"
    echo "   - FFN layers: 4 LoRA experts per layer (Top-2 routing)"
    echo "   - Attention layers: Standard LoRA on q_proj, v_proj (not routed)"
    echo "   - Parameter efficiency: Only LoRA weights are trainable"
    echo "   - Load balancing: Auxiliary loss ensures expert utilization balance"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ MixLoRA Training failed!"
    echo "============================================================"
    echo "Please check the error messages above and verify:"
    echo "  1. Model path is correct and accessible"
    echo "  2. Data file exists and is in the correct format"
    echo "  3. GPU memory is sufficient"
    echo "  4. All dependencies are properly installed"
    echo ""
    echo "Common issues:"
    echo "  - CUDA out of memory: Reduce batch_size or max_length"
    echo "  - Model loading error: Check model path and permissions"
    echo "  - Data format error: Verify JSON structure"
    echo "============================================================"
    exit 1
fi