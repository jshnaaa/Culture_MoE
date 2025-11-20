#!/bin/bash

# ============================================================
# 🚀 MiLoRA (Mixture of LoRA) TRAINING SCRIPT
# 基于提示感知路由机制的混合 LoRA 专家训练
# ============================================================
# MiLoRA 核心特性：
#
# ✅ 核心创新：
#   1. 每个 LoRA 模块视为一个专家（7个专家：Q, K, V, O, G, U, D）
#   2. 提示感知路由机制：只在生成第一个新词前计算一次路由，后续复用
#   3. 负载均衡损失 + 可学习的理性激活函数
#   4. Top-k 专家选择（默认 k=3）
#   5. 多种池化方式：last-token, average, max, self-attention
#
# 使用方法：
#   sh run_ft_milora_gen.sh <BACKBONE> <DATA_ID> <LORA_RANK> <TOP_K> <POOLING_TYPE>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 1=unified_all_datasets, 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 1)
#   LORA_RANK: LoRA 秩 (默认 16)
#   TOP_K: 选择的专家数量 (默认 3)
#   POOLING_TYPE: 池化类型 (默认 self_attention)
#
# 示例：
#   # 基本训练
#   sh run_ft_milora_gen.sh llama 1
#
#   # 自定义参数训练
#   sh run_ft_milora_gen.sh llama 1 32 4 average
#
#   # Qwen 模型训练
#   sh run_ft_milora_gen.sh qwen 1 16 3 self_attention
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-1}"                   # 默认 unified_all_datasets (1)
LORA_RANK="${3:-16}"                # 默认 LoRA 秩 16
TOP_K="${4:-3}"                     # 默认选择 3 个专家
POOLING_TYPE="${5:-self_attention}" # 默认自注意力池化

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
        DATASET_NAME="unified_all_datasets"
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_TAG="unified_all_datasets"
        echo "Using unified_all_datasets dataset"
        ;;
    2)
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        echo "Using CulturalBench dataset"
        ;;
    3)
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        echo "Using NormAD dataset"
        ;;
    4)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        echo "Using CultureLLM dataset"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 1, 2, 3, or 4."
        echo ""
        echo "DATA_ID options:"
        echo "  1 - unified_all_datasets (default)"
        echo "  2 - CulturalBench"
        echo "  3 - NormAD"
        echo "  4 - CultureLLM"
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-fs/data/ft/ft_milora_gen_${DATASET_TAG}_${BACKBONE}_rank${LORA_RANK}_topk${TOP_K}_${POOLING_TYPE}_$(date +%Y%m%d_%H%M)"

# 根据模型类型设置批次大小
if [ "$BACKBONE" = "qwen" ]; then
    TRAIN_BATCH_SIZE=4
    EVAL_BATCH_SIZE=4
else
    TRAIN_BATCH_SIZE=2
    EVAL_BATCH_SIZE=2
fi

echo "============================================================"
echo "🚀 MiLoRA (Mixture of LoRA) Training"
echo "============================================================"
echo "Training Mode: Prompt-Aware Routing with LoRA Experts"
echo "Innovation: Each LoRA module as an expert + One-time routing"
echo ""
echo "🔬 MiLoRA Architecture:"
echo "  1. LoRA Experts: 7 experts per layer (Q, K, V, O, G, U, D)"
echo "  2. Prompt-Aware Routing: Compute routing once, reuse for generation"
echo "  3. Rational Activation: Learnable activation function"
echo "  4. Load Balancing: Prevent expert collapse"
echo "  5. Top-k Selection: Activate top-k experts per layer"
echo ""
echo "📊 Training Strategy:"
echo "  - Routing Computation: Only at prompt encoding phase"
echo "  - Generation Phase: Reuse cached routing results"
echo "  - Load Balance Loss: λ_lb = 0.01"
echo "  - Expert Selection: Top-k = $TOP_K out of 7 experts"
echo ""
echo "🖥️  Model Configuration:"
echo "  Backbone: $BACKBONE ($MODEL_NAME)"
echo "  Dataset: $DATASET_NAME (ID: $DATA_ID)"
echo "  LoRA Rank: $LORA_RANK"
echo "  Top-k Experts: $TOP_K"
echo "  Pooling Type: $POOLING_TYPE"
echo ""
echo "📦 Batch Size Configuration:"
echo "  Train batch size: $TRAIN_BATCH_SIZE"
echo "  Eval batch size: $EVAL_BATCH_SIZE"
echo ""
echo "Learning Configuration:"
echo "  Base model: FROZEN"
echo "  MiLoRA experts: TRAINABLE (2e-4)"
echo "  Router networks: TRAINABLE (2e-4)"
echo "  Rational activation: TRAINABLE"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  Train file: $TRAIN_FILE"
echo "  Output: $OUTPUT_DIR"
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

echo "✅ All paths verified"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行 MiLoRA 训练
echo "🚀 Starting MiLoRA training..."
echo ""

python ft_milora_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha $(echo "$LORA_RANK * 2" | bc) \
    --top_k "$TOP_K" \
    --pooling_type "$POOLING_TYPE" \
    --load_balance_weight 0.01 \
    --num_epochs 5 \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dropout 0.1 \
    --max_length 512 \
    --val_split 0.1 \
    --device cuda \
    --fp16 \
    --num_workers 2 \
    --seed 42 \
    --save_interval 2

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ MiLoRA Training Completed Successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "📁 Generated Files:"
    echo "  ├── best_milora/ (Best MiLoRA weights)"
    echo "  ├── final_milora/ (Final MiLoRA weights)"
    echo "  ├── checkpoint-epoch-*/ (Intermediate checkpoints)"
    echo "  ├── training_history.json (Training metrics)"
    echo "  ├── config.json (Model configuration)"
    echo "  └── training.log (Detailed logs)"
    echo ""
    echo "🔬 MiLoRA Architecture Summary:"
    echo "  - LoRA Experts per Layer: 7 (Q, K, V, O, G, U, D)"
    echo "  - Active Experts per Layer: $TOP_K"
    echo "  - Routing Strategy: Prompt-aware (one-time computation)"
    echo "  - Pooling Method: $POOLING_TYPE"
    echo "  - LoRA Rank: $LORA_RANK"
    echo ""
    echo "💡 Key Innovations:"
    echo "  1. 🎯 Prompt-Aware Routing: Compute once, reuse for generation"
    echo "  2. 🧠 LoRA-as-Expert: Each LoRA module is an independent expert"
    echo "  3. ⚖️  Load Balancing: Prevent expert collapse with λ_lb loss"
    echo "  4. 🔀 Rational Activation: Learnable activation functions"
    echo "  5. 🎛️  Flexible Pooling: Multiple pooling strategies available"
    echo ""
    echo "💡 To view training results:"
    echo "   cat $OUTPUT_DIR/training_history.json | python -m json.tool"
    echo ""
    echo "💡 To view model statistics:"
    echo "   cat $OUTPUT_DIR/best_milora/config.json | python -c \"import json,sys; data=json.load(sys.stdin); print(json.dumps(data['model_statistics'], indent=2))\""
    echo ""
    echo "💡 Usage Examples:"
    echo "   # Different LoRA ranks:"
    echo "   sh run_ft_milora_gen.sh llama 1 8   # Rank 8"
    echo "   sh run_ft_milora_gen.sh llama 1 32  # Rank 32"
    echo ""
    echo "   # Different top-k values:"
    echo "   sh run_ft_milora_gen.sh llama 1 16 2  # Top-2 experts"
    echo "   sh run_ft_milora_gen.sh llama 1 16 5  # Top-5 experts"
    echo ""
    echo "   # Different pooling methods:"
    echo "   sh run_ft_milora_gen.sh llama 1 16 3 last_token"
    echo "   sh run_ft_milora_gen.sh llama 1 16 3 average"
    echo "   sh run_ft_milora_gen.sh llama 1 16 3 max"
    echo ""
    echo "🔍 Next Steps:"
    echo "   1. Evaluate MiLoRA model performance"
    echo "   2. Compare with traditional MoE approaches"
    echo "   3. Analyze expert specialization patterns"
    echo "   4. Test prompt-aware routing efficiency"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ MiLoRA Training Failed!"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. Base model path is correct and accessible"
    echo "  2. Training data file exists and is properly formatted"
    echo "  3. CUDA is available and working"
    echo "  4. Sufficient GPU memory for the model"
    echo "  5. All required Python packages are installed"
    echo ""
    echo "For debugging, you can run the Python script directly:"
    echo "  python ft_milora_gen.py --help"
    echo ""
    echo "Check training logs:"
    echo "  cat $OUTPUT_DIR/training.log"
    echo "============================================================"
    exit 1
fi