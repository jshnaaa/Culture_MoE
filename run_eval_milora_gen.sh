#!/bin/bash

# ============================================================
# 🧪 MiLoRA (Mixture of LoRA) EVALUATION SCRIPT
# 从 Base 模型 + MiLoRA 权重还原完整模型并评估
# 测试提示感知路由机制的效果
# ============================================================
#
# 功能：
#   1. 从基础模型和 MiLoRA 权重还原完整模型
#   2. 对比提示感知路由 vs 传统路由的性能
#   3. 分析专家使用模式和路由效率
#   4. 在测试集上进行详细评估
#
# 使用方法：
#   sh run_eval_milora_gen.sh <BACKBONE> <DATA_ID> <LORA_RANK> <TOP_K> <POOLING_TYPE>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 测试数据集ID (默认 1)
#   LORA_RANK: LoRA 秩 (默认 16)
#   TOP_K: 专家数量 (默认 3)
#   POOLING_TYPE: 池化类型 (默认 self_attention)
#
# 示例：
#   # 基本评估
#   sh run_eval_milora_gen.sh llama 1
#
#   # 指定参数评估
#   sh run_eval_milora_gen.sh llama 1 16 3 self_attention
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
        TEST_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_TAG="unified_all_datasets"
        ;;
    2)
        DATASET_NAME="CulturalBench"
        TEST_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        ;;
    3)
        DATASET_NAME="NormAD"
        TEST_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        ;;
    4)
        DATASET_NAME="CultureLLM"
        TEST_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 1, 2, 3, or 4."
        exit 1
        ;;
esac

# 查找最新的 MiLoRA 训练目录
MILORA_WEIGHTS_PATTERN="/root/autodl-fs/data/ft/ft_milora_gen_${DATASET_TAG}_${BACKBONE}_rank${LORA_RANK}_topk${TOP_K}_${POOLING_TYPE}_*"
MILORA_WEIGHTS_DIRS=($(ls -d $MILORA_WEIGHTS_PATTERN 2>/dev/null | sort -r))

if [ ${#MILORA_WEIGHTS_DIRS[@]} -eq 0 ]; then
    echo "❌ Error: MiLoRA training directory not found"
    echo "Expected pattern: $MILORA_WEIGHTS_PATTERN"
    echo ""
    echo "Please first run MiLoRA training:"
    echo "  sh run_ft_milora_gen.sh $BACKBONE $DATA_ID $LORA_RANK $TOP_K $POOLING_TYPE"
    exit 1
fi

MILORA_WEIGHTS_BASE="${MILORA_WEIGHTS_DIRS[0]}"
MILORA_WEIGHTS_PATH="$MILORA_WEIGHTS_BASE/best_milora"

# 输出目录
OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/eval_milora_${DATASET_TAG}_${BACKBONE}_rank${LORA_RANK}_topk${TOP_K}_${POOLING_TYPE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "🧪 MiLoRA (Mixture of LoRA) Evaluation"
echo "============================================================"
echo "Evaluation Focus: Prompt-Aware Routing Efficiency"
echo "Innovation Test: One-time routing vs Traditional routing"
echo ""
echo "🔬 MiLoRA Architecture Being Tested:"
echo "  1. LoRA Experts: 7 experts per layer (Q, K, V, O, G, U, D)"
echo "  2. Prompt-Aware Routing: Compute routing once, reuse for generation"
echo "  3. Expert Selection: Top-k = $TOP_K out of 7 experts"
echo "  4. Pooling Method: $POOLING_TYPE"
echo ""
echo "📊 Evaluation Strategy:"
echo "  - Mode 1: With Prompt-Aware Routing (Cache routing results)"
echo "  - Mode 2: Without Prompt-Aware Routing (Recompute each time)"
echo "  - Compare: Accuracy, Efficiency, Expert Usage Patterns"
echo ""
echo "🖥️  Model Configuration:"
echo "  Backbone: $BACKBONE ($MODEL_NAME)"
echo "  Dataset: $DATASET_NAME (ID: $DATA_ID)"
echo "  LoRA Rank: $LORA_RANK"
echo "  Top-k Experts: $TOP_K"
echo "  Pooling Type: $POOLING_TYPE"
echo ""
echo "📂 Component Paths:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  MiLoRA weights: $MILORA_WEIGHTS_PATH"
echo "  Training directory: $MILORA_WEIGHTS_BASE"
echo ""
echo "📁 Input/Output:"
echo "  Test file: $TEST_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查所有必需的路径
echo "🔍 Checking component paths..."

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi
echo "✅ Base model found"

if [ ! -d "$MILORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: MiLoRA weights directory not found: $MILORA_WEIGHTS_PATH"
    echo ""
    echo "Expected structure:"
    echo "  $MILORA_WEIGHTS_PATH/"
    echo "  ├── milora_weights.pth"
    echo "  └── config.json"
    exit 1
fi

if [ ! -f "$MILORA_WEIGHTS_PATH/milora_weights.pth" ]; then
    echo "❌ Error: MiLoRA weights file not found: $MILORA_WEIGHTS_PATH/milora_weights.pth"
    exit 1
fi

if [ ! -f "$MILORA_WEIGHTS_PATH/config.json" ]; then
    echo "❌ Error: MiLoRA config file not found: $MILORA_WEIGHTS_PATH/config.json"
    exit 1
fi
echo "✅ MiLoRA weights and config found"

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi
echo "✅ Test file found"

echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 强制单GPU评估
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo "0")
echo "🖥️  GPU Configuration:"
echo "  Detected GPUs: $NUM_GPUS"
echo "  Using: Single-GPU evaluation (forced for stability)"
echo ""

# 运行 MiLoRA 评估
echo "🚀 Starting MiLoRA evaluation..."
echo ""

python eval_milora_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --milora_weights_path "$MILORA_WEIGHTS_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 512 \
    --max_new_tokens 5 \
    --device cuda \
    --seed 42

# 检查评估结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ MiLoRA Evaluation Completed Successfully!"
    echo "============================================================"
    echo ""
    echo "📊 Model Configuration:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Dataset: $DATASET_NAME (ID: $DATA_ID)"
    echo "  LoRA Rank: $LORA_RANK"
    echo "  Top-k Experts: $TOP_K"
    echo "  Pooling Type: $POOLING_TYPE"
    echo ""
    echo "📁 Component Sources:"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  MiLoRA weights: $MILORA_WEIGHTS_PATH"
    echo ""
    echo "📋 Results saved to: $OUTPUT_DIR"
    echo "  ├── evaluation_summary.json (Overall performance comparison)"
    echo "  ├── with_prompt_routing_results.json (Prompt-aware routing results)"
    echo "  ├── without_prompt_routing_results.json (Traditional routing results)"
    echo "  ├── evaluation_config.json (Evaluation configuration)"
    echo "  └── evaluation.log (Detailed logs)"
    echo ""
    echo "🔬 Evaluation Modes Compared:"
    echo "  1. With Prompt-Aware Routing: Cache routing decisions"
    echo "  2. Without Prompt-Aware Routing: Recompute routing each time"
    echo ""
    echo "💡 Quick commands to view results:"
    echo ""
    echo "  # View evaluation summary:"
    echo "  cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "  # View efficiency comparison:"
    echo "  python -c \"import json; data=json.load(open('$OUTPUT_DIR/evaluation_summary.json')); eff=data['efficiency_analysis']; print(f'Efficiency Gain: {eff[\\\"efficiency_gain_percent\\\"]:.2f}%'); print(f'Time Reduction: {eff[\\\"routing_time_reduction\\\"]:.4f}s')\""
    echo ""
    echo "  # View accuracy comparison:"
    echo "  python -c \"import json; data=json.load(open('$OUTPUT_DIR/evaluation_summary.json')); comp=data['results_comparison']; print(f'With Prompt Routing: {comp[\\\"with_prompt_routing\\\"][\\\"accuracy\\\"]:.4f}'); print(f'Without Prompt Routing: {comp[\\\"without_prompt_routing\\\"][\\\"accuracy\\\"]:.4f}')\""
    echo ""
    echo "  # View detailed results (first 5 samples):"
    echo "  cat $OUTPUT_DIR/with_prompt_routing_results.json | python -c \"import json,sys; data=json.load(sys.stdin); [print(f'Q{i+1}: {item[\\\"instruction\\\"][:80]}...\\nTrue: {item[\\\"true_answer\\\"]} | Predicted: {item[\\\"predicted_answer\\\"]} | Correct: {item[\\\"is_correct\\\"]}\\nRouting Time: {item[\\\"routing_time\\\"]:.4f}s\\n---') for i, item in enumerate(data[:5])]\""
    echo ""
    echo "🔬 Key MiLoRA Insights:"
    echo "  1. 🎯 Prompt-Aware Routing Efficiency: Check routing time reduction"
    echo "  2. 🧠 Expert Specialization: Analyze expert usage patterns"
    echo "  3. ⚖️  Load Balancing: Review expert activation distribution"
    echo "  4. 🎛️  Pooling Impact: Compare different pooling strategies"
    echo "  5. 📈 Accuracy vs Efficiency: Trade-off analysis"
    echo ""
    echo "🔍 Next Steps:"
    echo "   1. Compare MiLoRA with traditional MoE approaches"
    echo "   2. Analyze expert specialization patterns"
    echo "   3. Test different top-k values and pooling methods"
    echo "   4. Evaluate on larger datasets for statistical significance"
    echo ""
    echo "💡 Usage Examples for Different Configurations:"
    echo "   # Different LoRA ranks:"
    echo "   sh run_eval_milora_gen.sh llama 1 8   # Rank 8"
    echo "   sh run_eval_milora_gen.sh llama 1 32  # Rank 32"
    echo ""
    echo "   # Different top-k values:"
    echo "   sh run_eval_milora_gen.sh llama 1 16 2  # Top-2 experts"
    echo "   sh run_eval_milora_gen.sh llama 1 16 5  # Top-5 experts"
    echo ""
    echo "   # Different pooling methods:"
    echo "   sh run_eval_milora_gen.sh llama 1 16 3 last_token"
    echo "   sh run_eval_milora_gen.sh llama 1 16 3 average"
    echo "   sh run_eval_milora_gen.sh llama 1 16 3 max"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ MiLoRA Evaluation Failed!"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. All component paths are correct"
    echo "  2. CUDA is available and working"
    echo "  3. Required Python packages are installed"
    echo "  4. Sufficient GPU memory is available"
    echo "  5. MiLoRA weights are properly saved"
    echo ""
    echo "For debugging, you can run the Python script directly:"
    echo "  python eval_milora_gen.py --help"
    echo ""
    echo "Check evaluation logs:"
    echo "  cat $OUTPUT_DIR/evaluation.log"
    echo ""
    echo "Verify MiLoRA weights:"
    echo "  ls -la $MILORA_WEIGHTS_PATH/"
    echo "  python -c \"import torch; print('Weights keys:', list(torch.load('$MILORA_WEIGHTS_PATH/milora_weights.pth').keys())[:5])\""
    echo "============================================================"
    exit 1
fi