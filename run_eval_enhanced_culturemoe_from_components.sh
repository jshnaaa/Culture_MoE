#!/bin/bash

# ============================================================
# 🧪 ENHANCED CULTUREMOE EVALUATION FROM COMPONENTS
# 从 Base 模型 + LoRA 权重 + Enhanced MoE 权重还原增强版 CultureMoE 模型并评估
# ============================================================
#
# 功能：
#   1. 从基础模型、LoRA权重、增强MoE权重三部分还原完整模型
#   2. 在测试集上进行评估，输出详细的评估结果
#   3. 保存生成答案、评估指标、模型配置等信息
#
# 使用方法：
#   sh run_eval_enhanced_culturemoe_from_components.sh <BACKBONE> <NUM_EXPERTS> <MOE_FUSION> <LAMBDA>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   NUM_EXPERTS: 专家数量 (默认 12)
#   MOE_FUSION: MoE融合系数 (默认 0.4)
#   LAMBDA: 文化损失权重 (默认 0.5)
#
# 注意：此脚本固定使用CultureLLM数据集训练的模型和特定时间戳
#
# 示例：
#   # 评估默认配置的LLaMA模型
#   sh run_eval_enhanced_culturemoe_from_components.sh llama 12 0.4 0.5
#
#   # 评估Qwen模型
#   sh run_eval_enhanced_culturemoe_from_components.sh qwen 12 0.4 0.5
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-12}"
NUM_EXPERTS="${3:-12}"              # 默认 12 个专家
MOE_FUSION="${4:-0.4}"              # 默认 MoE 融合系数 0.4
LAMBDA="${5:-0.5}"                  # 默认文化损失权重 0.5

# 设置数据集信息（固定为CultureLLM）
DATASET_TAG="cultureLLM"
DATA_ID="4"

# 确定共享专家标签（需要在使用前定义）
USE_SHARED="True"  # Enhanced CultureMoE 默认使用共享专家
if [ "$USE_SHARED" = "True" ]; then
    SHARED_TAG="shared"
else
    SHARED_TAG="noshared"
fi

# 根据 backbone 选择 base 模型路径和 LoRA 权重路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-fs/data/ft/ft_lora_only_gen_cultureLLM_qwen_20251114_1301/best_lora"
    MOE_WEIGHTS_BASE="/root/autodl-fs/data/ft/ft_enhanced_moe_gen_cultureLLM_qwen_experts${NUM_EXPERTS}_${SHARED_TAG}_fusion${MOE_FUSION}_lambda${LAMBDA}_20251116_1006"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_20251112_1551/best_lora"
    MOE_WEIGHTS_BASE="/root/autodl-fs/data/ft/ft_enhanced_moe_gen_cultureLLM_llama_experts${NUM_EXPERTS}_${SHARED_TAG}_fusion${MOE_FUSION}_lambda${LAMBDA}_20251116_0957"
fi

if [ -z "$MOE_WEIGHTS_BASE" ] || [ ! -d "$MOE_WEIGHTS_BASE" ]; then
    echo "❌ Error: Enhanced MoE training directory not found"
    echo "Expected directory: $MOE_WEIGHTS_BASE"
    echo ""
    echo "Please first run enhanced CultureMoE training:"
    echo "  sh run_ft_enhanced_culturemoe_gen.sh $BACKBONE $DATA_ID True $NUM_EXPERTS $MOE_FUSION $LAMBDA"
    exit 1
fi

MOE_WEIGHTS_PATH="$MOE_WEIGHTS_BASE/best_enhanced_moe"

# 测试数据集
#TEST_FILE="/root/autodl-fs/wvs_merge_gen.json"
# 根据 DATA_ID 选择数据集
case $DATA_ID in
    11)
        TEST_FILE="/root/autodl-fs/wvs_merge_gen.json"
        DATASET_NAME="WVS_Gen"
        ;;
    12)
        TEST_FILE="/root/autodl-fs/wvs_merge_gen_id.json"
        DATASET_NAME="WVS_Gen_ID"
        ;;
    13)
        TEST_FILE="/root/autodl-fs/wvs_merge_gen_ood.json"
        DATASET_NAME="WVS_Gen_OOD"
        ;;
    21)
        TEST_FILE="/autodl-fs/data/moral_stories_merge_gen.json"
        DATASET_NAME="moral_Gen"
        ;;
esac
# 输出目录
OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_enhanced_culturemoe_${BACKBONE}_cultureLLM_${DATASET_NAME}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "🧪 Enhanced CultureMoE Evaluation (From Components)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Training Dataset: $DATASET_TAG (ID: $DATA_ID)"
echo "Test Dataset: WVS (wvs_merge_gen.json)"
echo "Num experts: $NUM_EXPERTS"
echo "MoE fusion: $MOE_FUSION"
echo "Culture loss weight: $LAMBDA"
echo ""
echo "📂 Component Paths:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo "  Enhanced MoE weights: $MOE_WEIGHTS_PATH"
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

if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    exit 1
fi
echo "✅ LoRA weights found"

if [ ! -d "$MOE_WEIGHTS_PATH" ]; then
    echo "❌ Error: Enhanced MoE weights not found: $MOE_WEIGHTS_PATH"
    echo ""
    echo "Expected structure:"
    echo "  $MOE_WEIGHTS_PATH/"
    echo "  ├── moe_weights.pth"
    echo "  └── (other files)"
    exit 1
fi

if [ ! -f "$MOE_WEIGHTS_PATH/moe_weights.pth" ]; then
    echo "❌ Error: MoE weights file not found: $MOE_WEIGHTS_PATH/moe_weights.pth"
    exit 1
fi
echo "✅ Enhanced MoE weights found"

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi
echo "✅ Test file found"

echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 强制单GPU评估（避免DataParallel兼容性问题）
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo "0")
echo "🖥️  GPU Configuration:"
echo "  Detected GPUs: $NUM_GPUS"
echo "  Using: Single-GPU evaluation (forced for stability)"
echo "  Note: Multi-GPU disabled to avoid DataParallel issues with Enhanced CultureMoE"
USE_MULTI_GPU=""
echo ""

# 运行评估
echo "🚀 Starting Enhanced CultureMoE evaluation..."
echo ""

python eval_enhanced_culturemoe_from_components.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --lora_weights_path "$LORA_WEIGHTS_PATH" \
    --moe_weights_path "$MOE_WEIGHTS_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --num_experts "$NUM_EXPERTS" \
    --moe_fusion "$MOE_FUSION" \
    --culture_loss_lambda "$LAMBDA" \
    --device cuda \
    $USE_MULTI_GPU

# 检查评估结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Enhanced CultureMoE Evaluation Completed Successfully!"
    echo "============================================================"
    echo ""
    echo "📊 Model Configuration:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Training Dataset: $DATASET_TAG"
    echo "  Num experts: $NUM_EXPERTS"
    echo "  MoE fusion: $MOE_FUSION"
    echo "  Culture loss weight: $LAMBDA"
    echo ""
    echo "📁 Component Sources:"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo "  Enhanced MoE weights: $MOE_WEIGHTS_PATH"
    echo ""
    echo "📋 Results saved to: $OUTPUT_DIR"
    echo "  ├── evaluation_results.json (详细评估结果，包含按大洲统计)"
    echo "  ├── generated_answers.json (所有问题的完整回答信息)"
    echo "  ├── config.json (模型配置信息)"
    echo "  └── evaluation_summary.json (评估摘要)"
    echo ""
    echo "📊 generated_answers.json 包含内容："
    echo "  - 所有测试问题的完整信息"
    echo "  - 正确答案 vs 模型预测答案"
    echo "  - 回答是否正确的判断"
    echo "  - 文化背景信息（大洲、语言等）"
    echo "  - 专家权重分布（如果可用）"
    echo "  - 专家激活分析"
    echo ""
    echo "🔧 评估配置："
    echo "  - 单GPU模式：强制启用以确保稳定性"
    echo "  - DataParallel：已禁用（避免generate方法兼容性问题）"
    echo "  - 专家权重：尝试提取但不影响主要评估"
    echo ""
    echo "💡 Quick commands to view results:"
    echo ""
    echo "  # View evaluation summary:"
    echo "  cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "  # View model configuration:"
    echo "  cat $OUTPUT_DIR/config.json | python -m json.tool"
    echo ""
    echo "  # View all generated answers with details:"
    echo "  cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -50"
    echo ""
    echo "  # View generated answers (first 5 with key info):"
    echo "  cat $OUTPUT_DIR/generated_answers.json | python -c \"import json,sys; data=json.load(sys.stdin); [print(f'Q{i+1}: {item[\\\"question\\\"][:80]}...\\nTrue: {item[\\\"true_answer\\\"]} | Predicted: {item[\\\"predicted_answer\\\"]} | Correct: {item[\\\"is_correct\\\"]}\\nContinent: {item[\\\"culture_context\\\"][\\\"continent_name\\\"]}\\n---') for i, item in enumerate(data[:5])]\""
    echo ""
    echo "  # Check accuracy by continent:"
    echo "  cat $OUTPUT_DIR/evaluation_results.json | python -c \"import json,sys; data=json.load(sys.stdin); [print(f'{k}: {v[\\\"accuracy\\\"]:.4f} ({v[\\\"correct\\\"]}/{v[\\\"total\\\"]})') for k,v in data[\\\"continent_statistics\\\"].items()]\""
    echo ""
    echo "  # Check overall accuracy:"
    echo "  python -c \"import json; data=json.load(open('$OUTPUT_DIR/evaluation_summary.json')); print(f'Overall Accuracy: {data[\\\"accuracy\\\"]:.4f}')\""
    echo ""
    echo "  # View expert weight analysis (if available):"
    echo "  cat $OUTPUT_DIR/generated_answers.json | python -c \"import json,sys; data=json.load(sys.stdin); correct=[item for item in data if item.get('expert_analysis') and item['is_correct']]; wrong=[item for item in data if item.get('expert_analysis') and not item['is_correct']]; print(f'Correct answers: avg dominant expert weight = {sum(item[\\\"expert_analysis\\\"][\\\"max_weight\\\"] for item in correct)/len(correct):.3f}'); print(f'Wrong answers: avg dominant expert weight = {sum(item[\\\"expert_analysis\\\"][\\\"max_weight\\\"] for item in wrong)/len(wrong):.3f}') if len(correct)>0 and len(wrong)>0 else None\""
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Enhanced CultureMoE Evaluation Failed!"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. All component paths are correct"
    echo "  2. CUDA is available and working"
    echo "  3. Required Python packages are installed"
    echo "  4. Sufficient GPU memory is available"
    echo ""
    echo "For debugging, you can run the Python script directly:"
    echo "  python eval_enhanced_culturemoe_from_components.py --help"
    echo "============================================================"
    exit 1
fi