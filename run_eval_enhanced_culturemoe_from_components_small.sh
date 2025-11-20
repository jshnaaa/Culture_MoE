#!/bin/bash

# ============================================================
# 🧪 ENHANCED CULTUREMOE EVALUATION FROM COMPONENTS (权重压缩版本)
# 从 Base 模型 + LoRA 权重 + Enhanced MoE 权重还原增强版 CultureMoE 模型并评估
# 支持压缩权重格式的智能加载
# ============================================================
#
# 功能：
#   1. 从基础模型、LoRA权重、增强MoE权重三部分还原完整模型
#   2. 智能加载压缩或未压缩的MoE权重
#   3. 在测试集上进行评估，输出详细的评估结果
#   4. 保存生成答案、评估指标、模型配置等信息
#
# 使用方法：
#   sh run_eval_enhanced_culturemoe_from_components_small.sh <BACKBONE> <DATA_ID> <NUM_EXPERTS> <MOE_FUSION> <LAMBDA>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 数据集ID (默认 1，对应unified_all_datasets)
#   NUM_EXPERTS: 专家数量 (默认 12)
#   MOE_FUSION: MoE融合系数 (默认 0.4)
#   LAMBDA: 文化损失权重 (默认 0.5)
#
# 权重压缩支持：
#   - 自动检测压缩格式 (FP16, Gzip, FP16+Gzip)
#   - 智能加载最佳可用权重文件
#   - 兼容原版本和压缩版本权重
#
# 示例：
#   # 评估默认配置的LLaMA模型（压缩权重版本）
#   sh run_eval_enhanced_culturemoe_from_components_small.sh llama 1 12 0.4 0.5
#
#   # 评估Qwen模型（压缩权重版本）
#   sh run_eval_enhanced_culturemoe_from_components_small.sh qwen 1 12 0.4 0.5
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-1}"                   # 默认 unified_all_datasets (1)
NUM_EXPERTS="${3:-12}"              # 默认 12 个专家
MOE_FUSION="${4:-0.4}"              # 默认 MoE 融合系数 0.4
LAMBDA="${5:-0.5}"                  # 默认文化损失权重 0.5

# 确定共享专家标签（需要在使用前定义）
USE_SHARED="True"  # Enhanced CultureMoE 默认使用共享专家
USE_MASK="True"    # 默认使用 MASK 机制

if [ "$USE_SHARED" = "True" ]; then
    SHARED_TAG="shared"
else
    SHARED_TAG="noshared"
fi

if [ "$USE_MASK" = "True" ]; then
    MASK_TAG="mask"
else
    MASK_TAG="nomask"
fi

# 根据 backbone 选择 base 模型路径和 LoRA 权重路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_qwen_20251111_1421/best_lora"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/autodl-fs/data/data/ft/ft_lora_only_gen_unified_all_datasets_llama_20251117_1218/best_lora"
fi

# 根据 DATA_ID 选择数据集和对应的模型权重路径
case $DATA_ID in
    0)
        TEST_FILE="/root/autodl-fs/unified_all_datasets_small.json"
        DATASET_NAME="unified_all_datasets_small"
        DATASET_TAG="unified_all_datasets_small"
        NUM_CLASSES=6
        ;;
    1)
        TEST_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_NAME="unified_all_datasets"
        DATASET_TAG="unified_all_datasets"
        NUM_CLASSES=6
        ;;
    2)
        TEST_FILE="/autodl-fs/data/culturalBench_merge_gen_small.json"
        DATASET_NAME="culturalBench_small"
        DATASET_TAG="CulturalBench"
        NUM_CLASSES=2
        ;;
    3)
        TEST_FILE="/autodl-fs/data/normad_merge_gen_small.json"
        DATASET_NAME="normad_small"
        DATASET_TAG="normad"
        NUM_CLASSES=3
        ;;
    4)
        TEST_FILE="/autodl-fs/data/cultureLLM_merge_gen_small.json"
        DATASET_NAME="cultureLLM_small"
        DATASET_TAG="cultureLLM"
        NUM_CLASSES=10
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 0, 1, 2, 3, or 4."
        echo ""
        echo "DATA_ID options:"
        echo "  0 - unified_all_datasets_small"
        echo "  1 - unified_all_datasets (default)"
        echo "  2 - CulturalBench"
        echo "  3 - NormAD"
        echo "  4 - CultureLLM"
        exit 1
        ;;
esac

# 构建MoE权重路径（支持压缩版本）
MOE_WEIGHTS_BASE="/root/autodl-fs/data/ft/ft_enhanced_moe_gen_small_${DATASET_TAG}_${BACKBONE}_experts${NUM_EXPERTS}_${SHARED_TAG}_${MASK_TAG}_fusion${MOE_FUSION}_lambda${LAMBDA}_*"

# 查找最新的训练目录
MOE_WEIGHTS_DIRS=($(ls -d $MOE_WEIGHTS_BASE 2>/dev/null | sort -r))

if [ ${#MOE_WEIGHTS_DIRS[@]} -eq 0 ]; then
    echo "❌ Error: Enhanced MoE training directory not found"
    echo "Expected pattern: $MOE_WEIGHTS_BASE"
    echo ""
    echo "Please first run enhanced CultureMoE training (compressed version):"
    echo "  sh run_ft_enhanced_culturemoe_gen_small.sh $BACKBONE $DATA_ID True $NUM_EXPERTS $MOE_FUSION $LAMBDA"
    exit 1
fi

MOE_WEIGHTS_BASE="${MOE_WEIGHTS_DIRS[0]}"
MOE_WEIGHTS_PATH="$MOE_WEIGHTS_BASE/best_enhanced_moe"

# 输出目录
OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_enhanced_culturemoe_small_${DATASET_NAME}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "🧪 Enhanced CultureMoE Evaluation (权重压缩版本) - From Components"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME (ID: $DATA_ID)"
echo "Num experts: $NUM_EXPERTS"
echo "MoE fusion: $MOE_FUSION"
echo "Culture loss weight: $LAMBDA"
echo ""
echo "🔧 Applied Fixes:"
echo "  ✅ Increased max_length to 2048 (prevent instruction truncation)"
echo "  ✅ Uses dataset's native ### Answer: format"
echo "  ✅ Truncation warning system for debugging"
echo "  ✅ Maintains original prompt structure"
echo "  ✅ Intelligent compressed weight loading"
echo ""
echo "💾 Weight Compression Features:"
echo "  - Auto-detection of compression format"
echo "  - Support for FP16, Gzip, and FP16+Gzip formats"
echo "  - Fallback to uncompressed weights if needed"
echo "  - Memory-efficient loading"
echo ""
echo "📂 Component Paths:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo "  Enhanced MoE weights: $MOE_WEIGHTS_PATH"
echo "  Training directory: $MOE_WEIGHTS_BASE"
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
    echo "❌ Error: Enhanced MoE weights directory not found: $MOE_WEIGHTS_PATH"
    echo ""
    echo "Expected structure:"
    echo "  $MOE_WEIGHTS_PATH/"
    echo "  ├── moe_weights.pth (or compressed variants)"
    echo "  └── (other files)"
    exit 1
fi

# 检查权重文件（支持压缩格式）
MOE_WEIGHT_FILES=(
    "$MOE_WEIGHTS_PATH/moe_weights_fp16.pth.gz"
    "$MOE_WEIGHTS_PATH/moe_weights_fp16.pth"
    "$MOE_WEIGHTS_PATH/moe_weights.pth.gz"
    "$MOE_WEIGHTS_PATH/moe_weights.pth"
)

FOUND_WEIGHT_FILE=""
for weight_file in "${MOE_WEIGHT_FILES[@]}"; do
    if [ -f "$weight_file" ]; then
        FOUND_WEIGHT_FILE="$weight_file"
        break
    fi
done

if [ -z "$FOUND_WEIGHT_FILE" ]; then
    echo "❌ Error: No MoE weights file found in: $MOE_WEIGHTS_PATH"
    echo "Searched for:"
    for weight_file in "${MOE_WEIGHT_FILES[@]}"; do
        echo "  - $(basename $weight_file)"
    done
    exit 1
fi

echo "✅ Enhanced MoE weights found: $(basename $FOUND_WEIGHT_FILE)"

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

# 运行评估（使用压缩版本的脚本）
echo "🚀 Starting Enhanced CultureMoE evaluation (compressed weights)..."
echo ""

python eval_enhanced_culturemoe_from_components_small.py \
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
    echo "✅ Enhanced CultureMoE Evaluation (权重压缩版本) Completed Successfully!"
    echo "============================================================"
    echo ""
    echo "📊 Model Configuration:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Dataset: $DATASET_NAME (ID: $DATA_ID)"
    echo "  Num experts: $NUM_EXPERTS"
    echo "  MoE fusion: $MOE_FUSION"
    echo "  Culture loss weight: $LAMBDA"
    echo ""
    echo "📁 Component Sources:"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo "  Enhanced MoE weights: $FOUND_WEIGHT_FILE"
    echo ""
    echo "💾 Weight Compression Info:"
    echo "  Used weight file: $(basename $FOUND_WEIGHT_FILE)"
    echo "  Compression format: $(if [[ $FOUND_WEIGHT_FILE == *"fp16"* ]]; then echo "FP16"; fi)$(if [[ $FOUND_WEIGHT_FILE == *".gz" ]]; then echo "+Gzip"; fi)"
    echo "  Loading method: Intelligent auto-detection"
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
    echo "  - 权重格式：智能检测和加载压缩格式"
    echo "  - 多答案支持：支持逗号分隔的多个正确答案（如 '1,2'）"
    echo "  - 简洁回答模式：max_new_tokens=5，贪婪解码，只提取数字答案"
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
    echo "  # Check overall accuracy:"
    echo "  python -c \"import json; data=json.load(open('$OUTPUT_DIR/evaluation_summary.json')); print(f'Overall Accuracy: {data[\\\"accuracy\\\"]:.4f}')\""
    echo ""
    echo "  # Check compression savings (if available):"
    echo "  python model_utils_small.py --weight_path '$FOUND_WEIGHT_FILE' --action info"
    echo ""
    echo "🔬 Usage Examples (权重压缩版本):"
    echo "   # Evaluate different datasets:"
    echo "   sh run_eval_enhanced_culturemoe_from_components_small.sh llama 1  # unified_all_datasets"
    echo "   sh run_eval_enhanced_culturemoe_from_components_small.sh llama 2  # CulturalBench"
    echo "   sh run_eval_enhanced_culturemoe_from_components_small.sh llama 3  # NormAD"
    echo "   sh run_eval_enhanced_culturemoe_from_components_small.sh llama 4  # CultureLLM"
    echo ""
    echo "   # Evaluate with Qwen backbone:"
    echo "   sh run_eval_enhanced_culturemoe_from_components_small.sh qwen 1"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Enhanced CultureMoE Evaluation (权重压缩版本) Failed!"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. All component paths are correct"
    echo "  2. CUDA is available and working"
    echo "  3. Required Python packages are installed"
    echo "  4. Sufficient GPU memory is available"
    echo "  5. Compressed weight files are accessible"
    echo ""
    echo "For debugging, you can run the Python script directly:"
    echo "  python eval_enhanced_culturemoe_from_components_small.py --help"
    echo ""
    echo "To check weight file status:"
    echo "  python model_utils_small.py --weight_path '$MOE_WEIGHTS_PATH/moe_weights.pth' --action find"
    echo "============================================================"
    exit 1
fi