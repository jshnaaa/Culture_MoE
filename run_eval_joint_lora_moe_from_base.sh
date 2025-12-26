#!/bin/bash

# ============================================================
# 🧪 SIMPLIFIED CULTUREMOE MODEL EVALUATION
# 专门用于Simplified CultureMoE模型的评估
# ============================================================
#
# 功能：
#   1. 从基础模型和Simplified CultureMoE权重还原完整模型
#   2. 在指定测试集上进行评估，输出详细的评估结果
#   3. 保存生成答案、评估指标、模型配置等信息
#   4. 使用训练时的数据划分确保测试集一致性
#
# 使用方法：
#   bash run_eval_joint_lora_moe_from_base.sh <SIMPLIFIED_MODEL_PATH> <BACKBONE> <DATA_ID>
#
# 参数说明：
#   SIMPLIFIED_MODEL_PATH: Simplified CultureMoE模型路径 (必需)
#   BACKBONE: llama 或 qwen (必需)
#   DATA_ID: 测试数据集ID (必需)
#     - 2: CulturalBench (NUM_CLASSES=2)
#     - 3: normad (NUM_CLASSES=3)
#     - 4: cultureLLM (NUM_CLASSES=10)
#     - 6: moral_stories (NUM_CLASSES=2)
#     - 7: cultureAtlas (NUM_CLASSES=3)
#     - 8: culemo (NUM_CLASSES=6)
#
# 示例：
#   bash run_eval_joint_lora_moe_from_base.sh \
#     /autodl-fs/data/simplified_culturemoe/llama_blend_20251226_121046/best_simplified_culturemoe \
#     llama 2
# ============================================================

# ✅ 参数检查
if [ $# -ne 3 ]; then
    echo "❌ 错误: 需要提供三个参数"
    echo "用法: bash run_eval_joint_lora_moe_from_base.sh <SIMPLIFIED_MODEL_PATH> <BACKBONE> <DATA_ID>"
    echo ""
    echo "参数说明:"
    echo "  SIMPLIFIED_MODEL_PATH: Simplified CultureMoE模型路径"
    echo "  BACKBONE: llama 或 qwen"
    echo "  DATA_ID:"
    echo "    2 - CulturalBench (NUM_CLASSES=2)"
    echo "    3 - normad (NUM_CLASSES=3)"
    echo "    4 - cultureLLM (NUM_CLASSES=10)"
    echo "    6 - moral_stories (NUM_CLASSES=2)"
    echo "    7 - cultureAtlas (NUM_CLASSES=3)"
    echo "    8 - culemo (NUM_CLASSES=6)"
    echo ""
    echo "示例:"
    echo "  bash run_eval_joint_lora_moe_from_base.sh \\"
    echo "    /autodl-fs/data/simplified_culturemoe/llama_blend_20251226_121046/best_simplified_culturemoe \\"
    echo "    llama 2"
    exit 1
fi

SIMPLIFIED_MODEL_PATH="$1"
BACKBONE="$2"
DATA_ID="$3"


# ✅ 根据 backbone 设置基础模型路径
if [ "$BACKBONE" = "llama" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
elif [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    echo "❌ 错误: 不支持的backbone: $BACKBONE (支持: llama, qwen)"
    exit 1
fi

# ✅ 根据 DATA_ID 设置测试数据集
case $DATA_ID in
    2)
        TEST_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        NUM_CLASSES=2
        ;;
    3)
        TEST_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        NUM_CLASSES=3
        ;;
    4)
        TEST_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        NUM_CLASSES=10
        ;;
    6)
        TEST_FILE="/autodl-fs/data/moral_stories_merge_gen.json"
        DATASET_NAME="moral_Gen"
        NUM_CLASSES=2
        ;;
    7)
        TEST_FILE="/autodl-fs/data/cultureAtlas_merge_gen.json"
        DATASET_NAME="cultureAtlas"
        NUM_CLASSES=3
        ;;
    8)
        TEST_FILE="/autodl-fs/data/culemo_merge_gen.json"
        DATASET_NAME="culemo"
        NUM_CLASSES=6
        ;;
esac

# ✅ 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/simplified_culturemoe_eval_${DATASET_NAME}_${BACKBONE}_${TIMESTAMP}"

echo "============================================================"
echo "🧪 Simplified CultureMoE Model Evaluation"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Test Dataset: $DATASET_NAME (ID: $DATA_ID)"
echo "Number of Classes: $NUM_CLASSES"
echo ""
echo "📂 Model Paths:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  Simplified model: $SIMPLIFIED_MODEL_PATH"
echo ""
echo "📁 Input/Output:"
echo "  Test file: $TEST_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# ✅ 检查所有必需的路径
echo "🔍 Checking component paths..."

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi
echo "✅ Base model found"

if [ ! -d "$SIMPLIFIED_MODEL_PATH" ]; then
    echo "❌ Error: Simplified model not found: $SIMPLIFIED_MODEL_PATH"
    exit 1
fi

# 🔍 检查Simplified CultureMoE模型文件
echo "🔍 Verifying Simplified CultureMoE model structure..."

if [ ! -f "$SIMPLIFIED_MODEL_PATH/moe_weights.pt" ]; then
    echo "❌ Error: MoE weights file not found: $SIMPLIFIED_MODEL_PATH/moe_weights.pt"
    exit 1
fi

if [ ! -f "$SIMPLIFIED_MODEL_PATH/simplified_culturemoe_config.json" ]; then
    echo "❌ Error: Simplified config file not found: $SIMPLIFIED_MODEL_PATH/simplified_culturemoe_config.json"
    exit 1
fi

echo "✅ Simplified CultureMoE model components found"
echo "  ├── moe_weights.pt"
echo "  ├── simplified_culturemoe_config.json"
echo "  └── tokenizer files"

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi
echo "✅ Test file found"

# 🔍 查找数据划分文件以确保使用训练时的测试集
echo "🔍 Looking for data split file..."
SPLIT_FILE=""

# 推断训练输出目录（模型路径的父目录）
TRAINING_OUTPUT_DIR="$(dirname "$SIMPLIFIED_MODEL_PATH")"

# 查找数据划分文件的优先级顺序
if [ -f "$TRAINING_OUTPUT_DIR/data_split_8_1_1.pkl" ]; then
    SPLIT_FILE="$TRAINING_OUTPUT_DIR/data_split_8_1_1.pkl"
    echo "✅ Found data split file: $SPLIT_FILE"
elif [ -f "$SIMPLIFIED_MODEL_PATH/data_split_8_1_1.pkl" ]; then
    SPLIT_FILE="$SIMPLIFIED_MODEL_PATH/data_split_8_1_1.pkl"
    echo "✅ Found data split file in model dir: $SPLIT_FILE"
else
    echo "⚠️ Data split file not found. Using default validation split."
    echo "  Searched locations:"
    echo "    - $TRAINING_OUTPUT_DIR/data_split_8_1_1.pkl"
    echo "    - $SIMPLIFIED_MODEL_PATH/data_split_8_1_1.pkl"
fi

echo ""

# ✅ 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ✅ GPU配置
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo "0")
echo "🖥️  GPU Configuration:"
echo "  Detected GPUs: $NUM_GPUS"
echo "  Using: Single-GPU evaluation (for stability)"
echo ""

# ✅ 运行Simplified CultureMoE模型评估
echo "🚀 Starting Simplified CultureMoE model evaluation..."
echo ""

# 构建评估命令参数
EVAL_ARGS="--model_path \"$SIMPLIFIED_MODEL_PATH\" \
    --base_model_path \"$BASE_MODEL_PATH\" \
    --data_file \"$TEST_FILE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --device cuda \
    --use_fixed_split"

# 如果找到了数据划分文件，添加到参数中
if [ -n "$SPLIT_FILE" ]; then
    EVAL_ARGS="$EVAL_ARGS --split_file \"$SPLIT_FILE\""
    echo "📊 Using training-time data split: $SPLIT_FILE"
else
    echo "📊 Using default validation split"
fi

echo ""

# 执行评估
eval "python eval_simplified_culturemoe.py $EVAL_ARGS"

# ✅ 检查评估结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Simplified CultureMoE Model Evaluation Completed Successfully!"
    echo "============================================================"
    echo ""
    echo "📊 Model Configuration:"
    echo "  Model Type: Simplified CultureMoE"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Test Dataset: $DATASET_NAME"
    echo "  Number of Classes: $NUM_CLASSES"
    if [ -n "$SPLIT_FILE" ]; then
        echo "  Data Split: Training-time split (consistent test set)"
    else
        echo "  Data Split: Default validation split"
    fi
    echo ""
    echo "📁 Component Sources:"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  Simplified model: $SIMPLIFIED_MODEL_PATH"
    if [ -n "$SPLIT_FILE" ]; then
        echo "  Data split file: $SPLIT_FILE"
    fi
    echo ""
    echo "📋 Results saved to: $OUTPUT_DIR"
    echo "  ├── evaluation_results.json (详细评估结果)"
    echo "  ├── generated_answers.json (所有问题的完整回答信息)"
    echo "  ├── config.json (模型配置信息)"
    echo "  └── evaluation_summary.json (评估摘要)"
    echo ""
    echo "💡 Quick commands to view results:"
    echo ""
    echo "  # View evaluation summary:"
    echo "  cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "  # View overall accuracy:"
    echo "  python -c \"import json; data=json.load(open('$OUTPUT_DIR/evaluation_summary.json')); print(f'Overall Accuracy: {data[\\\"accuracy\\\"]:.4f}')\""
    echo ""
    echo "  # View detailed metrics:"
    echo "  cat $OUTPUT_DIR/evaluation_results.json | python -m json.tool"
    echo ""
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Simplified CultureMoE Model Evaluation Failed!"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. All component paths are correct"
    echo "  2. CUDA is available and working"
    echo "  3. Required Python packages are installed"
    echo "  4. Sufficient GPU memory is available"
    echo "  5. Model files are valid and complete"
    echo ""
    echo "For debugging, you can run the Python script directly:"
    echo "  python eval_simplified_culturemoe.py --help"
    echo ""
    echo "Model structure should be:"
    echo "  $SIMPLIFIED_MODEL_PATH/"
    echo "  ├── moe_weights.pt"
    echo "  ├── simplified_culturemoe_config.json"
    echo "  └── tokenizer files..."
    echo "============================================================"
    exit 1
fi