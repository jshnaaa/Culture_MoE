#!/bin/bash

# ============================================================
# 🧪 CULTUREMOE MODEL EVALUATION FROM BASE MODEL
# 支持Joint LoRA+MoE和Simplified CultureMoE两种模型类型的评估
# ============================================================
#
# 功能：
#   1. 自动检测模型类型（Joint vs Simplified）
#   2. 从基础模型和训练权重还原完整模型
#   3. 在指定测试集上进行评估，输出详细的评估结果
#   4. 保存生成答案、评估指标、模型配置等信息
#
# 使用方法：
#   bash run_eval_joint_lora_moe_from_base.sh <JOINT_MODEL_PATH> <BACKBONE> <DATA_ID>
#
# 参数说明：
#   MODEL_PATH: 训练模型路径 (支持Joint或Simplified模型) (必需)
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
#   # Joint模型评估
#   bash run_eval_joint_lora_moe_from_base.sh /path/to/joint_model llama 6
#   # Simplified模型评估
#   bash run_eval_joint_lora_moe_from_base.sh /path/to/simplified_model llama 2
# ============================================================

# ✅ 参数检查
if [ $# -ne 3 ]; then
    echo "❌ 错误: 需要提供三个参数"
    echo "用法: bash run_eval_joint_lora_moe_from_base.sh <JOINT_MODEL_PATH> <BACKBONE> <DATA_ID>"
    echo ""
    echo "参数说明:"
    echo "  JOINT_MODEL_PATH: 联合训练模型路径"
    echo "  BACKBONE: llama 或 qwen"
    echo "  DATA_ID:"
    echo "    6 - moral_stories (NUM_CLASSES=2)"
    echo "    7 - cultureAtlas (NUM_CLASSES=3)"
    echo "    8 - culemo (NUM_CLASSES=6)"
    echo ""
    echo "示例:"
    echo "  bash run_eval_joint_lora_moe_from_base.sh /path/to/joint_model llama 6"
    echo "  bash run_eval_joint_lora_moe_from_base.sh /path/to/joint_model qwen 7"
    exit 1
fi

MODEL_PATH="$1"
BACKBONE="$2"
DATA_ID="$3"

# 保持向后兼容性的变量别名
JOINT_MODEL_PATH="$MODEL_PATH"


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
OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/joint_lora_moe_eval_${DATASET_NAME}_${BACKBONE}_${TIMESTAMP}"

echo "============================================================"
echo "🧪 CultureMoE Model Evaluation"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Test Dataset: $DATASET_NAME (ID: $DATA_ID)"
echo "Number of Classes: $NUM_CLASSES"
echo ""
echo "📂 Model Paths:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  Joint trained model: $JOINT_MODEL_PATH"
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

if [ ! -d "$JOINT_MODEL_PATH" ]; then
    echo "❌ Error: Model not found: $JOINT_MODEL_PATH"
    exit 1
fi

# 🔍 检测模型类型：joint vs simplified
MODEL_TYPE=""
if [ -f "$JOINT_MODEL_PATH/joint_config.json" ] && [ -d "$JOINT_MODEL_PATH/lora_weights" ]; then
    MODEL_TYPE="joint"
    echo "🔍 Detected: Joint LoRA+MoE model"
elif [ -f "$JOINT_MODEL_PATH/simplified_culturemoe_config.json" ] && [ -f "$JOINT_MODEL_PATH/moe_weights.pt" ]; then
    MODEL_TYPE="simplified"
    echo "🔍 Detected: Simplified CultureMoE model"
else
    echo "❌ Error: Unknown model type. Expected either:"
    echo ""
    echo "Joint model structure:"
    echo "  $JOINT_MODEL_PATH/"
    echo "  ├── lora_weights/"
    echo "  │   ├── adapter_config.json"
    echo "  │   └── adapter_model.safetensors"
    echo "  ├── moe_weights.pt"
    echo "  ├── joint_config.json"
    echo "  └── tokenizer files..."
    echo ""
    echo "Simplified model structure:"
    echo "  $JOINT_MODEL_PATH/"
    echo "  ├── moe_weights.pt"
    echo "  ├── simplified_culturemoe_config.json"
    echo "  └── tokenizer files..."
    exit 1
fi

# 根据模型类型检查必要文件
if [ "$MODEL_TYPE" = "joint" ]; then
    if [ ! -d "$JOINT_MODEL_PATH/lora_weights" ]; then
        echo "❌ Error: LoRA weights directory not found: $JOINT_MODEL_PATH/lora_weights"
        exit 1
    fi

    if [ ! -f "$JOINT_MODEL_PATH/moe_weights.pt" ]; then
        echo "❌ Error: MoE weights file not found: $JOINT_MODEL_PATH/moe_weights.pt"
        exit 1
    fi

    if [ ! -f "$JOINT_MODEL_PATH/joint_config.json" ]; then
        echo "❌ Error: Joint config file not found: $JOINT_MODEL_PATH/joint_config.json"
        exit 1
    fi
    echo "✅ Joint model components found"

elif [ "$MODEL_TYPE" = "simplified" ]; then
    if [ ! -f "$JOINT_MODEL_PATH/moe_weights.pt" ]; then
        echo "❌ Error: MoE weights file not found: $JOINT_MODEL_PATH/moe_weights.pt"
        exit 1
    fi

    if [ ! -f "$JOINT_MODEL_PATH/simplified_culturemoe_config.json" ]; then
        echo "❌ Error: Simplified config file not found: $JOINT_MODEL_PATH/simplified_culturemoe_config.json"
        exit 1
    fi
    echo "✅ Simplified model components found"
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi
echo "✅ Test file found"

echo ""

# ✅ 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ✅ GPU配置
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo "0")
echo "🖥️  GPU Configuration:"
echo "  Detected GPUs: $NUM_GPUS"
echo "  Using: Single-GPU evaluation (for stability)"
echo ""

# ✅ 根据模型类型运行相应的评估
if [ "$MODEL_TYPE" = "joint" ]; then
    echo "🚀 Starting Joint LoRA+MoE model evaluation..."
    echo ""

    python eval_joint_lora_moe_from_base.py \
        --base_model_path "$BASE_MODEL_PATH" \
        --joint_model_path "$JOINT_MODEL_PATH" \
        --test_file "$TEST_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --backbone "$BACKBONE" \
        --num_classes "$NUM_CLASSES" \
        --device cuda

elif [ "$MODEL_TYPE" = "simplified" ]; then
    echo "🚀 Starting Simplified CultureMoE model evaluation..."
    echo ""

    python eval_simplified_culturemoe.py \
        --model_path "$JOINT_MODEL_PATH" \
        --base_model_path "$BASE_MODEL_PATH" \
        --data_file "$TEST_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --device cuda \
        --use_fixed_split

else
    echo "❌ Error: Unknown model type: $MODEL_TYPE"
    exit 1
fi

# ✅ 检查评估结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    if [ "$MODEL_TYPE" = "joint" ]; then
        echo "✅ Joint LoRA+MoE Model Evaluation Completed Successfully!"
    else
        echo "✅ Simplified CultureMoE Model Evaluation Completed Successfully!"
    fi
    echo "============================================================"
    echo ""
    echo "📊 Model Configuration:"
    echo "  Model Type: $MODEL_TYPE"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Test Dataset: $DATASET_NAME"
    echo "  Number of Classes: $NUM_CLASSES"
    echo ""
    echo "📁 Component Sources:"
    echo "  Base model: $BASE_MODEL_PATH"
    if [ "$MODEL_TYPE" = "joint" ]; then
        echo "  Joint trained model: $JOINT_MODEL_PATH"
    else
        echo "  Simplified trained model: $JOINT_MODEL_PATH"
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
    if [ "$MODEL_TYPE" = "joint" ]; then
        echo "❌ Joint LoRA+MoE Model Evaluation Failed!"
    else
        echo "❌ Simplified CultureMoE Model Evaluation Failed!"
    fi
    echo "============================================================"
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. All component paths are correct"
    echo "  2. CUDA is available and working"
    echo "  3. Required Python packages are installed"
    echo "  4. Sufficient GPU memory is available"
    echo ""
    echo "For debugging, you can run the Python script directly:"
    if [ "$MODEL_TYPE" = "joint" ]; then
        echo "  python eval_joint_lora_moe_from_base.py --help"
    else
        echo "  python eval_simplified_culturemoe.py --help"
    fi
    echo "============================================================"
    exit 1
fi