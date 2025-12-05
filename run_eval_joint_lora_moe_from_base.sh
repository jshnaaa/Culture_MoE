#!/bin/bash

# ============================================================
# 🧪 JOINT LORA+MOE EVALUATION FROM BASE MODEL
# 从 Base 模型 + 联合训练的 LoRA+MoE 权重还原完整模型并评估
# ============================================================
#
# 功能：
#   1. 从基础模型和联合训练的LoRA+MoE权重还原完整模型
#   2. 在指定测试集上进行评估，输出详细的评估结果
#   3. 保存生成答案、评估指标、模型配置等信息
#
# 使用方法：
#   bash run_eval_joint_lora_moe_from_base.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (必需)
#   DATA_ID: 测试数据集ID (必需)
#     - 6: moral_stories (NUM_CLASSES=2)
#     - 7: cultureAtlas (NUM_CLASSES=3)
#     - 8: culemo (NUM_CLASSES=6)
#
# 示例：
#   bash run_eval_joint_lora_moe_from_base.sh llama 6
#   bash run_eval_joint_lora_moe_from_base.sh qwen 7
# ============================================================

# ✅ 参数检查
if [ $# -ne 2 ]; then
    echo "❌ 错误: 需要提供两个参数"
    echo "用法: bash run_eval_joint_lora_moe_from_base.sh <BACKBONE> <DATA_ID>"
    echo ""
    echo "参数说明:"
    echo "  BACKBONE: llama 或 qwen"
    echo "  DATA_ID:"
    echo "    6 - moral_stories (NUM_CLASSES=2)"
    echo "    7 - cultureAtlas (NUM_CLASSES=3)"
    echo "    8 - culemo (NUM_CLASSES=6)"
    echo ""
    echo "示例:"
    echo "  bash run_eval_joint_lora_moe_from_base.sh llama 6"
    echo "  bash run_eval_joint_lora_moe_from_base.sh qwen 7"
    exit 1
fi

BACKBONE="$1"
DATA_ID="$2"

# ✅ 验证参数
if [ "$BACKBONE" != "llama" ] && [ "$BACKBONE" != "qwen" ]; then
    echo "❌ 错误: BACKBONE 必须是 'llama' 或 'qwen'"
    exit 1
fi

if [ "$DATA_ID" != "6" ] && [ "$DATA_ID" != "7" ] && [ "$DATA_ID" != "8" ]; then
    echo "❌ 错误: DATA_ID 必须是 6, 7, 或 8"
    exit 1
fi

# ✅ 根据 backbone 设置基础模型路径和联合训练模型路径（固定时间戳）
if [ "$BACKBONE" = "llama" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    JOINT_MODEL_PATH="/autodl-fs/data/joint_lora_moe/llama_cultureLLM_sharedfalse_gatefalse_20251203_174136/best_joint_model"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    JOINT_MODEL_PATH="/autodl-fs/data/joint_lora_moe/qwen_cultureLLM_sharedfalse_gatefalse_20251204_152323/best_joint_model"
fi

# ✅ 根据 DATA_ID 设置测试数据集
case $DATA_ID in
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
echo "🧪 Joint LoRA+MoE Model Evaluation"
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
    echo "❌ Error: Joint trained model not found: $JOINT_MODEL_PATH"
    echo ""
    echo "Expected structure:"
    echo "  $JOINT_MODEL_PATH/"
    echo "  ├── lora_weights/"
    echo "  │   ├── adapter_config.json"
    echo "  │   └── adapter_model.safetensors"
    echo "  ├── moe_weights.pt"
    echo "  ├── joint_config.json"
    echo "  └── tokenizer files..."
    exit 1
fi

# 检查联合训练模型的必要文件
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
echo "✅ Joint trained model components found"

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

# ✅ 运行评估
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

# ✅ 检查评估结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Joint LoRA+MoE Model Evaluation Completed Successfully!"
    echo "============================================================"
    echo ""
    echo "📊 Model Configuration:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Test Dataset: $DATASET_NAME"
    echo "  Number of Classes: $NUM_CLASSES"
    echo ""
    echo "📁 Component Sources:"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  Joint trained model: $JOINT_MODEL_PATH"
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
    echo "❌ Joint LoRA+MoE Model Evaluation Failed!"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. All component paths are correct"
    echo "  2. CUDA is available and working"
    echo "  3. Required Python packages are installed"
    echo "  4. Sufficient GPU memory is available"
    echo ""
    echo "For debugging, you can run the Python script directly:"
    echo "  python eval_joint_lora_moe_from_base.py --help"
    echo "============================================================"
    exit 1
fi