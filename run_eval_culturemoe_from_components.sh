#!/bin/bash

# ============================================================
# 从 Base 模型 + LoRA 权重 + MoE 权重还原 CultureMoE 模型并评估
#
# 使用方法：
#   sh run_eval_culturemoe_from_components.sh <BACKBONE> <NUM_CLASSES> <LORA_DIR> <MOE_DIR>
#
# 示例：
#   sh run_eval_culturemoe_from_components.sh llama 2 /path/to/lora_output /path/to/moe_output
#   sh run_eval_culturemoe_from_components.sh qwen 4 /path/to/lora_output /path/to/moe_output
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"           # 默认使用 llama
NUM_CLASSES="${2:-2}"            # 默认 2 分类
# LORA_OUTPUT_DIR="${3}"           # LoRA 训练输出目录
MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_output/culturemoe_${BACKBONE}_${NUM_CLASSES}class_experts${NUM_EXPERTS}_20251105_2216)"          # MoE 权重路径

# 检查必需参数
if [ -z "$LORA_OUTPUT_DIR" ]; then
    echo "❌ Error: LORA_OUTPUT_DIR is required"
    echo ""
    echo "Usage: sh run_eval_culturemoe_from_components.sh <BACKBONE> <NUM_CLASSES> <LORA_DIR> <MOE_DIR>"
    echo ""
    echo "Example:"
    echo "  sh run_eval_culturemoe_from_components.sh llama 2 \\"
    echo "    /root/autodl-tmp/.../lora_only_output \\"
    echo "    /root/autodl-tmp/.../model_moe_llama_2"
    exit 1
fi

if [ -z "$MOE_WEIGHTS_PATH" ]; then
    echo "❌ Error: MOE_WEIGHTS_PATH is required"
    echo ""
    echo "Usage: sh run_eval_culturemoe_from_components.sh <BACKBONE> <NUM_CLASSES> <LORA_DIR> <MOE_DIR>"
    exit 1
fi

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# LoRA 权重路径（best_lora 目录）
LORA_WEIGHTS_PATH="/root/autodl-fs/model/llama_lora_only_${NUM_CLASSES}"

# 根据 num_classes 选择测试数据集
case $NUM_CLASSES in
    2)
        TEST_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
        DATASET_NAME="CulturalBench_Hard"
        ;;
    3)
        TEST_FILE="/root/autodl-fs/normad_ed_merge.json"
        DATASET_NAME="NormAD_ED"
        ;;
    4)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_4.json"
        DATASET_NAME="WVS_4class"
        ;;
    5)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_5.json"
        DATASET_NAME="WVS_5class"
        ;;
    *)
        echo "❌ Error: Invalid num_classes=$NUM_CLASSES. Must be 2, 3, 4, or 5."
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/eval_results/${BACKBONE}_${NUM_CLASSES}class_from_components_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "CultureMoE Evaluation (From Components)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Num classes: $NUM_CLASSES"
echo "Dataset: $DATASET_NAME"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo "  MoE weights: $MOE_WEIGHTS_PATH"
echo ""
echo "Test file: $TEST_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    echo ""
    echo "Expected path: ${LORA_OUTPUT_DIR}/best_lora"
    echo ""
    echo "Please ensure you have completed:"
    echo "  sh run_train_lora_only.sh $BACKBONE $NUM_CLASSES true"
    echo ""
    echo "The training should create a 'best_lora' directory containing:"
    echo "  - adapter_config.json"
    echo "  - adapter_model.safetensors"
    echo "  - tokenizer files"
    exit 1
fi

if [ ! -d "$MOE_WEIGHTS_PATH" ]; then
    echo "❌ Error: MoE weights not found: $MOE_WEIGHTS_PATH"
    echo ""
    echo "Please ensure you have completed:"
    echo "  sh run_train_culturemoe_from_merged.sh $BACKBONE $NUM_CLASSES True 6 true"
    echo ""
    echo "The training should create a directory containing:"
    echo "  - moe_config.json"
    echo "  - moe_state_dict.pt"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行评估
python eval_culturemoe_from_components.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_weights_path $LORA_WEIGHTS_PATH \
    --moe_weights_path $MOE_WEIGHTS_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size 8 \
    --max_length 512 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Num classes: $NUM_CLASSES"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo "  MoE weights: $MOE_WEIGHTS_PATH"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果)"
    echo "  - evaluation_summary.json (摘要)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "💡 To compare with merged model evaluation:"
    echo "   sh run_load_culturemoe.sh $BACKBONE $NUM_CLASSES"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

