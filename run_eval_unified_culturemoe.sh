#!/bin/bash

# ============================================================
# 评估统一标签的通用 CultureMoE 模型
#
# 使用方法：
#   sh run_eval_unified_culturemoe.sh <BACKBONE> <DATASET>
#
# 参数说明：
#   BACKBONE: llama 或 qwen
#   DATASET: culturellm, normad, culturalbench, 或 all
#
# 示例：
#   sh run_eval_unified_culturemoe.sh llama culturellm
#   sh run_eval_unified_culturemoe.sh llama all
# ============================================================

BACKBONE="${1:-llama}"
DATASET="${2:-all}"

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    # 使用最新的统一 LoRA 模型
    LORA_WEIGHTS_PATH=$(ls -td /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_unified_qwen_*/best_lora 2>/dev/null | head -1)
    # 使用最新的统一 MoE 模型
    MOE_WEIGHTS_PATH=$(ls -td /root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/moe_unified_qwen_*/best_moe 2>/dev/null | head -1)
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    # 使用最新的统一 LoRA 模型
    LORA_WEIGHTS_PATH=$(ls -td /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_unified_llama_*/best_lora 2>/dev/null | head -1)
    # 使用最新的统一 MoE 模型
    MOE_WEIGHTS_PATH=$(ls -td /root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/moe_unified_llama_*/best_moe 2>/dev/null | head -1)
fi

# 检查模型是否存在
if [ -z "$LORA_WEIGHTS_PATH" ] || [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: No unified LoRA model found for $BACKBONE"
    echo "Please train the unified LoRA model first:"
    echo "  sh run_train_lora_only_gen_unified.sh $BACKBONE"
    exit 1
fi

if [ -z "$MOE_WEIGHTS_PATH" ] || [ ! -d "$MOE_WEIGHTS_PATH" ]; then
    echo "❌ Error: No unified MoE model found for $BACKBONE"
    echo "Please train the unified MoE model first:"
    echo "  sh run_train_culturemoe_from_base_gen_unified.sh $BACKBONE"
    exit 1
fi

echo "============================================================"
echo "Evaluating Unified CultureMoE Model"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "LoRA weights: $LORA_WEIGHTS_PATH"
echo "MoE weights: $MOE_WEIGHTS_PATH"
echo "Dataset: $DATASET"
echo "============================================================"
echo ""

# 根据数据集选择测试文件和标签映射
case $DATASET in
    culturellm)
        TEST_FILE="/root/autodl-fs/wvs_merge_gen.json"
        DATASET_NAME="CultureLLM (WVS)"
        NUM_CLASSES=10
        LABEL_OFFSET=0  # 1-10 保持不变
        echo "Evaluating on CultureLLM (labels 1-10)"
        ;;
    normad)
        TEST_FILE="/root/autodl-fs/normad_test_gen.json"
        DATASET_NAME="NormAD"
        NUM_CLASSES=3
        LABEL_OFFSET=10  # 11, 12, 13
        echo "Evaluating on NormAD (labels 11-13: yes/no/neutral)"
        ;;
    culturalbench)
        TEST_FILE="/root/autodl-fs/CulturalBench_test_gen.json"
        DATASET_NAME="CulturalBench"
        NUM_CLASSES=2
        LABEL_OFFSET=13  # 14, 15
        echo "Evaluating on CulturalBench (labels 14-15: TRUE/FALSE)"
        ;;
    all)
        echo "Evaluating on all datasets..."
        echo ""

        # 递归调用自己评估每个数据集
        sh run_eval_unified_culturemoe.sh $BACKBONE culturellm
        echo ""
        echo "============================================================"
        echo ""
        sh run_eval_unified_culturemoe.sh $BACKBONE normad
        echo ""
        echo "============================================================"
        echo ""
        sh run_eval_unified_culturemoe.sh $BACKBONE culturalbench

        exit 0
        ;;
    *)
        echo "❌ Error: Invalid dataset: $DATASET"
        echo "Valid options: culturellm, normad, culturalbench, all"
        exit 1
        ;;
esac

# 检查测试文件
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/eval_unified_moe_${BACKBONE}_${DATASET}_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Test file: $TEST_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# 运行评估
python eval_culturemoe_from_components_gen.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_weights_path $LORA_WEIGHTS_PATH \
    --moe_weights_path $MOE_WEIGHTS_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --num_classes $NUM_CLASSES \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "💡 To view generated answers:"
    echo "   cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -100"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

