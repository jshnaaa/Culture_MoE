#!/bin/bash

# ============================================================
# 评估 Base 模型（生成式版本）
#
# 使用方法：
#   sh run_eval_base_gen.sh <BACKBONE> <NUM_CLASSES>
#
# 示例：
#   sh run_eval_base_gen.sh llama 11
#   sh run_eval_base_gen.sh qwen 21
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"
DATASET_ID="${2:-11}"  # 数据集标识符（不是实际的分类数量）

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 根据数据集标识符选择数据集
case $DATASET_ID in
    11)
        TEST_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"  # 生成式验证数据
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/base_gen_cultureLLM_results/${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    21)
        TEST_FILE="/root/autodl-fs/wvs_merge_gen.json"  # 生成式测试数据
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/base_gen_wvs_results/${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    12)
        TEST_FILE="/root/autodl-fs/cultureLLM_merge_rp_gen.json"  # 生成式验证数据
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/rp_gen_cultureLLM_results/${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    22)
        TEST_FILE="/root/autodl-fs/wvs_merge_rp_gen.json"  # 生成式测试数据
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/rp_gen_wvs_results/${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    *)
        echo "❌ Error: Invalid dataset_id=$DATASET_ID. Must be 11,21,12,22."
        exit 1
        ;;
esac

echo "============================================================"
echo "Base Model Evaluation (Generative Version)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Model path: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

# 运行评估脚本（num_classes 将从数据文件中自动推断）
python eval_base_gen.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Model path: $MODEL_PATH"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果)"
    echo "  - evaluation_summary.json (摘要)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json | jq"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

