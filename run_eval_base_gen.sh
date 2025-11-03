#!/bin/bash

# ============================================================
# 评估 Base 模型（生成式版本）
#
# 使用方法：
#   sh run_eval_base_gen.sh <BACKBONE>
#
# 示例：
#   sh run_eval_base_gen.sh llama
#   sh run_eval_base_gen.sh qwen
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"
NUM_CLASSES="${2:-1}" # 生成式测试数据

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 根据 num_classes 选择数据集
case $NUM_CLASSES in
    1)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_gen.json"  # 生成式测试数据
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/base_gen_cultureLLM_results/${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    2)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_gen.json"  # 生成式测试数据
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/base_gen_wvs_results/${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    *)
        echo "❌ Error: Invalid num_classes=$NUM_CLASSES. Must be 2, 3, 4, or 5."
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

# 运行评估脚本
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

