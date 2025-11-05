#!/bin/bash

# ============================================================
# 评估生成式 LoRA Only 模型
# 只生成数字（选项编号），方便计算准确率
# ============================================================

# 参数
BACKBONE="${1:-llama}"  # llama 或 qwen

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    LORA_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/lora_only_gen_qwen"
    TEST_FILE="/root/autodl-fs/wvs_gen_test.json"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    LORA_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/lora_only_gen_llama"
    TEST_FILE="/root/autodl-fs/wvs_gen_test.json"
fi

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/eval_output/lora_gen_${BACKBONE}_$(date +%Y%m%d_%H%M)"

# 类别数量（根据数据集调整）
NUM_CLASSES=5

echo "============================================================"
echo "Evaluating Generative LoRA Model"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Base model: $BASE_MODEL_PATH"
echo "LoRA path: $LORA_PATH"
echo "Test file: $TEST_FILE"
echo "Num classes: $NUM_CLASSES"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 运行评估
python eval_lora_only_gen.py \
    --model_path $BASE_MODEL_PATH \
    --lora_path $LORA_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --num_classes $NUM_CLASSES \
    --device cuda:0 \
    --save_answers

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - eval_metrics.json       (Overall metrics)"
    echo "  - generated_answers.json  (All answers with details)"
    echo "  - error_cases.json        (Only incorrect predictions)"
    echo "  - predictions.txt         (Simple format for review)"
    echo "============================================================"
    echo ""
    echo "Quick view of results:"
    cat $OUTPUT_DIR/eval_metrics.json | jq '.metrics'
    echo ""
    echo "Sample answers:"
    head -10 $OUTPUT_DIR/predictions.txt
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

