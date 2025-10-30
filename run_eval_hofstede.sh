#!/bin/bash

# 评估 Base LLaMA 模型的霍夫斯泰德文化维度一致性

# ✅ 配置参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
DATA_PATH="/root/autodl-fs/VSM2013_questions.json"  # VSM2013 问题文件路径
OUTPUT_DIR="/root/autodl-fs/output/hofstede_evaluation/$(date +%Y%m%d_%H%M%S)"
DEVICE="cuda"  # 使用 GPU

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "Hofstede Cultural Dimensions Evaluation"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "============================================================"
echo ""

# 运行评估
python eval_hofstede_culture.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - VSM2013_<Country>_<Timestamp>.json (answered questions for each country)"
    echo "  - hofstede_evaluation_summary_<Timestamp>.json (overall summary)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

