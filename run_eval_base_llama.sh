#!/bin/bash

# 评估 Base 模型（支持不同类别数）
# 支持 LLaMA 3.1 和 Qwen 2.5
# 适用于 WVS 等多分类数据集

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama，可以通过第一个参数指定 qwen
DATASET_ID="${2:-21}"   # 数据集标识符（不是实际的分类数量）

# 根据数据集标识符选择数据集
case $DATASET_ID in
    21)
        TEST_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
        ;;
    31)
        TEST_FILE="/root/autodl-fs/normad_ed_merge.json"
        ;;
    41)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_4.json"
        ;;
    51)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_5.json"
        ;;
    221)
        TEST_FILE="/root/autodl-fs/wvs_2_merge.json"
        ;;
    2221)
        TEST_FILE="/root/autodl-fs/wvs_2c_merge.json"
        ;;
    441)
        TEST_FILE="/root/autodl-fs/wvs_4_merge.json"
        ;;
    22)
        TEST_FILE="/root/autodl-fs/CulturalBench_Hard_merge_rp.json"
        ;;
    32)
        TEST_FILE="/root/autodl-fs/normad_ed_merge_rp.json"
        ;;
    42)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_4_rp.json"
        ;;
    52)
        TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_5_rp.json"
        ;;
    222)
        TEST_FILE="/root/autodl-fs/wvs_2_merge_rp.json"
        ;;
    2222)
        TEST_FILE="/root/autodl-fs/wvs_2c_merge_rp.json"
        ;;
    442)
        TEST_FILE="/root/autodl-fs/wvs_4_merge_rp.json"
        ;;
    *)
        echo "❌ Error: Invalid dataset_id=$DATASET_ID."
        exit 1
        ;;
esac

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    OUTPUT_FILE="/root/autodl-fs/output/base_qwen/eval_results_base_qwen_${DATASET_ID}_$(date +%Y%m%d_%H%M).json"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_FILE="/root/autodl-fs/output/base_llama/eval_results_base_llama_${DATASET_ID}_$(date +%Y%m%d_%H%M).json"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

echo "============================================================"
echo "Evaluating Base $MODEL_NAME Model"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Dataset ID: $DATASET_ID"
echo "Model: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output: $OUTPUT_FILE"
echo "============================================================"
echo ""

# ✅ 运行评估（num_classes 将从数据中自动推断）
python eval_base_llama.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --batch_size 8 \
    --max_length 512 \
    --output_file $OUTPUT_FILE \
    --device cuda:0 \
    --backbone $BACKBONE

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_FILE"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

