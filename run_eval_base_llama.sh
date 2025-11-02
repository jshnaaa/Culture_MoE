#!/bin/bash

# 评估 Base 模型（支持不同类别数）
# 支持 LLaMA 3.1 和 Qwen 2.5
# 适用于 WVS 等多分类数据集

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama，可以通过第一个参数指定 qwen
NUM_CLASSES="${2:-21}"   # 默认 4 分类，可以通过第二个参数指定其他值（2/3/4/5）

# 根据 num_classes 选择数据集
case $NUM_CLASSES in
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
    442)
        TEST_FILE="/root/autodl-fs/wvs_4_merge_rp.json"
        ;;
    *)
        echo "❌ Error: Invalid num_classes=$NUM_CLASSES. Must be 2, 3, 4, or 5."
        exit 1
        ;;
esac

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    OUTPUT_FILE="/root/autodl-fs/output/base_qwen/eval_results_base_qwen_${NUM_CLASSES}class_$(date +%Y%m%d_%H%M).json"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_FILE="/root/autodl-fs/output/base_llama/eval_results_base_llama_${NUM_CLASSES}class_$(date +%Y%m%d_%H%M).json"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

echo "============================================================"
echo "Evaluating Base $MODEL_NAME Model (${NUM_CLASSES}-class)"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Num classes: $NUM_CLASSES"
echo "Model: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output: $OUTPUT_FILE"
echo "============================================================"
echo ""

# ✅ 运行评估
python eval_base_llama.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --batch_size 8 \
    --max_length 512 \
    --output_file $OUTPUT_FILE \
    --device cuda:0 \
    --backbone $BACKBONE \
    --num_classes $NUM_CLASSES

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

