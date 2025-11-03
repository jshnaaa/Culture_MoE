#!/bin/bash

# ============================================================
# 评估 LoRA Only 模型（生成式版本）
#
# 使用方法：
#   sh run_eval_lora_only_gen.sh <BACKBONE>
#
# 示例：
#   sh run_eval_lora_only_gen.sh llama
#   sh run_eval_lora_only_gen.sh qwen
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"

# 根据 backbone 构建路径
MERGED_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/${BACKBONE}_merge_gen"
TEST_FILE="/root/autodl-fs/wvs_all_llama_merge_gen.json"  # 生成式验证数据
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/%lora_only_gen_wvs_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "LoRA Only Model Evaluation (Generative Version)"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Merged model: $MERGED_MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$MERGED_MODEL_PATH" ]; then
    echo "❌ Error: Merged model not found: $MERGED_MODEL_PATH"
    echo ""
    echo "Please ensure you have completed the following steps:"
    echo "  1. Train LoRA Only: sh run_train_lora_only_gen.sh $BACKBONE"
    echo "  2. Merge weights: sh run_merge_lora_gen.sh $BACKBONE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

# 运行评估脚本
python eval_lora_only_gen.py \
    --model_path $MERGED_MODEL_PATH \
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
    echo "  Backbone: $BACKBONE"
    echo "  Merged model: $MERGED_MODEL_PATH"
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

