#!/bin/bash

# 合并 LoRA 权重的脚本

# ✅ 配置参数
BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
LORA_CHECKPOINT_PATH="/root/autodl-fs/output/classi_dual_20251019_1822"  # 或 checkpoint-xxx
OUTPUT_PATH="/root/autodl-fs/merge/llama_lora_merged_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Merging LoRA Weights"
echo "============================================================"
echo "Base model: $BASE_MODEL_PATH"
echo "LoRA checkpoint: $LORA_CHECKPOINT_PATH"
echo "Output: $OUTPUT_PATH"
echo "============================================================"

# 运行合并
python merge_lora_and_save.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_checkpoint_path $LORA_CHECKPOINT_PATH \
    --output_path $OUTPUT_PATH \
    --device cuda:0

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Merge completed successfully!"
    echo "============================================================"
    echo ""
    echo "Merged model saved to: $OUTPUT_PATH"
    echo ""
    echo "Next steps:"
    echo "  1. Use the merged model for evaluation:"
    echo "     MODEL_PATH=\"$OUTPUT_PATH\""
    echo "     bash run_eval_dual_binary.sh"
    echo ""
    echo "  2. Or update run_eval_dual_binary.sh:"
    echo "     vim run_eval_dual_binary.sh"
    echo "     # Change MODEL_PATH to: $OUTPUT_PATH"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Merge failed!"
    echo "============================================================"
    echo ""
    echo "Please check:"
    echo "  1. Does the checkpoint contain LoRA weights?"
    echo "     bash check_checkpoint.sh $LORA_CHECKPOINT_PATH"
    echo ""
    echo "  2. Is the base model path correct?"
    echo "     ls -la $BASE_MODEL_PATH"
    echo "============================================================"
fi

