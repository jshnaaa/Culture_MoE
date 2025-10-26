#!/bin/bash

# 训练 LoRA 微调的 LLaMA 3.1 模型（无 MoE）
# 90% 训练，10% 验证

# ✅ 配置参数
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
OUTPUT_DIR="/root/autodl-fs/output/lora_only/lora_only_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Training LLaMA 3.1 with LoRA (No MoE)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Train file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# ✅ 运行训练
python train_and_eval_lora_only.py \
    --model_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --lora_rank 8 \
    --max_length 512 \
    --val_split 0.1

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training and evaluation completed successfully!"
    echo "============================================================"
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Files:"
    echo "  - adapter_config.json (LoRA 配置)"
    echo "  - adapter_model.bin (LoRA 权重)"
    echo "  - pytorch_model.bin (分类头权重)"
    echo "  - eval_results.json (评估结果)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

