#!/bin/bash

# 验证 checkpoint 保存是否包含 LoRA 权重的脚本

echo "============================================================"
echo "Checkpoint LoRA Weights Verification"
echo "============================================================"

if [ -z "$1" ]; then
    echo "Usage: bash verify_checkpoint_save.sh <output_directory>"
    echo ""
    echo "Example:"
    echo "  bash verify_checkpoint_save.sh /root/autodl-fs/output/classi_dual_20251019_131545"
    exit 1
fi

OUTPUT_DIR="$1"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Error: Directory does not exist: $OUTPUT_DIR"
    exit 1
fi

echo ""
echo "Checking output directory: $OUTPUT_DIR"
echo ""

# 检查最终模型
echo "📁 Final Model:"
if [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
    echo "  ✅ adapter_config.json found"
else
    echo "  ❌ adapter_config.json NOT found"
fi

if [ -f "$OUTPUT_DIR/adapter_model.bin" ] || [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
    echo "  ✅ adapter_model found"
else
    echo "  ❌ adapter_model NOT found"
fi

# 检查所有 checkpoint
echo ""
echo "📁 Checkpoints:"

checkpoint_count=0
lora_count=0

for checkpoint_dir in "$OUTPUT_DIR"/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then
        checkpoint_count=$((checkpoint_count + 1))
        checkpoint_name=$(basename "$checkpoint_dir")

        has_lora=false
        if [ -f "$checkpoint_dir/adapter_config.json" ]; then
            if [ -f "$checkpoint_dir/adapter_model.bin" ] || [ -f "$checkpoint_dir/adapter_model.safetensors" ]; then
                has_lora=true
                lora_count=$((lora_count + 1))
            fi
        fi

        if [ "$has_lora" = true ]; then
            echo "  ✅ $checkpoint_name - Has LoRA weights"
        else
            echo "  ❌ $checkpoint_name - Missing LoRA weights"
        fi
    fi
done

echo ""
echo "============================================================"
echo "Summary:"
echo "============================================================"
echo "Total checkpoints: $checkpoint_count"
echo "Checkpoints with LoRA: $lora_count"

if [ $checkpoint_count -eq 0 ]; then
    echo ""
    echo "⚠️  No checkpoints found. Training may not have saved any checkpoints yet."
elif [ $lora_count -eq $checkpoint_count ]; then
    echo ""
    echo "✅ SUCCESS! All checkpoints contain LoRA weights!"
elif [ $lora_count -eq 0 ]; then
    echo ""
    echo "❌ FAILED! No checkpoints contain LoRA weights."
    echo ""
    echo "Possible reasons:"
    echo "  1. Training was done without LoRA (--use_llama_lora False)"
    echo "  2. SaveFullModelCallback was not added to the trainer"
    echo "  3. Training was interrupted before any checkpoint was saved"
else
    echo ""
    echo "⚠️  PARTIAL! Only $lora_count out of $checkpoint_count checkpoints have LoRA weights."
    echo ""
    echo "This might happen if:"
    echo "  1. Training was restarted with different settings"
    echo "  2. Some checkpoints were saved before adding the callback"
fi

echo ""

