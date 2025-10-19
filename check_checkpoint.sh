#!/bin/bash

# 检查 checkpoint 目录内容的脚本

if [ -z "$1" ]; then
    echo "Usage: bash check_checkpoint.sh <checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  bash check_checkpoint.sh /root/autodl-fs/output/classi_dual_20251019_131545/checkpoint-3000"
    exit 1
fi

CHECKPOINT_PATH="$1"

echo "============================================================"
echo "Checking checkpoint: $CHECKPOINT_PATH"
echo "============================================================"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Directory does not exist!"
    exit 1
fi

echo ""
echo "📁 Directory contents:"
ls -lh "$CHECKPOINT_PATH"

echo ""
echo "============================================================"
echo "Checking required files:"
echo "============================================================"

# 检查 LoRA 文件
echo ""
echo "🔍 LoRA weights:"
if [ -f "$CHECKPOINT_PATH/adapter_config.json" ]; then
    echo "  ✅ adapter_config.json found"
else
    echo "  ❌ adapter_config.json NOT found"
fi

if [ -f "$CHECKPOINT_PATH/adapter_model.bin" ]; then
    echo "  ✅ adapter_model.bin found"
    ls -lh "$CHECKPOINT_PATH/adapter_model.bin"
elif [ -f "$CHECKPOINT_PATH/adapter_model.safetensors" ]; then
    echo "  ✅ adapter_model.safetensors found"
    ls -lh "$CHECKPOINT_PATH/adapter_model.safetensors"
else
    echo "  ❌ adapter_model.bin/safetensors NOT found"
fi

# 检查 MoE 权重
echo ""
echo "🔍 MoE weights:"
if [ -f "$CHECKPOINT_PATH/pytorch_model.bin" ]; then
    echo "  ✅ pytorch_model.bin found"
    ls -lh "$CHECKPOINT_PATH/pytorch_model.bin"
elif [ -f "$CHECKPOINT_PATH/model.safetensors" ]; then
    echo "  ✅ model.safetensors found"
    ls -lh "$CHECKPOINT_PATH/model.safetensors"
else
    echo "  ❌ pytorch_model.bin/model.safetensors NOT found"
fi

# 检查其他文件
echo ""
echo "🔍 Other files:"
if [ -f "$CHECKPOINT_PATH/trainer_state.json" ]; then
    echo "  ✅ trainer_state.json found"
else
    echo "  ⚠️  trainer_state.json NOT found"
fi

if [ -f "$CHECKPOINT_PATH/training_args.bin" ]; then
    echo "  ✅ training_args.bin found"
else
    echo "  ⚠️  training_args.bin NOT found"
fi

if [ -f "$CHECKPOINT_PATH/config.json" ]; then
    echo "  ✅ config.json found"
else
    echo "  ⚠️  config.json NOT found"
fi

echo ""
echo "============================================================"
echo "Recommendation:"
echo "============================================================"

# 判断是否可用
has_lora=false
has_moe=false

if [ -f "$CHECKPOINT_PATH/adapter_config.json" ] && ([ -f "$CHECKPOINT_PATH/adapter_model.bin" ] || [ -f "$CHECKPOINT_PATH/adapter_model.safetensors" ]); then
    has_lora=true
fi

if [ -f "$CHECKPOINT_PATH/pytorch_model.bin" ] || [ -f "$CHECKPOINT_PATH/model.safetensors" ]; then
    has_moe=true
fi

if [ "$has_lora" = true ] && [ "$has_moe" = true ]; then
    echo "✅ This checkpoint is COMPLETE and can be used for evaluation"
    echo ""
    echo "Run evaluation with:"
    echo "  MODEL_PATH=\"$CHECKPOINT_PATH\""
elif [ "$has_lora" = false ] && [ "$has_moe" = true ]; then
    echo "⚠️  This checkpoint has MoE weights but NO LoRA weights"
    echo ""
    echo "Options:"
    echo "  1. Use the final model (not a checkpoint):"
    echo "     MODEL_PATH=\"${CHECKPOINT_PATH%/checkpoint-*}\""
    echo ""
    echo "  2. Run evaluation without LoRA (not recommended):"
    echo "     Remove --use_lora flag"
elif [ "$has_lora" = true ] && [ "$has_moe" = false ]; then
    echo "⚠️  This checkpoint has LoRA weights but NO MoE weights"
    echo ""
    echo "This is unusual. Check if training completed successfully."
else
    echo "❌ This checkpoint is INCOMPLETE"
    echo ""
    echo "Please use a different checkpoint or the final model."
fi

echo ""

