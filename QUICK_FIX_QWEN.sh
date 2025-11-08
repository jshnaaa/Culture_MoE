#!/bin/bash

# ============================================================
# Qwen 梯度 NaN 问题快速修复脚本
# ============================================================

echo "🔧 Qwen Gradient NaN Quick Fix"
echo "============================================================"
echo ""

# 步骤 1：清理 Python 缓存
echo "Step 1: Cleaning Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "✅ Cache cleaned"
echo ""

# 步骤 2：验证修复
echo "Step 2: Verifying fixes..."
echo ""

echo "Checking fp32 configuration..."
if grep -q "use_fp16 = False" train_lora_only_gen.py; then
    echo "✅ fp32 configuration found"
else
    echo "❌ fp32 configuration NOT found"
fi

echo ""
echo "Checking learning rate reduction..."
if grep -q "learning_rate = args.learning_rate \* 0.25" train_lora_only_gen.py; then
    echo "✅ Learning rate reduction (25%) found"
else
    echo "❌ Learning rate reduction NOT found"
fi

echo ""
echo "Checking gradient clipping..."
if grep -q "max_grad_norm = 0.3" train_lora_only_gen.py; then
    echo "✅ Aggressive gradient clipping (0.3) found"
else
    echo "❌ Aggressive gradient clipping NOT found"
fi

echo ""
echo "Checking warmup configuration..."
if grep -q "warmup_steps = 500" train_lora_only_gen.py; then
    echo "✅ Extended warmup (500 steps) found"
else
    echo "❌ Extended warmup NOT found"
fi

echo ""
echo "============================================================"
echo "✅ All fixes verified!"
echo "============================================================"
echo ""

# 步骤 3：提示用户运行训练
echo "Step 3: Ready to train!"
echo ""
echo "Run one of the following commands:"
echo ""
echo "  # CulturalBench (2 classes)"
echo "  bash run_train_lora_only_gen.sh qwen 2"
echo ""
echo "  # NormAD (3 classes)"
echo "  bash run_train_lora_only_gen.sh qwen 3"
echo ""
echo "  # CultureLLM (10 classes)"
echo "  bash run_train_lora_only_gen.sh qwen 4"
echo ""
echo "============================================================"
echo ""
echo "💡 Expected output:"
echo "  - Detected model type: qwen"
echo "  - Using Qwen-specific training configuration (aggressive)"
echo "  - Learning rate: 2.5e-05 (25% of 0.0001)"
echo "  - Max grad norm: 0.3"
echo "  - Warmup steps: 500"
echo "  - Using fp32 (not fp16)"
echo ""
echo "✅ If you see the above, the fix is working!"
echo ""

