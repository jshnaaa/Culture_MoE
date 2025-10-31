#!/bin/bash

# ============================================================
# 两阶段训练 CultureMoE（自动化）
# 自动依次运行：
# 1. run_train_lora_only.sh - 训练 LoRA Only
# 2. run_merge_lora.sh - 合并权重
# 3. run_train_culturemoe_from_merged.sh - 训练 MoE
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama
NUM_CLASSES="${2:-2}"   # 默认 2 分类
USE_CULTURE_LOSS="${3:-True}"  # 默认使用文化损失
NUM_EXPERTS="${4:-6}"   # 默认 6 个专家

echo "============================================================"
echo "Two-Stage CultureMoE Training (Automated)"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Num classes: $NUM_CLASSES"
echo "Use culture loss: $USE_CULTURE_LOSS"
echo "Num experts: $NUM_EXPERTS"
echo "============================================================"
echo ""

# ============================================================
# 步骤 1：训练 LoRA Only（保存最佳模型）
# ============================================================
echo "============================================================"
echo "STEP 1: Training LoRA Only"
echo "============================================================"
echo "Running: sh run_train_lora_only.sh $BACKBONE $NUM_CLASSES true"
echo ""

sh run_train_lora_only.sh $BACKBONE $NUM_CLASSES true

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Step 1 failed: LoRA Only training failed!"
    exit 1
fi

echo ""
echo "✅ Step 1 completed: LoRA Only training successful"
echo ""

# ============================================================
# 步骤 2：合并 LoRA 权重
# ============================================================
echo "============================================================"
echo "STEP 2: Merging LoRA Weights"
echo "============================================================"
echo "Running: sh run_merge_lora.sh $BACKBONE"
echo ""

sh run_merge_lora.sh $BACKBONE

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Step 2 failed: LoRA merging failed!"
    exit 1
fi

echo ""
echo "✅ Step 2 completed: LoRA weights merged successfully"
echo ""

# ============================================================
# 步骤 3：训练 CultureMoE（从合并后的模型）
# ============================================================
echo "============================================================"
echo "STEP 3: Training CultureMoE (From Merged Model)"
echo "============================================================"
echo "Running: sh run_train_culturemoe_from_merged.sh $BACKBONE $NUM_CLASSES $USE_CULTURE_LOSS $NUM_EXPERTS"
echo ""

sh run_train_culturemoe_from_merged.sh $BACKBONE $NUM_CLASSES $USE_CULTURE_LOSS $NUM_EXPERTS

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Step 3 failed: CultureMoE training failed!"
    exit 1
fi

echo ""
echo "✅ Step 3 completed: CultureMoE training successful"
echo ""

# ============================================================
# 完成
# ============================================================
echo "============================================================"
echo "✅ Two-Stage Training Completed Successfully!"
echo "============================================================"
echo ""
echo "Summary:"
echo "  Step 1: LoRA Only training ✅"
echo "  Step 2: LoRA weights merging ✅"
echo "  Step 3: CultureMoE training ✅"
echo ""
echo "Output locations:"
if [ "$BACKBONE" = "qwen" ]; then
    echo "  LoRA model: /root/autodl-tmp/CultureMoE/Culture_Alignment/qwen_lora_only"
    echo "  Merged model: /root/autodl-tmp/CultureMoE/Culture_Alignment/qwen_merge"
else
    echo "  LoRA model: /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only"
    echo "  Merged model: /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_merge"
fi
echo "  CultureMoE: /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_output/culturemoe_${NUM_CLASSES}class_experts${NUM_EXPERTS}_*"
echo "============================================================"

