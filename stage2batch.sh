#!/bin/bash

# ============================================================
# 批量执行两阶段训练 CultureMoE
# 依次执行多个训练任务，每个任务间隔 2 秒
# ============================================================

echo "============================================================"
echo "Batch Two-Stage CultureMoE Training"
echo "============================================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ============================================================
# 任务 1：Qwen + 2分类 + 6个专家
# ============================================================
echo "============================================================"
echo "TASK 1/3: Qwen + 2-class + 6 experts"
echo "============================================================"
echo "Running: sh run_two_stage_culturemoe.sh qwen 2 True 6"
echo ""

#sh run_two_stage_culturemoe.sh qwen 2 True 6

#if [ $? -ne 0 ]; then
#    echo ""
#    echo "❌ Task 1 failed!"
#    exit 1
#fi

echo ""
echo "✅ Task 1 completed successfully"
echo ""
echo "Waiting 2 seconds before next task..."
sleep 2
echo ""

# ============================================================
# 任务 2：Qwen + 4分类 + 6个专家
# ============================================================
echo "============================================================"
echo "TASK 2/3: Qwen + 4-class + 6 experts"
echo "============================================================"
echo "Running: sh run_two_stage_culturemoe.sh qwen 4 True 6"
echo ""

#sh run_two_stage_culturemoe.sh qwen 4 True 6

#if [ $? -ne 0 ]; then
#    echo ""
#    echo "❌ Task 2 failed!"
#    exit 1
#fi

echo ""
echo "✅ Task 2 completed successfully"
echo ""
echo "Waiting 2 seconds before next task..."
sleep 2
echo ""

# ============================================================
# 任务 3：Qwen + 4分类 + 6个专家（重复）
# ============================================================
echo "============================================================"
echo "TASK 3/3: llama + 4-class + 6 experts (repeat)"
echo "============================================================"
echo "Running: sh run_two_stage_culturemoe.sh llama 4 True 6"
echo ""

sh run_two_stage_culturemoe.sh llama 4 True 6

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Task 3 failed!"
    exit 1
fi

echo ""
echo "✅ Task 3 completed successfully"
echo ""

# ============================================================
# 完成
# ============================================================
echo "============================================================"
echo "✅ All Tasks Completed Successfully!"
echo "============================================================"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Summary:"
echo "  Task 1: Qwen + 2-class + 6 experts ✅"
echo "  Task 2: Qwen + 4-class + 6 experts ✅"
echo "  Task 3: llama + 4-class + 6 experts (repeat) ✅"
echo ""
echo "Output locations:"
echo "  Task 1: /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_output/culturemoe_2class_experts6_*"
echo "  Task 2: /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_output/culturemoe_4class_experts6_*"
echo "  Task 3: /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_output/culturemoe_4class_experts6_*"
echo "============================================================"

