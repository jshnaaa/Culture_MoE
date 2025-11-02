#!/bin/bash

# 记录开始时间
START_TIME=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/root/autodl-fs/output/batch_base_test_logs"
mkdir -p $LOG_DIR
BATCH_LOG="$LOG_DIR/batch_training_${TIMESTAMP}.log"

echo "Batch log will be saved to: $BATCH_LOG"
echo ""

# 任务 1: Qwen + 2分类
echo "============================================================" | tee -a $BATCH_LOG
echo "Start time: $(date)" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

TASK1_START=$(date +%s)
sh run_eval_base_llama.sh llama 221 2>&1 | tee -a $BATCH_LOG
TASK1_STATUS=$?
TASK1_END=$(date +%s)
TASK1_DURATION=$((TASK1_END - TASK1_START))

if [ $TASK1_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 1 completed successfully!" | tee -a $BATCH_LOG
    echo "Duration: $((TASK1_DURATION / 60)) minutes $((TASK1_DURATION % 60)) seconds" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 1 failed with exit code $TASK1_STATUS" | tee -a $BATCH_LOG
    echo "Duration: $((TASK1_DURATION / 60)) minutes $((TASK1_DURATION % 60)) seconds" | tee -a $BATCH_LOG
    echo "Stopping batch execution." | tee -a $BATCH_LOG
    exit 1
fi

# 任务 2: Qwen + 4分类
echo "============================================================" | tee -a $BATCH_LOG
echo "Start time: $(date)" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

TASK2_START=$(date +%s)
sh run_eval_base_llama.sh qwen 221 2>&1 | tee -a $BATCH_LOG
TASK2_STATUS=$?
TASK2_END=$(date +%s)
TASK2_DURATION=$((TASK2_END - TASK2_START))

if [ $TASK2_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 2 completed successfully!" | tee -a $BATCH_LOG
    echo "Duration: $((TASK2_DURATION / 60)) minutes $((TASK2_DURATION % 60)) seconds" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 2 failed with exit code $TASK2_STATUS" | tee -a $BATCH_LOG
    echo "Duration: $((TASK2_DURATION / 60)) minutes $((TASK2_DURATION % 60)) seconds" | tee -a $BATCH_LOG
    echo "Stopping batch execution." | tee -a $BATCH_LOG
    exit 1
fi

# 任务 3: Qwen + 5分类
echo "============================================================" | tee -a $BATCH_LOG
echo "Start time: $(date)" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

TASK3_START=$(date +%s)
sh run_eval_base_llama.sh qwen 5 True false 2>&1 | tee -a $BATCH_LOG
TASK3_STATUS=$?
TASK3_END=$(date +%s)
TASK3_DURATION=$((TASK3_END - TASK3_START))

if [ $TASK3_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 3 completed successfully!" | tee -a $BATCH_LOG
    echo "Duration: $((TASK3_DURATION / 60)) minutes $((TASK3_DURATION % 60)) seconds" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 3 failed with exit code $TASK3_STATUS" | tee -a $BATCH_LOG
    echo "Duration: $((TASK3_DURATION / 60)) minutes $((TASK3_DURATION % 60)) seconds" | tee -a $BATCH_LOG
    exit 1
fi

# 计算总时间
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "Batch Training Summary" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 1 (Qwen 2-class): $((TASK1_DURATION / 60))m $((TASK1_DURATION % 60))s" | tee -a $BATCH_LOG
echo "Task 2 (Qwen 4-class): $((TASK2_DURATION / 60))m $((TASK2_DURATION % 60))s" | tee -a $BATCH_LOG
echo "Task 3 (Qwen 5-class): $((TASK3_DURATION / 60))m $((TASK3_DURATION % 60))s" | tee -a $BATCH_LOG
echo "------------------------------------------------------------" | tee -a $BATCH_LOG
echo "Total time: $((TOTAL_DURATION / 60)) minutes $((TOTAL_DURATION % 60)) seconds" | tee -a $BATCH_LOG
echo "End time: $(date)" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG
echo "✅ All tasks completed successfully!" | tee -a $BATCH_LOG
echo "Batch log saved to: $BATCH_LOG" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG

# 查找并显示所有输出目录
echo "" | tee -a $BATCH_LOG
echo "Output directories:" | tee -a $BATCH_LOG
find /root/autodl-fs/output/CultureMoE -name "culturemoe_qwen_*class_*" -type d -mmin -$((TOTAL_DURATION / 60 + 10)) | tee -a $BATCH_LOG

echo "" | tee -a $BATCH_LOG
echo "To view epoch evaluation results:" | tee -a $BATCH_LOG
echo "  cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_*class_*/epoch_eval_results.json | jq" | tee -a $BATCH_LOG

