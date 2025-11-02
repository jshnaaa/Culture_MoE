#!/bin/bash

LOG_DIR="/root/autodl-fs/output/batch_base_test1_logs"
mkdir -p $LOG_DIR
BATCH_LOG="$LOG_DIR/batch_training_${TIMESTAMP}.log"

echo "Batch log will be saved to: $BATCH_LOG"
echo ""

# 任务 1
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 1" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_base_llama.sh llama 22 2>&1 | tee -a $BATCH_LOG
TASK1_STATUS=$?

if [ $TASK1_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 1 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 1 failed with exit code $TASK1_STATUS" | tee -a $BATCH_LOG
    exit 1
fi

# 任务 2
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 2" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_base_llama.sh qwen 22 2>&1 | tee -a $BATCH_LOG
TASK2_STATUS=$?

if [ $TASK2_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 2 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 2 failed with exit code $TASK2_STATUS" | tee -a $BATCH_LOG
    exit 1
fi

# 任务 3
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 3" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_base_llama.sh llama 42 2>&1 | tee -a $BATCH_LOG
TASK3_STATUS=$?

if [ $TASK3_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 3 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    exit 1
fi

# 任务 4
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 4" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_base_llama.sh qwen 42 2>&1 | tee -a $BATCH_LOG
TASK4_STATUS=$?

if [ $TASK4_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 4 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 4 failed with exit code $TASK3_STATUS" | tee -a $BATCH_LOG
    exit 1
fi

echo "" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
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

