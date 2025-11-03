#!/bin/bash

# ============================================================
# 批量运行 VSM13 文化一致性测试
#
# 使用方法：
#   sh run_batch_vsm13.sh
#
# 说明：
#   依次测试所有模型和骨干网络的组合
# ============================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/root/autodl-fs/vsm13_output/logs"
mkdir -p $LOG_DIR
BATCH_LOG="$LOG_DIR/batch_vsm13_${TIMESTAMP}.log"

echo "============================================================" | tee -a $BATCH_LOG
echo "Batch VSM13 Cultural Consistency Test" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $BATCH_LOG
echo "Batch log: $BATCH_LOG" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

START_TIME=$(date +%s)

# 任务 1: Base LLaMA
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 1: Base LLaMA" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh base llama 2>&1 | tee -a $BATCH_LOG
TASK1_STATUS=$?

if [ $TASK1_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 1 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 1 failed with exit code $TASK1_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 任务 2: Base Qwen
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 2: Base Qwen" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh base qwen 2>&1 | tee -a $BATCH_LOG
TASK2_STATUS=$?

if [ $TASK2_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 2 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 2 failed with exit code $TASK2_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 任务 3: LoRA LLaMA 2-class
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 3: LoRA LLaMA 2-class" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh lora llama 2 2>&1 | tee -a $BATCH_LOG
TASK3_STATUS=$?

if [ $TASK3_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 3 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 3 failed with exit code $TASK3_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 任务 4: LoRA LLaMA 4-class
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 4: LoRA LLaMA 4-class" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh lora llama 4 2>&1 | tee -a $BATCH_LOG
TASK4_STATUS=$?

if [ $TASK4_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 4 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 4 failed with exit code $TASK4_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 任务 5: LoRA Qwen 2-class
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 5: LoRA Qwen 2-class" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh lora qwen 2 2>&1 | tee -a $BATCH_LOG
TASK5_STATUS=$?

if [ $TASK5_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 5 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 5 failed with exit code $TASK5_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 任务 6: LoRA Qwen 4-class
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 6: LoRA Qwen 4-class" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh lora qwen 4 2>&1 | tee -a $BATCH_LOG
TASK6_STATUS=$?

if [ $TASK6_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 6 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 6 failed with exit code $TASK6_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 任务 7: MoE LLaMA 2-class
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 7: MoE LLaMA 2-class" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh moe llama 2 2>&1 | tee -a $BATCH_LOG
TASK7_STATUS=$?

if [ $TASK7_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 7 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 7 failed with exit code $TASK7_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 任务 8: MoE LLaMA 4-class
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 8: MoE LLaMA 4-class" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh moe llama 4 2>&1 | tee -a $BATCH_LOG
TASK8_STATUS=$?

if [ $TASK8_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 8 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 8 failed with exit code $TASK8_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 任务 9: MoE Qwen 2-class
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 9: MoE Qwen 2-class" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh moe qwen 2 2>&1 | tee -a $BATCH_LOG
TASK9_STATUS=$?

if [ $TASK9_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 9 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 9 failed with exit code $TASK9_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 任务 10: MoE Qwen 4-class
echo "============================================================" | tee -a $BATCH_LOG
echo "Task 10: MoE Qwen 4-class" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

sh run_eval_vsm13.sh moe qwen 4 2>&1 | tee -a $BATCH_LOG
TASK10_STATUS=$?

if [ $TASK10_STATUS -eq 0 ]; then
    echo "" | tee -a $BATCH_LOG
    echo "✅ Task 10 completed successfully!" | tee -a $BATCH_LOG
else
    echo "" | tee -a $BATCH_LOG
    echo "❌ Task 10 failed with exit code $TASK10_STATUS" | tee -a $BATCH_LOG
fi
echo "" | tee -a $BATCH_LOG

# 计算总时间
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# 汇总结果
echo "============================================================" | tee -a $BATCH_LOG
echo "Batch Test Summary" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $BATCH_LOG
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG
echo "Task Results:" | tee -a $BATCH_LOG
echo "  Task 1 (Base LLaMA):        $([ $TASK1_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "  Task 2 (Base Qwen):         $([ $TASK2_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "  Task 3 (LoRA LLaMA 2):      $([ $TASK3_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "  Task 4 (LoRA LLaMA 4):      $([ $TASK4_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "  Task 5 (LoRA Qwen 2):       $([ $TASK5_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "  Task 6 (LoRA Qwen 4):       $([ $TASK6_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "  Task 7 (MoE LLaMA 2):       $([ $TASK7_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "  Task 8 (MoE LLaMA 4):       $([ $TASK8_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "  Task 9 (MoE Qwen 2):        $([ $TASK9_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "  Task 10 (MoE Qwen 4):       $([ $TASK10_STATUS -eq 0 ] && echo '✅ Success' || echo '❌ Failed')" | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG
echo "Batch log saved to: $BATCH_LOG" | tee -a $BATCH_LOG
echo "============================================================" | tee -a $BATCH_LOG

# 显示所有结果文件
echo "" | tee -a $BATCH_LOG
echo "Result files:" | tee -a $BATCH_LOG
find /root/autodl-fs/vsm13_output -name "vsm13_test_results_*.json" -type f | tee -a $BATCH_LOG

echo "" | tee -a $BATCH_LOG
echo "💡 To view a specific result:" | tee -a $BATCH_LOG
echo "   cat /root/autodl-fs/vsm13_output/<model>_<backbone>/vsm13_test_results_<model>_<backbone>.json | jq" | tee -a $BATCH_LOG

