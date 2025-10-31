#!/bin/bash

# ============================================================
# 对比原始训练策略 vs 两阶段训练策略
# ============================================================

BACKBONE="${1:-llama}"
NUM_CLASSES="${2:-2}"
NUM_EXPERTS="${3:-6}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMPARISON_DIR="/root/autodl-fs/output/strategy_comparison_${TIMESTAMP}"
mkdir -p $COMPARISON_DIR

LOG_FILE="$COMPARISON_DIR/comparison_log.txt"

echo "============================================================" | tee $LOG_FILE
echo "Training Strategy Comparison" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "Backbone: $BACKBONE" | tee -a $LOG_FILE
echo "Num classes: $NUM_CLASSES" | tee -a $LOG_FILE
echo "Num experts: $NUM_EXPERTS" | tee -a $LOG_FILE
echo "Comparison directory: $COMPARISON_DIR" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ============================================================
# 策略 1：原始端到端训练
# ============================================================
echo "============================================================" | tee -a $LOG_FILE
echo "[Strategy 1] End-to-End Training (Original)" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

STRATEGY1_START=$(date +%s)

sh run_train_ddp_lora_dual.sh $BACKBONE $NUM_CLASSES True false 2>&1 | tee -a $LOG_FILE

STRATEGY1_END=$(date +%s)
STRATEGY1_DURATION=$((STRATEGY1_END - STRATEGY1_START))

echo "" | tee -a $LOG_FILE
echo "✅ Strategy 1 completed!" | tee -a $LOG_FILE
echo "Duration: $((STRATEGY1_DURATION / 60)) minutes $((STRATEGY1_DURATION % 60)) seconds" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 等待30秒
echo "Waiting 30 seconds before next strategy..." | tee -a $LOG_FILE
sleep 30

# ============================================================
# 策略 2：两阶段训练
# ============================================================
echo "============================================================" | tee -a $LOG_FILE
echo "[Strategy 2] Two-Stage Training (New)" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

STRATEGY2_START=$(date +%s)

sh run_two_stage_culturemoe.sh $BACKBONE $NUM_CLASSES $NUM_EXPERTS 2>&1 | tee -a $LOG_FILE

STRATEGY2_END=$(date +%s)
STRATEGY2_DURATION=$((STRATEGY2_END - STRATEGY2_START))

echo "" | tee -a $LOG_FILE
echo "✅ Strategy 2 completed!" | tee -a $LOG_FILE
echo "Duration: $((STRATEGY2_DURATION / 60)) minutes $((STRATEGY2_DURATION % 60)) seconds" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ============================================================
# 生成对比报告
# ============================================================
echo "============================================================" | tee -a $LOG_FILE
echo "Comparison Report" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# 查找最新的输出目录
STRATEGY1_DIR=$(find /root/autodl-fs/output/CultureMoE -name "culturemoe_${BACKBONE}_${NUM_CLASSES}class_*" -type d -mmin -$((STRATEGY1_DURATION / 60 + 10)) | sort -r | head -1)
STRATEGY2_DIR=$(find /root/autodl-fs/output/two_stage_moe -name "${BACKBONE}_${NUM_CLASSES}class_experts${NUM_EXPERTS}_*" -type d -mmin -$((STRATEGY2_DURATION / 60 + 10)) | sort -r | head -1)

echo "Strategy 1 output: $STRATEGY1_DIR" | tee -a $LOG_FILE
echo "Strategy 2 output: $STRATEGY2_DIR" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 提取结果
if [ -f "$STRATEGY1_DIR/epoch_eval_results.json" ]; then
    STRATEGY1_ACC=$(python3 -c "
import json
with open('$STRATEGY1_DIR/epoch_eval_results.json') as f:
    results = json.load(f)
best = max(results, key=lambda x: x.get('eval_accuracy', 0))
print(f\"{best['eval_accuracy']:.4f}\")
")
else
    STRATEGY1_ACC="N/A"
fi

if [ -f "$STRATEGY2_DIR/final_report.json" ]; then
    STRATEGY2_ACC=$(python3 -c "
import json
with open('$STRATEGY2_DIR/final_report.json') as f:
    report = json.load(f)
print(f\"{report['stage2']['eval_accuracy']:.4f}\")
")
    STRATEGY2_STAGE1_ACC=$(python3 -c "
import json
with open('$STRATEGY2_DIR/final_report.json') as f:
    report = json.load(f)
print(f\"{report['stage1']['eval_accuracy']:.4f}\")
")
else
    STRATEGY2_ACC="N/A"
    STRATEGY2_STAGE1_ACC="N/A"
fi

# 打印对比表格
echo "------------------------------------------------------------" | tee -a $LOG_FILE
echo "Performance Comparison" | tee -a $LOG_FILE
echo "------------------------------------------------------------" | tee -a $LOG_FILE
printf "%-30s | %-15s | %-15s\n" "Metric" "Strategy 1" "Strategy 2" | tee -a $LOG_FILE
echo "------------------------------------------------------------" | tee -a $LOG_FILE
printf "%-30s | %-15s | %-15s\n" "Training Time" "$((STRATEGY1_DURATION / 60))m ${STRATEGY1_DURATION % 60}s" "$((STRATEGY2_DURATION / 60))m ${STRATEGY2_DURATION % 60}s" | tee -a $LOG_FILE
printf "%-30s | %-15s | %-15s\n" "Final Accuracy" "$STRATEGY1_ACC" "$STRATEGY2_ACC" | tee -a $LOG_FILE
printf "%-30s | %-15s | %-15s\n" "Stage 1 Accuracy (LoRA)" "N/A" "$STRATEGY2_STAGE1_ACC" | tee -a $LOG_FILE
echo "------------------------------------------------------------" | tee -a $LOG_FILE

# 计算改进
if [ "$STRATEGY1_ACC" != "N/A" ] && [ "$STRATEGY2_ACC" != "N/A" ]; then
    IMPROVEMENT=$(python3 -c "print(f'{float('$STRATEGY2_ACC') - float('$STRATEGY1_ACC'):.4f}')")
    echo "" | tee -a $LOG_FILE
    echo "Improvement (Strategy 2 - Strategy 1): $IMPROVEMENT" | tee -a $LOG_FILE

    if (( $(echo "$IMPROVEMENT > 0" | bc -l) )); then
        echo "✅ Strategy 2 is better!" | tee -a $LOG_FILE
    elif (( $(echo "$IMPROVEMENT < 0" | bc -l) )); then
        echo "⚠️  Strategy 1 is better!" | tee -a $LOG_FILE
    else
        echo "➖ Both strategies are equal" | tee -a $LOG_FILE
    fi
fi

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "Comparison completed!" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "Log saved to: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 生成 JSON 报告
python3 << EOF
import json

report = {
    "timestamp": "$TIMESTAMP",
    "config": {
        "backbone": "$BACKBONE",
        "num_classes": $NUM_CLASSES,
        "num_experts": $NUM_EXPERTS
    },
    "strategy1": {
        "name": "End-to-End Training",
        "duration_seconds": $STRATEGY1_DURATION,
        "accuracy": "$STRATEGY1_ACC",
        "output_dir": "$STRATEGY1_DIR"
    },
    "strategy2": {
        "name": "Two-Stage Training",
        "duration_seconds": $STRATEGY2_DURATION,
        "stage1_accuracy": "$STRATEGY2_STAGE1_ACC",
        "stage2_accuracy": "$STRATEGY2_ACC",
        "output_dir": "$STRATEGY2_DIR"
    }
}

if "$STRATEGY1_ACC" != "N/A" and "$STRATEGY2_ACC" != "N/A":
    report["improvement"] = float("$STRATEGY2_ACC") - float("$STRATEGY1_ACC")

with open("$COMPARISON_DIR/comparison_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("JSON report saved to: $COMPARISON_DIR/comparison_report.json")
EOF

