#!/bin/bash

# ============================================================
# 搜索 CultureMoE 最佳专家数
# ============================================================
# 阶段 1：粗粒度搜索（2, 4, 6, 8, 12 个专家）
# 阶段 2：细粒度搜索（在最佳值附近）
# 阶段 3：使用最佳专家数训练完整模型
# ============================================================

set -e  # 遇到错误立即退出

# ============================================================
# 配置参数
# ============================================================
BACKBONE="llama"
NUM_CLASSES=2
TRAIN_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEARCH_OUTPUT_DIR="/root/autodl-fs/output/expert_search_${TIMESTAMP}"
mkdir -p $SEARCH_OUTPUT_DIR

# 日志文件
SEARCH_LOG="$SEARCH_OUTPUT_DIR/search_log.txt"
RESULTS_JSON="$SEARCH_OUTPUT_DIR/search_results.json"

echo "============================================================" | tee $SEARCH_LOG
echo "CultureMoE Expert Number Search" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG
echo "Start time: $(date)" | tee -a $SEARCH_LOG
echo "Backbone: $BACKBONE" | tee -a $SEARCH_LOG
echo "Num classes: $NUM_CLASSES" | tee -a $SEARCH_LOG
echo "Dataset: $TRAIN_FILE" | tee -a $SEARCH_LOG
echo "Output directory: $SEARCH_OUTPUT_DIR" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG

# ============================================================
# 阶段 1：粗粒度搜索
# ============================================================
echo "============================================================" | tee -a $SEARCH_LOG
echo "Phase 1: Coarse-grained Search" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG
echo "Testing num_experts: 2, 4, 6, 8, 12" | tee -a $SEARCH_LOG
echo "Training: 5 epochs, no model saving" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG

# 粗粒度搜索的专家数列表
COARSE_EXPERTS=(2 4 6 8 12)

# 存储每个专家数的最佳准确率
declare -A BEST_ACCURACY
declare -A BEST_EPOCH

# 遍历每个专家数
for NUM_EXPERTS in "${COARSE_EXPERTS[@]}"; do
    echo "------------------------------------------------------------" | tee -a $SEARCH_LOG
    echo "[Phase 1] Testing num_experts=$NUM_EXPERTS" | tee -a $SEARCH_LOG
    echo "Start time: $(date)" | tee -a $SEARCH_LOG
    echo "------------------------------------------------------------" | tee -a $SEARCH_LOG

    # 创建输出目录
    OUTPUT_DIR="$SEARCH_OUTPUT_DIR/phase1_experts_${NUM_EXPERTS}"
    mkdir -p $OUTPUT_DIR

    # 修改配置文件中的专家数
    # 假设配置在 src/llamafactory/model/moe_args.py 中
    # 这里使用临时环境变量传递
    export NUM_EXPERTS=$NUM_EXPERTS

    # 运行训练
    echo "Running training with $NUM_EXPERTS experts..." | tee -a $SEARCH_LOG

    torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        examples/train_classification.py \
        --model_name_or_path $MODEL_PATH \
        --train_file $TRAIN_FILE \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 5 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-5 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --fp16 \
        --gradient_checkpointing \
        --use_dual_input True \
        --freeze_llama False \
        --use_llama_lora True \
        --llama_lora_rank 16 \
        --llama_lora_alpha 32 \
        --llama_lora_dropout 0.1 \
        --llama_lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
        --logging_steps 10 \
        --eval_steps 100 \
        --max_length 512 \
        --val_split 0.1 \
        --dataloader_num_workers 4 \
        --culture_loss_lambda 0.05 \
        --num_classes $NUM_CLASSES \
        --use_culture_loss True \
        --num_experts $NUM_EXPERTS \
        2>&1 | tee -a $SEARCH_LOG

    # 提取最佳准确率
    if [ -f "$OUTPUT_DIR/epoch_eval_results.json" ]; then
        # 使用 Python 提取最佳准确率
        BEST_ACC=$(python3 -c "
import json
import sys

try:
    with open('$OUTPUT_DIR/epoch_eval_results.json', 'r') as f:
        results = json.load(f)

    if not results:
        print('0.0')
        sys.exit(0)

    # 找到最高准确率
    best_result = max(results, key=lambda x: x.get('eval_accuracy', 0))
    best_acc = best_result.get('eval_accuracy', 0)
    best_epoch = best_result.get('epoch', 0)

    print(f'{best_acc:.4f}')
    print(f'{best_epoch}', file=sys.stderr)
except Exception as e:
    print('0.0')
    print('0', file=sys.stderr)
" 2>&1)

        BEST_EPOCH_VAL=$(echo "$BEST_ACC" | tail -1)
        BEST_ACC_VAL=$(echo "$BEST_ACC" | head -1)

        BEST_ACCURACY[$NUM_EXPERTS]=$BEST_ACC_VAL
        BEST_EPOCH[$NUM_EXPERTS]=$BEST_EPOCH_VAL

        echo "" | tee -a $SEARCH_LOG
        echo "✅ Completed: num_experts=$NUM_EXPERTS" | tee -a $SEARCH_LOG
        echo "   Best accuracy: $BEST_ACC_VAL (at epoch $BEST_EPOCH_VAL)" | tee -a $SEARCH_LOG
        echo "   Results saved to: $OUTPUT_DIR/epoch_eval_results.json" | tee -a $SEARCH_LOG
    else
        echo "❌ Error: epoch_eval_results.json not found" | tee -a $SEARCH_LOG
        BEST_ACCURACY[$NUM_EXPERTS]=0.0
        BEST_EPOCH[$NUM_EXPERTS]=0
    fi

    echo "" | tee -a $SEARCH_LOG

    # 等待 30 秒
    echo "Waiting 30 seconds before next experiment..." | tee -a $SEARCH_LOG
    sleep 30
done

# ============================================================
# 分析粗粒度搜索结果
# ============================================================
echo "============================================================" | tee -a $SEARCH_LOG
echo "Phase 1 Results Summary" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG

# 打印所有结果
echo "Num Experts | Best Accuracy | Best Epoch" | tee -a $SEARCH_LOG
echo "------------|---------------|------------" | tee -a $SEARCH_LOG
for NUM_EXPERTS in "${COARSE_EXPERTS[@]}"; do
    printf "%11d | %13s | %10s\n" $NUM_EXPERTS "${BEST_ACCURACY[$NUM_EXPERTS]}" "${BEST_EPOCH[$NUM_EXPERTS]}" | tee -a $SEARCH_LOG
done
echo "============================================================" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG

# 找到最佳专家数
BEST_NUM_EXPERTS=0
BEST_OVERALL_ACC=0.0

for NUM_EXPERTS in "${COARSE_EXPERTS[@]}"; do
    ACC=${BEST_ACCURACY[$NUM_EXPERTS]}
    if (( $(echo "$ACC > $BEST_OVERALL_ACC" | bc -l) )); then
        BEST_OVERALL_ACC=$ACC
        BEST_NUM_EXPERTS=$NUM_EXPERTS
    fi
done

echo "🏆 Best num_experts from Phase 1: $BEST_NUM_EXPERTS (accuracy: $BEST_OVERALL_ACC)" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG

# 保存粗粒度搜索结果到 JSON
python3 -c "
import json

results = {
    'phase': 'coarse_grained',
    'tested_experts': [2, 4, 6, 8, 12],
    'results': {}
}

for num_experts in [2, 4, 6, 8, 12]:
    results['results'][str(num_experts)] = {
        'best_accuracy': ${BEST_ACCURACY[$num_experts]:-0.0},
        'best_epoch': ${BEST_EPOCH[$num_experts]:-0}
    }

results['best_num_experts'] = $BEST_NUM_EXPERTS
results['best_accuracy'] = $BEST_OVERALL_ACC

with open('$SEARCH_OUTPUT_DIR/phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)
"

# ============================================================
# 阶段 2：细粒度搜索
# ============================================================
echo "============================================================" | tee -a $SEARCH_LOG
echo "Phase 2: Fine-grained Search" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG
echo "Searching around best value: $BEST_NUM_EXPERTS" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG

# 确定细粒度搜索范围
if [ $BEST_NUM_EXPERTS -eq 2 ]; then
    FINE_EXPERTS=(3)
elif [ $BEST_NUM_EXPERTS -eq 12 ]; then
    FINE_EXPERTS=(10 11)
else
    # 在最佳值附近搜索
    LOWER=$((BEST_NUM_EXPERTS - 1))
    UPPER=$((BEST_NUM_EXPERTS + 1))
    FINE_EXPERTS=()

    # 只测试粗粒度搜索中没有测试过的值
    for i in $(seq $LOWER $UPPER); do
        if [[ ! " ${COARSE_EXPERTS[@]} " =~ " ${i} " ]]; then
            FINE_EXPERTS+=($i)
        fi
    done
fi

if [ ${#FINE_EXPERTS[@]} -eq 0 ]; then
    echo "No additional values to test in fine-grained search." | tee -a $SEARCH_LOG
    echo "Skipping Phase 2." | tee -a $SEARCH_LOG
else
    echo "Testing num_experts: ${FINE_EXPERTS[@]}" | tee -a $SEARCH_LOG
    echo "" | tee -a $SEARCH_LOG

    for NUM_EXPERTS in "${FINE_EXPERTS[@]}"; do
        echo "------------------------------------------------------------" | tee -a $SEARCH_LOG
        echo "[Phase 2] Testing num_experts=$NUM_EXPERTS" | tee -a $SEARCH_LOG
        echo "Start time: $(date)" | tee -a $SEARCH_LOG
        echo "------------------------------------------------------------" | tee -a $SEARCH_LOG

        OUTPUT_DIR="$SEARCH_OUTPUT_DIR/phase2_experts_${NUM_EXPERTS}"
        mkdir -p $OUTPUT_DIR

        export NUM_EXPERTS=$NUM_EXPERTS

        echo "Running training with $NUM_EXPERTS experts..." | tee -a $SEARCH_LOG

        torchrun \
            --nproc_per_node=2 \
            --master_port=29500 \
            examples/train_classification.py \
            --model_name_or_path $MODEL_PATH \
            --train_file $TRAIN_FILE \
            --output_dir $OUTPUT_DIR \
            --num_train_epochs 5 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-5 \
            --weight_decay 0.01 \
            --warmup_ratio 0.1 \
            --fp16 \
            --gradient_checkpointing \
            --use_dual_input True \
            --freeze_llama False \
            --use_llama_lora True \
            --llama_lora_rank 16 \
            --llama_lora_alpha 32 \
            --llama_lora_dropout 0.1 \
            --llama_lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
            --logging_steps 10 \
            --eval_steps 100 \
            --max_length 512 \
            --val_split 0.1 \
            --dataloader_num_workers 4 \
            --culture_loss_lambda 0.05 \
            --num_classes $NUM_CLASSES \
            --use_culture_loss True \
            --num_experts $NUM_EXPERTS \
            2>&1 | tee -a $SEARCH_LOG

        # 提取最佳准确率
        if [ -f "$OUTPUT_DIR/epoch_eval_results.json" ]; then
            BEST_ACC=$(python3 -c "
import json
import sys

try:
    with open('$OUTPUT_DIR/epoch_eval_results.json', 'r') as f:
        results = json.load(f)

    if not results:
        print('0.0')
        sys.exit(0)

    best_result = max(results, key=lambda x: x.get('eval_accuracy', 0))
    best_acc = best_result.get('eval_accuracy', 0)
    best_epoch = best_result.get('epoch', 0)

    print(f'{best_acc:.4f}')
    print(f'{best_epoch}', file=sys.stderr)
except Exception as e:
    print('0.0')
    print('0', file=sys.stderr)
" 2>&1)

            BEST_EPOCH_VAL=$(echo "$BEST_ACC" | tail -1)
            BEST_ACC_VAL=$(echo "$BEST_ACC" | head -1)

            BEST_ACCURACY[$NUM_EXPERTS]=$BEST_ACC_VAL
            BEST_EPOCH[$NUM_EXPERTS]=$BEST_EPOCH_VAL

            echo "" | tee -a $SEARCH_LOG
            echo "✅ Completed: num_experts=$NUM_EXPERTS" | tee -a $SEARCH_LOG
            echo "   Best accuracy: $BEST_ACC_VAL (at epoch $BEST_EPOCH_VAL)" | tee -a $SEARCH_LOG

            # 更新全局最佳值
            if (( $(echo "$BEST_ACC_VAL > $BEST_OVERALL_ACC" | bc -l) )); then
                BEST_OVERALL_ACC=$BEST_ACC_VAL
                BEST_NUM_EXPERTS=$NUM_EXPERTS
                echo "   🏆 New best!" | tee -a $SEARCH_LOG
            fi
        else
            echo "❌ Error: epoch_eval_results.json not found" | tee -a $SEARCH_LOG
        fi

        echo "" | tee -a $SEARCH_LOG
        sleep 30
    done

    # 打印细粒度搜索结果
    echo "============================================================" | tee -a $SEARCH_LOG
    echo "Phase 2 Results Summary" | tee -a $SEARCH_LOG
    echo "============================================================" | tee -a $SEARCH_LOG
    echo "Num Experts | Best Accuracy | Best Epoch" | tee -a $SEARCH_LOG
    echo "------------|---------------|------------" | tee -a $SEARCH_LOG
    for NUM_EXPERTS in "${FINE_EXPERTS[@]}"; do
        printf "%11d | %13s | %10s\n" $NUM_EXPERTS "${BEST_ACCURACY[$NUM_EXPERTS]}" "${BEST_EPOCH[$NUM_EXPERTS]}" | tee -a $SEARCH_LOG
    done
    echo "============================================================" | tee -a $SEARCH_LOG
    echo "" | tee -a $SEARCH_LOG
fi

echo "🏆 Final best num_experts: $BEST_NUM_EXPERTS (accuracy: $BEST_OVERALL_ACC)" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG

# ============================================================
# 阶段 3：使用最佳专家数训练完整模型
# ============================================================
echo "============================================================" | tee -a $SEARCH_LOG
echo "Phase 3: Training Final Model" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG
echo "Using best num_experts: $BEST_NUM_EXPERTS" | tee -a $SEARCH_LOG
echo "Training: 10 epochs" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG

FINAL_OUTPUT_DIR="$SEARCH_OUTPUT_DIR/final_model_experts_${BEST_NUM_EXPERTS}"
mkdir -p $FINAL_OUTPUT_DIR

export NUM_EXPERTS=$BEST_NUM_EXPERTS

echo "Running final training with $BEST_NUM_EXPERTS experts..." | tee -a $SEARCH_LOG

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    examples/train_classification.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $FINAL_OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --fp16 \
    --gradient_checkpointing \
    --use_dual_input True \
    --freeze_llama False \
    --use_llama_lora True \
    --llama_lora_rank 16 \
    --llama_lora_alpha 32 \
    --llama_lora_dropout 0.1 \
    --llama_lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
    --logging_steps 10 \
    --eval_steps 100 \
    --max_length 512 \
    --val_split 0.1 \
    --dataloader_num_workers 4 \
    --culture_loss_lambda 0.05 \
    --num_classes $NUM_CLASSES \
    --use_culture_loss True \
    --num_experts $BEST_NUM_EXPERTS \
    2>&1 | tee -a $SEARCH_LOG

echo "" | tee -a $SEARCH_LOG
echo "✅ Final model training completed!" | tee -a $SEARCH_LOG
echo "   Results saved to: $FINAL_OUTPUT_DIR/epoch_eval_results.json" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG

# ============================================================
# 生成最终报告
# ============================================================
echo "============================================================" | tee -a $SEARCH_LOG
echo "Final Report" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG
echo "Search completed at: $(date)" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG
echo "🏆 Best num_experts: $BEST_NUM_EXPERTS" | tee -a $SEARCH_LOG
echo "📊 Best accuracy (5 epochs): $BEST_OVERALL_ACC" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG
echo "📁 Output directory: $SEARCH_OUTPUT_DIR" | tee -a $SEARCH_LOG
echo "📄 Search log: $SEARCH_LOG" | tee -a $SEARCH_LOG
echo "📊 Final model results: $FINAL_OUTPUT_DIR/epoch_eval_results.json" | tee -a $SEARCH_LOG
echo "============================================================" | tee -a $SEARCH_LOG

# 生成 JSON 报告
python3 << EOF
import json
import os

# 收集所有结果
all_results = {}

# 粗粒度搜索结果
for num_experts in [2, 4, 6, 8, 12]:
    phase1_file = f"$SEARCH_OUTPUT_DIR/phase1_experts_{num_experts}/epoch_eval_results.json"
    if os.path.exists(phase1_file):
        with open(phase1_file, 'r') as f:
            results = json.load(f)
        all_results[num_experts] = {
            'phase': 'coarse',
            'epochs': results,
            'best_accuracy': max(r['eval_accuracy'] for r in results),
            'best_epoch': max(results, key=lambda x: x['eval_accuracy'])['epoch']
        }

# 细粒度搜索结果
for root, dirs, files in os.walk("$SEARCH_OUTPUT_DIR"):
    if 'phase2_experts_' in root and 'epoch_eval_results.json' in files:
        num_experts = int(root.split('_')[-1])
        with open(os.path.join(root, 'epoch_eval_results.json'), 'r') as f:
            results = json.load(f)
        all_results[num_experts] = {
            'phase': 'fine',
            'epochs': results,
            'best_accuracy': max(r['eval_accuracy'] for r in results),
            'best_epoch': max(results, key=lambda x: x['eval_accuracy'])['epoch']
        }

# 最终模型结果
final_file = f"$FINAL_OUTPUT_DIR/epoch_eval_results.json"
if os.path.exists(final_file):
    with open(final_file, 'r') as f:
        final_results = json.load(f)
    final_best = max(final_results, key=lambda x: x['eval_accuracy'])
else:
    final_results = []
    final_best = {}

# 生成报告
report = {
    'search_timestamp': '$TIMESTAMP',
    'best_num_experts': $BEST_NUM_EXPERTS,
    'best_accuracy_5epochs': $BEST_OVERALL_ACC,
    'final_model': {
        'num_experts': $BEST_NUM_EXPERTS,
        'epochs': 10,
        'best_accuracy': final_best.get('eval_accuracy', 0),
        'best_epoch': final_best.get('epoch', 0),
        'all_epochs': final_results
    },
    'search_results': {
        str(k): {
            'phase': v['phase'],
            'best_accuracy': v['best_accuracy'],
            'best_epoch': v['best_epoch']
        }
        for k, v in sorted(all_results.items())
    },
    'output_directory': '$SEARCH_OUTPUT_DIR'
}

with open('$SEARCH_OUTPUT_DIR/final_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("Final Report saved to: $SEARCH_OUTPUT_DIR/final_report.json")
print("="*60)
EOF

echo "" | tee -a $SEARCH_LOG
echo "✅ All tasks completed successfully!" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG
echo "To view results:" | tee -a $SEARCH_LOG
echo "  cat $SEARCH_OUTPUT_DIR/final_report.json | jq" | tee -a $SEARCH_LOG
echo "" | tee -a $SEARCH_LOG

