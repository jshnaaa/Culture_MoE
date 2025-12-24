#!/bin/bash

# 简化版CultureMoE消融实验批处理脚本
#
# 使用方法：
#   bash run_ablation_study.sh /path/to/trained_model /path/to/data.json /path/to/output

echo "======================================="
echo "简化版CultureMoE消融实验"
echo "自动化测试不同组件配置的性能"
echo "======================================="

# 参数检查
if [ "$#" -ne 3 ]; then
    echo "❌ 用法: $0 <model_path> <data_file> <output_dir>"
    echo ""
    echo "参数说明:"
    echo "  model_path: 训练好的模型路径（包含best_simplified_culturemoe目录）"
    echo "  data_file:  评估数据文件路径"
    echo "  output_dir: 输出结果目录"
    echo ""
    echo "示例:"
    echo "  bash run_ablation_study.sh \\"
    echo "    /root/autodl-fs/simplified_culturemoe/llama_CulturalBench_20251224_153803/best_simplified_culturemoe \\"
    echo "    /root/autodl-fs/CulturalBench_merge_gen.json \\"
    echo "    /root/autodl-fs/ablation_results"
    exit 1
fi

MODEL_PATH="$1"
DATA_FILE="$2"
OUTPUT_DIR="$3"

# 验证路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 数据文件不存在: $DATA_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="$OUTPUT_DIR/ablation_$TIMESTAMP"
mkdir -p "$EXPERIMENT_DIR"

echo "配置信息:"
echo "  模型路径: $MODEL_PATH"
echo "  数据文件: $DATA_FILE"
echo "  输出目录: $EXPERIMENT_DIR"
echo ""

# 保存实验配置
cat > "$EXPERIMENT_DIR/experiment_config.json" << EOF
{
    "model_path": "$MODEL_PATH",
    "data_file": "$DATA_FILE",
    "output_dir": "$EXPERIMENT_DIR",
    "timestamp": "$TIMESTAMP",
    "experiments": [
        {"name": "full", "disable_shared": false, "disable_gate": false, "description": "完整模型（所有组件）"},
        {"name": "no_shared", "disable_shared": true, "disable_gate": false, "description": "无共享专家"},
        {"name": "no_gate", "disable_shared": false, "disable_gate": true, "description": "无Gate网络"},
        {"name": "no_shared_no_gate", "disable_shared": true, "disable_gate": true, "description": "无共享专家和Gate网络"}
    ]
}
EOF

echo "🔄 开始消融实验..."
echo ""

# 实验1: 完整模型（基线）
echo "📊 实验1: 完整模型（基线）"
echo "----------------------------------------"
python eval_simplified_culturemoe.py \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_dir "$EXPERIMENT_DIR" \
    --experiment_name "full" \
    --use_fixed_split

EXPERIMENT_1_SUCCESS=$?
if [ $EXPERIMENT_1_SUCCESS -eq 0 ]; then
    echo "✅ 实验1完成"
else
    echo "❌ 实验1失败"
fi
echo ""

# 实验2: 无共享专家
echo "📊 实验2: 无共享专家"
echo "----------------------------------------"
python eval_simplified_culturemoe.py \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_dir "$EXPERIMENT_DIR" \
    --disable_shared \
    --experiment_name "no_shared" \
    --use_fixed_split

EXPERIMENT_2_SUCCESS=$?
if [ $EXPERIMENT_2_SUCCESS -eq 0 ]; then
    echo "✅ 实验2完成"
else
    echo "❌ 实验2失败"
fi
echo ""

# 实验3: 无Gate网络
echo "📊 实验3: 无Gate网络"
echo "----------------------------------------"
python eval_simplified_culturemoe.py \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_dir "$EXPERIMENT_DIR" \
    --disable_gate \
    --experiment_name "no_gate" \
    --use_fixed_split

EXPERIMENT_3_SUCCESS=$?
if [ $EXPERIMENT_3_SUCCESS -eq 0 ]; then
    echo "✅ 实验3完成"
else
    echo "❌ 实验3失败"
fi
echo ""

# 实验4: 无共享专家和Gate网络
echo "📊 实验4: 无共享专家和Gate网络"
echo "----------------------------------------"
python eval_simplified_culturemoe.py \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_dir "$EXPERIMENT_DIR" \
    --disable_shared \
    --disable_gate \
    --experiment_name "no_shared_no_gate" \
    --use_fixed_split

EXPERIMENT_4_SUCCESS=$?
if [ $EXPERIMENT_4_SUCCESS -eq 0 ]; then
    echo "✅ 实验4完成"
else
    echo "❌ 实验4失败"
fi
echo ""

# 汇总结果
echo "======================================="
echo "📊 消融实验结果汇总"
echo "======================================="

# 创建结果汇总脚本
python << EOF
import json
import os
import pandas as pd

experiment_dir = "$EXPERIMENT_DIR"
results = []

# 读取各个实验的结果
experiments = [
    {"name": "full", "description": "完整模型（基线）"},
    {"name": "no_shared", "description": "无共享专家"},
    {"name": "no_gate", "description": "无Gate网络"},
    {"name": "no_shared_no_gate", "description": "无共享专家和Gate网络"}
]

print("实验结果对比:")
print("-" * 80)
print(f"{'实验名称':<20} {'描述':<25} {'准确率':<10} {'损失':<10} {'文化损失':<10}")
print("-" * 80)

for exp in experiments:
    result_file = os.path.join(experiment_dir, f"evaluation_results_{exp['name']}.json")

    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)

        accuracy = result.get('accuracy', 0)
        eval_loss = result.get('eval_loss', 0)
        culture_loss = result.get('eval_culture_loss', 0)

        print(f"{exp['name']:<20} {exp['description']:<25} {accuracy:<10.4f} {eval_loss:<10.4f} {culture_loss:<10.4f}")

        results.append({
            'experiment': exp['name'],
            'description': exp['description'],
            'accuracy': accuracy,
            'eval_loss': eval_loss,
            'culture_loss': culture_loss,
            'correct': result.get('correct', 0),
            'total': result.get('total', 0)
        })
    else:
        print(f"{exp['name']:<20} {exp['description']:<25} {'失败':<10} {'N/A':<10} {'N/A':<10}")

print("-" * 80)

# 保存汇总结果
if results:
    summary_file = os.path.join(experiment_dir, "ablation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 计算性能变化
    baseline = next((r for r in results if r['experiment'] == 'full'), None)
    if baseline:
        print(f"\n性能变化分析（相对于基线 {baseline['accuracy']:.4f}）:")
        print("-" * 60)
        for result in results:
            if result['experiment'] != 'full':
                accuracy_change = result['accuracy'] - baseline['accuracy']
                change_pct = (accuracy_change / baseline['accuracy']) * 100
                change_sign = "+" if accuracy_change > 0 else ""
                print(f"{result['description']:<25}: {change_sign}{accuracy_change:.4f} ({change_sign}{change_pct:.2f}%)")
        print("-" * 60)

    print(f"\n✅ 详细结果已保存到: {summary_file}")

EOF

# 统计成功的实验数量
TOTAL_SUCCESS=$((EXPERIMENT_1_SUCCESS == 0 ? 1 : 0))
TOTAL_SUCCESS=$((TOTAL_SUCCESS + (EXPERIMENT_2_SUCCESS == 0 ? 1 : 0)))
TOTAL_SUCCESS=$((TOTAL_SUCCESS + (EXPERIMENT_3_SUCCESS == 0 ? 1 : 0)))
TOTAL_SUCCESS=$((TOTAL_SUCCESS + (EXPERIMENT_4_SUCCESS == 0 ? 1 : 0)))

echo ""
echo "实验完成情况:"
echo "  实验1 (完整模型): $([ $EXPERIMENT_1_SUCCESS -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
echo "  实验2 (无共享专家): $([ $EXPERIMENT_2_SUCCESS -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
echo "  实验3 (无Gate网络): $([ $EXPERIMENT_3_SUCCESS -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
echo "  实验4 (无共享+Gate): $([ $EXPERIMENT_4_SUCCESS -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
echo ""
echo "成功完成: $TOTAL_SUCCESS/4 个实验"

echo ""
echo "📁 所有结果文件位置:"
echo "  实验目录: $EXPERIMENT_DIR"
echo "  配置文件: $EXPERIMENT_DIR/experiment_config.json"
echo "  汇总结果: $EXPERIMENT_DIR/ablation_summary.json"
echo "  详细结果: $EXPERIMENT_DIR/evaluation_results_*.json"
echo "  生成答案: $EXPERIMENT_DIR/generated_answers_*.json"
echo ""

if [ $TOTAL_SUCCESS -eq 4 ]; then
    echo "🎉 所有消融实验成功完成！"
    echo "======================================="
    exit 0
else
    echo "⚠️ 部分实验失败，请检查日志"
    echo "======================================="
    exit 1
fi