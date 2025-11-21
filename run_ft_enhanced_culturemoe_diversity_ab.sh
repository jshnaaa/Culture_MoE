#!/bin/bash

# ============================================================
# Enhanced CultureMoE 多样性增强实验脚本
#
# 目标：增强专家功能分化，减少专家输出相似度
#
# 使用方法：
#   bash run_ft_enhanced_culturemoe_diversity_ab.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
#
# 示例：
#   bash run_ft_enhanced_culturemoe_diversity_ab.sh llama 4
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-4}"                   # 默认 CultureLLM (4)

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 根据 DATA_ID 选择数据集
case $DATA_ID in
    2)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        echo "Using CulturalBench dataset"
        ;;
    3)
        # NormAD
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        echo "Using NormAD dataset"
        ;;
    4)
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        echo "Using CultureLLM dataset"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 2, 3, or 4."
        exit 1
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ablation_results/diversity_${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Enhanced CultureMoE 多样性增强实验"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo ""
echo "多样性增强策略："
echo "  1. 增强多样性损失权重 (×10倍)"
echo "  2. 对抗性多样性训练"
echo "  3. 专家输出正交化约束"
echo "  4. 专家容量增加"
echo "  5. 差异化初始化"
echo ""
echo "多样性配置："
echo "  - Diversity weight: 0.5 (原来0.05)"
echo "  - Orthogonal constraint: 0.1 (新增)"
echo "  - Expert capacity: 增加50%"
echo "  - Contrastive learning: 启用"
echo "  - Adversarial diversity: 启用"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo ""
echo "Train file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Train file not found: $TRAIN_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/visualization"
mkdir -p "$OUTPUT_DIR/logs"

echo "Starting Enhanced CultureMoE diversity enhancement experiment..."
echo ""
echo "🎯 多样性增强设置："
echo "  - 对抗性训练：专家输出差异最大化"
echo "  - 正交化约束：强制专家学习正交特征"
echo "  - 对比学习：增强专家间的区分度"
echo "  - 容量扩展：为专家提供更大学习空间"
echo "  - 差异化初始化：避免专家收敛到相同解"
echo ""

# 运行多样性增强实验
python ft_enhanced_culturemoe_diversity_ab.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 10 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-4 \
    --weight_decay 0.001 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2 \
    --lora_r 96 \
    --lora_alpha 24 \
    --lora_dropout 0.1 \
    --num_experts 6 \
    --top_k 2 \
    --culture_loss_weight 1.0 \
    --load_balance_weight 0.01 \
    --entropy_weight 0.01 \
    --specialization_weight 0.1 \
    --diversity_weight 0.5 \
    --orthogonal_weight 0.1 \
    --contrastive_weight 0.2 \
    --adversarial_diversity_weight 0.1 \
    --expert_capacity_multiplier 1.5 \
    --diversity_enhancement_mode \
    --use_adversarial_diversity \
    --use_orthogonal_constraint \
    --use_contrastive_learning \
    --differentiated_initialization \
    --eval_interval 2 \
    --monitor_diversity \
    --save_diversity_evolution

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Enhanced CultureMoE 多样性增强实验完成！"
    echo "============================================================"
    echo "多样性增强结果保存到: $OUTPUT_DIR"
    echo ""
    echo "生成的多样性文件："
    echo "  - diversity_enhancement_report.json (多样性增强报告)"
    echo "  - expert_similarity_evolution.json (专家相似度演化)"
    echo "  - diversity_metrics.json (多样性指标)"
    echo "  - ablation_comparison.json (消融实验对比)"
    echo "  - visualization/ (可视化图表)"
    echo "    ├── similarity_reduction.png"
    echo "    ├── expert_diversity_evolution.png"
    echo "    ├── orthogonality_progress.png"
    echo "    └── ablation_improvement.png"
    echo ""
    echo "📊 关键多样性指标查看："
    echo "  1. 专家相似度降低："
    echo "     cat $OUTPUT_DIR/diversity_enhancement_report.json | python -c \"import sys,json; data=json.load(sys.stdin); print(f'相似度降低: {data[\\\"similarity_reduction\\\"]:.4f}')\" 2>/dev/null || echo '需要查看diversity_enhancement_report.json'"
    echo ""
    echo "  2. 正交性改善："
    echo "     cat $OUTPUT_DIR/diversity_enhancement_report.json | python -c \"import sys,json; data=json.load(sys.stdin); print(f'正交性提升: {data[\\\"orthogonality_improvement\\\"]:.4f}')\" 2>/dev/null || echo '需要查看diversity_enhancement_report.json'"
    echo ""
    echo "  3. 消融实验改善："
    echo "     cat $OUTPUT_DIR/ablation_comparison.json | python -c \"import sys,json; data=json.load(sys.stdin); print(f'消融效果增强: {data[\\\"ablation_improvement\\\"]:.4f}')\" 2>/dev/null || echo '需要查看ablation_comparison.json'"
    echo ""
    echo "💡 多样性增强效果评估："
    echo "  - 相似度降低 > 0.2: 专家功能分化显著改善"
    echo "  - 正交性提升 > 0.3: 专家学习到不同特征空间"
    echo "  - 消融效果增强 > 0.03: 消融实验结果更明显"
    echo ""
    echo "📈 下一步建议："
    echo "  1. 如果多样性问题已解决，运行文化信号增强实验"
    echo "  2. 如果仍需改进，尝试更激进的多样性约束"
    echo "  3. 查看专家功能分化详情: cat $OUTPUT_DIR/expert_similarity_evolution.json | python -m json.tool"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Enhanced CultureMoE 多样性增强实验失败！"
    echo "============================================================"
    echo "请检查错误信息并确认："
    echo "  1. 路由优化实验是否已完成"
    echo "  2. 模型是否支持多样性增强功能"
    echo "  3. 内存是否充足支持增强的专家容量"
    echo "  4. 多样性损失权重是否合适"
    echo ""
    echo "常见多样性增强问题："
    echo "  - 对抗性训练不稳定"
    echo "  - 正交化约束过强导致性能下降"
    echo "  - 对比学习与主任务冲突"
    echo "  - 专家容量增加导致过拟合"
    echo "============================================================"
    exit 1
fi