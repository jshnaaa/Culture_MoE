#!/bin/bash

# ============================================================
# Enhanced CultureMoE 路由优化实验脚本
#
# 目标：修复路由塌陷问题，改善专家利用率分布
#
# 使用方法：
#   bash run_ft_enhanced_culturemoe_routing_fix_ab.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
#
# 示例：
#   bash run_ft_enhanced_culturemoe_routing_fix_ab.sh llama 4
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
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ablation_results/routing_fix_${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Enhanced CultureMoE 路由优化实验"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo ""
echo "路由优化策略："
echo "  1. 增强负载均衡损失权重 (×5倍)"
echo "  2. 增加路由熵正则化"
echo "  3. 训练初期冻结shared专家"
echo "  4. 提高路由器学习率 (×2倍)"
echo "  5. Gumbel-Softmax温度调度"
echo ""
echo "优化配置："
echo "  - Load balance weight: 0.05 (原来0.01)"
echo "  - Entropy weight: 0.1 (新增)"
echo "  - Router learning rate: 4e-4 (×2倍)"
echo "  - Shared freeze epochs: 1"
echo "  - Gumbel temperature: 2.0->0.5"
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

echo "Starting Enhanced CultureMoE routing optimization experiment..."
echo ""
echo "🔧 路由优化设置："
echo "  - 阶段1：冻结shared专家，训练路由器和其他专家 (1 epoch)"
echo "  - 阶段2：解冻所有参数，联合训练 (剩余epochs)"
echo "  - 增强的负载均衡和熵正则化"
echo "  - 自适应温度调度促进专家分化"
echo "  - 详细的路由监控和可视化"
echo ""

# 运行路由优化实验
python ft_enhanced_culturemoe_routing_fix_ab.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 8 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-4 \
    --router_learning_rate 4e-4 \
    --weight_decay 0.001 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_experts 6 \
    --top_k 2 \
    --culture_loss_weight 1.0 \
    --load_balance_weight 0.05 \
    --entropy_weight 0.1 \
    --specialization_weight 0.1 \
    --diversity_weight 0.05 \
    --shared_freeze_epochs 1 \
    --gumbel_temperature_start 2.0 \
    --gumbel_temperature_end 0.5 \
    --temperature_decay_steps 1000 \
    --eval_interval 2 \
    --routing_fix_mode \
    --monitor_routing \
    --save_routing_evolution

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Enhanced CultureMoE 路由优化实验完成！"
    echo "============================================================"
    echo "优化结果保存到: $OUTPUT_DIR"
    echo ""
    echo "生成的优化文件："
    echo "  - routing_optimization_report.json (优化效果报告)"
    echo "  - routing_evolution.json (路由演化过程)"
    echo "  - expert_utilization_comparison.json (专家利用率对比)"
    echo "  - ablation_results.json (消融实验结果)"
    echo "  - visualization/ (可视化图表)"
    echo "    ├── routing_evolution.png"
    echo "    ├── expert_utilization_improvement.png"
    echo "    ├── entropy_evolution.png"
    echo "    └── ablation_comparison.png"
    echo ""
    echo "📊 关键优化指标查看："
    echo "  1. 专家利用率改善："
    echo "     cat $OUTPUT_DIR/routing_optimization_report.json | python -c \"import sys,json; data=json.load(sys.stdin); print(f'Gini改善: {data[\\\"gini_improvement\\\"]:.4f}')\" 2>/dev/null || echo '需要查看routing_optimization_report.json'"
    echo ""
    echo "  2. 路由熵提升："
    echo "     cat $OUTPUT_DIR/routing_optimization_report.json | python -c \"import sys,json; data=json.load(sys.stdin); print(f'熵提升: {data[\\\"entropy_improvement\\\"]:.4f}')\" 2>/dev/null || echo '需要查看routing_optimization_report.json'"
    echo ""
    echo "  3. 消融实验改善："
    echo "     cat $OUTPUT_DIR/ablation_results.json | python -c \"import sys,json; data=json.load(sys.stdin); print(f'消融差异扩大: {data[\\\"max_ablation_difference\\\"]:.4f}')\" 2>/dev/null || echo '需要查看ablation_results.json'"
    echo ""
    echo "💡 优化效果评估："
    echo "  - Gini改善 > 0.2: 专家利用率显著改善"
    echo "  - 熵提升 > 0.5: 路由分布更加均衡"
    echo "  - 消融差异 > 0.02: 消融实验效果更明显"
    echo ""
    echo "📈 下一步建议："
    echo "  1. 如果路由问题已解决，运行多样性增强实验"
    echo "  2. 如果仍有问题，尝试更激进的优化参数"
    echo "  3. 查看详细演化过程: cat $OUTPUT_DIR/routing_evolution.json | python -m json.tool"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Enhanced CultureMoE 路由优化实验失败！"
    echo "============================================================"
    echo "请检查错误信息并确认："
    echo "  1. 诊断实验是否已完成并确认存在路由问题"
    echo "  2. 模型路径和数据文件是否正确"
    echo "  3. 内存是否充足支持优化实验"
    echo "  4. 所有依赖模块是否正确安装"
    echo ""
    echo "常见路由优化问题："
    echo "  - 学习率过高导致训练不稳定"
    echo "  - 温度调度参数不当"
    echo "  - 负载均衡权重过大影响主任务"
    echo "  - 冻结策略与模型架构不兼容"
    echo "============================================================"
    exit 1
fi