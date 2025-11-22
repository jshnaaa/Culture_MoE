#!/bin/bash

# ============================================================
# Enhanced CultureMoE 诊断实验脚本
#
# 目标：诊断路由塌陷和专家功能重叠问题
#
# 使用方法：
#   bash run_ft_enhanced_culturemoe_diagnostic_ab.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
#
# 示例：
#   bash run_ft_enhanced_culturemoe_diagnostic_ab.sh llama 4
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
OUTPUT_DIR="/root/autodl-fs/data/ablation_results/diagnostic_${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "Enhanced CultureMoE 诊断实验"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME"
echo ""
echo "诊断目标："
echo "  1. 路由分布分析 - 检测专家利用率不均衡"
echo "  2. 专家输出相似度 - 检测功能重叠"
echo "  3. 文化信号强度 - 评估数据集文化差异"
echo "  4. 按国家分组分析 - 识别文化特异性"
echo ""
echo "诊断配置："
echo "  - 短期训练：3个epoch（快速诊断）"
echo "  - 详细日志：记录每个batch的路由分布"
echo "  - 专家分析：计算输出相似度和激活模式"
echo "  - 统计分析：Gini系数、熵、JS散度等"
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

echo "Starting Enhanced CultureMoE diagnostic experiment..."
echo ""
echo "🔍 诊断实验设置："
echo "  - 快速训练模式：3个epoch"
echo "  - 详细路由日志：每个batch记录专家分配"
echo "  - 专家输出保存：用于后续相似度分析"
echo "  - 按国家统计：分析文化特异性路由模式"
echo "  - 可视化生成：专家利用率和相似度热图"
echo ""

# 设置单卡运行和内存优化环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

echo "🔧 强制单卡运行和内存优化设置:"
echo "  - CUDA_VISIBLE_DEVICES=0 (只使用第一个GPU)"
echo "  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True (内存碎片优化)"
echo "  - PYTORCH_NO_CUDA_MEMORY_CACHING=1 (禁用内存缓存)"
echo ""
echo "🏭 完整生产架构配置:"
echo "  - 专家数量=12, LoRA rank=32 (与生产一致)"
echo "  - 专家隐藏维度=4096, 路由器隐藏维度=2048 (与生产一致)"
echo "  - 文化数量=6, 文化维度=256 (与生产一致)"
echo "  - 门控机制=启用 (与生产一致)"
echo ""
echo "⚡ 内存优化技术:"
echo "  - 混合精度训练 (FP16) - 减少50%内存"
echo "  - 梯度累积 (8步) - 模拟batch_size=8"
echo "  - 实时内存清理 - 防止内存泄漏"
echo "  - batch_size=1, max_length=512 (与生产一致的序列长度)"
echo ""

# 运行诊断实验
python ft_enhanced_culturemoe_diagnostic_ab.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 3 \
    --batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
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
    --load_balance_weight 0.01 \
    --entropy_weight 0.01 \
    --specialization_weight 0.1 \
    --diversity_weight 0.05 \
    --eval_interval 1 \
    --diagnostic_mode \
    --save_router_logs \
    --save_expert_outputs \
    --analyze_cultural_routing

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Enhanced CultureMoE 诊断实验完成！"
    echo "============================================================"
    echo "诊断结果保存到: $OUTPUT_DIR"
    echo ""
    echo "生成的诊断文件："
    echo "  - diagnostic_report.json (总体诊断报告)"
    echo "  - router_distribution.json (路由分布统计)"
    echo "  - expert_similarity.json (专家相似度分析)"
    echo "  - cultural_routing.json (按文化分组的路由分析)"
    echo "  - visualization/ (可视化图表)"
    echo "    ├── expert_utilization.png"
    echo "    ├── router_entropy_dist.png"
    echo "    ├── expert_similarity_heatmap.png"
    echo "    └── cultural_routing_pattern.png"
    echo ""
    echo "📊 关键诊断指标查看："
    echo "  1. 专家利用率不均衡程度："
    echo "     cat $OUTPUT_DIR/diagnostic_report.json | python -c \"import sys,json; data=json.load(sys.stdin); print(f'Gini系数: {data[\\\"expert_utilization\\\"][\\\"gini_coefficient\\\"]:.4f}')\" 2>/dev/null || echo '需要查看diagnostic_report.json'"
    echo ""
    echo "  2. 路由熵统计："
    echo "     cat $OUTPUT_DIR/diagnostic_report.json | python -c \"import sys,json; data=json.load(sys.stdin); print(f'平均熵: {data[\\\"router_entropy\\\"][\\\"mean\\\"]:.4f}')\" 2>/dev/null || echo '需要查看diagnostic_report.json'"
    echo ""
    echo "  3. 专家相似度："
    echo "     cat $OUTPUT_DIR/diagnostic_report.json | python -c \"import sys,json; data=json.load(sys.stdin); print(f'平均相似度: {data[\\\"expert_similarity\\\"][\\\"mean_similarity\\\"]:.4f}')\" 2>/dev/null || echo '需要查看diagnostic_report.json'"
    echo ""
    echo "💡 诊断结果解读："
    echo "  - Gini系数 > 0.5: 严重的专家利用不均衡"
    echo "  - 平均熵 < 0.5: 路由过于确定，可能存在塌陷"
    echo "  - 平均相似度 > 0.8: 专家功能重叠严重"
    echo ""
    echo "📈 下一步行动建议："
    echo "  1. 查看完整诊断报告: cat $OUTPUT_DIR/diagnostic_report.json | python -m json.tool"
    echo "  2. 查看可视化图表: ls $OUTPUT_DIR/visualization/"
    echo "  3. 根据诊断结果选择对应的改进实验脚本"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Enhanced CultureMoE 诊断实验失败！"
    echo "============================================================"
    echo "请检查错误信息并确认："
    echo "  1. 模型路径是否正确且可访问"
    echo "  2. 数据文件是否存在且格式正确"
    echo "  3. 所有依赖是否已正确安装"
    echo "  4. 内存是否充足"
    echo ""
    echo "常见问题解决："
    echo "  - CUDA out of memory: 减小batch_size"
    echo "  - 文件权限错误: 检查输出目录权限"
    echo "  - 模块导入错误: 检查Python路径和依赖"
    echo "============================================================"
    exit 1
fi