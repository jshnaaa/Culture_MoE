#!/bin/bash
# 动态文化感知 MoE 训练执行脚本
# 专门针对动态聚类功能优化的训练配置

set -e  # 遇到错误立即退出

# 脚本参数检查
if [ "$#" -ne 11 ]; then
    echo "用法: $0 <model_type> <num_experts> <use_gate> <lora_rank> <culture_loss_alpha> <culture_loss_beta> <culture_loss_lambda> <moe_fusion> <use_shared_experts> <router_temperature> <learning_rate>"
    echo "示例: $0 llama 12 True 8 0.4 0.5 0.5 1.0 True 2.0 0.001"
    echo ""
    echo "参数说明:"
    echo "  model_type: 模型类型 (llama 或 qwen)"
    echo "  num_experts: 专家数量 (建议12, 支持动态聚类)"
    echo "  use_gate: 是否使用门控机制 (True/False)"
    echo "  lora_rank: LoRA秩 (建议8)"
    echo "  culture_loss_alpha: 文化损失alpha参数 (建议0.4)"
    echo "  culture_loss_beta: 文化损失beta参数 (建议0.5)"
    echo "  culture_loss_lambda: 文化损失权重 (建议0.5)"
    echo "  moe_fusion: MoE融合权重 (建议1.0)"
    echo "  use_shared_experts: 是否使用共享专家 (True/False)"
    echo "  router_temperature: 路由温度 (建议2.0)"
    echo "  learning_rate: 学习率 (建议0.001)"
    exit 1
fi

# 解析参数
MODEL_TYPE=$1
NUM_EXPERTS=$2
USE_GATE=$3
LORA_RANK=$4
CULTURE_LOSS_ALPHA=$5
CULTURE_LOSS_BETA=$6
CULTURE_LOSS_LAMBDA=$7
MOE_FUSION=$8
USE_SHARED_EXPERTS=$9
ROUTER_TEMPERATURE=${10}
LEARNING_RATE=${11}

# 🔧 自动检测环境和路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 检测运行环境 (云服务器 vs 本地)
if [ -d "/root/autodl-tmp" ]; then
    # 云服务器环境
    ENVIRONMENT="cloud"
    echo "🌐 检测到云服务器环境"
    if [ "$MODEL_TYPE" = "llama" ]; then
        BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
        LORA_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_*/best_lora"
    else
        BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
        LORA_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_*/best_lora"
    fi
    DATA_PATH="/root/autodl-fs/cultureLLM_merge_gen.json"
else
    # 本地环境 - 使用相对路径和灵活检测
    ENVIRONMENT="local"
    echo "💻 检测到本地环境"
    echo "⚠️  注意: 请确保已准备好模型和数据文件"

    # 尝试多个可能的路径
    POSSIBLE_BASE_PATHS=(
        "$PROJECT_ROOT/models/Meta-Llama-3.1-8B-Instruct"
        "$PROJECT_ROOT/models/llama3.1-8b-instruct"
        "$PROJECT_ROOT/models/Qwen2.5-7B-Instruct"
        "$PROJECT_ROOT/models/qwen2.5-7b-instruct"
        "/path/to/your/base/model"  # 占位符
    )

    POSSIBLE_LORA_PATHS=(
        "$PROJECT_ROOT/models/llama3.1-8b-instruct-lora"
        "$PROJECT_ROOT/models/qwen2.5-7b-instruct-lora"
        "$PROJECT_ROOT/lora_weights"
        "/path/to/your/lora/model"  # 占位符
    )

    POSSIBLE_DATA_PATHS=(
        "$PROJECT_ROOT/data/culture_train_data.jsonl"
        "$PROJECT_ROOT/data/cultureLLM_merge_gen.json"
        "$PROJECT_ROOT/data/train_data.jsonl"
        "/path/to/your/data.jsonl"  # 占位符
    )

    # 查找存在的路径
    BASE_MODEL_PATH=""
    for path in "${POSSIBLE_BASE_PATHS[@]}"; do
        if [ -d "$path" ]; then
            BASE_MODEL_PATH="$path"
            break
        fi
    done

    LORA_MODEL_PATH=""
    for path in "${POSSIBLE_LORA_PATHS[@]}"; do
        if [ -d "$path" ]; then
            LORA_MODEL_PATH="$path"
            break
        fi
    done

    DATA_PATH=""
    for path in "${POSSIBLE_DATA_PATHS[@]}"; do
        if [ -f "$path" ]; then
            DATA_PATH="$path"
            break
        fi
    done
fi

# 路径验证和错误提示
if [ -z "$BASE_MODEL_PATH" ] || [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ 错误: 找不到基础模型路径"
    echo ""
    echo "请设置正确的模型路径:"
    echo "1. 创建 models/ 目录"
    echo "2. 下载或链接模型到以下位置之一:"
    if [ "$MODEL_TYPE" = "llama" ]; then
        echo "   - $PROJECT_ROOT/models/Meta-Llama-3.1-8B-Instruct"
        echo "   - $PROJECT_ROOT/models/llama3.1-8b-instruct"
    else
        echo "   - $PROJECT_ROOT/models/Qwen2.5-7B-Instruct"
        echo "   - $PROJECT_ROOT/models/qwen2.5-7b-instruct"
    fi
    echo "3. 或修改此脚本中的路径设置"
    exit 1
fi

if [ -z "$LORA_MODEL_PATH" ] || [ ! -d "$LORA_MODEL_PATH" ]; then
    echo "❌ 错误: 找不到LoRA模型路径"
    echo ""
    echo "请设置正确的LoRA权重路径:"
    echo "1. 训练或下载LoRA权重"
    echo "2. 将权重放置到以下位置之一:"
    echo "   - $PROJECT_ROOT/models/${MODEL_TYPE}-lora"
    echo "   - $PROJECT_ROOT/lora_weights"
    echo "3. 或修改此脚本中的路径设置"
    exit 1
fi

if [ -z "$DATA_PATH" ] || [ ! -f "$DATA_PATH" ]; then
    echo "❌ 错误: 找不到训练数据文件"
    echo ""
    echo "请设置正确的数据文件:"
    echo "1. 准备训练数据文件 (JSON/JSONL格式)"
    echo "2. 将文件放置到以下位置之一:"
    echo "   - $PROJECT_ROOT/data/culture_train_data.jsonl"
    echo "   - $PROJECT_ROOT/data/train_data.jsonl"
    echo "3. 或修改此脚本中的路径设置"
    exit 1
fi

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 设置输出目录（包含动态聚类标识）
OUTPUT_DIR="/Users/yzl/ownCode/Culture_Moe/outputs/dynamic_culturemoe_${MODEL_TYPE}_${NUM_EXPERTS}experts_${TIMESTAMP}"

echo "=========================================="
echo "🚀 动态文化感知 MoE 训练启动"
echo "=========================================="
echo "模型类型: $MODEL_TYPE"
echo "专家数量: $NUM_EXPERTS"
echo "动态聚类: 启用"
echo "基础模型: $BASE_MODEL_PATH"
echo "LoRA模型: $LORA_MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "时间戳: $TIMESTAMP"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行训练（使用动态增强的训练脚本）
echo "开始训练..."

python ft_enhanced_culturemoe_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --lora_weights_path "$LORA_MODEL_PATH" \
    --train_file "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 4 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_length 512 \
    --eval_interval 1 \
    --save_interval 1 \
    --logging_steps 10 \
    --use_amp True \
    --num_workers 4 \
    --seed 42 \
    --freeze_base_model True \
    --use_culture_loss True \
    --culture_loss_lambda $CULTURE_LOSS_LAMBDA \
    --culture_loss_alpha $CULTURE_LOSS_ALPHA \
    --culture_loss_beta $CULTURE_LOSS_BETA \
    --use_shared_experts $USE_SHARED_EXPERTS \
    --router_temperature $ROUTER_TEMPERATURE \
    --load_balance_weight 0.01 \
    --entropy_weight 0.1 \
    --num_experts $NUM_EXPERTS \
    --experts_hidden_dim 2048 \
    --router_hidden_dim 2048 \
    --moe_lora_rank $LORA_RANK \
    --dropout 0.1 \
    --moe_fusion $MOE_FUSION \
    --use_gate $USE_GATE \
    --moe_lr_multiplier 2.0 \
    --router_lr_multiplier 3.0 \
    --shared_lr_multiplier 1.0

# 训练完成后的处理
TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 动态文化感知 MoE 训练完成!"
    echo "输出目录: $OUTPUT_DIR"

    # 检查关键输出文件
    echo ""
    echo "📁 输出文件检查:"

    if [ -d "$OUTPUT_DIR/best_enhanced_moe" ]; then
        echo "✅ 最佳模型: $OUTPUT_DIR/best_enhanced_moe"
    else
        echo "⚠️  最佳模型目录未找到"
    fi

    if [ -d "$OUTPUT_DIR/routing_logs" ]; then
        echo "✅ 路由日志: $OUTPUT_DIR/routing_logs"
        ROUTING_FILES=$(find "$OUTPUT_DIR/routing_logs" -name "*.json" | wc -l)
        echo "   - 路由汇总文件数: $ROUTING_FILES"
    else
        echo "⚠️  路由日志目录未找到"
    fi

    if [ -f "$OUTPUT_DIR/training_log.txt" ]; then
        echo "✅ 训练日志: $OUTPUT_DIR/training_log.txt"
    else
        echo "⚠️  训练日志文件未找到"
    fi

    # 🆕 动态聚类特定检查
    echo ""
    echo "🔍 动态聚类功能检查:"

    # 检查路由汇总中是否包含动态聚类信息
    LATEST_SUMMARY=$(find "$OUTPUT_DIR/routing_logs" -name "epoch_*_summary.json" | sort | tail -1)
    if [ -f "$LATEST_SUMMARY" ]; then
        if grep -q "dynamic_clustering" "$LATEST_SUMMARY"; then
            echo "✅ 动态聚类统计信息已记录"

            # 提取关键指标
            DYNAMIC_WEIGHT=$(python -c "
import json
with open('$LATEST_SUMMARY', 'r') as f:
    data = json.load(f)
    dc = data.get('dynamic_clustering', {})
    if 'dynamic_weight_progression' in dc:
        print(f\"最终动态权重: {dc['dynamic_weight_progression']['final']:.2f}\")
    if 'cluster_centers' in dc:
        print(f\"聚类中心范数: {dc['cluster_centers']['mean_center_norm']:.4f}\")
    if 'clustering_parameters' in dc:
        print(f\"最终温度: {dc['clustering_parameters']['final_temperature']:.4f}\")
" 2>/dev/null)
            if [ -n "$DYNAMIC_WEIGHT" ]; then
                echo "   $DYNAMIC_WEIGHT"
            fi
        else
            echo "⚠️  动态聚类信息未在路由汇总中找到"
        fi
    else
        echo "⚠️  无法找到路由汇总文件进行检查"
    fi

    echo ""
    echo "📊 训练摘要:"
    echo "   模型类型: $MODEL_TYPE"
    echo "   专家数量: $NUM_EXPERTS"
    echo "   LoRA秩: $LORA_RANK"
    echo "   学习率: $LEARNING_RATE"
    echo "   文化损失权重: $CULTURE_LOSS_LAMBDA"
    echo "   路由温度: $ROUTER_TEMPERATURE"
    echo "   动态聚类: 启用 (4轮渐进式学习)"
    echo ""
    echo "🎯 下一步建议:"
    echo "1. 查看训练日志: cat '$OUTPUT_DIR/training_log.txt'"
    echo "2. 分析路由演变: ls '$OUTPUT_DIR/routing_logs/'"
    echo "3. 评估模型性能: 运行评估脚本"
    echo "4. 可视化聚类演变: 分析 dynamic_clustering 字段"

else
    echo "❌ 动态文化感知 MoE 训练失败 (退出码: $TRAIN_EXIT_CODE)"
    echo "请检查训练日志: $OUTPUT_DIR/training_log.txt"
fi

echo "=========================================="
echo "训练时间: $(date)"
echo "=========================================="

exit $TRAIN_EXIT_CODE