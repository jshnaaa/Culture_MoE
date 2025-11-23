#!/bin/bash
# 动态文化感知 MoE 训练脚本
# 基于 ft_enhanced_culturemoe_gen.py 的动态聚类版本
#
# 功能特性:
# 1. 动态文化聚类：数据驱动的专家分配
# 2. 渐进式学习：从固定分配到动态聚类的4轮平滑过渡
# 3. 完整监控：详细的聚类演变追踪和可视化
# 4. 兼容现有架构：保持与静态版本的完全兼容
#
# ============================================================
# 🔧 使用说明:
#
# 参数顺序 (与 run_ft_enhanced_culturemoe_gen.sh 完全一致):
#   BACKBONE: 模型骨干 (llama, qwen)
#   DATA_ID: 数据集ID (1=unified, 2=CulturalBench, 3=NormAD, 4=CultureLLM)
#   USE_CULTURE_LOSS: 是否使用文化损失 (True/False)
#   NUM_EXPERTS: 专家数量 (建议6/12/24)
#   MOE_FUSION: MoE融合系数 (0.1-1.0)
#   LAMBDA: 文化损失权重 (0.1-1.0)
#   MARGIN: 文化损失margin (0.1-1.0)
#   LAMBDA_DIFF: 文化损失lambda_diff (0.5-2.0)
#   USE_SHARED: 是否使用共享专家 (True/False)
#   ROUTER_TEMP: 路由器温度 (1.0-5.0)
#   LOAD_BAL: 负载均衡权重 (0.001-0.1)
#   ENTROPY: 熵正则化权重 (0.01-0.5)
#   NUM_GPUS: GPU数量 (1/2)
#   USE_MASK: MASK机制开关 (True/False)
#   USE_GATE: GATE机制开关 (True/False)
#
# 📦 动态聚类特性:
#   Epoch 1: 0% 动态聚类权重 (完全固定分配)
#   Epoch 2: 30% 动态聚类权重 (开始学习)
#   Epoch 3: 70% 动态聚类权重 (动态主导)
#   Epoch 4: 100% 动态聚类权重 (完全动态)
#   温度退火: 2.0 → 1.5 → 1.0 → 0.7
#
# 示例：
#   # 标准动态聚类训练 (LLaMA + CultureLLM + 12专家)
#   sh run_dynamic_culturemoe.sh llama 4 True 12 0.4 0.5 0.5 1.0 True 3.0 0.01 0.1 1 True True
#
#   # 双GPU动态聚类训练
#   sh run_dynamic_culturemoe.sh llama 4 True 12 0.4 0.5 0.5 1.0 True 3.0 0.01 0.1 2 True True
#
#   # 消融实验：6个专家
#   sh run_dynamic_culturemoe.sh llama 4 True 6 0.4 0.5 0.5 1.0 True 3.0 0.01 0.1 1 True True
#
#   # 消融实验：24个专家
#   sh run_dynamic_culturemoe.sh llama 4 True 24 0.4 0.5 0.5 1.0 True 2.0 0.001 0.01 2 True True
# ============================================================

# ✅ 配置参数 (与现有脚本完全一致)
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-2}"                   # 默认 CulturalBench (2)
USE_CULTURE_LOSS="${3:-True}"       # 默认使用文化损失
NUM_EXPERTS="${4:-8}"              # 默认 8 个专家（支持消融实验）
MOE_FUSION="${5:-0.4}"              # 默认 MoE 融合系数 0.4
LAMBDA="${6:-0.5}"                  # 默认文化损失权重 0.5
MARGIN="${7:-0.5}"                  # 默认 margin 0.5
LAMBDA_DIFF="${8:-1.0}"             # 默认 lambda_diff 1.0
USE_SHARED="${9:-True}"             # 默认使用共享专家
ROUTER_TEMP="${10:-3.0}"            # 默认 Router 温度参数 3.0 (提高增强路由多样性)
LOAD_BAL="${11:-0.1}"             # 默认负载均衡权重 0.1 (增强专家均衡)
ENTROPY="${12:-0.2}"               # 默认熵正则化权重 0.2 (增强路由多样性)
NUM_GPUS="${13:-1}"                 # 默认使用 1 个 GPU
USE_MASK="${14:-True}"              # 默认使用 MASK 机制 (True=共享专家使用instruction_mask, False=共享专家使用instruction)
USE_GATE="${15:-True}"              # 默认使用 GATE 机制 (True=使用文化感知门控, False=不使用门控)

# 根据 backbone 选择 base 模型路径和 LoRA 权重路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_*/best_lora"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_*/best_lora"
fi

# 根据 DATA_ID 选择数据集 (与现有脚本完全一致)
case $DATA_ID in
    0)
        # unified_all_datasets small
        DATASET_NAME="unified_all_datasets_small"
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets_small_merge_gen.json"
        DATASET_TAG="unified_small"
        echo "Using unified_all_datasets_small (enhanced format)"
        ;;
    1)
        # unified_all_datasets
        DATASET_NAME="unified_all_datasets"
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets_merge_gen.json"
        DATASET_TAG="unified"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_qwen_*/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_llama_*/best_lora"
        fi
        echo "Using unified_all_datasets (enhanced format)"
        ;;
    2)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_qwen_20251112_1228/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_llama_20251112_1141/best_lora"
        fi
        echo "Using CulturalBench dataset (enhanced format)"
        ;;
    3)
        # NormAD
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_qwen_20251111_1204/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_llama_20251112_1335/best_lora"
        fi
        echo "Using NormAD dataset (enhanced format)"
        ;;
    4)
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_*/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_*/best_lora"
        fi
        echo "Using CultureLLM dataset (enhanced format)"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID: $DATA_ID"
        echo "Valid options: 0=unified_small, 1=unified, 2=CulturalBench, 3=NormAD, 4=CultureLLM"
        exit 1
        ;;
esac

# 设置标签
if [ "$USE_SHARED" = "True" ] || [ "$USE_SHARED" = "true" ]; then
    SHARED_TAG="shared"
else
    SHARED_TAG="noshared"
fi

if [ "$USE_MASK" = "True" ] || [ "$USE_MASK" = "true" ]; then
    MASK_TAG="mask"
else
    MASK_TAG="nomask"
fi

if [ "$USE_GATE" = "True" ] || [ "$USE_GATE" = "true" ]; then
    GATE_TAG="gate"
else
    GATE_TAG="nogate"
fi

# 🆕 动态聚类标签
DYNAMIC_TAG="dynamic"

OUTPUT_DIR="/root/autodl-fs/data/ft/ft_${DYNAMIC_TAG}_moe_gen_${DATASET_TAG}_${BACKBONE}_experts${NUM_EXPERTS}_${SHARED_TAG}_${MASK_TAG}_${GATE_TAG}_fusion${MOE_FUSION}_lambda${LAMBDA}_$(date +%Y%m%d_%H%M)"

# 设置 GPU - 明确的单卡/双卡逻辑
export NUM_GPUS="$NUM_GPUS"  # 传递给Python脚本

if [ "$NUM_GPUS" = "1" ]; then
#    export CUDA_VISIBLE_DEVICES=0
    GPU_INFO="Single GPU"
    echo "🔧 GPU Configuration: Single GPU training"
elif [ "$NUM_GPUS" = "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    GPU_INFO="Dual GPUs (GPU 0,1)"
    echo "🔧 GPU Configuration: Dual GPU training"
else
    echo "❌ Error: Invalid NUM_GPUS: $NUM_GPUS (only 1 or 2 supported)"
    exit 1
fi

echo ""
echo "=========================================="
echo "🚀 动态文化感知 MoE 训练启动"
echo "=========================================="
echo "🧬 Model: $MODEL_NAME ($BACKBONE)"
echo "📊 Dataset: $DATASET_NAME (ID: $DATA_ID)"
echo "🔬 Experts: $NUM_EXPERTS"
echo "🎯 MoE Fusion: $MOE_FUSION"
echo "⚖️  Culture Loss: $USE_CULTURE_LOSS (λ=$LAMBDA, α=$MARGIN, β=$LAMBDA_DIFF)"
echo "🌡️  Router Temp: $ROUTER_TEMP"
echo "⚖️  Load Balance: $LOAD_BAL"
echo "🎲 Entropy: $ENTROPY"
echo "👥 Shared Experts: $USE_SHARED"
echo "🎭 MASK: $USE_MASK"
echo "🚪 GATE: $USE_GATE"
echo "🖥️  GPUs: $GPU_INFO"
echo ""
echo "🆕 动态聚类特性:"
echo "   - Epoch 1: 0% 动态权重 (完全固定分配)"
echo "   - Epoch 2: 30% 动态权重 (开始学习)"
echo "   - Epoch 3: 70% 动态权重 (动态主导)"
echo "   - Epoch 4: 100% 动态权重 (完全动态)"
echo "   - 温度退火: 2.0 → 1.5 → 1.0 → 0.7"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo "  → Merged LoRA-finetuned model: FROZEN"
echo "  → Dynamic Enhanced MoE + Cultural components: TRAINABLE"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

# 检查 LoRA 权重
LORA_PATH=$(ls -d $LORA_WEIGHTS_PATH 2>/dev/null | head -1)
if [ -z "$LORA_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    echo ""
    echo "Please first run:"
    echo "  sh run_ft_lora_only_gen.sh $BACKBONE $DATA_ID"
    exit 1
fi

echo "Found LoRA weights: $LORA_PATH"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "Created output directory: $OUTPUT_DIR"
echo ""

# 🚀 开始训练 (使用动态增强的训练脚本)
echo "🚀 Starting Dynamic Culture MoE Training..."
echo ""

python ft_enhanced_culturemoe_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --lora_weights_path "$LORA_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --use_culture_loss "$USE_CULTURE_LOSS" \
    --culture_loss_lambda "$LAMBDA" \
    --culture_loss_alpha "$MARGIN" \
    --culture_loss_beta "$LAMBDA_DIFF" \
    --num_experts "$NUM_EXPERTS" \
    --moe_fusion "$MOE_FUSION" \
    --use_shared_experts "$USE_SHARED" \
    --use_mask "$USE_MASK" \
    --use_gate "$USE_GATE" \
    --router_temperature "$ROUTER_TEMP" \
    --load_balance_weight "$LOAD_BAL" \
    --entropy_weight "$ENTROPY" \
    --num_epochs 3 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --max_length 1024 \
    --eval_interval 1 \
    --num_workers 4 \
    --freeze_base_model True \
    --experts_hidden_dim 2048 \
    --router_hidden_dim 2048 \
    --moe_lora_rank 8 \
    --dropout 0.1 \
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
            DYNAMIC_METRICS=$(python3 -c "
import json
import sys
try:
    with open('$LATEST_SUMMARY', 'r') as f:
        data = json.load(f)
        dc = data.get('dynamic_clustering', {})
        if 'dynamic_weight_progression' in dc:
            print(f'最终动态权重: {dc[\"dynamic_weight_progression\"][\"final\"]:.2f}')
        if 'cluster_centers' in dc:
            print(f'聚类中心范数: {dc[\"cluster_centers\"][\"mean_center_norm\"]:.4f}')
        if 'clustering_parameters' in dc:
            print(f'最终温度: {dc[\"clustering_parameters\"][\"final_temperature\"]:.4f}')
        if 'culture_affinities' in dc:
            print(f'亲和性平衡: {dc[\"culture_affinities\"][\"affinity_balance_ratio\"]:.4f}')
except Exception as e:
    print(f'无法解析聚类信息: {e}')
" 2>/dev/null)
            if [ -n "$DYNAMIC_METRICS" ]; then
                echo "   $DYNAMIC_METRICS"
            fi
        else
            echo "⚠️  动态聚类信息未在路由汇总中找到"
        fi
    else
        echo "⚠️  无法找到路由汇总文件进行检查"
    fi

    echo ""
    echo "📊 训练摘要:"
    echo "   模型类型: $MODEL_NAME"
    echo "   数据集: $DATASET_NAME"
    echo "   专家数量: $NUM_EXPERTS"
    echo "   文化损失权重: $LAMBDA"
    echo "   路由温度: $ROUTER_TEMP"
    echo "   动态聚类: 启用 (4轮渐进式学习)"
    echo ""
    echo "🎯 下一步建议:"
    echo "1. 查看训练日志: cat '$OUTPUT_DIR/training_log.txt'"
    echo "2. 分析路由演变: ls '$OUTPUT_DIR/routing_logs/'"
    echo "3. 评估模型性能: 使用评估脚本"
    echo "4. 可视化聚类演变: 分析 dynamic_clustering 字段"
    echo "5. 比较与静态版本: 对比专家利用率和文化理解能力"

else
    echo "❌ 动态文化感知 MoE 训练失败 (退出码: $TRAIN_EXIT_CODE)"
    echo "请检查训练日志: $OUTPUT_DIR/training_log.txt"
    echo ""
    echo "常见问题排查:"
    echo "1. 检查GPU内存是否足够"
    echo "2. 检查数据文件格式是否正确"
    echo "3. 检查LoRA权重是否兼容"
    echo "4. 检查动态聚类参数设置"
fi

echo "=========================================="
echo "训练时间: $(date)"
echo "脚本位置: $(realpath $0)"
echo "=========================================="

exit $TRAIN_EXIT_CODE