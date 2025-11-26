#!/bin/bash

# 简化版FFN CultureMoE训练脚本
# 基于MixLoRA实现，所有层都使用MoE，添加文化损失
# 针对48GB×2卡优化

echo "======================================="
echo "简化版FFN CultureMoE训练"
echo "基于MixLoRA + 文化损失，所有层都使用MoE"
echo "针对48GB×2卡优化"
echo "======================================="

# 参数设置
BACKBONE=${1:-"qwen"}  # 默认使用qwen2.5-7B
DATA_ID=${2:-"2"}
NUM_ROUTING_EXPERTS=${3:-"2"}  # 减少到2个路由专家
USE_CULTURE_LOSS=${4:-"true"}
NUM_GPUS=${5:-"2"}

# 检查参数
if [ "$#" -gt 5 ]; then
    echo "❌ 参数过多！用法: $0 [backbone] [data_id] [num_routing_experts] [use_culture_loss] [num_gpus]"
    exit 1
fi

# 验证专家数参数
if ! [[ "$NUM_ROUTING_EXPERTS" =~ ^[1-4]$ ]]; then
    echo "❌ 路由专家数必须是1-4: $NUM_ROUTING_EXPERTS"
    exit 1
fi

# 设置基础模型路径
if [ "$BACKBONE" = "llama" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="llama"
    TOTAL_LAYERS=32
    MoE_LAYERS="30,31"  # LLaMA的最后2层
elif [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="qwen"
    TOTAL_LAYERS=28
    MoE_LAYERS="26,27"  # Qwen的最后2层
else
    echo "❌ 不支持的backbone: $BACKBONE (支持: llama, qwen)"
    exit 1
fi

# 设置数据文件路径
case $DATA_ID in
    2)
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        ;;
    3)
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        ;;
    4)
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        ;;
    *)
        echo "❌ 无效的DATA_ID: $DATA_ID (支持: 2, 3, 4)"
        exit 1
        ;;
esac

# 检查文件存在性
if [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "❌ 基础模型不存在: $BASE_MODEL"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ 数据文件不存在: $TRAIN_FILE"
    exit 1
fi

# 检查GPU数量
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 8 ]; then
    echo "❌ GPU数量必须在1-8之间: $NUM_GPUS"
    exit 1
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
        echo "❌ 请求的GPU数量($NUM_GPUS)超过可用数量($AVAILABLE_GPUS)"
        exit 1
    fi
fi

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/simplified_culturemoe/${MODEL_NAME}_${DATASET_TAG}_${TIMESTAMP}"

echo "配置信息:"
echo "  模型: $MODEL_NAME ($BASE_MODEL)"
echo "  数据: $DATASET_TAG ($TRAIN_FILE)"
echo "  总层数: $TOTAL_LAYERS"
echo "  MoE层: $MoE_LAYERS (最后2层)"
echo "  专家配置: ${NUM_ROUTING_EXPERTS}个路由专家"
echo "  文化损失: $USE_CULTURE_LOSS"
echo "  GPU: $NUM_GPUS卡"
echo "  输出: $OUTPUT_DIR"
echo ""

# 内存优化的训练参数
BATCH_SIZE=1              # 最小batch size
GRADIENT_ACCUMULATION=8   # 梯度累积
LEARNING_RATE=1e-4        # 学习率
NUM_EPOCHS=6              # 训练轮数
MAX_SEQ_LEN=512          # 序列长度

echo "训练参数:"
echo "  Batch Size: $BATCH_SIZE (per GPU)"
echo "  梯度累积: $GRADIENT_ACCUMULATION"
echo "  有效Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))"
echo "  学习率: $LEARNING_RATE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  最大序列长度: $MAX_SEQ_LEN"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 保存配置
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "model_config": {
        "backbone": "$BACKBONE",
        "base_model": "$BASE_MODEL",
        "model_name": "$MODEL_NAME",
        "total_layers": $TOTAL_LAYERS,
        "moe_layers": "$MoE_LAYERS"
    },
    "data_config": {
        "data_id": "$DATA_ID",
        "data_file": "$TRAIN_FILE",
        "dataset_tag": "$DATASET_TAG",
        "max_seq_length": $MAX_SEQ_LEN
    },
    "moe_config": {
        "moe_layers": "ALL layers (like MixLoRA)",
        "routing_experts": $NUM_ROUTING_EXPERTS,
        "use_culture_loss": $USE_CULTURE_LOSS,
        "architecture": "simplified_mixlora_based"
    },
    "training_config": {
        "num_epochs": $NUM_EPOCHS,
        "batch_size": $BATCH_SIZE,
        "gradient_accumulation_steps": $GRADIENT_ACCUMULATION,
        "learning_rate": $LEARNING_RATE,
        "num_gpus": $NUM_GPUS,
        "memory_optimized": true
    },
    "timestamp": "$TIMESTAMP"
}
EOF

# 设置内存优化环境变量（与MixLoRA一致）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "开始简化版FFN CultureMoE训练..."

# 训练命令
TRAINING_SUCCESS=0

if [ "$NUM_GPUS" -eq 1 ]; then
    # 单卡训练
    python train_simplified_culturemoe.py \
        --base_model_path "$BASE_MODEL" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_SEQ_LEN \
        --backbone $BACKBONE \
        --num_routing_experts $NUM_ROUTING_EXPERTS \
        --use_culture_loss $USE_CULTURE_LOSS \
        --lora_r 16 \
        --lora_alpha 8 \
        --memory_efficient \
        2>&1 | tee "$OUTPUT_DIR/training.log"
else
    # 多卡训练
    export CUDA_VISIBLE_DEVICES=0,1
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_simplified_culturemoe.py \
        --base_model_path "$BASE_MODEL" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_SEQ_LEN \
        --backbone $BACKBONE \
        --num_routing_experts $NUM_ROUTING_EXPERTS \
        --use_culture_loss $USE_CULTURE_LOSS \
        --lora_r 16 \
        --lora_alpha 8 \
        --memory_efficient \
        2>&1 | tee "$OUTPUT_DIR/training.log"
fi

TRAINING_SUCCESS=$?

# 检查训练结果
echo "======================================="
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ 简化版FFN CultureMoE训练成功！"

    # 检查最佳模型
    BEST_MODEL="$OUTPUT_DIR/best_simplified_culturemoe"
    if [ -d "$BEST_MODEL" ]; then
        echo "✅ 最佳模型已保存: $BEST_MODEL"

        echo ""
        echo "🎉 训练完成！模型特点:"
        echo "  - 基于MixLoRA架构，内存优化"
        echo "  - 所有层都使用MoE (与MixLoRA一致)"
        echo "  - ${NUM_ROUTING_EXPERTS}个路由专家每层"
        echo "  - 添加了文化损失"
        echo "  - LoRA rank=16, alpha=8"
        echo "  - 序列长度=$MAX_SEQ_LEN (内存优化)"
    else
        echo "⚠️  训练完成但未找到最佳模型"
    fi
else
    echo "❌ 简化版FFN CultureMoE训练失败！退出码: $TRAINING_SUCCESS"
    echo ""
    echo "故障排查："
    echo "1. 检查显存使用: nvidia-smi"
    echo "2. 查看详细日志: $OUTPUT_DIR/training.log"
fi

echo ""
echo "文件位置:"
echo "  训练日志: $OUTPUT_DIR/training.log"
echo "  配置文件: $OUTPUT_DIR/config.json"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "  最佳模型: $OUTPUT_DIR/best_simplified_culturemoe/"
fi
echo "======================================="

exit $TRAINING_SUCCESS