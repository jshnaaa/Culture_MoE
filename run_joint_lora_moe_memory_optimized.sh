#!/bin/bash

# 联合LoRA+MoE训练脚本 - 内存优化版本
# 针对48GB×2卡的极限内存优化
# 大幅减少模型参数以避免OOM

echo "======================================="
echo "联合LoRA+MoE训练 - 内存优化版本"
echo "极限内存优化，针对48GB×2卡"
echo "======================================="

# 参数设置 - 内存优化
BACKBONE=${1:-"llama"}  # 默认使用llama
DATA_ID=${2:-"2"}
NUM_MOE_EXPERTS=${3:-"2"}  # 仅2个专家
USE_CULTURE_LOSS=${4:-"false"}
NUM_GPUS=${5:-"2"}
LORA_RANK=${6:-"2"}   # 极小的LoRA rank
LORA_ALPHA=${7:-"4"}  # 极小的LoRA alpha

# 检查参数
if [ "$#" -gt 7 ]; then
    echo "❌ 参数过多！用法: $0 [backbone] [data_id] [num_moe_experts] [use_culture_loss] [num_gpus] [lora_rank] [lora_alpha]"
    exit 1
fi

# 设置基础模型路径
if [ "$BACKBONE" = "llama" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="llama"
elif [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="qwen"
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
OUTPUT_DIR="/root/autodl-fs/joint_lora_moe_memory_optimized/${MODEL_NAME}_${DATASET_TAG}_${TIMESTAMP}"

echo "配置信息:"
echo "  模型: $MODEL_NAME ($BASE_MODEL)"
echo "  数据: $DATASET_TAG ($TRAIN_FILE)"
echo "  训练模式: 联合LoRA+MoE (内存优化)"
echo "  MoE专家数: $NUM_MOE_EXPERTS"
echo "  LoRA配置: rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "  文化损失: $USE_CULTURE_LOSS"
echo "  GPU: $NUM_GPUS卡"
echo "  输出: $OUTPUT_DIR"
echo ""

# 极限内存优化的训练参数
BATCH_SIZE=1              # 最小batch size
GRADIENT_ACCUMULATION=32  # 大幅增加梯度累积
LEARNING_RATE_BASE=2e-5   # 更低的基础学习率
LEARNING_RATE_MOE=1e-4    # 重新设计的路由器可以使用更高学习率
NUM_EPOCHS=3              # 减少训练轮数
MAX_SEQ_LEN=256          # 大幅减少序列长度

echo "训练参数 (内存优化):"
echo "  Batch Size: $BATCH_SIZE (per GPU)"
echo "  梯度累积: $GRADIENT_ACCUMULATION"
echo "  有效Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))"
echo "  基础LoRA学习率: $LEARNING_RATE_BASE"
echo "  MoE学习率: $LEARNING_RATE_MOE"
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
        "model_name": "$MODEL_NAME"
    },
    "data_config": {
        "data_id": "$DATA_ID",
        "data_file": "$TRAIN_FILE",
        "dataset_tag": "$DATASET_TAG",
        "max_seq_length": $MAX_SEQ_LEN
    },
    "training_config": {
        "training_mode": "joint_lora_moe_memory_optimized",
        "moe_experts": $NUM_MOE_EXPERTS,
        "lora_rank": $LORA_RANK,
        "lora_alpha": $LORA_ALPHA,
        "use_culture_loss": $USE_CULTURE_LOSS,
        "num_epochs": $NUM_EPOCHS,
        "batch_size": $BATCH_SIZE,
        "gradient_accumulation_steps": $GRADIENT_ACCUMULATION,
        "learning_rate_base": $LEARNING_RATE_BASE,
        "learning_rate_moe": $LEARNING_RATE_MOE,
        "num_gpus": $NUM_GPUS,
        "memory_optimized": true
    },
    "timestamp": "$TIMESTAMP"
}
EOF

# 设置极限内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:8
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export TORCH_USE_CUDA_DSA=1
export NCCL_P2P_DISABLE=1  # 禁用P2P以节省内存

echo "开始联合训练 LoRA + MoE (内存优化版本)..."

# 训练命令
TRAINING_SUCCESS=0

if [ "$NUM_GPUS" -eq 1 ]; then
    # 单卡训练
    python train_joint_lora_moe.py \\
        --base_model_path "$BASE_MODEL" \\
        --train_file "$TRAIN_FILE" \\
        --output_dir "$OUTPUT_DIR" \\
        --num_epochs $NUM_EPOCHS \\
        --batch_size $BATCH_SIZE \\
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION \\
        --learning_rate_base $LEARNING_RATE_BASE \\
        --learning_rate_moe $LEARNING_RATE_MOE \\
        --max_length $MAX_SEQ_LEN \\
        --backbone $BACKBONE \\
        --num_moe_experts $NUM_MOE_EXPERTS \\
        --lora_rank $LORA_RANK \\
        --lora_alpha $LORA_ALPHA \\
        --eval_interval 1 \\
        --memory_efficient \\
        $([ "$USE_CULTURE_LOSS" = "true" ] && echo "--use_culture_loss") \\
        2>&1 | tee "$OUTPUT_DIR/training.log"
else
    # 多卡训练
    export CUDA_VISIBLE_DEVICES=0,1
    torchrun \\
        --nproc_per_node=$NUM_GPUS \\
        --master_port=29502 \\
        train_joint_lora_moe.py \\
        --base_model_path "$BASE_MODEL" \\
        --train_file "$TRAIN_FILE" \\
        --output_dir "$OUTPUT_DIR" \\
        --num_epochs $NUM_EPOCHS \\
        --batch_size $BATCH_SIZE \\
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION \\
        --learning_rate_base $LEARNING_RATE_BASE \\
        --learning_rate_moe $LEARNING_RATE_MOE \\
        --max_length $MAX_SEQ_LEN \\
        --backbone $BACKBONE \\
        --num_moe_experts $NUM_MOE_EXPERTS \\
        --lora_rank $LORA_RANK \\
        --lora_alpha $LORA_ALPHA \\
        --eval_interval 1 \\
        --memory_efficient \\
        $([ "$USE_CULTURE_LOSS" = "true" ] && echo "--use_culture_loss") \\
        2>&1 | tee "$OUTPUT_DIR/training.log"
fi

TRAINING_SUCCESS=$?

# 检查训练结果
echo "======================================="
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ 联合LoRA+MoE训练 (内存优化版本) 成功！"

    # 检查最佳模型
    BEST_MODEL="$OUTPUT_DIR/best_joint_lora_moe"
    if [ -d "$BEST_MODEL" ]; then
        echo "✅ 最佳模型已保存: $BEST_MODEL"

        echo ""
        echo "🎉 训练完成！模型特点:"
        echo "  - 训练模式: 联合LoRA+MoE (内存优化)"
        echo "  - MoE专家数: $NUM_MOE_EXPERTS"
        echo "  - LoRA配置: rank=$LORA_RANK, alpha=$LORA_ALPHA"
        echo "  - 文化损失: $USE_CULTURE_LOSS"
        echo "  - 序列长度: $MAX_SEQ_LEN"
        echo "  - 内存优化: 极限优化版本"
    else
        echo "⚠️  训练完成但未找到最佳模型"
    fi
else
    echo "❌ 联合LoRA+MoE训练 (内存优化版本) 失败！退出码: $TRAINING_SUCCESS"
    echo "故障排查："
    echo "1. 检查显存使用: nvidia-smi"
    echo "2. 查看详细日志: $OUTPUT_DIR/training.log"
    echo "3. 考虑进一步减少参数: LoRA rank, 专家数, 序列长度"
fi

echo ""
echo "文件位置:"
echo "  训练日志: $OUTPUT_DIR/training.log"
echo "  配置文件: $OUTPUT_DIR/config.json"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "  最佳模型: $OUTPUT_DIR/best_joint_lora_moe/"
fi

exit $TRAINING_SUCCESS