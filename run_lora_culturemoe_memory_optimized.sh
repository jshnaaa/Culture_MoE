#!/bin/bash

# LoRA CultureMoE 内存优化版本训练脚本
# 针对48GB显存优化，支持完整功能但减少内存使用

echo "======================================"
echo "LoRA CultureMoE 内存优化训练"
echo "针对48GB显存优化"
echo "======================================"

# 默认参数（内存优化）
BACKBONE=${1:-"llama"}
DATA_ID=${2:-"2"}
NUM_EXPERTS=${3:-"4"}      # 减少专家数量
USE_SHARED=${4:-"true"}
USE_MASK=${5:-"true"}
USE_GATE=${6:-"true"}
USE_CULTURE_LOSS=${7:-"true"}
NUM_GPUS=${8:-"1"}         # 默认单卡

# 设置基础模型路径
if [ "$BACKBONE" = "llama" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="llama-8b"
elif [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="qwen-7b"
else
    echo "错误：不支持的backbone类型: $BACKBONE"
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
        echo "❌ Error: Invalid DATA_ID=$DATA_ID"
        exit 1
        ;;
esac

# 检查文件存在性
if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误：数据文件不存在: $TRAIN_FILE"
    exit 1
fi

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/memory_optimized/${BACKBONE}_${DATASET_TAG}_experts${NUM_EXPERTS}_${TIMESTAMP}"

echo "配置信息:"
echo "Backbone: $BACKBONE ($BASE_MODEL)"
echo "Data: $DATASET_TAG ($TRAIN_FILE)"
echo "Experts: $NUM_EXPERTS (减少以节省内存)"
echo "Use Shared: $USE_SHARED"
echo "Use Mask: $USE_MASK"
echo "Use Gate: $USE_GATE"
echo "Use Culture Loss: $USE_CULTURE_LOSS"
echo "Num GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "======================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 保存配置
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "backbone": "$BACKBONE",
    "base_model": "$BASE_MODEL",
    "data_id": "$DATA_ID",
    "data_file": "$TRAIN_FILE",
    "dataset_tag": "$DATASET_TAG",
    "num_experts": $NUM_EXPERTS,
    "use_shared": $USE_SHARED,
    "use_mask": $USE_MASK,
    "use_gate": $USE_GATE,
    "use_culture_loss": $USE_CULTURE_LOSS,
    "num_gpus": $NUM_GPUS,
    "memory_optimized": true,
    "timestamp": "$TIMESTAMP"
}
EOF

# 内存优化的训练参数
BATCH_SIZE=1           # 最小batch size
LEARNING_RATE=2e-4     # 稍高学习率补偿小batch
NUM_EPOCHS=3           # 减少epoch数

echo "内存优化参数:"
echo "Batch Size: $BATCH_SIZE (最小)"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

echo "开始内存优化训练..."

TRAINING_SUCCESS=0

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "使用单卡训练（内存优化）..."
    python train_lora_culturemoe_ffn_integrated_ddp.py \
        --base_model "$BASE_MODEL" \
        --data_path "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_experts $NUM_EXPERTS \
        --use_shared $USE_SHARED \
        --use_mask $USE_MASK \
        --use_gate $USE_GATE \
        --use_culture_loss $USE_CULTURE_LOSS \
        --num_gpus 1 \
        --seed 42 \
        2>&1 | tee "$OUTPUT_DIR/training.log"
    TRAINING_SUCCESS=$?
else
    echo "使用DDP多卡训练（内存优化），GPU数量: $NUM_GPUS"
    python train_lora_culturemoe_ffn_integrated_ddp.py \
        --base_model "$BASE_MODEL" \
        --data_path "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_experts $NUM_EXPERTS \
        --use_shared $USE_SHARED \
        --use_mask $USE_MASK \
        --use_gate $USE_GATE \
        --use_culture_loss $USE_CULTURE_LOSS \
        --num_gpus $NUM_GPUS \
        --seed 42 \
        2>&1 | tee "$OUTPUT_DIR/training.log"
    TRAINING_SUCCESS=$?
fi

# 检查训练结果
echo "======================================"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ 内存优化训练成功！"

    # 检查权重文件
    LORA_WEIGHTS="$OUTPUT_DIR/final_lora_weights.pt"
    if [ -f "$LORA_WEIGHTS" ]; then
        echo "✅ LoRA权重已保存: $LORA_WEIGHTS"
        echo ""
        echo "成功！可以尝试："
        echo "1. 增加专家数量: 4 -> 6 -> 8"
        echo "2. 增加batch_size: 1 -> 2"
        echo "3. 使用双卡训练提升效率"
        echo ""
        echo "双卡训练命令:"
        echo "sh run_lora_culturemoe_memory_optimized.sh llama 2 4 true true true true 2"
    else
        echo "⚠️ 训练完成但未找到LoRA权重文件"
    fi
else
    echo "❌ 内存优化训练失败！退出码: $TRAINING_SUCCESS"
    echo "请检查日志: $OUTPUT_DIR/training.log"
    echo ""
    echo "如果仍然OOM，建议："
    echo "1. 进一步减少专家数量: 4 -> 2"
    echo "2. 使用更小的基础模型"
    echo "3. 考虑使用DeepSpeed ZeRO"
fi

echo "训练日志: $OUTPUT_DIR/training.log"
echo "配置文件: $OUTPUT_DIR/config.json"
echo "======================================"