#!/bin/bash

# 梯度累积版 CultureMoE 训练脚本
# 使用梯度累积技术在小显存上实现大batch_size效果

echo "======================================"
echo "梯度累积 CultureMoE 训练"
echo "小显存大batch_size解决方案"
echo "======================================"

# 配置参数
BACKBONE=${1:-"llama"}
DATA_ID=${2:-"2"}
NUM_EXPERTS=${3:-"2"}      # 保守的专家数量
USE_SHARED=${4:-"false"}   # 暂时关闭共享专家
USE_MASK=${5:-"false"}     # 暂时关闭mask
USE_GATE=${6:-"false"}     # 暂时关闭门控
USE_CULTURE_LOSS=${7:-"false"}  # 暂时关闭文化损失
NUM_GPUS=${8:-"1"}         # 单卡训练

# 路径设置
BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"

# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/gradient_accumulation/grad_accum_${TIMESTAMP}"

echo "梯度累积配置："
echo "  专家数量: $NUM_EXPERTS"
echo "  额外功能: 暂时关闭以节省内存"
echo "  GPU数量: $NUM_GPUS"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 梯度累积训练参数
MICRO_BATCH_SIZE=1         # 每步的实际batch size
GRADIENT_ACCUMULATION_STEPS=8  # 累积8步 = 有效batch size 8
EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
LEARNING_RATE=5e-5         # 稍微降低学习率
NUM_EPOCHS=2               # 减少epoch以快速测试

echo "梯度累积参数："
echo "  微批次大小: $MICRO_BATCH_SIZE"
echo "  累积步数: $GRADIENT_ACCUMULATION_STEPS"
echo "  有效批次大小: $EFFECTIVE_BATCH_SIZE"
echo "  学习率: $LEARNING_RATE"
echo "  训练轮数: $NUM_EPOCHS"

# 保存配置
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "training_method": "gradient_accumulation",
    "micro_batch_size": $MICRO_BATCH_SIZE,
    "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": $EFFECTIVE_BATCH_SIZE,
    "num_experts": $NUM_EXPERTS,
    "memory_optimized": true
}
EOF

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1

echo "开始梯度累积训练..."

python train_lora_culturemoe_ffn_integrated_ddp.py \
    --base_model "$BASE_MODEL" \
    --data_path "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_experts $NUM_EXPERTS \
    --use_shared $USE_SHARED \
    --use_mask $USE_MASK \
    --use_gate $USE_GATE \
    --use_culture_loss $USE_CULTURE_LOSS \
    --num_gpus $NUM_GPUS \
    --seed 42 \
    > "$OUTPUT_DIR/training.log" 2>&1

TRAINING_SUCCESS=$?

echo "======================================"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ 梯度累积训练成功！"
    echo ""
    echo "48GB显存可以通过梯度累积运行训练"
    echo "下一步可以尝试："
    echo "1. 逐步启用功能: shared -> mask -> gate -> culture_loss"
    echo "2. 增加专家数量: 2 -> 4 -> 6"
    echo "3. 尝试双卡分布式训练"
    echo ""
    echo "启用功能的命令示例："
    echo "sh run_gradient_accumulation_culturemoe.sh llama 2 2 true false false false 1"
else
    echo "❌ 梯度累积训练失败！"
    echo ""
    echo "如果仍然OOM，可以："
    echo "1. 增加梯度累积步数: 8 -> 16 -> 32"
    echo "2. 减少序列长度"
    echo "3. 使用DeepSpeed ZeRO"
fi

echo "训练日志: $OUTPUT_DIR/training.log"
echo "配置文件: $OUTPUT_DIR/config.json"
echo "======================================"