#!/bin/bash

# 超轻量级 CultureMoE 训练脚本
# 针对48GB显存的极限优化版本

echo "======================================"
echo "超轻量级 CultureMoE 训练"
echo "48GB显存极限优化"
echo "======================================"

# 超轻量级配置
BACKBONE="llama"
DATA_ID="2"
NUM_EXPERTS="2"        # 最少专家数
USE_SHARED="false"     # 关闭所有额外功能
USE_MASK="false"
USE_GATE="false"
USE_CULTURE_LOSS="false"
NUM_GPUS="1"           # 强制单卡

# 路径设置
BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"

# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/ultra_lightweight/ultra_${TIMESTAMP}"

echo "超轻量级配置："
echo "  专家数量: $NUM_EXPERTS (最少)"
echo "  所有CultureMoE功能: 关闭"
echo "  训练模式: 仅LoRA"
echo "  GPU数量: $NUM_GPUS (单卡)"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 保存配置
cat > "$OUTPUT_DIR/config.json" << 'EOF'
{
    "mode": "ultra_lightweight",
    "num_experts": 2,
    "all_features_disabled": true,
    "lora_only": true,
    "memory_optimized": true
}
EOF

# 超轻量级训练参数
BATCH_SIZE=1           # 最小batch
LEARNING_RATE=1e-4     # 降低学习率
NUM_EPOCHS=1           # 只训练1个epoch测试

echo "训练参数："
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"

# 设置极限内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "开始超轻量级训练..."

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
    > "$OUTPUT_DIR/training.log" 2>&1

TRAINING_SUCCESS=$?

echo "======================================"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ 超轻量级训练成功！"
    echo ""
    echo "48GB显存可以运行基础LoRA训练"
    echo "可以考虑逐步增加功能："
    echo "1. 增加专家数量: 2 -> 3 -> 4"
    echo "2. 启用单个功能: shared -> mask -> gate"
    echo "3. 尝试双卡分布式训练"
else
    echo "❌ 超轻量级训练失败！"
    echo "48GB显存可能仍然不足以运行LLaMA-8B训练"
    echo ""
    echo "最后的建议："
    echo "1. 使用更小的基础模型 (LLaMA-2-7B或更小)"
    echo "2. 使用DeepSpeed ZeRO-3 + CPU offloading"
    echo "3. 考虑使用云端更大显存的GPU"
fi

echo "训练日志: $OUTPUT_DIR/training.log"
echo "配置文件: $OUTPUT_DIR/config.json"
echo "======================================"