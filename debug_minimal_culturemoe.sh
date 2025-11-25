#!/bin/bash

# 极简版 LoRA CultureMoE 调试脚本
# 针对48GB显存优化的最小配置

echo "======================================"
echo "极简版 LoRA CultureMoE 调试"
echo "适配 48GB 显存限制"
echo "======================================"

# 极简参数设置
BACKBONE="llama"
DATA_ID="2"
NUM_EXPERTS="2"     # 大幅减少专家数量
USE_SHARED="false"  # 关闭共享专家以节省内存
USE_MASK="false"    # 关闭mask机制
USE_GATE="false"    # 关闭门控融合
USE_CULTURE_LOSS="false"  # 关闭文化损失
NUM_GPUS="1"        # 强制单卡

# 路径设置
BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"

# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/minimal_debug/minimal_${TIMESTAMP}"

echo "配置信息:"
echo "  专家数量: $NUM_EXPERTS (最少)"
echo "  共享专家: $USE_SHARED (关闭)"
echo "  Mask机制: $USE_MASK (关闭)"
echo "  门控融合: $USE_GATE (关闭)"
echo "  文化损失: $USE_CULTURE_LOSS (关闭)"
echo "======================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查文件
if [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "❌ 错误：基础模型不存在: $BASE_MODEL"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ 错误：数据文件不存在: $TRAIN_FILE"
    exit 1
fi

# 极简训练参数
BATCH_SIZE=1        # 最小batch
LEARNING_RATE=5e-5  # 更小学习率
NUM_EPOCHS=1        # 只训练1个epoch
MAX_LENGTH=256      # 减少序列长度

echo "训练参数:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Length: $MAX_LENGTH (减少内存)"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"

# 设置环境变量优化内存
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

echo "开始极简调试训练..."

# 运行训练
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
    2>&1 | tee "$OUTPUT_DIR/minimal_debug.log"

TRAINING_SUCCESS=${PIPESTATUS[0]}

echo "======================================"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ 极简调试成功！"
    echo ""
    echo "下一步建议："
    echo "1. 逐步增加专家数量: 2 -> 4 -> 6 -> 8"
    echo "2. 启用单个功能测试: shared -> mask -> gate -> culture_loss"
    echo "3. 增加batch_size: 1 -> 2"
    echo "4. 最后尝试双卡训练"
else
    echo "❌ 极简调试失败！退出码: $TRAINING_SUCCESS"
    echo ""
    echo "如果仍然OOM，可能需要："
    echo "1. 使用更小的基础模型(如LLaMA-2-7B)"
    echo "2. 使用gradient checkpointing"
    echo "3. 使用DeepSpeed ZeRO"
    echo "4. 考虑使用CPU offloading"
fi

echo "调试日志: $OUTPUT_DIR/minimal_debug.log"
echo "======================================"