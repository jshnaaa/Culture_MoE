#!/bin/bash

# V100 极简训练脚本
# 使用最小配置和内存优化技巧

echo "======================================"
echo "V100 极简 CultureMoE 训练"
echo "适配16GB/32GB显存限制"
echo "======================================"

# 检查显存大小
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "检测到GPU显存: ${GPU_MEMORY}MB"

if [ "$GPU_MEMORY" -lt 15000 ]; then
    echo "❌ 显存不足15GB，无法运行LLaMA-8B"
    echo "建议使用更大显存的GPU或更小的模型"
    exit 1
fi

# 极简配置
BACKBONE="llama"
DATA_ID="2"
NUM_EXPERTS="2"        # 最少专家数
USE_SHARED="false"     # 关闭所有额外功能
USE_MASK="false"
USE_GATE="false"
USE_CULTURE_LOSS="false"

# 路径设置
BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"

# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/v100_minimal/minimal_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "极简配置："
echo "  专家数量: $NUM_EXPERTS (最少)"
echo "  所有额外功能: 关闭"
echo "  Batch Size: 1"
echo "  序列长度: 256 (减少)"
echo "  梯度累积: 4 (补偿小batch)"

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# 极简训练参数
BATCH_SIZE=1
LEARNING_RATE=2e-4
NUM_EPOCHS=1
GRADIENT_ACCUMULATION_STEPS=4

echo "开始V100极简训练..."

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
    2>&1 | tee "$OUTPUT_DIR/v100_training.log"

TRAINING_SUCCESS=$?

echo "======================================"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ V100极简训练成功！"
    echo ""
    echo "V100可以运行极简版CultureMoE"
    echo "但建议使用更大显存的GPU进行完整训练"
else
    echo "❌ V100训练失败"
    echo ""
    echo "V100显存不足以运行LLaMA-8B + CultureMoE"
    echo "建议："
    echo "1. 使用A100/H100等大显存GPU"
    echo "2. 使用更小的基础模型(如LLaMA-2-7B)"
    echo "3. 使用模型并行/DeepSpeed"
fi

echo "训练日志: $OUTPUT_DIR/v100_training.log"
echo "======================================"