#!/bin/bash

# LoRA CultureMoE 调试脚本 - 单卡训练，小batch size
# 用于调试和测试模型是否能正常运行

echo "======================================"
echo "LoRA CultureMoE 调试模式"
echo "======================================"

# 设置基础参数
BACKBONE="llama"
DATA_ID="2"
NUM_EXPERTS="4"  # 减少专家数量
USE_SHARED="true"
USE_MASK="true"
USE_GATE="true"
USE_CULTURE_LOSS="true"
NUM_GPUS="1"  # 强制单卡

# 设置基础模型路径
BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/debug_moe/debug_${TIMESTAMP}"

echo "Base Model: $BASE_MODEL"
echo "Data File: $TRAIN_FILE"
echo "Experts: $NUM_EXPERTS"
echo "Output: $OUTPUT_DIR"
echo "======================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查文件是否存在
if [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "❌ 错误：基础模型不存在: $BASE_MODEL"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ 错误：数据文件不存在: $TRAIN_FILE"
    exit 1
fi

# 调试训练参数（极小设置）
BATCH_SIZE=1  # 最小batch size
LEARNING_RATE=1e-4  # 较小学习率
NUM_EPOCHS=1  # 只训练1个epoch用于调试

echo "开始调试训练..."
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"

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
    2>&1 | tee "$OUTPUT_DIR/debug.log"

TRAINING_SUCCESS=${PIPESTATUS[0]}

echo "======================================"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ 调试训练成功完成！"

    # 检查是否有LoRA权重文件
    LORA_WEIGHTS="$OUTPUT_DIR/final_lora_weights.pt"
    if [ -f "$LORA_WEIGHTS" ]; then
        echo "✅ LoRA权重已保存: $LORA_WEIGHTS"
        echo ""
        echo "模型可以正常训练！可以尝试："
        echo "1. 增加batch_size和epochs进行正式训练"
        echo "2. 使用双卡DDP训练"
        echo ""
        echo "正式训练命令："
        echo "sh run_lora_culturemoe_ddp.sh llama 2 8 true true true true 2"
    else
        echo "⚠️ 训练完成但未找到LoRA权重文件"
    fi
else
    echo "❌ 调试训练失败！退出码: $TRAINING_SUCCESS"
    echo "请检查日志: $OUTPUT_DIR/debug.log"
    echo ""
    echo "常见问题排查："
    echo "1. 检查GPU显存是否足够"
    echo "2. 检查基础模型路径是否正确"
    echo "3. 检查数据文件格式是否正确"
    echo "4. 检查Python环境和依赖是否完整"
fi

echo "调试日志: $OUTPUT_DIR/debug.log"
echo "======================================"