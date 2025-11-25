#!/bin/bash

# FFN集成CultureMoE训练脚本 - 优化版
# 仅在Layer 17-32使用MoE，Layer 1-16保持原始FFN
# 支持LLaMA-3.1-8B和Qwen2.5-7B，针对48GB×2卡优化

echo "======================================="
echo "FFN集成CultureMoE训练 - 优化版"
echo "Layer 17-24: 3个专家/层, Layer 25-32: 5个专家/层"
echo "针对48GB×2卡优化"
echo "======================================="

# 参数设置
BACKBONE=${1:-"qwen"}  # 默认使用qwen2.5-7B（显存需求更小）
DATA_ID=${2:-"2"}
USE_SHARED=${3:-"true"}
USE_MASK=${4:-"true"}
USE_GATE=${5:-"true"}
USE_CULTURE_LOSS=${6:-"true"}
NUM_GPUS=${7:-"2"}

# 检查参数
if [ "$#" -gt 7 ]; then
    echo "❌ 参数过多！用法: $0 [backbone] [data_id] [use_shared] [use_mask] [use_gate] [use_culture_loss] [num_gpus]"
    exit 1
fi

# 设置基础模型路径
if [ "$BACKBONE" = "llama" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="llama-3.1-8b"
    echo "⚠️  LLaMA-3.1-8B显存需求: ~52GB/卡，接近48GB限制"
elif [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="qwen2.5-7b"
    echo "✅ Qwen2.5-7B显存需求: ~52GB/卡，推荐选择"
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
OUTPUT_DIR="/root/autodl-fs/ffn_culturemoe/${MODEL_NAME}_${DATASET_TAG}_${TIMESTAMP}"

echo "配置信息:"
echo "  模型: $MODEL_NAME ($BASE_MODEL)"
echo "  数据: $DATASET_TAG ($TRAIN_FILE)"
echo "  MoE层: Layer 17-32 (16层)"
echo "  专家分配: Layer 17-24(3个), Layer 25-32(5个)"
echo "  功能: shared=$USE_SHARED, mask=$USE_MASK, gate=$USE_GATE, culture_loss=$USE_CULTURE_LOSS"
echo "  GPU: $NUM_GPUS卡"
echo "  输出: $OUTPUT_DIR"
echo "======================================="

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
    "moe_layers": "17-32",
    "layer_17_24_experts": 3,
    "layer_25_32_experts": 5,
    "use_shared": $USE_SHARED,
    "use_mask": $USE_MASK,
    "use_gate": $USE_GATE,
    "use_culture_loss": $USE_CULTURE_LOSS,
    "num_gpus": $NUM_GPUS,
    "optimization": "lora_only",
    "memory_optimized": true,
    "timestamp": "$TIMESTAMP"
}
EOF

# 内存优化的训练参数
BATCH_SIZE=1              # 最小batch size以节省显存
GRADIENT_ACCUMULATION=4   # 梯度累积补偿小batch
LEARNING_RATE=2e-4        # 适中的学习率
NUM_EPOCHS=3              # 适中的epoch数
MAX_SEQ_LEN=512          # 控制序列长度

echo "训练参数:"
echo "  Batch Size: $BATCH_SIZE (per GPU)"
echo "  梯度累积: $GRADIENT_ACCUMULATION"
echo "  有效Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))"
echo "  学习率: $LEARNING_RATE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  最大序列长度: $MAX_SEQ_LEN"

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# 设置DDP环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=12355

echo "开始FFN集成CultureMoE训练..."

# 训练命令
TRAINING_SUCCESS=0

python train_lora_culturemoe_ffn_integrated_ddp.py \
    --base_model "$BASE_MODEL" \
    --data_path "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LEN \
    --moe_start_layer 17 \
    --layer_17_24_experts 3 \
    --layer_25_32_experts 5 \
    --use_shared $USE_SHARED \
    --use_mask $USE_MASK \
    --use_gate $USE_GATE \
    --use_culture_loss $USE_CULTURE_LOSS \
    --num_gpus $NUM_GPUS \
    --enable_activation_checkpointing true \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

TRAINING_SUCCESS=${PIPESTATUS[0]}

# 检查训练结果
echo "======================================="
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ FFN集成CultureMoE训练成功！"

    # 检查权重文件
    LORA_WEIGHTS="$OUTPUT_DIR/final_lora_weights.pt"
    if [ -f "$LORA_WEIGHTS" ]; then
        echo "✅ LoRA权重已保存: $LORA_WEIGHTS"

        # 显示文件大小
        WEIGHT_SIZE=$(du -h "$LORA_WEIGHTS" | cut -f1)
        echo "   权重文件大小: $WEIGHT_SIZE"

        echo ""
        echo "🎉 训练完成！可以进行推理测试"
        echo "模型特点:"
        echo "  - 仅Layer 17-32使用MoE"
        echo "  - Layer 17-24: 3个专家/层"
        echo "  - Layer 25-32: 5个专家/层"
        echo "  - 总计64个专家"
        echo "  - 仅训练LoRA参数，基础模型冻结"
    else
        echo "⚠️  训练完成但未找到LoRA权重文件"
    fi

    # 显示训练统计
    if [ -f "$OUTPUT_DIR/training.log" ]; then
        echo ""
        echo "训练统计:"
        grep -E "(Epoch|Loss|专家利用率)" "$OUTPUT_DIR/training.log" | tail -10
    fi

else
    echo "❌ FFN集成CultureMoE训练失败！退出码: $TRAINING_SUCCESS"
    echo ""
    echo "故障排查："
    echo "1. 检查显存使用: nvidia-smi"
    echo "2. 查看详细日志: $OUTPUT_DIR/training.log"
    echo "3. 检查错误信息:"

    if [ -f "$OUTPUT_DIR/training.log" ]; then
        echo ""
        echo "最近的错误信息:"
        tail -20 "$OUTPUT_DIR/training.log" | grep -E "(Error|Exception|Failed|OOM)"
    fi

    echo ""
    echo "可能的解决方案:"
    echo "1. 减少batch_size: --batch_size 1"
    echo "2. 增加梯度累积: --gradient_accumulation_steps 8"
    echo "3. 减少序列长度: --max_seq_length 256"
    echo "4. 使用单卡训练: $0 $BACKBONE $DATA_ID $USE_SHARED $USE_MASK $USE_GATE $USE_CULTURE_LOSS 1"
fi

echo ""
echo "文件位置:"
echo "  训练日志: $OUTPUT_DIR/training.log"
echo "  配置文件: $OUTPUT_DIR/config.json"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "  模型权重: $OUTPUT_DIR/final_lora_weights.pt"
fi
echo "======================================="

# 清理GPU缓存
if command -v nvidia-smi &> /dev/null; then
    echo "清理GPU缓存..."
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi

exit $TRAINING_SUCCESS