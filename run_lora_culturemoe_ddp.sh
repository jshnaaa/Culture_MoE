#!/bin/bash

# LoRA增强FFN集成CultureMoE DDP训练脚本 - 支持消融实验
# 使用方法:
# sh run_lora_culturemoe_ddp.sh [BACKBONE] [DATA_ID] [NUM_EXPERTS] [USE_SHARED] [USE_MASK] [USE_GATE] [USE_CULTURE_LOSS] [NUM_GPUS]
#
# 参数说明:
# BACKBONE       - 基础模型类型: llama 或 qwen (默认: llama)
# DATA_ID        - 数据集ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认: 2)
# NUM_EXPERTS    - 路由专家数量 (默认: 8)
# USE_SHARED     - 是否使用共享专家: true/false (默认: true)
# USE_MASK       - 是否使用mask机制: true/false (默认: true)
# USE_GATE       - 是否使用门控融合: true/false (默认: true)
# USE_CULTURE_LOSS - 是否使用文化损失: true/false (默认: true)
# NUM_GPUS       - GPU数量: 1=单卡, 2+=多卡DDP (默认: 2)
#
# 示例:
# sh run_lora_culturemoe_ddp.sh llama 2 8 true true true true 2    # 完整配置双卡训练
# sh run_lora_culturemoe_ddp.sh llama 2 8 false true true true 1   # 不使用共享专家，单卡训练
# sh run_lora_culturemoe_ddp.sh llama 2 16 true false true false 4 # 消融实验：无mask+无文化损失

# 默认参数
BACKBONE=${1:-"llama"}           # llama 或 qwen
DATA_ID=${2:-"2"}               # 2, 3, 4, 5
NUM_EXPERTS=${3:-"8"}           # 路由专家数量
USE_SHARED=${4:-"true"}         # 是否使用共享专家
USE_MASK=${5:-"true"}           # 是否使用mask机制
USE_GATE=${6:-"true"}           # 是否使用门控融合
USE_CULTURE_LOSS=${7:-"true"}   # 是否使用文化损失
NUM_GPUS=${8:-"2"}              # GPU数量，默认双卡

# 设置基础模型路径
if [ "$BACKBONE" = "llama" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="llama-7b"
elif [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="qwen-7b"
else
    echo "错误：不支持的backbone类型: $BACKBONE"
    echo "支持的类型: llama, qwen"
    exit 1
fi

# 设置数据文件路径
case $DATA_ID in
    0)
        # unified_all_datasets small
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets_small.json"
        DATASET_TAG="unified_small"
        echo "Using unified_all_datasets dataset (enhanced format)"
        ;;
    1)
        # unified_all_datasets
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_TAG="unified"
        echo "Using unified_all_datasets dataset (enhanced format)"
        ;;
    2)
        # CulturalBench
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        echo "Using CulturalBench dataset (enhanced format)"
        ;;
    3)
        # NormAD
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        echo "Using NormAD dataset (enhanced format)"
        ;;
    4)
        # CultureLLM
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        echo "Using CultureLLM dataset (enhanced format)"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 1, 2, 3, or 4."
        echo ""
        echo "DATA_ID options:"
        echo "  1 - unified_all_datasets"
        echo "  2 - CulturalBench"
        echo "  3 - NormAD"
        echo "  4 - CultureLLM"
        exit 1
        ;;
esac

# 检查数据文件是否存在
if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误：数据文件不存在: $TRAIN_FILE"
    exit 1
fi

LORA_RANK=32

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/ffn_moe/${BACKBONE}_${DATASET_TAG}_experts${NUM_EXPERTS}_${TIMESTAMP}"

echo "======================================"
echo "LoRA Enhanced CultureMoE FFN Integrated Training - Ablation Study"
echo "======================================"
echo "Backbone: $BACKBONE ($BASE_MODEL)"
echo "Data: $DATASET_TAG ($TRAIN_FILE)"
echo "LoRA Rank: $LORA_RANK"
echo "Routing Experts: $NUM_EXPERTS"
echo "Use Shared Expert: $USE_SHARED"
echo "Use Mask Mechanism: $USE_MASK"
echo "Use Gate Fusion: $USE_GATE"
echo "Use Culture Loss: $USE_CULTURE_LOSS"
echo "Num GPUs: $NUM_GPUS"
echo "Output Directory: $OUTPUT_DIR"
echo "======================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 保存配置信息
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "backbone": "$BACKBONE",
    "base_model": "$BASE_MODEL",
    "data_id": "$DATA_ID",
    "data_file": "$TRAIN_FILE",
    "dataset_tag": "$DATASET_TAG",
    "lora_rank": $LORA_RANK,
    "num_experts": $NUM_EXPERTS,
    "use_shared": $USE_SHARED,
    "use_mask": $USE_MASK,
    "use_gate": $USE_GATE,
    "use_culture_loss": $USE_CULTURE_LOSS,
    "num_gpus": $NUM_GPUS,
    "timestamp": "$TIMESTAMP"
}
EOF

# 设置训练参数（为了避免OOM，使用较小的batch_size）
BATCH_SIZE=2  # 减少batch_size以避免内存不足
LEARNING_RATE=5e-4
NUM_EPOCHS=8

# 开始训练
echo "开始训练..."

TRAINING_SUCCESS=0

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "使用单卡训练..."
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
    TRAINING_SUCCESS=${PIPESTATUS[0]}
else
    echo "使用DDP多卡训练，GPU数量: $NUM_GPUS"
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
    TRAINING_SUCCESS=${PIPESTATUS[0]}
fi

# 检查训练是否成功
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "训练完成成功！"

    # 检查是否有LoRA权重文件
    LORA_WEIGHTS="$OUTPUT_DIR/final_lora_weights.pt"
    if [ -f "$LORA_WEIGHTS" ]; then
        echo "LoRA权重已保存: $LORA_WEIGHTS"

        # 自动运行评估（如果有测试数据）
        TEST_DATA_FILE="${TRAIN_FILE%_merge_gen.json}_test.json"
        if [ -f "$TEST_DATA_FILE" ]; then
            echo "发现测试数据，开始评估..."

            EVAL_OUTPUT_DIR="$OUTPUT_DIR/evaluation"
            if [ "$NUM_GPUS" -eq 1 ]; then
                echo "使用单卡评估..."
                python eval_lora_culturemoe_ffn_integrated_ddp.py \
                    --base_model "$BASE_MODEL" \
                    --lora_weights "$LORA_WEIGHTS" \
                    --test_data "$TEST_DATA_FILE" \
                    --output_dir "$EVAL_OUTPUT_DIR" \
                    --batch_size 8 \
                    --num_experts $NUM_EXPERTS \
                    --num_gpus 1 \
                    2>&1 | tee "$OUTPUT_DIR/evaluation.log"
            else
                echo "使用DDP多卡评估，GPU数量: $NUM_GPUS"
                python eval_lora_culturemoe_ffn_integrated_ddp.py \
                    --base_model "$BASE_MODEL" \
                    --lora_weights "$LORA_WEIGHTS" \
                    --test_data "$TEST_DATA_FILE" \
                    --output_dir "$EVAL_OUTPUT_DIR" \
                    --batch_size 8 \
                    --num_experts $NUM_EXPERTS \
                    --num_gpus $NUM_GPUS \
                    2>&1 | tee "$OUTPUT_DIR/evaluation.log"
            fi

            if [ $? -eq 0 ]; then
                echo "评估完成！结果保存在: $EVAL_OUTPUT_DIR"
            else
                echo "评估失败，请检查日志。"
            fi
        else
            echo "未找到测试数据文件: $TEST_DATA_FILE"
            echo "如需评估，请手动运行评估脚本。"
        fi
    else
        echo "警告：未找到LoRA权重文件"
    fi
else
    echo "❌ 训练失败！退出码: $TRAINING_SUCCESS"
    echo "请检查错误日志: $OUTPUT_DIR/training.log"
    echo ""
    echo "常见失败原因："
    echo "1. 内存不足（OOM）- 尝试减少batch_size"
    echo "2. GPU显存不足 - 尝试减少batch_size或使用单卡训练"
    echo "3. 数据文件问题 - 检查数据文件路径和格式"
    echo "4. 模型路径问题 - 检查base_model路径是否正确"
    echo "5. DDP通信问题 - 尝试使用单卡训练调试"
    echo ""
    echo "建议调试步骤："
    echo "1. 先尝试单卡训练: sh run_lora_culturemoe_ddp.sh llama 2 8 true true true true 1"
    echo "2. 减少batch_size: 修改脚本中的BATCH_SIZE=2"
    echo "3. 减少专家数量: 使用较少的experts数量"
    exit 1
fi

echo "======================================"
echo "训练流程完成"
echo "输出目录: $OUTPUT_DIR"
echo "配置文件: $OUTPUT_DIR/config.json"
echo "训练日志: $OUTPUT_DIR/training.log"
if [ -f "$OUTPUT_DIR/evaluation.log" ]; then
    echo "评估日志: $OUTPUT_DIR/evaluation.log"
fi
echo "======================================"