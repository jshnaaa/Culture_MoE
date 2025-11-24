#!/bin/bash

# LoRA增强FFN集成CultureMoE训练脚本
# 使用方法: sh run_lora_culturemoe_ffn_integrated.sh <BACKBONE> <DATA_ID> <USE_PROGRESSIVE> [LORA_RANK] [NUM_EXPERTS]

set -e

# 默认参数
BACKBONE=${1:-"llama"}  # llama 或 qwen
DATA_ID=${2:-"2"}       # 2, 3, 4, 5
USE_PROGRESSIVE=${3:-"true"}  # true 或 false
NUM_EXPERTS=${4:-"8"}   # 专家数量

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
        DATASET_TAG="unified_all_datasets_small"
        echo "Using unified_all_datasets dataset (enhanced format)"
        ;;
    1)
        # unified_all_datasets
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_TAG="unified_all_datasets"
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
        echo "  4 - CultureLLM (default)"
        exit 1
        ;;
esac

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "错误：数据文件不存在: $DATA_FILE"
    exit 1
fi

LORA_RANK=32

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/ffn_moe/${BACKBONE}_${DATASET_TAG}_experts${NUM_EXPERTS}_${TIMESTAMP}"

echo "======================================"
echo "LoRA Enhanced CultureMoE FFN Integrated Training"
echo "======================================"
echo "Backbone: $BACKBONE ($BASE_MODEL)"
echo "Data: $TASK_NAME ($DATA_FILE)"
echo "LoRA Rank: $LORA_RANK"
echo "Experts: $NUM_EXPERTS"
echo "Progressive Training: $USE_PROGRESSIVE"
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
    "data_file": "$DATA_FILE",
    "task_name": "$TASK_NAME",
    "lora_rank": $LORA_RANK,
    "num_experts": $NUM_EXPERTS,
    "use_progressive": $USE_PROGRESSIVE,
    "timestamp": "$TIMESTAMP"
}
EOF

# 设置训练参数
BATCH_SIZE=4
LEARNING_RATE=5e-4
NUM_EPOCHS=8

if [ "$USE_PROGRESSIVE" = "true" ]; then
    PROGRESSIVE_FLAG="--progressive_training"
else
    PROGRESSIVE_FLAG=""
fi

# 开始训练
echo "开始训练..."
python train_lora_culturemoe_ffn_integrated.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_rank $LORA_RANK \
    --lora_alpha 32.0 \
    --num_experts $NUM_EXPERTS \
    $PROGRESSIVE_FLAG \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "训练完成成功！"

    # 检查是否有LoRA权重文件
    LORA_WEIGHTS="$OUTPUT_DIR/final_lora_weights.pt"
    if [ -f "$LORA_WEIGHTS" ]; then
        echo "LoRA权重已保存: $LORA_WEIGHTS"

        # 自动运行评估（如果有测试数据）
        TEST_DATA_FILE="${DATA_FILE%_merge.json}_test.json"
        if [ -f "$TEST_DATA_FILE" ]; then
            echo "发现测试数据，开始评估..."

            EVAL_OUTPUT_DIR="$OUTPUT_DIR/evaluation"
            python eval_lora_culturemoe_ffn_integrated.py \
                --base_model "$BASE_MODEL" \
                --lora_weights "$LORA_WEIGHTS" \
                --test_data "$TEST_DATA_FILE" \
                --output_dir "$EVAL_OUTPUT_DIR" \
                --batch_size 8 \
                --lora_rank $LORA_RANK \
                --lora_alpha 32.0 \
                --num_experts $NUM_EXPERTS \
                2>&1 | tee "$OUTPUT_DIR/evaluation.log"

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
    echo "训练失败！请检查错误日志。"
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