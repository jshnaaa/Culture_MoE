#!/bin/bash

# CultureMoE训练脚本
# 使用方法: ./run_train_culturemoe.sh [BACKBONE] [DATA_ID] [USE_SHARED] [USE_MASK] [USE_GATE] [USE_CULTURE_LOSS] [LAMBDA] [ALPHA] [BETA] [NUM_MOE_EXPERTS] [NUM_ACTIVATED_EXPERTS] [LORA_RANK] [LORA_ALPHA]

set -e  # 遇到错误立即退出

# ===== 参数配置 =====
BACKBONE=${1:-"llama"}  # 默认使用llama，还可以指定qwen
DATA_ID=${2:-"5"}
USE_SHARED=${3:-"true"}   # 是否使用共享专家，默认为true
USE_MASK=${4:-"false"}    # MASK机制已禁用，仅作为占位符保留
USE_GATE=${5:-"true"}     # 是否使用MoE内部融合Gate，默认为true
USE_CULTURE_LOSS=${6:-"new"}  # new/false，默认为new
LAMBDA=${7:-"1.0"}  # 🔧 提升lambda让辅助损失有意义
ALPHA=${8:-"0.1"}
BETA=${9:-"0.5"}  # 🔧 进一步提升文化损失权重：强化辅助学习信号
NUM_MOE_EXPERTS=${10:-"4"}  # MoE专家数量
NUM_ACTIVATED_EXPERTS=${11:-"2"}  # 激活的专家数量，默认为top-2
LORA_RANK=${12:-"16"}   # LoRA rank
LORA_ALPHA=${13:-"32"}  # 🔧 恢复LoRA alpha：确保LoRA有足够影响力

# ===== 基座模型路径 =====
if [ "$BACKBONE" == "llama" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
elif [ "$BACKBONE" == "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
else
    echo "❌ 无效的BACKBONE: $BACKBONE (支持: llama, qwen)"
    exit 1
fi

# ===== 数据集路径和MAX_LENGTH设置 =====
case $DATA_ID in
    1)
        TRAIN_FILE="/root/autodl-fs/blend_merge_gen.json"
        DATASET_TAG="blend"
        MAX_LENGTH=512
        ;;
    2)
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        MAX_LENGTH=512
        ;;
    3)
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        MAX_LENGTH=850
        ;;
    4)
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        MAX_LENGTH=512
        ;;
    5)
        TRAIN_FILE="/autodl-fs/data/cultureAtlas_merge_gen.json"
        DATASET_TAG="cultureAtlas"
        MAX_LENGTH=512
        ;;
    *)
        echo "❌ 无效的DATA_ID: $DATA_ID (支持: 1, 2, 3, 4, 5)"
        exit 1
        ;;
esac

# ===== 输出目录配置 =====
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="/root/autodl-fs/culturemoe/${BACKBONE}_${DATASET_TAG}_experts${NUM_MOE_EXPERTS}_top${NUM_ACTIVATED_EXPERTS}_lambda${LAMBDA}_alpha${ALPHA}_beta${BETA}_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

# ===== 训练参数 =====
NUM_EPOCHS=6
BATCH_SIZE=4  # 48GB*2卡配置下可以使用更大的批次
LEARNING_RATE=1e-4
# MAX_LENGTH在数据集配置中设置
SEED=42
EVAL_STEPS=1  # 每个epoch都进行评估

# ===== 日志配置 =====
LOG_FILE="${OUTPUT_DIR}/training.log"

echo "========================================"
echo "🚀 开始训练 CultureMoE 模型"
echo "========================================"
echo "📊 配置信息:"
echo "  - 基座模型: $BACKBONE ($BASE_MODEL_PATH)"
echo "  - 数据集: $DATASET_TAG ($TRAIN_FILE)"
echo "  - 最大长度: $MAX_LENGTH"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - MoE配置: ${NUM_MOE_EXPERTS}个专家, Top-${NUM_ACTIVATED_EXPERTS}激活"
echo "  - 共享专家: $USE_SHARED"
echo "  - 门控融合: $USE_GATE"
echo "  - 文化损失: $USE_CULTURE_LOSS"
echo "  - LoRA配置: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  - 损失权重: λ=${LAMBDA}, α=${ALPHA}, β=${BETA}"
echo "  - 训练参数: epochs=${NUM_EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LEARNING_RATE}"
echo "========================================"

# ===== 检查文件存在性 =====
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ 训练数据文件不存在: $TRAIN_FILE"
    exit 1
fi

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ 基座模型目录不存在: $BASE_MODEL_PATH"
    exit 1
fi

# ===== 保存配置到文件 =====
cat > "${OUTPUT_DIR}/run_config.txt" << EOF
CultureMoE 训练配置信息
==========================================
基座模型: $BACKBONE
模型路径: $BASE_MODEL_PATH
数据集: $DATASET_TAG
数据文件: $TRAIN_FILE
最大长度: $MAX_LENGTH
输出目录: $OUTPUT_DIR

MoE配置:
- 专家数量: $NUM_MOE_EXPERTS
- 激活专家数: $NUM_ACTIVATED_EXPERTS
- 使用共享专家: $USE_SHARED
- 使用门控融合: $USE_GATE

LoRA配置:
- Rank: $LORA_RANK
- Alpha: $LORA_ALPHA

损失函数:
- 文化损失: $USE_CULTURE_LOSS
- Lambda权重: $LAMBDA
- Alpha权重: $ALPHA
- Beta权重: $BETA

训练参数:
- Epochs: $NUM_EPOCHS
- Batch Size: $BATCH_SIZE
- Learning Rate: $LEARNING_RATE
- Max Length: $MAX_LENGTH
- Seed: $SEED
- Eval Steps: $EVAL_STEPS

数据处理:
- 数据划分比例: 8:1:1
- label字段: 文化所属大洲标签
- input字段: 空（占位符）
- output字段: 答案标签
- 损失计算: 从生成文本提取数字与output比对

评估指标:
- eval_loss: 验证损失
- eval_acc: 验证准确率
- precision: 精确率
- recall: 召回率
- F1: F1分数

输出文件结构:
- config.json: 训练配置
- dataset_info.json: 数据集信息
- data_split_8_1_1.pkl: 数据划分索引
- training_history.json: 训练历史
- final_summary.json: 最终总结
- best_model/: 最佳模型（基于eval_acc）
- checkpoint-epoch-*/: 各epoch检查点
- generated_answers_epoch_*.json: 生成答案
- eval_results_epoch_*.json: 评估结果
- training.log: 训练日志

开始时间: $(date)
==========================================
EOF

# ===== 开始训练 =====
echo "🏃‍♂️ 开始训练..."
echo "📝 日志文件: $LOG_FILE"

python train_culturemoe.py \
    --backbone "$BACKBONE" \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_moe_experts "$NUM_MOE_EXPERTS" \
    --num_activated_experts "$NUM_ACTIVATED_EXPERTS" \
    --use_shared "$USE_SHARED" \
    --use_gate "$USE_GATE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --apply_to_attention "false" \
    --use_culture_loss "$USE_CULTURE_LOSS" \
    --lambda_weight "$LAMBDA" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_length "$MAX_LENGTH" \
    --seed "$SEED" \
    --eval_steps "$EVAL_STEPS" \
    2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=$?

# ===== 训练完成处理 =====
echo "" >> "${OUTPUT_DIR}/run_config.txt"
echo "结束时间: $(date)" >> "${OUTPUT_DIR}/run_config.txt"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成!"
    echo "📁 模型保存在: $OUTPUT_DIR"

    # 显示最终结果
    if [ -f "${OUTPUT_DIR}/final_summary.json" ]; then
        echo "📊 训练总结:"
        python3 -c "
import json
with open('${OUTPUT_DIR}/final_summary.json', 'r') as f:
    summary = json.load(f)
print(f\"  - 最佳Epoch: {summary['best_epoch']}\")
print(f\"  - 最佳准确率: {summary['best_accuracy']:.4f}\")
print(f\"  - 总Epochs: {summary['total_epochs']}\")
print(f\"  - 训练集大小: {summary['dataset_info']['train_size']}\")
print(f\"  - 验证集大小: {summary['dataset_info']['val_size']}\")
print(f\"  - 测试集大小: {summary['dataset_info']['test_size']}\")
"
    fi


else
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
    echo "📝 请检查日志文件: $LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

echo "========================================"
echo "🎉 CultureMoE 训练流程完成"
echo "📁 输出目录: $OUTPUT_DIR"
echo "📝 配置文件: ${OUTPUT_DIR}/run_config.txt"
echo "📊 训练日志: $LOG_FILE"
echo "🏆 最佳模型: ${OUTPUT_DIR}/best_model/"
echo "========================================"