#!/bin/bash

# Phase 1: 带知识蒸馏的增强CultureMoE训练脚本
# 通过从LoRA teacher模型蒸馏知识来改善OOD性能

# 使用方法:
# bash run_ft_enhanced_culturemoe_with_distill.sh <model_type> <num_gpus> [其他参数...]
#
# 示例:
# bash run_ft_enhanced_culturemoe_with_distill.sh llama 1
# bash run_ft_enhanced_culturemoe_with_distill.sh llama 2 True 12 0.4 0.5 0.5 1.0 True 3.0 0.01 0.1 1 True False 4.0 0.7

# 检查参数数量
if [ $# -lt 2 ]; then
    echo "用法: $0 <model_type> <num_gpus> [freeze_base_model] [num_experts] [moe_fusion] [culture_loss_lambda] [culture_loss_alpha] [culture_loss_beta] [use_culture_loss] [router_temperature] [load_balance_weight] [entropy_weight] [eval_interval] [use_shared_experts] [use_gate] [distill_temperature] [distill_alpha]"
    echo ""
    echo "参数说明:"
    echo "  model_type: 模型类型 (llama)"
    echo "  num_gpus: GPU数量 (1 或 2)"
    echo "  freeze_base_model: 是否冻结基础模型 (默认: True)"
    echo "  num_experts: 专家数量 (默认: 12)"
    echo "  moe_fusion: MoE融合系数 (默认: 0.4)"
    echo "  culture_loss_lambda: 文化损失权重 (默认: 0.5)"
    echo "  culture_loss_alpha: 文化损失margin (默认: 0.5)"
    echo "  culture_loss_beta: 文化损失lambda_diff (默认: 1.0)"
    echo "  use_culture_loss: 是否使用文化损失 (默认: True)"
    echo "  router_temperature: 路由器温度 (默认: 3.0)"
    echo "  load_balance_weight: 负载均衡权重 (默认: 0.01)"
    echo "  entropy_weight: 熵正则化权重 (默认: 0.1)"
    echo "  eval_interval: 评估间隔 (默认: 1)"
    echo "  use_shared_experts: 是否使用共享专家 (默认: True)"
    echo "  use_gate: 是否使用门控机制 (默认: False)"
    echo "  distill_temperature: 蒸馏温度 (默认: 4.0)"
    echo "  distill_alpha: 蒸馏权重 (默认: 0.7)"
    echo ""
    echo "示例:"
    echo "  $0 llama 1"
    echo "  $0 llama 2 True 12 0.4 0.5 0.5 1.0 True 3.0 0.01 0.1 1 True False 4.0 0.7"
    exit 1
fi

# 解析参数
MODEL_TYPE=$1
NUM_GPUS=$2
FREEZE_BASE_MODEL=${3:-"True"}
NUM_EXPERTS=${4:-12}
MOE_FUSION=${5:-0.4}
CULTURE_LOSS_LAMBDA=${6:-0.5}
CULTURE_LOSS_ALPHA=${7:-0.5}
CULTURE_LOSS_BETA=${8:-1.0}
USE_CULTURE_LOSS=${9:-"True"}
ROUTER_TEMPERATURE=${10:-3.0}
LOAD_BALANCE_WEIGHT=${11:-0.01}
ENTROPY_WEIGHT=${12:-0.1}
EVAL_INTERVAL=${13:-1}
USE_SHARED_EXPERTS=${14:-"True"}
USE_GATE=${15:-"False"}
DISTILL_TEMPERATURE=${16:-4.0}
DISTILL_ALPHA=${17:-0.7}

echo "==============================================="
echo "Phase 1: Enhanced CultureMoE with Distillation"
echo "==============================================="
echo "Model Type: $MODEL_TYPE"
echo "Number of GPUs: $NUM_GPUS"
echo "Freeze Base Model: $FREEZE_BASE_MODEL"
echo "Number of Experts: $NUM_EXPERTS"
echo "MoE Fusion: $MOE_FUSION"
echo "Culture Loss Lambda: $CULTURE_LOSS_LAMBDA"
echo "Culture Loss Alpha: $CULTURE_LOSS_ALPHA"
echo "Culture Loss Beta: $CULTURE_LOSS_BETA"
echo "Use Culture Loss: $USE_CULTURE_LOSS"
echo "Router Temperature: $ROUTER_TEMPERATURE"
echo "Load Balance Weight: $LOAD_BALANCE_WEIGHT"
echo "Entropy Weight: $ENTROPY_WEIGHT"
echo "Evaluation Interval: $EVAL_INTERVAL"
echo "Use Shared Experts: $USE_SHARED_EXPERTS"
echo "Use Gate: $USE_GATE"
echo "Distillation Temperature: $DISTILL_TEMPERATURE"
echo "Distillation Alpha: $DISTILL_ALPHA"
echo "==============================================="

# 根据模型类型设置路径
if [ "$MODEL_TYPE" = "llama" ]; then
    BASE_MODEL_PATH="/Users/yzl/models/Meta-Llama-3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/Users/yzl/ownCode/Culture_Moe/output/ft_llama_culturalign_lora_2024-11-18_16-09-59/final"
    TRAIN_FILE="/Users/yzl/ownCode/Culture_Moe/data/train_culture_align.json"
    OUTPUT_DIR="/Users/yzl/ownCode/Culture_Moe/output/ft_enhanced_culturemoe_with_distill_$(date +%Y-%m-%d_%H-%M-%S)"
else
    echo "错误: 不支持的模型类型 '$MODEL_TYPE'"
    echo "支持的模型类型: llama"
    exit 1
fi

# 检查必要文件是否存在
echo "检查必要文件..."
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "错误: 基础模型路径不存在: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "错误: LoRA权重路径不存在: $LORA_WEIGHTS_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练数据文件不存在: $TRAIN_FILE"
    exit 1
fi

echo "✅ 所有必要文件检查通过"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "输出目录: $OUTPUT_DIR"

# 设置GPU环境变量
export NUM_GPUS=$NUM_GPUS
if [ "$NUM_GPUS" = "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "使用GPU: 0,1"
elif [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "使用GPU: 0"
else
    echo "错误: 不支持的GPU数量 '$NUM_GPUS'"
    echo "支持的GPU数量: 1, 2"
    exit 1
fi

# 记录开始时间
START_TIME=$(date)
echo "开始时间: $START_TIME"

# 运行训练脚本
echo ""
echo "🚀 开始Phase 1知识蒸馏训练..."
echo ""

python ft_enhanced_culturemoe_with_distill.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --lora_weights_path "$LORA_WEIGHTS_PATH" \
    --teacher_model_path "$BASE_MODEL_PATH" \
    --teacher_lora_path "$LORA_WEIGHTS_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --freeze_base_model "$FREEZE_BASE_MODEL" \
    --num_experts $NUM_EXPERTS \
    --shared_hidden_dim 4096 \
    --router_hidden_dim 2048 \
    --experts_hidden_dim 4096 \
    --moe_lora_rank 32 \
    --dropout 0.05 \
    --moe_fusion $MOE_FUSION \
    --use_culture_loss "$USE_CULTURE_LOSS" \
    --culture_loss_lambda $CULTURE_LOSS_LAMBDA \
    --culture_loss_alpha $CULTURE_LOSS_ALPHA \
    --culture_loss_beta $CULTURE_LOSS_BETA \
    --use_shared_experts "$USE_SHARED_EXPERTS" \
    --use_mask "True" \
    --use_gate "$USE_GATE" \
    --router_temperature $ROUTER_TEMPERATURE \
    --load_balance_weight $LOAD_BALANCE_WEIGHT \
    --entropy_weight $ENTROPY_WEIGHT \
    --distill_temperature $DISTILL_TEMPERATURE \
    --distill_alpha $DISTILL_ALPHA \
    --num_epochs 6 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-4 \
    --moe_lr_multiplier 1.0 \
    --router_lr_multiplier 0.1 \
    --shared_lr_multiplier 1.0 \
    --weight_decay 0.01 \
    --max_length 1024 \
    --num_workers 2 \
    --eval_interval $EVAL_INTERVAL \
    --device cuda

# 记录结束时间
END_TIME=$(date)
EXIT_CODE=$?

echo ""
echo "==============================================="
echo "Phase 1 知识蒸馏训练完成"
echo "==============================================="
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "输出目录: $OUTPUT_DIR"
echo "退出代码: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成!"
    echo ""
    echo "📁 输出文件:"
    echo "  - 最佳模型: $OUTPUT_DIR/best_distill_moe/"
    echo "  - 训练日志: $OUTPUT_DIR/distill_training.log"
    echo "  - 训练配置: $OUTPUT_DIR/distill_config.json"
    echo "  - Epoch结果: $OUTPUT_DIR/distill_epoch_results.json"
    echo ""
    echo "🔍 接下来可以:"
    echo "  1. 检查训练日志查看蒸馏损失和不变性损失"
    echo "  2. 在OOD测试集上评估模型性能"
    echo "  3. 对比原始MoE和蒸馏MoE的OOD性能"
    echo "  4. 如果Phase 1效果良好，可以考虑实施Phase 2"
else
    echo "❌ 训练失败 (退出代码: $EXIT_CODE)"
    echo "请检查错误日志: $OUTPUT_DIR/distill_training.log"
fi

echo "==============================================="