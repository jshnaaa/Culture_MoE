#!/bin/bash

# 联合训练脚本：同时训练预训练LoRA适配器 + 新增MoE处理层
# 实现端到端的LoRA+MoE联合优化训练
# 针对48GB×2卡优化

echo "======================================="
echo "联合训练：LoRA + MoE 端到端优化"
echo "同时训练预训练LoRA适配器和新增MoE层"
echo "======================================="

# 参数设置
BACKBONE=${1:-"llama"}  # 默认使用llama
DATA_ID=${2:-"2"}
USE_SHARED=${3:-"false"}  # 是否使用共享专家，默认为false
USE_GATE=${4:-"false"}    # 是否使用MoE内部融合Gate，默认为false
NUM_MOE_EXPERTS=${5:-"4"}  # MoE专家数量
USE_CULTURE_LOSS=${6:-"csl"}  # ori/new/kl/csl/false，默认使用CSL文化相似性损失
NUM_ACTIVATED_EXPERTS=${7:-"2"}  # 激活的专家数量，默认为top-2
USE_CULTURE_ROUTER=${8:-"false"}  # 是否使用文化感知冲突检测路由，默认为false
USE_LORA=${9:-"true"}   # 是否启用预训练LoRA微调，默认为true
NUM_GPUS=${10:-"2"}
LORA_RANK=${11:-"16"}   # LoRA rank
LORA_ALPHA=${12:-"32"}  # LoRA alpha

# 检查参数
if [ "$#" -gt 12 ]; then
    echo "❌ 参数过多！用法: $0 [backbone] [data_id] [use_shared] [use_gate] [num_moe_experts] [use_culture_loss] [num_activated_experts] [use_culture_router] [use_lora] [num_gpus] [lora_rank] [lora_alpha]"
    exit 1
fi

# 设置基础模型路径
if [ "$BACKBONE" = "llama" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="llama"
    TOTAL_LAYERS=32
elif [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="qwen"
    TOTAL_LAYERS=28
else
    echo "❌ 不支持的backbone: $BACKBONE (支持: llama, qwen)"
    exit 1
fi

# 设置数据文件路径
case $DATA_ID in
    1)
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_TAG="unified"
        ;;
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

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/joint_lora_moe/${MODEL_NAME}_${DATASET_TAG}_shared${USE_SHARED}_gate${USE_GATE}_${TIMESTAMP}"

echo "配置信息:"
echo "  模型: $MODEL_NAME ($BASE_MODEL)"
echo "  数据: $DATASET_TAG ($TRAIN_FILE)"
echo "  总层数: $TOTAL_LAYERS"
if [ "$USE_LORA" = "true" ]; then
    echo "  训练模式: 联合训练 (预训练LoRA + MoE专家LoRA + Router)"
else
    echo "  训练模式: MoE专家训练 (基座冻结，仅训练MoE专家LoRA + Router)"
fi
echo "  共享专家: $USE_SHARED"
echo "  MoE内部Gate: $USE_GATE"
echo "  MoE专家数: $NUM_MOE_EXPERTS"
echo "  激活专家数: $NUM_ACTIVATED_EXPERTS (top-k激活，如果等于总专家数则为dense模式)"
echo "  文化损失模式: $USE_CULTURE_LOSS (ori=原始L_o, new=文化感知L_o, kl=KL散度L_o, csl=CSL三组件损失, false=仅L_aux)"
echo "  文化感知路由: $USE_CULTURE_ROUTER (true=启用文化感知冲突检测路由, false=标准路由)"
echo "  启用预训练LoRA: $USE_LORA"
echo "  LoRA配置: rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "  GPU: $NUM_GPUS卡"
echo "  输出: $OUTPUT_DIR"
echo ""

# 内存优化的训练参数 - 针对长序列优化
BATCH_SIZE=2              # 调整为2，支持culture loss多样本计算
GRADIENT_ACCUMULATION=16   # 相应增加梯度累积，保持有效batch size

# 🔧 根据backbone设置不同的学习率
if [ "$BACKBONE" = "llama" ]; then
    LEARNING_RATE_BASE=2e-4   # LLaMA: 基础LoRA学习率
    LEARNING_RATE_MOE=8e-5    # LLaMA: MoE学习率
    echo "🔧 LLaMA学习率: Base=${LEARNING_RATE_BASE}, MoE=${LEARNING_RATE_MOE}"
elif [ "$BACKBONE" = "qwen" ]; then
    LEARNING_RATE_BASE=1e-4   # Qwen: 更保守的基础LoRA学习率
    LEARNING_RATE_MOE=2e-5    # Qwen: 更保守的MoE学习率
    echo "🔧 Qwen学习率: Base=${LEARNING_RATE_BASE}, MoE=${LEARNING_RATE_MOE}"
fi

NUM_EPOCHS=8              # 训练轮数

# 动态设置max_seq_len：normad等长文本数据集需要更长的序列长度
echo "🔧 调试信息: DATA_ID='$DATA_ID'"
if [ "$DATA_ID" = "3" ] || [ "$DATA_ID" = "0" ] || [ "$DATA_ID" = "1" ]; then
    MAX_SEQ_LEN=850       # 长文本数据集使用850，给答案部分留更多空间
    echo "🔧 检测到长文本数据集(DATA_ID=$DATA_ID)，使用MAX_SEQ_LEN=850"
else
    MAX_SEQ_LEN=384       # 其他数据集使用384
    echo "🔧 使用标准序列长度MAX_SEQ_LEN=384"
fi

# 验证变量设置
echo "🔧 最终MAX_SEQ_LEN设置为: $MAX_SEQ_LEN"

echo "训练参数:"
echo "  Batch Size: $BATCH_SIZE (per GPU)"
echo "  梯度累积: $GRADIENT_ACCUMULATION"
echo "  有效Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))"
echo "  基础LoRA学习率: $LEARNING_RATE_BASE (针对${BACKBONE}优化)"
echo "  MoE学习率: $LEARNING_RATE_MOE (针对${BACKBONE}优化)"
echo "  训练轮数: $NUM_EPOCHS"
echo "  最大序列长度: $MAX_SEQ_LEN (动态设置)"
if [ "$BACKBONE" = "llama" ]; then
    echo "  MoE影响权重: 0.5 (LLaMA配置)"
elif [ "$BACKBONE" = "qwen" ]; then
    case $DATA_ID in
        2)
            echo "  MoE影响权重: 0.05 (Qwen + CulturalBench配置)"
            ;;
        4|1)
            echo "  MoE影响权重: 0.1 (Qwen + CultureLLM配置)"
            ;;
        3)
            echo "  MoE影响权重: 0.2 (Qwen + Normad配置)"
            ;;
        *)
            echo "  MoE影响权重: 0.2 (Qwen默认配置)"
            ;;
    esac
fi
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 保存配置
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "model_config": {
        "backbone": "$BACKBONE",
        "base_model": "$BASE_MODEL",
        "model_name": "$MODEL_NAME",
        "total_layers": $TOTAL_LAYERS
    },
    "data_config": {
        "data_id": "$DATA_ID",
        "data_file": "$TRAIN_FILE",
        "dataset_tag": "$DATASET_TAG",
        "max_seq_length": $MAX_SEQ_LEN
    },
    "training_config": {
        "training_mode": "joint_lora_moe",
        "use_shared_expert": $USE_SHARED,
        "use_moe_gate": $USE_GATE,
        "moe_experts": $NUM_MOE_EXPERTS,
        "activated_experts": $NUM_ACTIVATED_EXPERTS,
        "use_culture_loss": $USE_CULTURE_LOSS,
        "use_culture_router": $USE_CULTURE_ROUTER,
        "use_lora": $USE_LORA,
        "lora_rank": $LORA_RANK,
        "lora_alpha": $LORA_ALPHA,
        "num_epochs": $NUM_EPOCHS,
        "batch_size": $BATCH_SIZE,
        "gradient_accumulation_steps": $GRADIENT_ACCUMULATION,
        "learning_rate_base": $LEARNING_RATE_BASE,
        "learning_rate_moe": $LEARNING_RATE_MOE,
        "num_gpus": $NUM_GPUS,
        "memory_optimized": true
    },
    "timestamp": "$TIMESTAMP"
}
EOF

# 设置内存优化环境变量 - 修复CUDA内存分配器问题
# 移除expandable_segments配置，避免与某些PyTorch版本冲突
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
# 移除TORCH_USE_CUDA_DSA，可能导致内存分配问题
# export TORCH_USE_CUDA_DSA=1

echo "开始联合训练 LoRA + MoE..."

# 训练命令
TRAINING_SUCCESS=0

if [ "$NUM_GPUS" -eq 1 ]; then
    # 单卡训练
    python train_joint_lora_moe.py \
        --base_model_path "$BASE_MODEL" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --data_id "$DATA_ID" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
        --learning_rate_base $LEARNING_RATE_BASE \
        --learning_rate_moe $LEARNING_RATE_MOE \
        --max_length $MAX_SEQ_LEN \
        --backbone $BACKBONE \
        --num_moe_experts $NUM_MOE_EXPERTS \
        --num_activated_experts $NUM_ACTIVATED_EXPERTS \
        --use_culture_loss $USE_CULTURE_LOSS \
        --use_culture_router $USE_CULTURE_ROUTER \
        --use_lora $USE_LORA \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --eval_interval 1 \
        --memory_efficient \
        2>&1 | tee "$OUTPUT_DIR/training.log"
else
    # 多卡训练
    export CUDA_VISIBLE_DEVICES=0,1
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_joint_lora_moe.py \
        --base_model_path "$BASE_MODEL" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --data_id "$DATA_ID" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
        --learning_rate_base $LEARNING_RATE_BASE \
        --learning_rate_moe $LEARNING_RATE_MOE \
        --max_length $MAX_SEQ_LEN \
        --backbone $BACKBONE \
        --num_moe_experts $NUM_MOE_EXPERTS \
        --num_activated_experts $NUM_ACTIVATED_EXPERTS \
        --use_culture_loss $USE_CULTURE_LOSS \
        --use_culture_router $USE_CULTURE_ROUTER \
        --use_lora $USE_LORA \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --eval_interval 1 \
        --memory_efficient \
        2>&1 | tee "$OUTPUT_DIR/training.log"
fi

TRAINING_SUCCESS=$?

# 检查训练结果
echo "======================================="
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ 联合训练 LoRA + MoE 成功！"

    # 检查最佳模型
    BEST_MODEL="$OUTPUT_DIR/best_joint_model"
    if [ -d "$BEST_MODEL" ]; then
        echo "✅ 最佳模型已保存: $BEST_MODEL"

        echo ""
        echo "🎉 训练完成！模型特点:"
        echo "  - 联合训练: 同时优化预训练LoRA + 新增MoE层"
        echo "  - LoRA配置: rank=$LORA_RANK, alpha=$LORA_ALPHA"
        echo "  - MoE专家数: $NUM_MOE_EXPERTS"
        echo "  - 共享专家: $USE_SHARED"
        echo "  - MoE内部Gate: $USE_GATE"
        echo "  - 分层学习率: Base LoRA=$LEARNING_RATE_BASE, MoE=$LEARNING_RATE_MOE"
        echo "  - 文化损失模式: $USE_CULTURE_LOSS"
        echo "  - 文化感知路由: $USE_CULTURE_ROUTER"
        echo "  - 序列长度: $MAX_SEQ_LEN"
    else
        echo "⚠️  训练完成但未找到最佳模型"
    fi
else
    echo "❌ 联合训练 LoRA + MoE 失败！退出码: $TRAINING_SUCCESS"
    echo "故障排查："
    echo "1. 检查显存使用: nvidia-smi"
    echo "2. 查看详细日志: $OUTPUT_DIR/training.log"
fi

echo ""
echo "文件位置:"
echo "  训练日志: $OUTPUT_DIR/training.log"
echo "  配置文件: $OUTPUT_DIR/config.json"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "  最佳模型: $OUTPUT_DIR/best_joint_model/"
fi

exit $TRAINING_SUCCESS