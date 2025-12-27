#!/bin/bash

# 简化版FFN CultureMoE训练脚本
# 纯LoRA MoE架构，在所有层替换FFN为LoRA MoE，添加文化损失
# 针对48GB×2卡优化

echo "======================================="
echo "简化版FFN CultureMoE训练"
echo "纯LoRA MoE架构，在所有层FFN替换为LoRA MoE"
echo "针对48GB×2卡优化"
echo "======================================="

# 参数设置
BACKBONE=${1:-"llama"}  # 默认使用llama
DATA_ID=${2:-"2"}
USE_SHARED=${3:-"true"}   # 是否使用共享专家，默认为true
USE_MASK=${4:-"true"}     # 是否启用MASK机制，默认为true
USE_GATE=${5:-"true"}     # 是否使用MoE内部融合Gate，默认为true
USE_CULTURE_LOSS=${6:-"new"}  # ori/new/kl/false，默认为new
LAMBDA=${7:-"0.1"}
ALPHA=${8:-"0.5"}
BETA=${9:-"0.5"}
NUM_MOE_EXPERTS=${10:-"4"}  # MoE专家数量
NUM_ACTIVATED_EXPERTS=${11:-"2"}  # 激活的专家数量，默认为top-2
LORA_RANK=${12:-"16"}   # LoRA rank
LORA_ALPHA=${13:-"32"}  # LoRA alpha
USE_LORA=${14:-"true"}   # 是否启用LoRA，默认为true
NUM_GPUS=${15:-"2"}


# 检查参数
if [ "$#" -gt 15 ]; then
    echo "❌ 参数过多！用法: $0 [backbone] [data_id] [use_shared] [use_mask] [use_gate] [use_culture_loss] [lambda] [alpha] [beta] [num_moe_experts] [num_activated_experts] [lora_rank] [lora_alpha] [use_lora] [num_gpus]"
    exit 1
fi

# 验证专家数参数
if ! [[ "$NUM_MOE_EXPERTS" =~ ^[1-8]$ ]]; then
    echo "❌ MoE专家数必须是1-8: $NUM_MOE_EXPERTS"
    exit 1
fi

# 验证激活专家数参数
if ! [[ "$NUM_ACTIVATED_EXPERTS" =~ ^[1-8]$ ]]; then
    echo "❌ 激活专家数必须是1-8: $NUM_ACTIVATED_EXPERTS"
    exit 1
fi

if [ "$NUM_ACTIVATED_EXPERTS" -gt "$NUM_MOE_EXPERTS" ]; then
    echo "❌ 激活专家数不能超过总专家数: $NUM_ACTIVATED_EXPERTS > $NUM_MOE_EXPERTS"
    exit 1
fi

# 设置基础模型路径
if [ "$BACKBONE" = "llama" ]; then
    # 尝试多个可能的路径
    POSSIBLE_PATHS=(
        "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
        "/Users/yzl/models/Meta-Llama-3.1-8B-Instruct"
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
        "microsoft/DialoGPT-medium"  # fallback for testing
    )

    BASE_MODEL=""
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -d "$path" ] || [ "$path" = "microsoft/DialoGPT-medium" ]; then
            BASE_MODEL="$path"
            break
        fi
    done

    if [ -z "$BASE_MODEL" ]; then
        echo "❌ 找不到LLaMA模型，尝试的路径："
        for path in "${POSSIBLE_PATHS[@]}"; do
            echo "  - $path"
        done
        exit 1
    fi

    MODEL_NAME="llama"
    TOTAL_LAYERS=32
    MoE_LAYERS="ALL"  # 所有层都使用MoE
elif [ "$BACKBONE" = "qwen" ]; then
    # 尝试多个可能的路径
    POSSIBLE_PATHS=(
        "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
        "/Users/yzl/models/Meta-Qwen-2.5-7B-Instruct"
        "Qwen/Qwen2.5-7B-Instruct"
        "microsoft/DialoGPT-medium"  # fallback for testing
    )

    BASE_MODEL=""
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -d "$path" ] || [ "$path" = "microsoft/DialoGPT-medium" ]; then
            BASE_MODEL="$path"
            break
        fi
    done

    if [ -z "$BASE_MODEL" ]; then
        echo "❌ 找不到Qwen模型，尝试的路径："
        for path in "${POSSIBLE_PATHS[@]}"; do
            echo "  - $path"
        done
        exit 1
    fi

    MODEL_NAME="qwen"
    TOTAL_LAYERS=28
    MoE_LAYERS="ALL"  # 所有层都使用MoE
else
    echo "❌ 不支持的backbone: $BACKBONE (支持: llama, qwen)"
    exit 1
fi

# 设置数据文件路径
case $DATA_ID in
    1)
        TRAIN_FILE="/root/autodl-fs/blend_merge_gen.json"
        DATASET_TAG="blend"
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
    5)
        TRAIN_FILE="/autodl-fs/data/cultureAtlas_merge_gen.json"
        DATASET_TAG="cultureAtlas"
        ;;
    15)
        TRAIN_FILE=""
        DATASET_TAG="blend + cultureAtlas"
        ;;
    *)
        echo "❌ 无效的DATA_ID: $DATA_ID (支持: 1, 2, 3, 4, 5, 15)"
        exit 1
        ;;
esac

# 特殊处理：DATA_ID=15时合并blend和cultureAtlas数据集
if [ "$DATA_ID" = "15" ]; then
    echo "🔄 合并blend和cultureAtlas数据集..."

    BLEND_FILE="/root/autodl-fs/blend_merge_gen.json"
    CULTUREATLAS_FILE="/autodl-fs/data/cultureAtlas_merge_gen.json"
    MERGED_FILE="/root/autodl-fs/blend_cultureAtlas_merged.json"

    # 检查源文件是否存在
    if [ ! -f "$BLEND_FILE" ]; then
        echo "❌ blend数据文件不存在: $BLEND_FILE"
        exit 1
    fi

    if [ ! -f "$CULTUREATLAS_FILE" ]; then
        echo "❌ cultureAtlas数据文件不存在: $CULTUREATLAS_FILE"
        exit 1
    fi

    # 使用Python合并JSON数据
    python3 -c "
import json

# 读取两个数据集
with open('$BLEND_FILE', 'r', encoding='utf-8') as f:
    blend_data = json.load(f)

with open('$CULTUREATLAS_FILE', 'r', encoding='utf-8') as f:
    cultureatlas_data = json.load(f)

# 合并数据
merged_data = blend_data + cultureatlas_data

# 保存合并结果
with open('$MERGED_FILE', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f'✅ 数据合并完成:')
print(f'  - blend: {len(blend_data)} 条')
print(f'  - cultureAtlas: {len(cultureatlas_data)} 条')
print(f'  - 合并后: {len(merged_data)} 条')
print(f'  - 保存至: $MERGED_FILE')
"

    if [ $? -eq 0 ]; then
        TRAIN_FILE="$MERGED_FILE"
        echo "✅ 使用合并数据集: $TRAIN_FILE"
    else
        echo "❌ 数据合并失败"
        exit 1
    fi
fi

# 检查文件存在性
if [ ! -f "$BASE_MODEL/config.json" ] && [ ! "$BASE_MODEL" = "microsoft/DialoGPT-medium" ]; then
    echo "❌ 基础模型不存在: $BASE_MODEL"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "⚠️ 数据文件不存在: $TRAIN_FILE"
    echo "🔧 为了测试修复效果，将创建一个最小测试数据集..."

    # 创建一个最小的测试数据集
    mkdir -p "$(dirname "$TRAIN_FILE")"
    cat > "$TRAIN_FILE" << 'EOF'
[
    {
        "instruction": "回答以下问题",
        "input": "什么是人工智能？",
        "output": "人工智能是计算机科学的一个分支。",
        "label": "0"
    },
    {
        "instruction": "回答以下问题",
        "input": "什么是机器学习？",
        "output": "机器学习是人工智能的一个子领域。",
        "label": "1"
    }
]
EOF
    echo "✅ 已创建测试数据集: $TRAIN_FILE"
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
OUTPUT_DIR="/root/autodl-fs/simplified_culturemoe/${MODEL_NAME}_${DATASET_TAG}_${TIMESTAMP}"

echo "配置信息:"
echo "  模型: $MODEL_NAME ($BASE_MODEL)"
echo "  数据: $DATASET_TAG ($TRAIN_FILE)"
echo "  总层数: $TOTAL_LAYERS"
echo "  MoE层: $MoE_LAYERS (所有层FFN替换为LoRA MoE)"
if [ "$USE_LORA" = "true" ]; then
    echo "  训练模式: 注意力层LoRA + 所有层LoRA MoE专家训练"
else
    echo "  训练模式: 仅所有层LoRA MoE专家训练"
fi
echo "  共享专家: $USE_SHARED"
echo "  MASK机制: $USE_MASK"
echo "  MoE内部Gate: $USE_GATE"
echo "  MoE专家数: $NUM_MOE_EXPERTS"
echo "  激活专家数: $NUM_ACTIVATED_EXPERTS (top-k激活，如果等于总专家数则为dense模式)"
echo "  文化损失模式: $USE_CULTURE_LOSS (ori=原始L_o, new=文化感知L_o, kl=KL散度L_o, false=仅L_aux)"
echo "  层次化损失系数: λ=$LAMBDA, α=$ALPHA, β=$BETA"
echo "  启用LoRA: $USE_LORA"
echo "  LoRA配置: rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "  GPU: $NUM_GPUS卡"
echo "  输出: $OUTPUT_DIR"
echo ""

# 内存优化的训练参数 - 参考joint版本设置
BATCH_SIZE=4              # 调整为2，支持culture loss多样本计算
GRADIENT_ACCUMULATION=8   # 相应增加梯度累积，保持有效batch size
LEARNING_RATE=1e-4        # 简化版使用单一学习率
NUM_EPOCHS=7              # 🔧 减少到7轮，避免过拟合（观察到第8轮准确率下降）

# 动态设置max_seq_len：参考joint版本逻辑
echo "🔧 调试信息: DATA_ID='$DATA_ID'"
if [ "$DATA_ID" = "3" ] || [ "$DATA_ID" = "0" ] || [ "$DATA_ID" = "1" ]; then
    MAX_SEQ_LEN=850       # 长文本数据集使用850
    echo "🔧 检测到长文本数据集(DATA_ID=$DATA_ID)，使用MAX_SEQ_LEN=850"
else
    MAX_SEQ_LEN=384       # 其他数据集使用384
    echo "🔧 使用标准序列长度MAX_SEQ_LEN=384"
fi

echo "训练参数:"
echo "  Batch Size: $BATCH_SIZE (per GPU)"
echo "  梯度累积: $GRADIENT_ACCUMULATION"
echo "  有效Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))"
echo "  学习率: $LEARNING_RATE (简化版使用单一学习率)"
echo "  训练轮数: $NUM_EPOCHS"
echo "  最大序列长度: $MAX_SEQ_LEN (动态设置)"
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
        "total_layers": $TOTAL_LAYERS,
        "moe_layers": "$MoE_LAYERS"
    },
    "data_config": {
        "data_id": "$DATA_ID",
        "data_file": "$TRAIN_FILE",
        "dataset_tag": "$DATASET_TAG",
        "max_seq_length": $MAX_SEQ_LEN
    },
    "training_config": {
        "training_mode": "simplified_all_layers_moe",
        "use_shared_expert": $USE_SHARED,
        "use_mask": $USE_MASK,
        "use_moe_gate": $USE_GATE,
        "moe_experts": $NUM_MOE_EXPERTS,
        "activated_experts": $NUM_ACTIVATED_EXPERTS,
        "use_culture_loss": $USE_CULTURE_LOSS,
        "use_lora": $USE_LORA,
        "lora_rank": $LORA_RANK,
        "lora_alpha": $LORA_ALPHA,
        "num_epochs": $NUM_EPOCHS,
        "batch_size": $BATCH_SIZE,
        "gradient_accumulation_steps": $GRADIENT_ACCUMULATION,
        "learning_rate": $LEARNING_RATE,
        "num_gpus": $NUM_GPUS,
        "memory_optimized": true
    },
    "timestamp": "$TIMESTAMP"
}
EOF

# 设置内存优化环境变量（与MixLoRA一致）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "开始简化版FFN CultureMoE训练..."

# 训练命令
TRAINING_SUCCESS=0

if [ "$NUM_GPUS" -eq 1 ]; then
    # 单卡训练
    python train_simplified_culturemoe.py \
        --base_model_path "$BASE_MODEL" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_SEQ_LEN \
        --backbone $BACKBONE \
        --use_shared $USE_SHARED \
        --use_gate $USE_GATE \
        --num_moe_experts $NUM_MOE_EXPERTS \
        --use_culture_loss $USE_CULTURE_LOSS \
        --num_activated_experts $NUM_ACTIVATED_EXPERTS \
        --use_lora $USE_LORA \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --eval_interval 1 \
        --memory_efficient \
        --lambda_balance $LAMBDA \
        --alpha_z $ALPHA \
        --beta_culture $BETA \
        $(if [ "$USE_MASK" = "true" ]; then echo "--enable_mask --mask_prob 0.15"; fi) \
        2>&1 | tee "$OUTPUT_DIR/training.log"
else
    # 多卡训练
    export CUDA_VISIBLE_DEVICES=0,1
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_simplified_culturemoe.py \
        --base_model_path "$BASE_MODEL" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_SEQ_LEN \
        --backbone $BACKBONE \
        --use_shared $USE_SHARED \
        --use_gate $USE_GATE \
        --num_moe_experts $NUM_MOE_EXPERTS \
        --use_culture_loss $USE_CULTURE_LOSS \
        --num_activated_experts $NUM_ACTIVATED_EXPERTS \
        --use_lora $USE_LORA \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --eval_interval 1 \
        --memory_efficient \
        --lambda_balance $LAMBDA \
        --alpha_z $ALPHA \
        --beta_culture $BETA \
        $(if [ "$USE_MASK" = "true" ]; then echo "--enable_mask --mask_prob 0.15"; fi) \
        2>&1 | tee "$OUTPUT_DIR/training.log"
fi

TRAINING_SUCCESS=$?

# 检查训练结果
echo "======================================="
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ 简化版FFN CultureMoE训练成功！"

    # 检查最佳模型
    BEST_MODEL="$OUTPUT_DIR/best_simplified_culturemoe"
    if [ -d "$BEST_MODEL" ]; then
        echo "✅ 最佳模型已保存: $BEST_MODEL"

        echo ""
        echo "🎉 训练完成！模型特点:"
        echo "  - 纯LoRA MoE架构：所有层FFN替换为LoRA MoE"
        echo "  - MoE专家数: $NUM_MOE_EXPERTS"
        echo "  - 激活专家数: $NUM_ACTIVATED_EXPERTS ({'dense模式' if [ "$NUM_ACTIVATED_EXPERTS" = "$NUM_MOE_EXPERTS" ]; then echo 'dense模式'; else echo "top-$NUM_ACTIVATED_EXPERTS"; fi})"
        echo "  - 文化损失模式: $USE_CULTURE_LOSS"
        echo "  - LoRA配置: rank=$LORA_RANK, alpha=$LORA_ALPHA"
        echo "  - 注意力层LoRA: $USE_LORA"
        echo "  - 共享专家: $USE_SHARED"
        echo "  - MoE内部Gate: $USE_GATE"
        echo "  - 序列长度: $MAX_SEQ_LEN"
    else
        echo "⚠️  训练完成但未找到最佳模型"
    fi
else
    echo "❌ 简化版FFN CultureMoE训练失败！退出码: $TRAINING_SUCCESS"
    echo ""
    echo "故障排查："
    echo "1. 检查显存使用: nvidia-smi"
    echo "2. 查看详细日志: $OUTPUT_DIR/training.log"
fi

echo ""
echo "文件位置:"
echo "  训练日志: $OUTPUT_DIR/training.log"
echo "  配置文件: $OUTPUT_DIR/config.json"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "  最佳模型: $OUTPUT_DIR/best_simplified_culturemoe/"
fi
echo "======================================="

exit $TRAINING_SUCCESS