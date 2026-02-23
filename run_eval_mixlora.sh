#!/bin/bash

# ============================================================
# MixLoRA 模型评估脚本
# 用于评估通过 run_ft_mixlora.sh 训练的 MixLoRA 模型
# 支持从保存的训练参数还原完整模型并在测试集上评估
#
# 使用方法：
#   sh run_eval_mixlora.sh <model_path> [backbone] [data_id] [num_gpus]
#
# 参数说明：
#   model_path: 训练输出目录（包含 best_mixlora/ 的目录）
#   backbone: llama 或 qwen (默认 llama)
#   data_id: 0=使用pkl文件测试集, 2=CulturalBench, 3=NormAD, 4=CultureLLM, 5=cultureAtlas (默认 0)
#   num_gpus: GPU数量 (默认 1)
#
# 示例：
#   # 使用 pkl 文件中的测试集评估（默认，自动查找所有pkl文件）
#   sh run_eval_mixlora.sh /root/autodl-fs/mixlora/LLaMA_3.1-8B-Instruct_CulturalBench_CultureLLM_20250220_123456
#
#   # 使用特定数据集评估
#   sh run_eval_mixlora.sh /path/to/model llama 2
#
#   # 使用 Qwen  backbone
#   sh run_eval_mixlora.sh /path/to/model qwen 0
# ============================================================

echo "======================================="
echo "MixLoRA 模型测试集评估"
echo "从训练参数还原完整模型并评估"
echo "======================================="

# 参数设置
MODEL_PATH=${1}  # 训练输出目录（必须提供）
BACKBONE=${2:-"llama"}  # 模型骨干：llama/qwen
DATA_ID=${3:-"0"}  # 数据集ID，0=使用pkl文件，2=CulturalBench，4=CultureLLM
NUM_GPUS=${4:-"1"}  # GPU数量

# 检查必需参数
if [ -z "$MODEL_PATH" ]; then
    echo "❌ 错误：必须提供模型路径"
    echo "用法: $0 <model_path> [backbone] [data_id] [num_gpus]"
    echo "示例: $0 /root/autodl-fs/mixlora/LLaMA_3.1-8B-Instruct_CulturalBench_CultureLLM_20250220_123456"
    exit 1
fi

# 检查参数数量
if [ "$#" -gt 4 ]; then
    echo "❌ 参数过多！用法: $0 <model_path> [backbone] [data_id] [num_gpus]"
    exit 1
fi

# 设置基础模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen"
    TOTAL_LAYERS=28
else
    BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA"
    TOTAL_LAYERS=32
fi

# 设置数据文件路径
 case $DATA_ID in
    0)
        # 使用pkl文件中的测试集
        TRAIN_FILE="pkl_split"  # 特殊标记，表示使用pkl文件
        DATASET_TAG="pkl_test"
        echo "📋 使用模型目录中的pkl文件测试集"
        ;;
    2)
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        echo "📋 使用 CulturalBench 完整数据集"
        ;;
    3)
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        echo "📋 使用 NormAD 完整数据集"
        ;;
    4)
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        echo "📋 使用 CultureLLM 完整数据集"
        ;;
    5)
        TRAIN_FILE="/root/autodl-fs/cultureAtlas_merge_gen.json"
        DATASET_TAG="cultureAtlas"
        echo "📋 使用 cultureAtlas 完整数据集"
        ;;
    *)
        echo "❌ 无效的DATA_ID: $DATA_ID (支持: 0=pkl文件, 2=CulturalBench, 3=NormAD, 4=CultureLLM, 5=cultureAtlas)"
        exit 1
        ;;
esac

# 检查提供的模型路径
echo "🔍 检查 MixLoRA 模型路径: $MODEL_PATH"

# 查找 best_mixlora 目录
if [ -d "$MODEL_PATH/best_mixlora" ]; then
    FOUND_MODEL="$MODEL_PATH/best_mixlora"
    echo "✅ 找到 MixLoRA 模型: $FOUND_MODEL"
elif [ -f "$MODEL_PATH/mixlora_config.json" ] && [ -f "$MODEL_PATH/mixlora_weights.pt" ]; then
    FOUND_MODEL="$MODEL_PATH"
    echo "✅ 用户提供的路径本身就是 MixLoRA 模型目录: $FOUND_MODEL"
else
    echo "❌ 未找到有效的 MixLoRA 模型目录"
    echo "   请确保路径包含 best_mixlora/ 目录或直接的 mixlora_config.json 和 mixlora_weights.pt 文件"
    exit 1
fi

# 验证模型文件
if [ ! -f "$FOUND_MODEL/mixlora_config.json" ]; then
    echo "❌ MixLoRA 配置文件不存在: $FOUND_MODEL/mixlora_config.json"
    exit 1
fi

if [ ! -f "$FOUND_MODEL/mixlora_weights.pt" ]; then
    echo "❌ MixLoRA 权重文件不存在: $FOUND_MODEL/mixlora_weights.pt"
    exit 1
fi

# 设置pkl文件路径（用于DATA_ID=0）
PKL_FILES=""
if [ "$DATA_ID" = "0" ]; then
    # pkl文件在模型父目录中
    MODEL_PARENT_DIR=$(dirname "$FOUND_MODEL")

    echo "🔍 查找pkl文件:"
    echo "  - FOUND_MODEL: $FOUND_MODEL"
    echo "  - MODEL_PARENT_DIR: $MODEL_PARENT_DIR"

    # 首先查找多数据集模式的pkl文件（data_split_8_1_1_*.pkl）
    MULTI_PKL_FILES=$(find "$MODEL_PARENT_DIR" -maxdepth 1 -name "data_split_8_1_1_*.pkl" 2>/dev/null | sort)
    SINGLE_PKL_FILE="$MODEL_PARENT_DIR/data_split_8_1_1.pkl"

    if [ -n "$MULTI_PKL_FILES" ]; then
        # 找到多个pkl文件（联合训练模式）
        PKL_FILES="$MULTI_PKL_FILES"
        echo "✅ 检测到多数据集训练模式，找到多个pkl文件:"
        for pkl_file in $PKL_FILES; do
            echo "  - $(basename "$pkl_file")"
        done
        echo "  - 将分别对每个pkl文件进行独立评估"
    elif [ -f "$SINGLE_PKL_FILE" ]; then
        # 找到单个pkl文件（传统模式）
        PKL_FILES="$SINGLE_PKL_FILE"
        echo "✅ 检测到单数据集训练模式，找到pkl文件:"
        echo "  - $(basename "$SINGLE_PKL_FILE")"
    else
        # 没有找到任何pkl文件
        echo "❌ 未找到数据集划分文件"
        echo "  - 期望的单数据集文件: $SINGLE_PKL_FILE"
        echo "  - 期望的多数据集文件: data_split_8_1_1_*.pkl"
        echo "请确保训练时保存了数据集划分信息"
        exit 1
    fi
fi

# 设置最大序列长度
MAX_SEQ_LEN=512

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/mixlora_eval_results/${MODEL_NAME}_${DATASET_TAG}_${TIMESTAMP}"

echo ""
echo "评估配置信息:"
echo "  基础模型: $MODEL_NAME ($BASE_MODEL)"
echo "  MixLoRA 模型: $FOUND_MODEL"
echo "  数据集: $DATASET_TAG (DATA_ID=$DATA_ID)"
if [ "$DATA_ID" = "0" ]; then
    echo "  数据来源: pkl测试集"
else
    echo "  数据来源: 完整数据集 ($TRAIN_FILE)"
fi
echo "  最大序列长度: $MAX_SEQ_LEN"
echo "  GPU数量: $NUM_GPUS"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 保存评估配置
cat > "$OUTPUT_DIR/eval_config.json" << EOF
{
    "model_config": {
        "backbone": "$BACKBONE",
        "base_model": "$BASE_MODEL",
        "model_name": "$MODEL_NAME",
        "total_layers": $TOTAL_LAYERS,
        "mixlora_model_path": "$FOUND_MODEL"
    },
    "data_config": {
        "data_id": "$DATA_ID",
        "dataset_tag": "$DATASET_TAG",
        "use_pkl_split": $(if [ "$DATA_ID" = "0" ]; then echo "true"; else echo "false"; fi)
    },
    "eval_config": {
        "num_gpus": $NUM_GPUS,
        "timestamp": "$TIMESTAMP",
        "max_seq_length": $MAX_SEQ_LEN
    }
}
EOF

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "开始评估 MixLoRA 模型..."

# 检查是否为多pkl文件模式
if [ "$DATA_ID" = "0" ] && [ $(echo "$PKL_FILES" | wc -w) -gt 1 ]; then
    # 多pkl文件模式：分别对每个pkl文件进行评估
    echo "🔧 多数据集评估模式：将分别评估每个数据集"

    OVERALL_SUCCESS=0
    PKL_COUNT=0

    for PKL_FILE in $PKL_FILES; do
        PKL_COUNT=$((PKL_COUNT + 1))
        PKL_BASENAME=$(basename "$PKL_FILE" .pkl)

        echo ""
        echo "======================================="
        echo "📊 评估数据集 $PKL_COUNT: $PKL_BASENAME"
        echo "======================================="

        # 为每个pkl文件创建独立的输出目录
        CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_${PKL_BASENAME}"
        mkdir -p "$CURRENT_OUTPUT_DIR"

        # 复制配置文件到当前输出目录
        cp "$OUTPUT_DIR/eval_config.json" "$CURRENT_OUTPUT_DIR/"

        # 评估当前pkl文件
        EVAL_SUCCESS=0

        if [ "$NUM_GPUS" -eq 1 ]; then
            # 单卡评估
            python eval_mixlora.py \
                --base_model_path "$BASE_MODEL" \
                --mixlora_model_path "$FOUND_MODEL" \
                --pkl_file "$PKL_FILE" \
                --output_dir "$CURRENT_OUTPUT_DIR" \
                --backbone "$BACKBONE" \
                --max_length "$MAX_SEQ_LEN" \
                2>&1 | tee "$CURRENT_OUTPUT_DIR/eval.log"
        else
            # 多卡评估
            export CUDA_VISIBLE_DEVICES=0,1
            torchrun \
                --nproc_per_node=$NUM_GPUS \
                --master_port=29700 \
                eval_mixlora.py \
                --base_model_path "$BASE_MODEL" \
                --mixlora_model_path "$FOUND_MODEL" \
                --pkl_file "$PKL_FILE" \
                --output_dir "$CURRENT_OUTPUT_DIR" \
                --backbone "$BACKBONE" \
                --max_length "$MAX_SEQ_LEN" \
                2>&1 | tee "$CURRENT_OUTPUT_DIR/eval.log"
        fi

        EVAL_SUCCESS=$?

        if [ $EVAL_SUCCESS -eq 0 ]; then
            echo "✅ 数据集 $PKL_COUNT ($PKL_BASENAME) 评估成功"
        else
            echo "❌ 数据集 $PKL_COUNT ($PKL_BASENAME) 评估失败"
            OVERALL_SUCCESS=$EVAL_SUCCESS
        fi
    done

    EVAL_SUCCESS=$OVERALL_SUCCESS

    echo ""
    echo "======================================="
    echo "📊 多数据集评估完成汇总"
    echo "======================================="
    echo "  - 总共评估了 $PKL_COUNT 个数据集"
    echo "  - 结果保存在各自的输出目录中:"
    for PKL_FILE in $PKL_FILES; do
        PKL_BASENAME=$(basename "$PKL_FILE" .pkl)
        echo "    * ${OUTPUT_DIR}_${PKL_BASENAME}/"
    done

else
    # 单pkl文件模式或完整数据集模式
    if [ "$DATA_ID" = "0" ]; then
        # 提取单个PKL_FILE
        PKL_FILE=$(echo "$PKL_FILES" | head -n1)
        echo "🔧 单数据集评估模式，使用pkl文件: $(basename "$PKL_FILE")"
        PKL_ARG="--pkl_file $PKL_FILE"
    else
        # 完整数据集模式
        PKL_ARG="--data_file $TRAIN_FILE"
    fi

    EVAL_SUCCESS=0

    if [ "$NUM_GPUS" -eq 1 ]; then
        # 单卡评估
        python eval_mixlora.py \
            --base_model_path "$BASE_MODEL" \
            --mixlora_model_path "$FOUND_MODEL" \
            $PKL_ARG \
            --output_dir "$OUTPUT_DIR" \
            --backbone "$BACKBONE" \
            --max_length "$MAX_SEQ_LEN" \
            2>&1 | tee "$OUTPUT_DIR/eval.log"
    else
        # 多卡评估
        export CUDA_VISIBLE_DEVICES=0,1
        torchrun \
            --nproc_per_node=$NUM_GPUS \
            --master_port=29700 \
            eval_mixlora.py \
            --base_model_path "$BASE_MODEL" \
            --mixlora_model_path "$FOUND_MODEL" \
            $PKL_ARG \
            --output_dir "$OUTPUT_DIR" \
            --backbone "$BACKBONE" \
            --max_length "$MAX_SEQ_LEN" \
            2>&1 | tee "$OUTPUT_DIR/eval.log"
    fi

    EVAL_SUCCESS=$?
fi

# 检查评估结果
echo "======================================="
if [ $EVAL_SUCCESS -eq 0 ]; then
    echo "✅ MixLoRA 模型评估成功！"

    # 显示结果摘要
    echo ""
    echo "📊 评估结果摘要:"
    if [ "$DATA_ID" = "0" ] && [ $(echo "$PKL_FILES" | wc -w) -gt 1 ]; then
        # 多数据集模式
        for PKL_FILE in $PKL_FILES; do
            PKL_BASENAME=$(basename "$PKL_FILE" .pkl)
            CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_${PKL_BASENAME}"
            RESULTS_FILE="$CURRENT_OUTPUT_DIR/eval_results.json"
            if [ -f "$RESULTS_FILE" ]; then
                echo ""
                echo "  $PKL_BASENAME:"
                python3 -c "
import json
try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    if 'accuracy' in results:
        print(f'    准确率: {results[\"accuracy\"]:.4f}')
    if 'total_samples' in results:
        print(f'    测试样本数: {results[\"total_samples\"]}')
    if 'correct_predictions' in results:
        print(f'    正确预测数: {results[\"correct_predictions\"]}')
except Exception as e:
    print(f'    无法解析结果: {e}')
"
            fi
        done
    else
        # 单数据集模式
        RESULTS_FILE="$OUTPUT_DIR/eval_results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 -c "
import json
try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    if 'accuracy' in results:
        print(f'  准确率: {results[\"accuracy\"]:.4f}')
    if 'total_samples' in results:
        print(f'  测试样本数: {results[\"total_samples\"]}')
    if 'correct_predictions' in results:
        print(f'  正确预测数: {results[\"correct_predictions\"]}')
except Exception as e:
    print(f'  无法解析结果: {e}')
"
        fi
    fi
else
    echo "❌ MixLoRA 模型评估失败！退出码: $EVAL_SUCCESS"
    echo "故障排查："
    echo "1. 检查GPU内存: nvidia-smi"
    echo "2. 查看详细日志: $OUTPUT_DIR/eval.log"
    echo "3. 验证模型文件完整性"
fi

echo ""
echo "文件位置:"
echo "  评估日志: $OUTPUT_DIR/eval.log"
echo "  评估配置: $OUTPUT_DIR/eval_config.json"
if [ $EVAL_SUCCESS -eq 0 ]; then
    echo "  评估结果: $OUTPUT_DIR/eval_results.json"
    echo "  详细输出: $OUTPUT_DIR/detailed_results.json"
fi

exit $EVAL_SUCCESS
