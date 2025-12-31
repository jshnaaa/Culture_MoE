#!/bin/bash

# Joint CultureMoE模型评估脚本
# 专门用于评估通过run_joint_lora_moe_training.sh训练的联合模型
# 支持8:1:1数据集划分的测试集评估

echo "======================================="
echo "Joint CultureMoE 模型测试集评估"
echo "专用于联合训练模型 (best_joint_model/)"
echo "======================================="

# 参数设置
MODEL_PATH=${1}  # 默认联合训练模型根目录
BACKBONE=${2:-"llama"}  # 模型骨干：llama/qwen
DATA_ID=${3:-"0"}  # 数据集ID，0=使用pkl文件，1-5=使用完整数据集
USE_SHARED=${4:-"true"}  # 是否使用共享专家，支持消融评估
USE_MASK=${5:-"true"}   # 是否启用MASK机制，支持消融评估
USE_GATE=${6:-"true"}   # 是否使用MoE内部融合Gate，支持消融评估
USE_CULTURE_LOSS=${7:-"csl"}  # 文化损失类型
NUM_MOE_EXPERTS=${8:-"4"}  # MoE专家数量
NUM_ACTIVATED_EXPERTS=${9:-"2"}  # 激活的专家数量
NUM_GPUS=${10:-"1"}  # GPU数量

# 检查参数
if [ "$#" -gt 10 ]; then
    echo "❌ 参数过多！用法: $0 [model_path] [data_id] [backbone] [use_shared] [use_gate] [num_moe_experts] [num_activated_experts] [use_culture_loss] [use_mask] [num_gpus]"
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
    0)
        # 使用pkl文件中的测试集
        TRAIN_FILE="pkl_split"  # 特殊标记，表示使用pkl文件
        DATASET_TAG="pkl_test"
        echo "📋 使用模型目录中的pkl文件测试集"
        ;;
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
        TRAIN_FILE="/root/autodl-fs/cultureAtlas_merge_gen.json"
        DATASET_TAG="cultureAtlas"
        ;;
    *)
        echo "❌ 无效的DATA_ID: $DATA_ID (支持: 0=pkl文件, 1=unified, 2=CulturalBench, 3=normad, 4=cultureLLM, 5=cultureAtlas)"
        exit 1
        ;;
esac

# 检查提供的模型路径
echo "🔍 检查联合训练模型路径: $MODEL_PATH"

# 🔧 修复：检查用户提供的路径是否直接包含best_joint_model
if [ -d "$MODEL_PATH/best_joint_model" ]; then
    # 用户直接提供了包含best_joint_model的目录（这是正确的情况）
    FOUND_MODEL="$MODEL_PATH/best_joint_model"
    echo "✅ 找到联合训练模型: $FOUND_MODEL"
else
    # 🔧 修复：如果用户提供的路径本身就是best_joint_model目录
    if [ -f "$MODEL_PATH/joint_config.json" ] && [ -f "$MODEL_PATH/moe_weights.pt" ]; then
        FOUND_MODEL="$MODEL_PATH"
        echo "✅ 用户提供的路径本身就是联合训练模型目录: $FOUND_MODEL"
    else
        # 在提供的路径下查找子目录
        echo "🔍 在 $MODEL_PATH 中查找联合训练模型..."

        # 查找所有可能的模型目录（按时间倒序）
        if [ "$DATA_ID" = "0" ]; then
            # 对于pkl文件模式，查找任何包含指定backbone的模型目录
            CANDIDATE_DIRS=$(find "$MODEL_PATH" -maxdepth 1 -type d -name "*${MODEL_NAME}_*" 2>/dev/null | sort -r)
            echo "   查找模式: *${MODEL_NAME}_* (pkl文件模式，匹配任何数据集)"
        else
            # 对于特定数据集，使用精确匹配
            CANDIDATE_DIRS=$(find "$MODEL_PATH" -maxdepth 1 -type d -name "*${MODEL_NAME}_${DATASET_TAG}*" 2>/dev/null | sort -r)
            echo "   查找模式: *${MODEL_NAME}_${DATASET_TAG}*"
        fi

        if [ -z "$CANDIDATE_DIRS" ]; then
            echo "❌ 在 $MODEL_PATH 中未找到匹配的模型目录"
            echo "   请检查模型路径和参数设置"
            exit 1
        fi

        # 查找包含best_joint_model的目录
        FOUND_MODEL=""
        for dir in $CANDIDATE_DIRS; do
            if [ -d "$dir/best_joint_model" ]; then
                FOUND_MODEL="$dir/best_joint_model"
                echo "✅ 找到联合训练模型: $FOUND_MODEL"
                break
            fi
        done

        if [ -z "$FOUND_MODEL" ]; then
            echo "❌ 未找到包含best_joint_model/的目录"
            echo "可用的候选目录:"
            for dir in $CANDIDATE_DIRS; do
                echo "  - $dir"
            done
            echo "请确保模型训练已完成并保存了最佳模型"
            exit 1
        fi
    fi
fi

# 验证模型文件
if [ ! -f "$FOUND_MODEL/joint_config.json" ]; then
    echo "❌ 联合模型配置文件不存在: $FOUND_MODEL/joint_config.json"
    exit 1
fi

# 验证MoE权重文件
if [ ! -f "$FOUND_MODEL/moe_weights.pt" ]; then
    echo "❌ MoE权重文件不存在: $FOUND_MODEL/moe_weights.pt"
    exit 1
fi

# 验证LoRA权重目录
if [ ! -d "$FOUND_MODEL/lora_weights" ]; then
    echo "❌ LoRA权重目录不存在: $FOUND_MODEL/lora_weights"
    exit 1
fi

# 设置pkl文件路径（用于DATA_ID=0）
PKL_FILE=""
if [ "$DATA_ID" = "0" ]; then
    # 🔧 修复：根据实际目录结构查找pkl文件
    # 用户提供的目录结构：MODEL_PATH/data_split_8_1_1.pkl
    # FOUND_MODEL指向：MODEL_PATH/best_joint_model/
    # 所以pkl文件在FOUND_MODEL的父目录中

    MODEL_PARENT_DIR=$(dirname "$FOUND_MODEL")
    PKL_FILE="$MODEL_PARENT_DIR/data_split_8_1_1.pkl"

    echo "🔍 查找pkl文件:"
    echo "  - FOUND_MODEL: $FOUND_MODEL"
    echo "  - MODEL_PARENT_DIR: $MODEL_PARENT_DIR"
    echo "  - 期望的PKL文件路径: $PKL_FILE"

    if [ ! -f "$PKL_FILE" ]; then
        echo "❌ 数据集划分文件不存在: $PKL_FILE"
        echo "请确保联合训练时保存了数据集划分信息"
        echo ""
        echo "🔍 调试信息：查找模型目录中的所有文件"
        echo "模型父目录内容:"
        ls -la "$MODEL_PARENT_DIR" 2>/dev/null || echo "无法列出目录内容"
        exit 1
    fi
    echo "✅ 找到数据集划分文件: $PKL_FILE"
fi

# 动态设置max_seq_len：与训练脚本保持一致
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

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/joint_eval_results/${MODEL_NAME}_${DATASET_TAG}_SHARED${USE_SHARED}_MASK_${USE_MASK}_GATE${USE_GATE}_CULTURELOSS_${USE_CULTURE_LOSS}_${TIMESTAMP}"

echo ""
echo "评估配置信息:"
echo "  基础模型: $MODEL_NAME ($BASE_MODEL)"
echo "  联合训练模型: $FOUND_MODEL"
echo "  数据集: $DATASET_TAG (DATA_ID=$DATA_ID)"
if [ "$DATA_ID" = "0" ]; then
    echo "  数据来源: pkl测试集 ($PKL_FILE)"
else
    echo "  数据来源: 完整数据集 ($TRAIN_FILE)"
fi
echo "  共享专家: $USE_SHARED"
echo "  MoE内部Gate: $USE_GATE"
echo "  MoE专家数: $NUM_MOE_EXPERTS"
echo "  激活专家数: $NUM_ACTIVATED_EXPERTS"
echo "  文化损失类型: $USE_CULTURE_LOSS"
echo "  MASK机制: $USE_MASK (支持消融评估)"
echo "  最大序列长度: $MAX_SEQ_LEN (动态设置)"
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
        "joint_model_path": "$FOUND_MODEL",
        "model_name": "$MODEL_NAME",
        "total_layers": $TOTAL_LAYERS
    },
    "data_config": {
        "data_id": "$DATA_ID",
        "dataset_tag": "$DATASET_TAG",
        "data_source": $(if [ "$DATA_ID" = "0" ]; then echo "\"$PKL_FILE\""; else echo "\"$TRAIN_FILE\""; fi),
        "use_pkl_split": $(if [ "$DATA_ID" = "0" ]; then echo "true"; else echo "false"; fi)
    },
    "model_params": {
        "use_shared_expert": $USE_SHARED,
        "use_moe_gate": $USE_GATE,
        "moe_experts": $NUM_MOE_EXPERTS,
        "activated_experts": $NUM_ACTIVATED_EXPERTS,
        "use_culture_loss": "$USE_CULTURE_LOSS",
        "use_mask": "$USE_MASK"
    },
    "eval_config": {
        "num_gpus": $NUM_GPUS,
        "timestamp": "$TIMESTAMP"
    }
}
EOF

echo "开始评估联合训练模型..."

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# 评估命令
EVAL_SUCCESS=0

if [ "$NUM_GPUS" -eq 1 ]; then
    # 单卡评估
    python eval_joint_culturemoe.py \
        --base_model_path "$BASE_MODEL" \
        --joint_model_path "$FOUND_MODEL" \
        --data_file "$TRAIN_FILE" \
        --pkl_file "$PKL_FILE" \
        --data_id "$DATA_ID" \
        --output_dir "$OUTPUT_DIR" \
        --backbone "$BACKBONE" \
        --num_moe_experts "$NUM_MOE_EXPERTS" \
        --num_activated_experts "$NUM_ACTIVATED_EXPERTS" \
        --use_shared "$USE_SHARED" \
        --use_gate "$USE_GATE" \
        --use_culture_loss "$USE_CULTURE_LOSS" \
        --use_mask "$USE_MASK" \
        --max_length "$MAX_SEQ_LEN" \
        2>&1 | tee "$OUTPUT_DIR/eval.log"
else
    # 多卡评估
    export CUDA_VISIBLE_DEVICES=0,1
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29600 \
        eval_joint_culturemoe.py \
        --base_model_path "$BASE_MODEL" \
        --joint_model_path "$FOUND_MODEL" \
        --data_file "$TRAIN_FILE" \
        --pkl_file "$PKL_FILE" \
        --data_id "$DATA_ID" \
        --output_dir "$OUTPUT_DIR" \
        --backbone "$BACKBONE" \
        --num_moe_experts "$NUM_MOE_EXPERTS" \
        --num_activated_experts "$NUM_ACTIVATED_EXPERTS" \
        --use_shared "$USE_SHARED" \
        --use_gate "$USE_GATE" \
        --use_culture_loss "$USE_CULTURE_LOSS" \
        --use_mask "$USE_MASK" \
        --max_length "$MAX_SEQ_LEN" \
        2>&1 | tee "$OUTPUT_DIR/eval.log"
fi

EVAL_SUCCESS=$?

# 检查评估结果
echo "======================================="
if [ $EVAL_SUCCESS -eq 0 ]; then
    echo "✅ 联合模型评估成功！"

    # 检查结果文件
    RESULTS_FILE="$OUTPUT_DIR/eval_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "✅ 评估结果已保存: $RESULTS_FILE"

        # 显示关键指标
        if command -v python3 >/dev/null 2>&1; then
            python3 -c "
import json
try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    print()
    print('📊 评估结果摘要:')
    if 'accuracy' in results:
        print(f'  准确率: {results[\"accuracy\"]:.4f}')
    if 'total_samples' in results:
        print(f'  测试样本数: {results[\"total_samples\"]}')
    if 'correct_predictions' in results:
        print(f'  正确预测数: {results[\"correct_predictions\"]}')
except:
    print('无法解析评估结果文件')
"
        fi
    else
        echo "⚠️  评估完成但未找到结果文件"
    fi
else
    echo "❌ 联合模型评估失败！退出码: $EVAL_SUCCESS"
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