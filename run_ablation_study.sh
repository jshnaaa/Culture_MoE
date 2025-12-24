#!/bin/bash

# 简化版CultureMoE消融实验批处理脚本
#
# 使用方法：
#   bash run_ablation_study.sh <model_path> [backbone] [data_id] [use_shared] [use_mask] [use_gate] [use_culture_loss]

echo "======================================="
echo "简化版CultureMoE消融实验"
echo "自动化测试不同组件配置的性能"
echo "======================================="

# 参数设置
MODEL_PATH="$1"
BACKBONE=${2:-"llama"}        # 默认llama
DATA_ID=${3:-"2"}            # 默认数据集2
USE_SHARED=${4:-"true"}      # 默认保留shared专家
USE_MASK=${5:-"true"}        # 默认保留MASK机制（占位符）
USE_GATE=${6:-"true"}        # 默认保留gate机制
USE_CULTURE_LOSS=${7:-"true"} # 默认保留文化损失

# 参数检查
if [ "$#" -lt 1 ]; then
    echo "❌ 用法: $0 <model_path> [backbone] [data_id] [use_shared] [use_mask] [use_gate] [use_culture_loss]"
    echo ""
    echo "参数说明:"
    echo "  model_path:      训练好的模型路径（必需）"
    echo "  backbone:        基座模型 (llama/qwen, 默认: llama)"
    echo "  data_id:         数据集ID (2/3/4, 默认: 2)"
    echo "  use_shared:      是否保留shared专家 (true/false, 默认: true)"
    echo "  use_mask:        是否保留MASK机制 (true/false, 默认: true, 占位符)"
    echo "  use_gate:        是否保留gate机制 (true/false, 默认: true)"
    echo "  use_culture_loss: 是否保留文化损失 (true/false, 默认: true)"
    echo ""
    echo "示例:"
    echo "  bash run_ablation_study.sh \\"
    echo "    /root/autodl-fs/simplified_culturemoe/llama_CulturalBench_20251224_153803/best_simplified_culturemoe"
    echo ""
    echo "  bash run_ablation_study.sh \\"
    echo "    /path/to/model llama 2 false true false true"
    exit 1
fi

# 验证路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 验证参数
if [[ "$BACKBONE" != "llama" && "$BACKBONE" != "qwen" ]]; then
    echo "❌ 不支持的backbone: $BACKBONE (支持: llama, qwen)"
    exit 1
fi

if [[ "$DATA_ID" != "2" && "$DATA_ID" != "3" && "$DATA_ID" != "4" ]]; then
    echo "❌ 无效的DATA_ID: $DATA_ID (支持: 2, 3, 4)"
    exit 1
fi

# 设置数据文件路径（与训练脚本一致）
case $DATA_ID in
    2)
        DATA_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        ;;
    3)
        DATA_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        ;;
    4)
        DATA_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        ;;
esac

# 验证数据文件存在
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 数据文件不存在: $DATA_FILE"
    exit 1
fi

# 设置基础模型路径（与训练脚本一致）
if [ "$BACKBONE" = "llama" ]; then
    POSSIBLE_PATHS=(
        "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
        "/Users/yzl/models/Meta-Llama-3.1-8B-Instruct"
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
elif [ "$BACKBONE" = "qwen" ]; then
    POSSIBLE_PATHS=(
        "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
        "/Users/yzl/models/Meta-Qwen-2.5-7B-Instruct"
        "Qwen/Qwen2.5-7B-Instruct"
    )
fi

BASE_MODEL_PATH=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ] || [[ "$path" == *"/"* ]]; then
        BASE_MODEL_PATH="$path"
        break
    fi
done

if [ -z "$BASE_MODEL_PATH" ]; then
    echo "❌ 找不到基础模型，尝试的路径："
    for path in "${POSSIBLE_PATHS[@]}"; do
        echo "  - $path"
    done
    exit 1
fi

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建输出目录（新的命名规则）
OUTPUT_DIR="/root/autodl-fs/simplified_culturemoe_ablation/${BACKBONE}_${DATA_ID}_${USE_SHARED}_${USE_MASK}_${USE_GATE}_${USE_CULTURE_LOSS}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "配置信息:"
echo "  模型路径: $MODEL_PATH"
echo "  基座模型: $BACKBONE ($BASE_MODEL_PATH)"
echo "  数据集: $DATASET_TAG ($DATA_FILE)"
echo "  使用shared专家: $USE_SHARED"
echo "  使用MASK机制: $USE_MASK (占位符)"
echo "  使用gate机制: $USE_GATE"
echo "  使用文化损失: $USE_CULTURE_LOSS"
echo "  输出目录: $OUTPUT_DIR"
echo "  数据划分: 8:1:1 (训练:验证:测试)"
echo ""

# 保存实验配置
cat > "$OUTPUT_DIR/experiment_config.json" << EOF
{
    "model_path": "$MODEL_PATH",
    "backbone": "$BACKBONE",
    "base_model_path": "$BASE_MODEL_PATH",
    "data_id": "$DATA_ID",
    "data_file": "$DATA_FILE",
    "dataset_tag": "$DATASET_TAG",
    "output_dir": "$OUTPUT_DIR",
    "timestamp": "$TIMESTAMP",
    "experiment_config": {
        "use_shared": $USE_SHARED,
        "use_mask": $USE_MASK,
        "use_gate": $USE_GATE,
        "use_culture_loss": $USE_CULTURE_LOSS
    }
}
EOF

echo "🔄 开始单一配置消融实验..."
echo ""

# 根据参数设置实验名称
EXPERIMENT_NAME="${BACKBONE}_${DATA_ID}"
if [ "$USE_SHARED" = "false" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_no_shared"
fi
if [ "$USE_MASK" = "false" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_no_mask"
fi
if [ "$USE_GATE" = "false" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_no_gate"
fi
if [ "$USE_CULTURE_LOSS" = "false" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_no_culture_loss"
fi

echo "📊 实验配置: $EXPERIMENT_NAME"
echo "----------------------------------------"

# 查找数据划分文件
SPLIT_FILE=""
# 首先检查模型目录中是否有划分文件
if [ -f "$MODEL_PATH/data_split_8_1_1.pkl" ]; then
    SPLIT_FILE="$MODEL_PATH/data_split_8_1_1.pkl"
    echo "✅ 找到模型目录中的数据划分文件: $SPLIT_FILE"
# 检查模型的父目录
elif [ -f "$(dirname "$MODEL_PATH")/data_split_8_1_1.pkl" ]; then
    SPLIT_FILE="$(dirname "$MODEL_PATH")/data_split_8_1_1.pkl"
    echo "✅ 找到父目录中的数据划分文件: $SPLIT_FILE"
# 检查常见的训练输出目录
elif [ -f "/root/autodl-fs/simplified_culturemoe/data_split_8_1_1.pkl" ]; then
    SPLIT_FILE="/root/autodl-fs/simplified_culturemoe/data_split_8_1_1.pkl"
    echo "✅ 找到通用目录中的数据划分文件: $SPLIT_FILE"
else
    echo "⚠️ 未找到数据划分文件，将使用原有的验证集划分逻辑"
fi

# 构建eval命令参数
EVAL_ARGS="--model_path \"$MODEL_PATH\" \
    --base_model_path \"$BASE_MODEL_PATH\" \
    --data_file \"$DATA_FILE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --experiment_name \"$EXPERIMENT_NAME\" \
    --use_fixed_split"

# 如果找到了数据划分文件，添加到参数中
if [ -n "$SPLIT_FILE" ]; then
    EVAL_ARGS="$EVAL_ARGS --split_file \"$SPLIT_FILE\""
fi

# 根据配置添加disable参数
if [ "$USE_SHARED" = "false" ]; then
    EVAL_ARGS="$EVAL_ARGS --disable_shared"
fi

if [ "$USE_MASK" = "false" ]; then
    EVAL_ARGS="$EVAL_ARGS --disable_mask"
fi

if [ "$USE_GATE" = "false" ]; then
    EVAL_ARGS="$EVAL_ARGS --disable_gate"
fi

if [ "$USE_CULTURE_LOSS" = "false" ]; then
    EVAL_ARGS="$EVAL_ARGS --disable_culture_loss"
fi

# 执行评估
eval "python eval_simplified_culturemoe.py $EVAL_ARGS"

EXPERIMENT_SUCCESS=$?
if [ $EXPERIMENT_SUCCESS -eq 0 ]; then
    echo "✅ 实验完成"
else
    echo "❌ 实验失败"
fi
echo ""

# 汇总结果
echo "======================================="
echo "📊 消融实验结果汇总"
echo "======================================="

# 显示实验结果
RESULT_FILE="$OUTPUT_DIR/evaluation_results_${EXPERIMENT_NAME}.json"

if [ -f "$RESULT_FILE" ]; then
    echo "实验配置: $EXPERIMENT_NAME"
    echo "----------------------------------------"

    # 使用Python读取结果
    python << EOF
import json
import os

result_file = "$RESULT_FILE"
config_file = "$OUTPUT_DIR/experiment_config.json"

if os.path.exists(result_file):
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)

    accuracy = result.get('accuracy', 0)
    eval_loss = result.get('eval_loss', 0)
    culture_loss = result.get('eval_culture_loss', 0)
    correct = result.get('correct', 0)
    total = result.get('total', 0)

    print(f"📊 实验结果:")
    print(f"  准确率: {accuracy:.4f} ({correct}/{total})")
    print(f"  验证损失: {eval_loss:.4f}")
    print(f"  文化损失: {culture_loss:.4f}")

    # 显示配置信息
    config = result.get('config', {})
    print(f"\\n🔧 有效配置:")
    print(f"  Shared专家: {'启用' if config.get('use_shared', True) else '禁用'}")
    print(f"  MASK机制: {'启用' if config.get('use_mask', True) else '禁用'} (占位符)")
    print(f"  Gate网络: {'启用' if config.get('use_gate', True) else '禁用'}")
    print(f"  文化损失: {'启用' if config.get('use_culture_loss', True) else '禁用'}")
else:
    print("❌ 未找到结果文件: $RESULT_FILE")

EOF
else
    echo "❌ 实验失败，未生成结果文件"
fi

echo ""
echo "📁 结果文件位置:"
echo "  实验目录: $OUTPUT_DIR"
echo "  配置文件: $OUTPUT_DIR/experiment_config.json"
echo "  详细结果: $OUTPUT_DIR/evaluation_results_${EXPERIMENT_NAME}.json"
echo "  生成答案: $OUTPUT_DIR/generated_answers_${EXPERIMENT_NAME}.json"
echo ""

if [ $EXPERIMENT_SUCCESS -eq 0 ]; then
    echo "🎉 消融实验成功完成！"
    echo "======================================="
    exit 0
else
    echo "⚠️ 实验失败，请检查日志"
    echo "======================================="
    exit 1
fi