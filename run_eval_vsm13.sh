#!/bin/bash

# ============================================================
# VSM13 评估脚本
#
# 使用方法：
#   sh run_eval_vsm13.sh <MODEL_PATH> [BACKBONE]
#
# 参数说明：
#   MODEL_PATH: joint训练模型目录路径 (必需)
#   BACKBONE: qwen 或 llama (默认从模型路径推断)
#
# 示例：
#   # 评估 Joint 模型 (自动推断backbone)
#   sh run_eval_vsm13.sh /root/autodl-fs/joint_lora_moe/llama_CulturalBench_cultureLLM_sharedtrue_gatetrue_masktrue_20260102_215142
#
#   # 评估 Joint 模型 (指定backbone)
#   sh run_eval_vsm13.sh /root/autodl-fs/joint_lora_moe/llama_CulturalBench_cultureLLM_sharedtrue_gatetrue_masktrue_20260102_215142 llama
#
#   # 评估 Joint 模型 (Qwen)
#   sh run_eval_vsm13.sh /root/autodl-fs/joint_lora_moe/qwen_CulturalBench_cultureLLM_sharedtrue_gatetrue_masktrue_20260102_215142 qwen
# ============================================================

# ✅ 配置参数
MODEL_PATH="$1"
BACKBONE="$2"

# 参数检查
if [ -z "$MODEL_PATH" ]; then
    echo "❌ 错误: 缺少MODEL_PATH参数"
    echo ""
    echo "使用方法: $0 <MODEL_PATH> [BACKBONE]"
    echo ""
    echo "示例:"
    echo "  $0 /root/autodl-fs/joint_lora_moe/llama_CulturalBench_cultureLLM_sharedtrue_gatetrue_masktrue_20260102_215142"
    echo ""
    exit 1
fi

# 🔧 从模型路径推断backbone（如果未指定）
if [ -z "$BACKBONE" ]; then
    if [[ "$MODEL_PATH" == *"llama"* ]]; then
        BACKBONE="llama"
        echo "🔍 自动推断backbone: llama"
    elif [[ "$MODEL_PATH" == *"qwen"* ]]; then
        BACKBONE="qwen"
        echo "🔍 自动推断backbone: qwen"
    else
        echo "❌ 无法从路径推断backbone，请手动指定: llama 或 qwen"
        exit 1
    fi
else
    echo "✅ 使用指定的backbone: $BACKBONE"
fi

# 🔧 验证模型目录和best_joint_model子目录
JOINT_MODEL_DIR="$MODEL_PATH/best_joint_model"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型目录不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$JOINT_MODEL_DIR" ]; then
    echo "❌ joint模型子目录不存在: $JOINT_MODEL_DIR"
    echo "请确保模型目录包含 best_joint_model/ 子目录"
    exit 1
fi

# 🔧 验证joint模型文件
REQUIRED_FILES=(
    "$JOINT_MODEL_DIR/joint_config.json"
    "$JOINT_MODEL_DIR/moe_weights.pt"
)

REQUIRED_DIRS=(
    "$JOINT_MODEL_DIR/lora_weights"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少必需文件: $file"
        exit 1
    fi
done

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ 缺少必需目录: $dir"
        exit 1
    fi
done

echo "✅ Joint模型验证通过: $JOINT_MODEL_DIR"

# 🔧 根据backbone设置基础模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 🔧 数据集路径
VSM13_DATA="/root/autodl-fs/vsm13.json"

# 🔧 输出目录（基于joint模型路径）
MODEL_DIR_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/vsm13/joint_${MODEL_DIR_NAME}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "VSM13 Evaluation - Joint Model"
echo "============================================================"
echo "Model type: joint (LoRA + MoE)"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Base model: $BASE_MODEL_PATH"
echo "Joint model: $JOINT_MODEL_DIR"
echo "Dataset: $VSM13_DATA"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 🔧 检查数据集
if [ ! -f "$VSM13_DATA" ]; then
    echo "❌ Error: VSM13 dataset not found: $VSM13_DATA"
    exit 1
fi

# 🔧 检查基础模型
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

echo "✅ 所有必需文件验证通过"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 🔧 运行评估
echo "Starting joint model evaluation..."
echo ""

python eval_vsm13.py \
    --model_type "joint" \
    --backbone "$BACKBONE" \
    --base_model_path "$BASE_MODEL_PATH" \
    --joint_model_path "$JOINT_MODEL_DIR" \
    --data_path "$VSM13_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Joint Model Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Model type: joint (LoRA + MoE)"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Joint model: $JOINT_MODEL_DIR"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - vsm13_results.json (详细结果)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/vsm13_results.json | python -m json.tool"
    echo ""
    echo "💡 To view average distance:"
    echo "   python -c \"import json; data = json.load(open('$OUTPUT_DIR/vsm13_results.json')); print(f'Average Distance: {data[\\\"average_euclidean_distance\\\"]:.2f}')\""
    echo ""
    echo "💡 To view country-wise scores:"
    echo "   python -c \"import json; data = json.load(open('$OUTPUT_DIR/vsm13_results.json')); [print(f'{country}: {scores}') for country, scores in data['country_scores'].items()]\""
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

