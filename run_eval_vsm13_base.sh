#!/bin/bash

# ============================================================
# VSM13 评估脚本 - 基座模型版本
#
# 使用方法：
#   sh run_eval_vsm13_base.sh [BACKBONE]
#
# 参数说明：
#   BACKBONE: qwen 或 llama (默认: llama)
#
# 示例：
#   # 评估 LLaMA 基座模型
#   sh run_eval_vsm13_base.sh llama
#
#   # 评估 Qwen 基座模型
#   sh run_eval_vsm13_base.sh qwen
#
#   # 默认评估 LLaMA 基座模型
#   sh run_eval_vsm13_base.sh
# ============================================================

# ✅ 配置参数
BACKBONE="$1"

# 🔧 设置默认backbone
if [ -z "$BACKBONE" ]; then
    BACKBONE="llama"
    echo "🔍 使用默认backbone: llama"
else
    echo "✅ 使用指定的backbone: $BACKBONE"
fi

# 🔧 验证backbone参数
if [ "$BACKBONE" != "llama" ] && [ "$BACKBONE" != "qwen" ]; then
    echo "❌ 错误: backbone参数必须是 'llama' 或 'qwen'"
    echo ""
    echo "使用方法: $0 [BACKBONE]"
    echo ""
    echo "示例:"
    echo "  $0 llama    # 评估LLaMA基座模型"
    echo "  $0 qwen     # 评估Qwen基座模型"
    echo "  $0          # 默认评估LLaMA基座模型"
    echo ""
    exit 1
fi

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

# 🔧 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/vsm13/base_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "VSM13 Evaluation - Base Model"
echo "============================================================"
echo "Model type: base (原始基座模型)"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Base model: $BASE_MODEL_PATH"
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
echo "Starting base model evaluation..."
echo ""

python eval_vsm13.py \
    --model_type "base" \
    --backbone "$BACKBONE" \
    --base_model_path "$BASE_MODEL_PATH" \
    --data_path "$VSM13_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Base Model Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Model type: base (原始基座模型)"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Base model: $BASE_MODEL_PATH"
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
    echo ""
    echo "🔍 Compare with Joint Model:"
    echo "   # 运行Joint模型评估进行对比"
    echo "   sh run_eval_vsm13.sh /path/to/joint/model $BACKBONE"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi