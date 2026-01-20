#!/bin/bash

# GPT API评测脚本
# 使用OpenAI GPT模型对文化数据集进行评测，作为baseline对比

echo "======================================="
echo "GPT API 数据集评测"
echo "使用OpenAI GPT作为baseline对比"
echo "======================================="

# 参数设置
DATA_ID=${1:-"2"}  # 数据集ID，默认2(CulturalBench)
GPT_MODEL_INPUT=${2:-"4o"}  # GPT模型简化名，默认4o
MAX_SAMPLES=${3:-"20"}  # 最大样本数，默认20（测试模式）

# 🔧 模型名称映射：简化名 → OpenAI完整模型名
case $GPT_MODEL_INPUT in
    3.5)
        GPT_MODEL="gpt-3.5-turbo"
        MODEL_TAG="gpt35"
        ;;
    4o)
        GPT_MODEL="gpt-4o"
        MODEL_TAG="gpt4o"
        ;;
    4omini)
        GPT_MODEL="gpt-4o-mini"
        MODEL_TAG="gpt4omini"
        ;;
    *)
        # 向后兼容：如果输入的是完整模型名，直接使用
        GPT_MODEL="$GPT_MODEL_INPUT"
        MODEL_TAG=$(echo "$GPT_MODEL" | tr '.' '_' | tr '-' '_')
        echo "ℹ️  使用完整模型名: $GPT_MODEL"
        ;;
esac

# 🔧 检查API KEY环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: 未设置OPENAI_API_KEY环境变量"
    echo "请先设置API KEY："
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi
echo "✅ 使用环境变量中的API KEY"

# 显示帮助信息（可选，使用 -h 或 --help 触发）
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "用法: $0 [data_id] [gpt_model] [max_samples]"
    echo ""
    echo "参数说明（所有参数都是可选的）:"
    echo "  data_id:     数据集ID (默认: 2)"
    echo "               2=CulturalBench, 3=normad, 4=cultureLLM, 5=cultureAtlas"
    echo "  gpt_model:   GPT模型简化名 (默认: 4o)"
    echo "               3.5=gpt-3.5-turbo, 4o=gpt-4o, 4omini=gpt-4o-mini"
    echo "  max_samples: 最大评测样本数 (默认: 20，用于快速测试)"
    echo ""
    echo "示例:"
    echo "  $0                      # 使用所有默认值 (DATA_ID=2, GPT_MODEL=4o, MAX_SAMPLES=20)"
    echo "  $0 2                    # 评测CulturalBench，使用gpt-4o，20个样本"
    echo "  $0 3 3.5                # 评测normad，使用gpt-3.5-turbo，20个样本"
    echo "  $0 4 4omini 100         # 评测cultureLLM，使用gpt-4o-mini，100个样本"
    echo "  $0 2 4o 0               # 评测CulturalBench全部样本（MAX_SAMPLES=0表示全部）"
    exit 0
fi

# 设置数据文件路径
case $DATA_ID in
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
        echo "❌ 无效的DATA_ID: $DATA_ID (支持: 2, 3, 4, 5)"
        echo ""
        echo "数据集说明:"
        echo "  2 - CulturalBench: 文化基准测试数据集"
        echo "  3 - normad: Normad数据集"
        echo "  4 - cultureLLM: CultureLLM数据集"
        echo "  5 - cultureAtlas: Culture Atlas数据集"
        exit 1
        ;;
esac

# 检查数据文件是否存在
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ 数据文件不存在: $TRAIN_FILE"
    exit 1
fi

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/gpt_eval_results/${MODEL_TAG}_${DATASET_TAG}_${TIMESTAMP}"

echo ""
echo "配置信息:"
echo "  模型简化名: $GPT_MODEL_INPUT"
echo "  实际模型: $GPT_MODEL"
echo "  数据集: $DATASET_TAG (DATA_ID=$DATA_ID)"
echo "  数据文件: $TRAIN_FILE"
if [ "$MAX_SAMPLES" = "0" ] || [ -z "$MAX_SAMPLES" ]; then
    echo "  最大样本数: 全部 (完整评测)"
else
    echo "  最大样本数: $MAX_SAMPLES (测试模式)"
fi
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 保存配置信息
cat > "$OUTPUT_DIR/eval_config.json" << EOF
{
    "eval_type": "gpt_api_baseline",
    "model_input": "$GPT_MODEL_INPUT",
    "model_actual": "$GPT_MODEL",
    "dataset": {
        "data_id": "$DATA_ID",
        "dataset_tag": "$DATASET_TAG",
        "data_file": "$TRAIN_FILE"
    },
    "max_samples": $(if [ "$MAX_SAMPLES" = "0" ]; then echo "null"; else echo "$MAX_SAMPLES"; fi),
    "timestamp": "$TIMESTAMP"
}
EOF

echo "开始GPT API评测..."
echo ""

# 构造Python命令
PYTHON_CMD="python eval_gpt.py \
    --data_file \"$TRAIN_FILE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --model_name \"$GPT_MODEL\""

# 添加可选参数（MAX_SAMPLES=0表示全部，不传递给Python脚本）
if [ "$MAX_SAMPLES" != "0" ] && [ -n "$MAX_SAMPLES" ]; then
    PYTHON_CMD="$PYTHON_CMD --max_samples $MAX_SAMPLES"
fi

# 执行评测
EVAL_SUCCESS=0

eval $PYTHON_CMD 2>&1 | tee "$OUTPUT_DIR/eval.log"

EVAL_SUCCESS=$?

# 检查评测结果
echo ""
echo "======================================="
if [ $EVAL_SUCCESS -eq 0 ]; then
    echo "✅ GPT评测成功！"

    # 检查结果文件
    RESULTS_FILE="$OUTPUT_DIR/eval_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "✅ 评测结果已保存: $RESULTS_FILE"

        # 显示关键指标
        if command -v python3 >/dev/null 2>&1; then
            python3 -c "
import json
try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    print()
    print('📊 评测结果摘要:')
    print(f'  模型: {results.get(\"model\", \"N/A\")}')
    print(f'  准确率: {results.get(\"accuracy\", 0):.4f} ({results.get(\"accuracy\", 0)*100:.2f}%)')
    print(f'  总样本数: {results.get(\"total_samples\", 0)}')
    print(f'  正确预测数: {results.get(\"correct_predictions\", 0)}')
    print(f'  API调用失败数: {results.get(\"failed_api_calls\", 0)}')
except Exception as e:
    print(f'无法解析评测结果文件: {e}')
"
        fi
    else
        echo "⚠️  评测完成但未找到结果文件"
    fi
else
    echo "❌ GPT评测失败！退出码: $EVAL_SUCCESS"
    echo ""
    echo "故障排查："
    echo "1. 检查OPENAI_API_KEY是否正确"
    echo "2. 检查网络连接是否正常"
    echo "3. 检查API配额是否充足"
    echo "4. 查看详细日志: $OUTPUT_DIR/eval.log"
fi

echo ""
echo "文件位置:"
echo "  评测日志: $OUTPUT_DIR/eval.log"
echo "  评测配置: $OUTPUT_DIR/eval_config.json"
if [ $EVAL_SUCCESS -eq 0 ]; then
    echo "  评测结果: $OUTPUT_DIR/eval_results.json"
    echo "  详细结果: $OUTPUT_DIR/detailed_results.json"
fi
echo ""

exit $EVAL_SUCCESS
