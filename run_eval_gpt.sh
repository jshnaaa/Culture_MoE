#!/bin/bash

# 多模型API评测脚本
# 支持评测 DeepSeek-R1, Mistral, GPT 等模型

echo "======================================="
echo "多模型 API 数据集评测"
echo "支持 DeepSeek, Mistral, GPT 等模型"
echo "======================================="

# 参数设置（调整后的顺序）
WHICH_MODEL=${1:-"ds"}  # 模型选择，默认ds(deepseek)
API_KEY=${2}  # API KEY（必需参数）
DATA_ID=${3:-"2"}  # 数据集ID，默认2(CulturalBench)
MAX_SAMPLES=${4:-"5"}  # 最大样本数，默认5（快速测试）

# 显示帮助信息
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "用法: $0 <which_model> <api_key> [data_id] [max_samples]"
    echo ""
    echo "参数说明:"
    echo "  which_model: 模型选择 (默认: ds)"
    echo "               ds=DeepSeek-R1, mistral=Mistral-Small-3.1, gpt=GPT-4o"
    echo "  api_key:     API KEY (必需参数)"
    echo "  data_id:     数据集ID (默认: 2)"
    echo "               2=CulturalBench, 3=normad, 4=cultureLLM, 5=cultureAtlas"
    echo "  max_samples: 最大评测样本数 (默认: 5，用于快速测试)"
    echo ""
    echo "示例:"
    echo "  $0 ds sk-or-v1-xxx                # 使用DeepSeek-R1评测CulturalBench，5个样本"
    echo "  $0 mistral sk-or-v1-xxx 3 10     # 使用Mistral评测normad，10个样本"
    echo "  $0 gpt sk-proj-xxx 2 20          # 使用GPT-4o评测CulturalBench，20个样本"
    echo ""
    echo "注意: API KEY作为第二个参数传入，不会被记录到日志或Git历史中"
    exit 0
fi

# 检查API KEY参数
if [ -z "$API_KEY" ]; then
    echo "❌ 错误: 未提供API KEY"
    echo ""
    echo "用法: $0 <which_model> <api_key> [data_id] [max_samples]"
    echo ""
    echo "示例:"
    echo "  $0 ds sk-or-v1-your-api-key 2 5"
    echo ""
    echo "提示: 使用 $0 --help 查看详细帮助"
    exit 1
fi

# 根据模型选择设置参数
case $WHICH_MODEL in
    ds)
        MODEL_NAME="deepseek-r1"
        MODEL_TAG="deepseek"
        EVAL_SCRIPT="eval_ds.py"
        MODEL_FULL_NAME="deepseek/deepseek-r1-0528:free"
        echo "✅ 使用模型: DeepSeek-R1"
        ;;
    mistral)
        MODEL_NAME="mistral-small-3.1"
        MODEL_TAG="mistral"
        EVAL_SCRIPT="eval_mistral.py"
        MODEL_FULL_NAME="mistralai/mistral-small-3.1-24b-instruct:free"
        echo "✅ 使用模型: Mistral-Small-3.1"
        ;;
    gpt|4o|3.5|4omini)
        # GPT模型保持原有逻辑
        MODEL_NAME="gpt"
        MODEL_TAG="gpt"
        EVAL_SCRIPT="eval_gpt.py"

        # GPT模型名称映射
        case $WHICH_MODEL in
            3.5)
                MODEL_FULL_NAME="gpt-3.5-turbo"
                MODEL_TAG="gpt35"
                ;;
            4o|gpt)
                MODEL_FULL_NAME="gpt-4o"
                MODEL_TAG="gpt4o"
                ;;
            4omini)
                MODEL_FULL_NAME="gpt-4o-mini"
                MODEL_TAG="gpt4omini"
                ;;
        esac
        echo "✅ 使用模型: $MODEL_FULL_NAME"
        ;;
    *)
        echo "❌ 无效的模型选择: $WHICH_MODEL"
        echo "支持的模型: ds (DeepSeek), mistral (Mistral), gpt/4o/3.5/4omini (GPT系列)"
        exit 1
        ;;
esac

# 设置数据文件路径（使用_gpt.json后缀）
case $DATA_ID in
    2)
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen_gpt.json"
        DATASET_TAG="CulturalBench"
        ;;
    3)
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen_gpt.json"
        DATASET_TAG="normad"
        ;;
    4)
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen_gpt.json"
        DATASET_TAG="cultureLLM"
        ;;
    5)
        TRAIN_FILE="/root/autodl-fs/cultureAtlas_merge_gen_gpt.json"
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

# 设置输出目录（新格式）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/gpt_eval_results/${WHICH_MODEL}_${DATASET_TAG}_${MAX_SAMPLES}_${TIMESTAMP}"

echo ""
echo "配置信息:"
echo "  模型选择: $WHICH_MODEL"
echo "  实际模型: $MODEL_FULL_NAME"
echo "  API KEY: ${API_KEY:0:20}... (已隐藏)"
echo "  数据集: $DATASET_TAG (DATA_ID=$DATA_ID)"
echo "  数据文件: $TRAIN_FILE"
if [ "$MAX_SAMPLES" = "0" ] || [ -z "$MAX_SAMPLES" ]; then
    echo "  最大样本数: 全部 (完整评测)"
else
    echo "  最大样本数: $MAX_SAMPLES (测试模式)"
fi
echo "  输出目录: $OUTPUT_DIR"
echo "  评测脚本: $EVAL_SCRIPT"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 保存配置信息
cat > "$OUTPUT_DIR/eval_config.json" << EOF
{
    "eval_type": "multi_model_api_eval",
    "which_model": "$WHICH_MODEL",
    "model_name": "$MODEL_NAME",
    "model_full_name": "$MODEL_FULL_NAME",
    "dataset": {
        "data_id": "$DATA_ID",
        "dataset_tag": "$DATASET_TAG",
        "data_file": "$TRAIN_FILE"
    },
    "max_samples": $(if [ "$MAX_SAMPLES" = "0" ]; then echo "null"; else echo "$MAX_SAMPLES"; fi),
    "timestamp": "$TIMESTAMP"
}
EOF

echo "开始模型评测..."
echo ""

# 构造Python命令
PYTHON_CMD="python $EVAL_SCRIPT \
    --api_key \"$API_KEY\" \
    --data_file \"$TRAIN_FILE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --model_name \"$MODEL_FULL_NAME\""

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
    echo "✅ 模型评测成功！"

    # 检查结果文件
    RESULTS_FILE="$OUTPUT_DIR/eval_result.json"
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
    print(f'  错误预测数: {results.get(\"wrong_predictions\", 0)}')
    if 'failed_api_calls' in results:
        print(f'  API调用失败数: {results.get(\"failed_api_calls\", 0)}')
except Exception as e:
    print(f'无法解析评测结果文件: {e}')
"
        fi
    else
        echo "⚠️  评测完成但未找到结果文件"
    fi
else
    echo "❌ 模型评测失败！退出码: $EVAL_SUCCESS"
    echo ""
    echo "故障排查："
    echo "1. 检查API_KEY是否正确"
    echo "2. 检查网络连接是否正常"
    echo "3. 检查API配额是否充足"
    echo "4. 查看详细日志: $OUTPUT_DIR/eval.log"
fi

echo ""
echo "文件位置:"
echo "  评测日志: $OUTPUT_DIR/eval.log"
echo "  评测配置: $OUTPUT_DIR/eval_config.json"
if [ $EVAL_SUCCESS -eq 0 ]; then
    echo "  评测结果: $OUTPUT_DIR/eval_result.json"
    echo "  详细答案: $OUTPUT_DIR/generated_answers.json"
fi
echo ""

exit $EVAL_SUCCESS
