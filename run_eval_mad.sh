#!/bin/bash

# ============================================================================
# Multi-Agent Debate (MAD) 评估脚本
# ============================================================================
#
# 使用方法:
#   bash run_eval_mad.sh MODEL_TYPE DATA_ID [MAX_SAMPLES]
#
# 参数说明:
#   MODEL_TYPE - 模型类型: 1=LLaMA 3.1, 2=Qwen 2.5
#   DATA_ID    - 数据集编号: 1=unified, 2=CulturalBench, 3=NORMAD, 4=CultureLLM
#   MAX_SAMPLES - (可选) 最大样本数，用于快速测试
#
# 示例:
#   # 使用LLaMA在CulturalBench上完整评估
#   bash run_eval_mad.sh 1 2
#
#   # 使用Qwen在NORMAD上测试100个样本
#   bash run_eval_mad.sh 2 3 100
#
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 参数解析
# ============================================================================

MODEL_TYPE=${1:-1}  # 默认使用LLaMA
DATA_ID=${2:-2}     # 默认使用CulturalBench
MAX_SAMPLES=${3:-""}

# 验证MODEL_TYPE
if [ "$MODEL_TYPE" -ne 1 ] && [ "$MODEL_TYPE" -ne 2 ]; then
    echo "❌ 错误: MODEL_TYPE 必须是 1 (LLaMA) 或 2 (Qwen)"
    echo ""
    echo "使用方法:"
    echo "  bash run_eval_mad.sh MODEL_TYPE DATA_ID [MAX_SAMPLES]"
    echo ""
    echo "参数:"
    echo "  MODEL_TYPE: 1=LLaMA 3.1, 2=Qwen 2.5"
    echo "  DATA_ID: 1=unified, 2=CulturalBench, 3=NORMAD, 4=CultureLLM"
    echo "  MAX_SAMPLES: (可选) 最大样本数"
    exit 1
fi

# 验证DATA_ID
if [ "$DATA_ID" -lt 1 ] || [ "$DATA_ID" -gt 4 ]; then
    echo "❌ 错误: DATA_ID 必须是 1-4"
    echo ""
    echo "数据集编号:"
    echo "  1: unified_all_datasets.json"
    echo "  2: CulturalBench_merge_gen.json"
    echo "  3: normad_merge_gen.json"
    echo "  4: cultureLLM_merge_gen.json"
    exit 1
fi

# ============================================================================
# 模型配置
# ============================================================================

if [ "$MODEL_TYPE" -eq 1 ]; then
    MODEL_NAME="llama3.1-8b"
    # 🔧 修改为实际的模型路径
    MODEL_PATH="/path/to/Meta-Llama-3.1-8B-Instruct"
    # 如果使用Hugging Face模型，可以使用以下路径:
    # MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
elif [ "$MODEL_TYPE" -eq 2 ]; then
    MODEL_NAME="qwen2.5-7b"
    # 🔧 修改为实际的模型路径
    MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"
    # 如果使用Hugging Face模型，可以使用以下路径:
    # MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
fi

# ============================================================================
# 数据集配置
# ============================================================================

# 🔧 修改为实际的数据文件路径
case $DATA_ID in
    1)
        DATA_FILE="/path/to/unified_all_datasets.json"
        DATA_NAME="unified"
        ;;
    2)
        DATA_FILE="/path/to/CulturalBench_merge_gen.json"
        DATA_NAME="culturalbench"
        ;;
    3)
        DATA_FILE="/path/to/normad_merge_gen.json"
        DATA_NAME="normad"
        ;;
    4)
        DATA_FILE="/path/to/cultureLLM_merge_gen.json"
        DATA_NAME="culturellm"
        ;;
esac

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "⚠️  警告: 数据文件不存在: $DATA_FILE"
    echo "请修改脚本中的数据文件路径"
    echo ""
    echo "当前配置的数据文件路径:"
    echo "  1: /path/to/unified_all_datasets.json"
    echo "  2: /path/to/CulturalBench_merge_gen.json"
    echo "  3: /path/to/normad_merge_gen.json"
    echo "  4: /path/to/cultureLLM_merge_gen.json"
    exit 1
fi

# ============================================================================
# 输出目录配置
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results/mad_${MODEL_NAME}_${DATA_NAME}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

LOG_FILE="${OUTPUT_DIR}/eval.log"

# ============================================================================
# 打印配置信息
# ============================================================================

echo "========================================" | tee $LOG_FILE
echo "Multi-Agent Debate (MAD) Evaluation" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "Model Type: $MODEL_TYPE ($MODEL_NAME)" | tee -a $LOG_FILE
echo "Model Path: $MODEL_PATH" | tee -a $LOG_FILE
echo "Data ID: $DATA_ID ($DATA_NAME)" | tee -a $LOG_FILE
echo "Data File: $DATA_FILE" | tee -a $LOG_FILE
echo "Output Dir: $OUTPUT_DIR" | tee -a $LOG_FILE
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples: $MAX_SAMPLES (test mode)" | tee -a $LOG_FILE
fi
echo "Start Time: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ============================================================================
# 环境变量设置（可选）
# ============================================================================

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# Hugging Face缓存（可选）
# export HF_HOME=/path/to/huggingface/cache
# export TRANSFORMERS_CACHE=/path/to/transformers/cache

# ============================================================================
# 运行评估
# ============================================================================

echo "Starting MAD evaluation..." | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 构建命令
CMD="python eval_mad.py \
    --model_type $MODEL_TYPE \
    --data_file $DATA_FILE \
    --output_dir $OUTPUT_DIR"

# 添加可选参数
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# 如果需要指定自定义模型路径
# CMD="$CMD --model_path $MODEL_PATH"

# 执行评估
echo "Command: $CMD" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

eval $CMD 2>&1 | tee -a $LOG_FILE

# ============================================================================
# 完成
# ============================================================================

echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "Evaluation completed!" | tee -a $LOG_FILE
echo "End Time: $(date)" | tee -a $LOG_FILE
echo "Results saved to: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# 打印结果文件列表
echo "" | tee -a $LOG_FILE
echo "Generated files:" | tee -a $LOG_FILE
ls -lh $OUTPUT_DIR | tee -a $LOG_FILE

# ============================================================================
# 快速查看结果
# ============================================================================

echo "" | tee -a $LOG_FILE
echo "Quick summary:" | tee -a $LOG_FILE
if [ -f "$OUTPUT_DIR/summary_statistics.json" ]; then
    echo "Accuracy:" | tee -a $LOG_FILE
    python -c "import json; data=json.load(open('$OUTPUT_DIR/summary_statistics.json')); print(f\"  Final: {data['accuracy']:.2%}\"); print(f\"  Agent A: {data['agent_a_accuracy']:.2%}\"); print(f\"  Agent B: {data['agent_b_accuracy']:.2%}\"); print(f\"  MAD Gain: {data['mad_gain']:+.2%}\")" | tee -a $LOG_FILE
fi

echo "" | tee -a $LOG_FILE
echo "✅ All done! Check $OUTPUT_DIR for detailed results." | tee -a $LOG_FILE
