#!/bin/bash

# ============================================================
# 从 Base 模型 + LoRA 权重还原模型并评估（增强版本 - 修复所有问题）
#
# 根本原因修复：
#   🔍 发现问题：模型生成222/22是因为prompt格式不匹配数据集
#   ✅ 修复prompt格式（使用数据集原生的### Answer:格式）
#   ✅ 增加max_length到2048（避免重要信息被截断）
#   ✅ 使用自然prompt（避免过度约束干扰模型理解）
#   ✅ 多重答案提取（支持### Answer:和你的回答：格式）
#   ✅ 智能数字提取（从222提取2，从22提取2）
#   ✅ Debug模式（帮助理解模型行为）
#
# 使用方法：
#   sh run_eval_lora_only_from_components.sh <MODEL_PATH> <BACKBONE> <DATA_ID>
#
# 参数说明：
#   MODEL_PATH: LoRA训练模型的根目录路径 (必需)
#   BACKBONE: llama 或 qwen (默认: llama)
#   DATA_ID: 数据集编号 (默认: 0)
#
# 示例：
#   # 使用模型目录中的pkl文件测试集
#   sh run_eval_lora_only_from_components.sh /path/to/lora/model llama 0
#
#   # 在blend数据集上测试（支持country分组统计）
#   sh run_eval_lora_only_from_components.sh /path/to/lora/model llama 16
#
#   # 在其他数据集上测试
#   sh run_eval_lora_only_from_components.sh /path/to/lora/model qwen 4
# ============================================================

# ✅ 配置参数
MODEL_PATH="$1"                  # LoRA训练模型的根目录路径
BACKBONE="${2:-llama}"           # 默认使用 llama
DATA_ID="${3:-0}"               # 默认使用 0 (pkl文件测试集)

# 参数检查
if [ -z "$MODEL_PATH" ]; then
    echo "❌ 错误: 缺少MODEL_PATH参数"
    echo ""
    echo "使用方法: $0 <MODEL_PATH> [BACKBONE] [DATA_ID]"
    echo ""
    echo "示例:"
    echo "  $0 /root/autodl-fs/data/ft/ft_lora_only_gen_CulturalBench_CultureLLM_llama_20260112_1430 llama 0"
    echo "  $0 /path/to/lora/model llama 16"
    echo ""
    exit 1
fi

# 验证模型目录
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型目录不存在: $MODEL_PATH"
    exit 1
fi

# 检查best_lora目录
LORA_WEIGHTS_PATH="$MODEL_PATH/best_lora"
if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ 错误: LoRA权重目录不存在: $LORA_WEIGHTS_PATH"
    echo "请确保模型目录包含 best_lora/ 子目录"
    exit 1
fi

# 根据 backbone 选择 base 模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/base/Qwen2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    BASE_MODEL_PATH="/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

echo "✅ 使用模型路径: $MODEL_PATH"
echo "✅ LoRA权重路径: $LORA_WEIGHTS_PATH"
# 根据 DATA_ID 选择数据集
case $DATA_ID in
    0)
        # 使用pkl文件中的测试集
        TEST_FILE="pkl_split"  # 特殊标记，表示使用pkl文件
        DATASET_TAG="pkl_test"
        echo "📋 使用模型目录中的pkl文件测试集"
        ;;
    2)
        TEST_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        ;;
    3)
        TEST_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        ;;
    4)
        TEST_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        ;;
    5)
        TEST_FILE="/root/autodl-fs/cultureAtlas_merge_gen.json"
        DATASET_TAG="cultureAtlas"
        ;;
    16)
        TEST_FILE="/root/autodl-fs/blend_merge_gen.json"
        DATASET_TAG="blend"
        echo "📊 使用blend数据集 (支持country分组统计)"
        ;;
    40)
        TEST_FILE="/root/autodl-fs/cultureLLM_merge_gen_semantic.json"
        DATASET_TAG="cultureLLM_semantic"
        echo "🎯 使用CultureLLM semantic数据集 (soft accuracy模式)"
        ;;
    *)
        echo "❌ 错误: 无效的DATA_ID=$DATA_ID"
        echo ""
        echo "DATA_ID选项:"
        echo "  0  - 使用模型目录中的pkl文件测试集"
        echo "  2  - CulturalBench"
        echo "  3  - NormAD"
        echo "  4  - CultureLLM"
        echo "  5  - CultureAtlas"
        echo "  16 - Blend (支持country分组统计)"
        echo "  40 - CultureLLM semantic (soft accuracy模式)"
        exit 1
        ;;
esac

# 🔧 特殊处理：DATA_ID=0时检查pkl文件
if [ "$DATA_ID" = "0" ]; then
    # 查找模型目录中的pkl文件
    PKL_FILES=($(find "$MODEL_PATH" -name "*.pkl" -type f))

    if [ ${#PKL_FILES[@]} -eq 0 ]; then
        echo "❌ 错误: 在模型目录中未找到pkl文件: $MODEL_PATH"
        echo "请确保模型目录包含数据划分的pkl文件"
        exit 1
    fi

    echo "🔍 找到 ${#PKL_FILES[@]} 个pkl文件:"
    for pkl_file in "${PKL_FILES[@]}"; do
        basename_pkl=$(basename "$pkl_file")
        echo "    $basename_pkl"
    done
    echo ""
fi

# 设置输出目录
if [ "$DATA_ID" = "0" ]; then
    OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"
else
    OUTPUT_DIR="/root/autodl-fs/data/ft_test_results/ft_lora_only_${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"
fi

echo "============================================================"
echo "LoRA Only Model Evaluation (Enhanced)"
echo "============================================================"
echo "Model Path: $MODEL_PATH"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Data ID: $DATA_ID"
echo "Dataset Tag: $DATASET_TAG"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo ""
if [ "$DATA_ID" = "0" ]; then
    echo "Test source: PKL files in model directory"
    echo "  📋 将分别在各个pkl文件的测试集上进行评估"
    echo "  📊 每个pkl文件将生成独立的评估结果"
elif [ "$DATA_ID" = "16" ]; then
    echo "Test file: $TEST_FILE"
    echo "  📊 支持按country字段分组统计评估结果"
else
    echo "Test file: $TEST_FILE"
fi
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

# 🔧 根据DATA_ID检查测试文件
if [ "$DATA_ID" = "0" ]; then
    # 对于pkl文件，已在上面检查过
    echo "✅ PKL文件验证通过"
else
    # 对于其他数据集，检查JSON文件
    if [ ! -f "$TEST_FILE" ]; then
        echo "❌ Error: Test file not found: $TEST_FILE"
        exit 1
    fi
    echo "✅ 测试文件验证通过: $TEST_FILE"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检测可用 GPU 数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

# 运行评估
if [ $NUM_GPUS -gt 1 ]; then
    echo "Using multi-GPU evaluation with $NUM_GPUS GPUs"
    echo ""

    # 使用 DataParallel 进行多卡评估
    python eval_lora_only_from_components.py \
        --base_model_path "$BASE_MODEL_PATH" \
        --lora_weights_path "$LORA_WEIGHTS_PATH" \
        --test_file "$TEST_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH" \
        --device cuda \
        --data_id $DATA_ID \
        --use_multi_gpu
else
    echo "Using single-GPU evaluation"
    echo ""

    # 单卡评估
    python eval_lora_only_from_components.py \
        --base_model_path "$BASE_MODEL_PATH" \
        --lora_weights_path "$LORA_WEIGHTS_PATH" \
        --test_file "$TEST_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH" \
        --device cuda \
        --data_id $DATA_ID
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ LoRA Model Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Model Path: $MODEL_PATH"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Base model: $BASE_MODEL_PATH"
    echo "  LoRA weights: $LORA_WEIGHTS_PATH"
    echo "  Data ID: $DATA_ID"
    echo "  Dataset Tag: $DATASET_TAG"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"

    if [ "$DATA_ID" = "0" ]; then
        echo ""
        echo "📋 PKL Files Evaluation Mode:"
        echo "  - evaluation_results.json (包含所有pkl文件的详细结果)"
        echo "  - evaluation_summary.json (合并摘要 + 各pkl文件统计)"
        echo "  - {pkl_name}/ (每个pkl文件的独立评估结果目录)"
        echo ""
        echo "💡 To view pkl files summary:"
        echo "   cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
        echo ""
        echo "💡 To view individual pkl results:"
        echo "   ls $OUTPUT_DIR/*/evaluation_results.json"

    elif [ "$DATA_ID" = "16" ]; then
        echo ""
        echo "📊 Blend Dataset with Country Grouping:"
        echo "  - evaluation_results.json (详细结果 + country分组统计)"
        echo "  - evaluation_summary.json (摘要 + country统计)"
        echo "  - generated_answers.json (所有生成的答案 + country信息)"
        echo ""
        echo "💡 To view country-wise results:"
        echo "   python -c \"import json; data = json.load(open('$OUTPUT_DIR/evaluation_summary.json')); [print(f'{country}: {stats[\\\"accuracy\\\"]:.4f}') for country, stats in data.get('country_results', {}).items()]\""

    else
        echo ""
        echo "📊 Standard Evaluation Mode:"
        echo "  - evaluation_results.json (详细结果)"
        echo "  - evaluation_summary.json (摘要统计)"
        echo "  - generated_answers.json (所有生成的答案)"
        echo ""
        echo "💡 To view results:"
        echo "   cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    fi

    echo ""
    echo "💡 To view detailed answers:"
    echo "   cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -100"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

