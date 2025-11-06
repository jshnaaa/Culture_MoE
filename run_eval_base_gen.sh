#!/bin/bash

# ============================================================
# 评估 Base 模型（生成式版本）
#
# 使用方法：
#   sh run_eval_base_gen.sh <BACKBONE> <DATA_ID>
#
# 参数说明：
#   BACKBONE: llama 或 qwen
#   DATA_ID: 数据集标识符
#     21 - CulturalBench (base)
#     31 - NormAD (base)
#     41 - CultureLLM (base, 默认)
#     51 - WVS (base)
#     22 - CulturalBench (RP)
#     32 - NormAD (RP)
#     42 - CultureLLM (RP)
#     52 - WVS (RP)
#
# 示例：
#   sh run_eval_base_gen.sh llama 41    # CultureLLM base
#   sh run_eval_base_gen.sh llama 42    # CultureLLM RP
#   sh run_eval_base_gen.sh qwen 21     # CulturalBench base
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"
DATA_ID="${2:-41}"  # 默认使用 CultureLLM base (41)

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
fi

# 根据 DATA_ID 选择数据集
case $DATA_ID in
    21)
        # CulturalBench (base)
        DATASET_NAME="CulturalBench"
        TEST_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/base_gen_CulturalBench_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    31)
        # NormAD (base)
        DATASET_NAME="NormAD"
        TEST_FILE="/root/autodl-fs/normad_merge_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/base_gen_normad_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    41)
        # CultureLLM (base, 默认)
        DATASET_NAME="CultureLLM"
        TEST_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/base_gen_cultureLLM_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    51)
        # WVS (base)
        DATASET_NAME="WVS"
        TEST_FILE="/root/autodl-fs/wvs_merge_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/base_gen_wvs_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    22)
        # CulturalBench (RP)
        DATASET_NAME="CulturalBench_RP"
        TEST_FILE="/root/autodl-fs/CulturalBench_rp_merge_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/rp_gen_CulturalBench_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    32)
        # NormAD (RP)
        DATASET_NAME="NormAD_RP"
        TEST_FILE="/root/autodl-fs/normad_merge_rp_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/rp_gen_normad_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    42)
        # CultureLLM (RP)
        DATASET_NAME="CultureLLM_RP"
        TEST_FILE="/root/autodl-fs/cultureLLM_merge_rp_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/rp_gen_cultureLLM_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    52)
        # WVS (RP)
        DATASET_NAME="WVS_RP"
        TEST_FILE="/root/autodl-fs/wvs_merge_rp_gen.json"
        OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/rp_gen_wvs_${BACKBONE}_$(date +%Y%m%d_%H%M)"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID."
        echo ""
        echo "DATA_ID options:"
        echo "  Base datasets:"
        echo "    21 - CulturalBench"
        echo "    31 - NormAD"
        echo "    41 - CultureLLM (default)"
        echo "    51 - WVS"
        echo ""
        echo "  RP (Role-Play) datasets:"
        echo "    22 - CulturalBench_RP"
        echo "    32 - NormAD_RP"
        echo "    42 - CultureLLM_RP"
        echo "    52 - WVS_RP"
        exit 1
        ;;
esac

echo "============================================================"
echo "Base Model Evaluation (Generative Version)"
echo "============================================================"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME (DATA_ID=$DATA_ID)"
echo "Model path: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

# 运行评估脚本
# 对于 CultureLLM 数据集（DATA_ID=41/42），手动指定 num_classes=10
if [ "$DATA_ID" = "41" ] || [ "$DATA_ID" = "42" ]; then
    NUM_CLASSES_ARG="--num_classes 10"
else
    NUM_CLASSES_ARG=""
fi

python eval_base_gen.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    $NUM_CLASSES_ARG

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
    echo "  Model path: $MODEL_PATH"
    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR"
    echo "  - evaluation_results.json (详细结果)"
    echo "  - evaluation_summary.json (摘要)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/evaluation_summary.json | jq"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

