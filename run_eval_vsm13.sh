#!/bin/bash

# ============================================================
# VSM13 评估脚本
#
# 使用方法：
#   sh run_eval_vsm13.sh <MODEL_TYPE> <BACKBONE>
#
# 参数说明：
#   MODEL_TYPE: base, lora_only, moe 或 culturemoe
#   BACKBONE: qwen 或 llama (默认 qwen)
#
# 示例：
#   # 评估 Base 模型 (Qwen)
#   sh run_eval_vsm13.sh base qwen
#
#   # 评估 LoRA Only 模型 (Qwen)
#   sh run_eval_vsm13.sh lora_only qwen
#
#   # 评估 MOE 模型 (Qwen)
#   sh run_eval_vsm13.sh moe qwen
#
#   # 评估 CultureMoE 模型 (Qwen)
#   sh run_eval_vsm13.sh culturemoe qwen
#
#   # 评估 MOE 模型 (LLaMA)
#   sh run_eval_vsm13.sh moe llama
# ============================================================

# ✅ 配置参数
MODEL_TYPE="${1:-lora_only}"
BACKBONE="${2:-qwen}"

# 根据 backbone 设置模型路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora"
    MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_qwen_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora"
    MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe"
fi

# 数据集路径
VSM13_DATA="/root/autodl-fs/vsm13.json"

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/vsm13/${MODEL_TYPE}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

echo "============================================================"
echo "VSM13 Evaluation"
echo "============================================================"
echo "Model type: $MODEL_TYPE"
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Base model: $BASE_MODEL_PATH"
if [ "$MODEL_TYPE" = "lora_only" ]; then
    echo "LoRA weights: $LORA_WEIGHTS_PATH"
fi
if [ "$MODEL_TYPE" = "moe" ]; then
    echo "LoRA weights: $LORA_WEIGHTS_PATH"
    echo "MOE weights: $MOE_WEIGHTS_PATH"
fi
echo "Dataset: $VSM13_DATA"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查数据集
if [ ! -f "$VSM13_DATA" ]; then
    echo "❌ Error: VSM13 dataset not found: $VSM13_DATA"
    exit 1
fi

# 检查 base 模型
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

# 检查 LoRA 权重（仅当 model_type 为 lora_only 或 moe 时）
if [ "$MODEL_TYPE" = "lora_only" ] || [ "$MODEL_TYPE" = "moe" ]; then
    if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
        echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
        exit 1
    fi
fi

# 检查 MOE 权重（仅当 model_type 为 moe 时）
if [ "$MODEL_TYPE" = "moe" ] && [ ! -d "$MOE_WEIGHTS_PATH" ]; then
    echo "❌ Error: MOE weights not found: $MOE_WEIGHTS_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行评估
echo "Starting evaluation..."
echo ""

if [ "$MODEL_TYPE" = "lora_only" ]; then
    python eval_vsm13.py \
        --model_type "$MODEL_TYPE" \
        --backbone "$BACKBONE" \
        --base_model_path "$BASE_MODEL_PATH" \
        --lora_weights_path "$LORA_WEIGHTS_PATH" \
        --data_path "$VSM13_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --device cuda
elif [ "$MODEL_TYPE" = "moe" ]; then
    python eval_vsm13.py \
        --model_type "$MODEL_TYPE" \
        --backbone "$BACKBONE" \
        --base_model_path "$BASE_MODEL_PATH" \
        --lora_weights_path "$LORA_WEIGHTS_PATH" \
        --moe_weights_path "$MOE_WEIGHTS_PATH" \
        --data_path "$VSM13_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --device cuda
else
    python eval_vsm13.py \
        --model_type "$MODEL_TYPE" \
        --backbone "$BACKBONE" \
        --base_model_path "$BASE_MODEL_PATH" \
        --data_path "$VSM13_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --device cuda
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model information:"
    echo "  Model type: $MODEL_TYPE"
    echo "  Backbone: $BACKBONE ($MODEL_NAME)"
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
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed!"
    echo "============================================================"
    exit 1
fi

