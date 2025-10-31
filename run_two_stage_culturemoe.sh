#!/bin/bash

# ============================================================
# 两阶段训练 CultureMoE
# 阶段 1：LoRA 微调 LLM
# 阶段 2：冻结 LLM，训练 MoE
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama
NUM_CLASSES="${2:-2}"   # 默认 2 分类
USE_CULTURE_LOSS="${3:-True}"  # 默认使用文化损失
SAVE_MODEL="${4:-false}"  # 默认不保存模型
NUM_EXPERTS="${5:-6}"   # 默认 6 个专家

# 根据 num_classes 选择数据集
case $NUM_CLASSES in
    2)
        TRAIN_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
        ;;
    3)
        TRAIN_FILE="/root/autodl-fs/normad_ed_merge.json"
        ;;
    4)
        TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge_4.json"
        ;;
    5)
        TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge_5.json"
        ;;
    *)
        echo "❌ Error: Invalid num_classes=$NUM_CLASSES. Must be 2, 3, 4, or 5."
        exit 1
        ;;
esac

# 根据 backbone 选择模型路径
if [ "$BACKBONE" = "qwen" ]; then
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    OUTPUT_DIR="/root/autodl-fs/output/two_stage_moe/qwen_${NUM_CLASSES}class_experts${NUM_EXPERTS}_$(date +%Y%m%d_%H%M)"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_DIR="/root/autodl-fs/output/two_stage_moe/llama_${NUM_CLASSES}class_experts${NUM_EXPERTS}_$(date +%Y%m%d_%H%M)"
fi

echo "============================================================"
echo "Two-Stage CultureMoE Training"
echo "============================================================"
echo "Backbone: $BACKBONE"
echo "Num classes: $NUM_CLASSES"
echo "Use culture loss: $USE_CULTURE_LOSS"
echo "Save model: $SAVE_MODEL"
echo "Num experts: $NUM_EXPERTS"
echo "Model: $MODEL_PATH"
echo "Dataset: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 运行两阶段训练
python train_two_stage_culturemoe.py \
    --model_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_classes $NUM_CLASSES \
    --use_culture_loss $USE_CULTURE_LOSS \
    --culture_loss_lambda 0.05 \
    --save_model $SAVE_MODEL \
    \
    --stage1_epochs 3 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    \
    --stage2_epochs 5 \
    --num_experts $NUM_EXPERTS \
    --shared_hidden_dim 2048 \
    --router_hidden_dim 1024 \
    --experts_hidden_dim 2048 \
    --moe_lora_rank 16 \
    --classification_hidden_dim 512 \
    --dropout 0.1 \
    --num_heads 8 \
    \
    --batch_size 4 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_length 512 \
    --val_split 0.1 \
    --logging_steps 10 \
    --num_workers 4

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Two-stage training completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - stage1_epoch_eval_results.json (Stage 1 epoch results)"
    echo "  - stage1_final_eval.json (Stage 1 final evaluation)"
    echo "  - stage2_epoch_eval_results.json (Stage 2 epoch results)"
    echo "  - stage2_final_eval.json (Stage 2 final evaluation)"
    echo "  - final_report.json (Overall summary)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

