#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1
export NCCL_DEBUG=INFO  # 调试信息（可选）

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama，可以通过第一个参数指定 qwen
NUM_CLASSES="${2:-2}"   # 默认 2 分类，可以通过第二个参数指定其他值（2/3/4/5）
USE_CULTURE_LOSS="${3:-True}"  # 默认使用文化损失，可以通过第三个参数指定 True/False
SAVE_MODEL="${4:-false}"  # 默认不保存模型，可以通过第四个参数指定 true/false

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
    OUTPUT_DIR="/root/autodl-fs/output/CultureMoE/culturemoe_qwen_${NUM_CLASSES}class_$(date +%Y%m%d_%H%M)"
else
    MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_DIR="/root/autodl-fs/output/CultureMoE/culturemoe_llama_${NUM_CLASSES}class_$(date +%Y%m%d_%H%M)"
fi

# ✅ 使用 torchrun 启动分布式训练（双路输入 + LLaMA LoRA 微调）
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    examples/train_classification.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --fp16 \
    --gradient_checkpointing \
    --use_dual_input True \
    --freeze_llama False \
    --use_llama_lora True \
    --llama_lora_rank 16 \
    --llama_lora_alpha 32 \
    --llama_lora_dropout 0.1 \
    --llama_lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 100 \
    --max_length 512 \
    --val_split 0.1 \
    --dataloader_num_workers 4 \
    --culture_loss_lambda 0.05 \
    --num_classes $NUM_CLASSES \
    --use_culture_loss $USE_CULTURE_LOSS \
    $([ "$SAVE_MODEL" = "true" ] && echo "--save_model" || echo "")

echo ""
echo "============================================================"
if [ "$SAVE_MODEL" = "true" ]; then
    echo "✅ Model will be saved to: $OUTPUT_DIR"
else
    echo "⚠️  Model will NOT be saved (only eval results)"
fi
echo "============================================================"

