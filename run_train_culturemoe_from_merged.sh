#!/bin/bash

# ============================================================
# 从合并后的模型训练 CultureMoE
# 假设已经完成：
# 1. LoRA Only 训练（sh run_train_lora_only.sh llama 2 true）
# 2. 权重合并（sh run_merge_lora.sh llama /path/to/lora /path/to/output）
#
# 本脚本只负责：冻结 LLM，训练 MoE 部分
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"  # 默认使用 llama
NUM_CLASSES="${2:-2}"     # 默认 2 分类
USE_CULTURE_LOSS="${3:-True}"  # 默认使用文化损失
NUM_EXPERTS="${4:-6}"     # 默认 6 个专家
SAVE_MODEL="${5:-false}"  # 默认不保存模型
NUM_GPUS="${6:-2}"        # 默认使用 2 个 GPU

# 根据 backbone 选择 合并后的 模型路径
if [ "$BACKBONE" = "qwen" ]; then
#    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    MERGED_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/qwen_merge_${NUM_CLASSES}"
else
#    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    MERGED_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/llama_merge_${NUM_CLASSES}"
fi

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

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_output/culturemoe_${BACKBONE}_${NUM_CLASSES}class_experts${NUM_EXPERTS}_$(date +%Y%m%d_%H%M)"

# 设置 GPU
if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
    GPU_INFO="Single GPU (GPU 0)"
elif [ "$NUM_GPUS" = "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    GPU_INFO="Dual GPUs (GPU 0,1)"
else
    export CUDA_VISIBLE_DEVICES=0,1
    GPU_INFO="Dual GPUs (GPU 0,1)"
fi

echo "============================================================"
echo "CultureMoE Training (From Merged Model)"
echo "============================================================"
echo "Merged model: $MERGED_MODEL_PATH"
echo "Num classes: $NUM_CLASSES"
echo "Use culture loss: $USE_CULTURE_LOSS"
echo "Num experts: $NUM_EXPERTS"
echo "Save model: $SAVE_MODEL"
echo "GPUs: $GPU_INFO"
echo "Dataset: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 构建训练命令
TRAIN_CMD="python train_culturemoe_from_merged.py \
    --merged_model_path $MERGED_MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_classes $NUM_CLASSES \
    --use_culture_loss $USE_CULTURE_LOSS \
    --culture_loss_lambda 0.5 \
    \
    --num_epochs 10 \
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
    --eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_length 512 \
    --val_split 0.1 \
    --logging_steps 10 \
    --num_workers 2"

# 添加 save_model 参数
if [ "$SAVE_MODEL" = "true" ]; then
    MODEL_SAVE_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_${BACKBONE}_${NUM_CLASSES}"
    TRAIN_CMD="$TRAIN_CMD --save_model --model_save_path $MODEL_SAVE_PATH"
    echo "Model will be saved to: $MODEL_SAVE_PATH"
    echo ""
fi

# 运行训练
eval $TRAIN_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - epoch_eval_results.json (Epoch-by-epoch results)"
    echo "  - final_eval_results.json (Final evaluation)"
    echo "  - config.json (Training configuration)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    exit 1
fi

