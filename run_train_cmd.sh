#!/bin/bash
# run_train_cmd.sh

# ✅ 设置 GPU 环境变量
export CUDA_VISIBLE_DEVICES=0,1

# 打印 GPU 信息
echo "=========================================="
echo "GPU Information:"
echo "=========================================="
nvidia-smi
echo ""

# 训练参数
MODEL_PATH="meta-llama/Llama-2-7b-hf"
TRAIN_FILE="data/global_opinions_train.json"  # ✅ 确保 output 字段是 int 类型
OUTPUT_DIR="./output/classification_$(date +%Y%m%d_%H%M%S)"

echo "Starting training..."
echo "Model: ${MODEL_PATH}"
echo "Train file: ${TRAIN_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# ✅ 运行训练（不需要 convert_labels 参数）
python examples/train_classification.py \
  --model_name_or_path ${MODEL_PATH} \
  --train_file ${TRAIN_FILE} \
  --val_split 0.1 \
  --max_length 512 \
  --num_experts 6 \
  --shared_hidden_dim 2048 \
  --router_hidden_dim 1024 \
  --experts_hidden_dim 2048 \
  --lora_rank 16 \
  --num_classes 3 \
  --classification_hidden_dim 512 \
  --dropout 0.1 \
  --freeze_llama \
  --output_dir ${OUTPUT_DIR} \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --save_total_limit 3 \
  --fp16 \
  --gradient_accumulation_steps 2 \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory \
  --seed 42

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output saved to ${OUTPUT_DIR}"
echo "=========================================="