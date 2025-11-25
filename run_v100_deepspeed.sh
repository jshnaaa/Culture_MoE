#!/bin/bash

# V100 DeepSpeed ZeRO 训练脚本
# 使用模型并行和CPU offloading来适配小显存

echo "======================================"
echo "V100 DeepSpeed ZeRO 训练配置"
echo "使用CPU offloading适配小显存"
echo "======================================"

# 检查DeepSpeed是否安装
if ! python -c "import deepspeed" 2>/dev/null; then
    echo "❌ DeepSpeed未安装，请先安装："
    echo "pip install deepspeed"
    exit 1
fi

# 基础配置
BACKBONE="llama"
DATA_ID="2"
NUM_EXPERTS="2"        # 最少专家
USE_SHARED="false"     # 关闭共享专家
USE_MASK="false"       # 关闭mask
USE_GATE="false"       # 关闭门控
USE_CULTURE_LOSS="false"  # 关闭文化损失

# 模型和数据路径
BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"

# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-fs/v100_deepspeed/ds_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# 创建DeepSpeed配置文件
cat > "$OUTPUT_DIR/deepspeed_config.json" << EOF
{
    "train_batch_size": 1,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 100
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "fp16": {
        "enabled": true,
        "auto_cast": false,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": false,
        "number_checkpoints": 4,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    },
    "wall_clock_breakdown": false
}
EOF

echo "DeepSpeed ZeRO-3 配置："
echo "  CPU Offloading: 启用 (优化器+参数)"
echo "  Activation Checkpointing: 启用"
echo "  FP16: 启用"
echo "  Batch Size: 1"

# 训练参数
BATCH_SIZE=1
LEARNING_RATE=1e-4
NUM_EPOCHS=1

echo "开始DeepSpeed训练..."

# 使用DeepSpeed启动训练
deepspeed --num_gpus=1 train_lora_culturemoe_ffn_integrated_ddp.py \
    --deepspeed "$OUTPUT_DIR/deepspeed_config.json" \
    --base_model "$BASE_MODEL" \
    --data_path "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_experts $NUM_EXPERTS \
    --use_shared $USE_SHARED \
    --use_mask $USE_MASK \
    --use_gate $USE_GATE \
    --use_culture_loss $USE_CULTURE_LOSS \
    --num_gpus 1 \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/deepspeed_training.log"

TRAINING_SUCCESS=$?

echo "======================================"
if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "✅ DeepSpeed训练成功！"
    echo "V100可以通过CPU offloading运行该模型"
else
    echo "❌ DeepSpeed训练失败"
    echo "可能需要进一步优化或使用更小模型"
fi

echo "训练日志: $OUTPUT_DIR/deepspeed_training.log"
echo "DeepSpeed配置: $OUTPUT_DIR/deepspeed_config.json"
echo "======================================"