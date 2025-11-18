#!/bin/bash

# ============================================================
# 🔄 ENHANCED CULTUREMOE RESUME TRAINING SCRIPT
# 从检查点恢复增强的文化感知 CultureMoE 模型训练
# ============================================================
# 恢复训练脚本
#
# ✅ 功能：
#   1. 从指定的检查点目录恢复训练
#   2. 自动加载模型权重、优化器状态和训练进度
#   3. 继续剩余的epoch训练
#   4. 保持与原训练一致的配置
#   5. 支持覆盖部分训练参数
#
# 使用方法：
#   sh run_ft_enhanced_culturemoe_gen_resume.sh <RESUME_DIR> [OPTIONS]
#
# 参数说明：
#   RESUME_DIR: 要恢复的训练目录（必需）
#   BASE_MODEL_PATH: 基础模型路径（可选，从原配置读取）
#   LORA_WEIGHTS_PATH: LoRA权重路径（可选，从原配置读取）
#   NUM_EPOCHS: 总训练轮数（可选，默认12）
#   LEARNING_RATE: 学习率（可选，从原配置读取）
#   BATCH_SIZE: 批次大小（可选，从原配置读取）
#
# 示例：
#   # 基本恢复（使用原始配置）
#   sh run_ft_enhanced_culturemoe_gen_resume.sh /root/autodl-fs/data/ft/ft_enhanced_moe_gen_unified_all_datasets_llama_experts12_shared_fusion0.4_lambda0.5_20251118_1740
#
#   # 恢复并修改训练轮数
#   sh run_ft_enhanced_culturemoe_gen_resume.sh /path/to/resume/dir 15
#
#   # 恢复并修改学习率
#   sh run_ft_enhanced_culturemoe_gen_resume.sh /path/to/resume/dir 12 1e-5
# ============================================================

# ✅ 配置参数
RESUME_DIR="${1:-1}"                    # 必需：恢复目录
NUM_EPOCHS="${2:-12}"                # 可选：总训练轮数
LEARNING_RATE="${3}"                 # 可选：学习率
BATCH_SIZE="${4}"                    # 可选：批次大小

RESUME_DIR="/root/autodl-fs/data/ft/ft_enhanced_moe_gen_unified_all_datasets_llama_experts12_shared_fusion0.4_lambda0.5_20251118_1740"

# 检查必需参数
if [ -z "$RESUME_DIR" ]; then
    echo "❌ Error: Resume directory is required"
    echo ""
    echo "Usage: sh run_ft_enhanced_culturemoe_gen_resume.sh <RESUME_DIR> [NUM_EPOCHS] [LEARNING_RATE] [BATCH_SIZE]"
    echo ""
    echo "Example:"
    echo "  sh run_ft_enhanced_culturemoe_gen_resume.sh /root/autodl-fs/data/ft/ft_enhanced_moe_gen_unified_all_datasets_llama_experts12_shared_fusion0.4_lambda0.5_20251118_1740"
    exit 1
fi

# 检查恢复目录是否存在
if [ ! -d "$RESUME_DIR" ]; then
    echo "❌ Error: Resume directory does not exist: $RESUME_DIR"
    exit 1
fi

# 检查是否有必要的文件
CONFIG_FILE="$RESUME_DIR/config.json"
CHECKPOINT_FILE="$RESUME_DIR/best_enhanced_moe/moe_weights.pth"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  Warning: Config file not found: $CONFIG_FILE"
    echo "Will use default parameters"
fi

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT_FILE"
    echo "Expected structure:"
    echo "  $RESUME_DIR/"
    echo "  ├── config.json"
    echo "  ├── best_enhanced_moe/"
    echo "  │   └── moe_weights.pth"
    echo "  └── epoch_eval_results.json (optional)"
    exit 1
fi

# 从config.json读取原始配置（如果存在）
if [ -f "$CONFIG_FILE" ]; then
    echo "📋 Reading original configuration..."

    # 提取关键信息
    BASE_MODEL_PATH=$(python -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('base_model_path', ''))" 2>/dev/null)
    LORA_WEIGHTS_PATH=$(python -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('lora_weights_path', ''))" 2>/dev/null)
    TRAIN_FILE=$(python -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('train_file', ''))" 2>/dev/null)
    ORIGINAL_NUM_EXPERTS=$(python -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('num_experts', 12))" 2>/dev/null)
    ORIGINAL_BATCH_SIZE=$(python -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('batch_size', 2))" 2>/dev/null)
    ORIGINAL_LEARNING_RATE=$(python -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('learning_rate', 2e-4))" 2>/dev/null)

    echo "  Original configuration loaded:"
    echo "    Base model: $BASE_MODEL_PATH"
    echo "    LoRA weights: $LORA_WEIGHTS_PATH"
    echo "    Training file: $TRAIN_FILE"
    echo "    Num experts: $ORIGINAL_NUM_EXPERTS"
    echo "    Batch size: $ORIGINAL_BATCH_SIZE"
    echo "    Learning rate: $ORIGINAL_LEARNING_RATE"
else
    echo "⚠️  No config.json found, will need to specify parameters manually"

    # 从目录名推断一些信息
    if [[ "$RESUME_DIR" == *"llama"* ]]; then
        BACKBONE="llama"
        BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
        LORA_WEIGHTS_PATH="/autodl-fs/data/data/ft/ft_lora_only_gen_unified_all_datasets_llama_20251117_1218/best_lora"
    elif [[ "$RESUME_DIR" == *"qwen"* ]]; then
        BACKBONE="qwen"
        BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
        LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_qwen_20251111_1421/best_lora"
    fi

    if [[ "$RESUME_DIR" == *"unified_all_datasets"* ]]; then
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
    fi

    echo "  Inferred from directory name:"
    echo "    Backbone: $BACKBONE"
    echo "    Base model: $BASE_MODEL_PATH"
    echo "    LoRA weights: $LORA_WEIGHTS_PATH"
    echo "    Training file: $TRAIN_FILE"
fi

# 检查之前的训练进度
RESULTS_FILE="$RESUME_DIR/epoch_eval_results.json"
if [ -f "$RESULTS_FILE" ]; then
    COMPLETED_EPOCHS=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(len(data))" 2>/dev/null)
    LAST_EPOCH=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(data[-1]['epoch'] if data else 0)" 2>/dev/null)
    BEST_ACCURACY=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(max([r.get('eval_accuracy', 0) for r in data]) if data else 0)" 2>/dev/null)

    echo "📊 Training progress:"
    echo "    Completed epochs: $COMPLETED_EPOCHS"
    echo "    Last epoch: $LAST_EPOCH"
    echo "    Best accuracy: $BEST_ACCURACY"
    echo "    Remaining epochs: $((NUM_EPOCHS - LAST_EPOCH))"
else
    echo "📊 No previous results found, will start from saved checkpoint"
fi

# 设置GPU配置
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo "0")
echo "🖥️  GPU Configuration:"
echo "  Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" = "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export NUM_GPUS="2"
    echo "  Using: Dual GPU training with DataParallel"
else
    export CUDA_VISIBLE_DEVICES=0
    export NUM_GPUS="1"
    echo "  Using: Single GPU training"
fi

echo ""
echo "============================================================"
echo "🔄 Enhanced CultureMoE Resume Training"
echo "============================================================"
echo "Resume directory: $RESUME_DIR"
echo "Training mode: Continue from checkpoint"
echo ""
echo "📁 Paths:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo "  Training file: $TRAIN_FILE"
echo "  Checkpoint: $CHECKPOINT_FILE"
echo ""
echo "⚙️  Training parameters:"
echo "  Total epochs: $NUM_EPOCHS"
if [ -n "$LEARNING_RATE" ]; then
    echo "  Learning rate: $LEARNING_RATE (override)"
else
    echo "  Learning rate: $ORIGINAL_LEARNING_RATE (from config)"
fi
if [ -n "$BATCH_SIZE" ]; then
    echo "  Batch size: $BATCH_SIZE (override)"
else
    echo "  Batch size: $ORIGINAL_BATCH_SIZE (from config)"
fi
echo ""
echo "🔄 Resume strategy:"
echo "  ✅ Load model weights from checkpoint"
echo "  ✅ Restore optimizer and scheduler state"
echo "  ✅ Continue from last completed epoch"
echo "  ✅ Preserve best accuracy record"
echo "  ✅ Append new results to existing log"
echo "============================================================"
echo ""

# 检查必需的路径
if [ -n "$BASE_MODEL_PATH" ] && [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ -n "$LORA_WEIGHTS_PATH" ] && [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    exit 1
fi

if [ -n "$TRAIN_FILE" ] && [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

echo "🚀 Starting resume training..."
echo ""

# 构建Python命令参数
PYTHON_ARGS="--resume_dir $RESUME_DIR --device cuda"

if [ -n "$BASE_MODEL_PATH" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --base_model_path $BASE_MODEL_PATH"
fi

if [ -n "$LORA_WEIGHTS_PATH" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --lora_weights_path $LORA_WEIGHTS_PATH"
fi

if [ -n "$TRAIN_FILE" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --train_file $TRAIN_FILE"
fi

if [ -n "$NUM_EPOCHS" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --num_epochs $NUM_EPOCHS"
fi

if [ -n "$LEARNING_RATE" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --learning_rate $LEARNING_RATE"
fi

if [ -n "$BATCH_SIZE" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --batch_size $BATCH_SIZE"
fi

# 运行恢复训练
python ft_enhanced_culturemoe_gen_resume.py $PYTHON_ARGS

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Enhanced CultureMoE Resume Training Completed!"
    echo "============================================================"
    echo ""
    echo "📁 Results location: $RESUME_DIR"
    echo ""
    echo "📊 Generated files:"
    echo "  - best_enhanced_moe/ (Updated best model weights)"
    echo "  - epoch_eval_results.json (Complete training history)"
    echo "  - training_resumed.log (Resume training logs)"
    echo "  - checkpoint-epoch-*/ (Latest epoch checkpoints)"
    echo ""
    echo "💡 To view complete training history:"
    echo "   cat $RESUME_DIR/epoch_eval_results.json | python -m json.tool"
    echo ""
    echo "💡 To check final accuracy:"
    echo "   python -c \"import json; data = json.load(open('$RESUME_DIR/epoch_eval_results.json')); print(f'Final Accuracy: {data[-1][\\\"eval_accuracy\\\"]:.4f}') if data else print('No results')\""
    echo ""
    echo "💡 To view resume logs:"
    echo "   tail -50 $RESUME_DIR/training_resumed.log"
    echo ""
    echo "🎯 Training continuation successful!"
    echo "   - Previous progress preserved"
    echo "   - New epochs completed"
    echo "   - Best model updated if improved"
    echo "   - All logs and results saved"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Enhanced CultureMoE Resume Training Failed!"
    echo "============================================================"
    echo ""
    echo "🔍 Debugging steps:"
    echo "  1. Check resume logs: tail -50 $RESUME_DIR/training_resumed.log"
    echo "  2. Verify checkpoint integrity: ls -la $RESUME_DIR/best_enhanced_moe/"
    echo "  3. Check GPU memory: nvidia-smi"
    echo "  4. Verify all paths are accessible"
    echo ""
    echo "💡 Common solutions:"
    echo "  - Reduce batch size if out of memory"
    echo "  - Check disk space in output directory"
    echo "  - Ensure checkpoint files are not corrupted"
    echo "  - Try single GPU if DataParallel issues"
    echo "============================================================"
    exit 1
fi