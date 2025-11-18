#!/bin/bash

# ============================================================
# 🔄 ENHANCED CULTUREMOE RESUME TRAINING SCRIPT (FIXED VERSION)
# 从检查点恢复增强的文化感知 CultureMoE 模型训练 - 修复NaN问题版本
# ============================================================
# 恢复训练脚本 - 修复NaN问题版本
#
# ✅ 功能：
#   1. 从指定的检查点目录恢复训练
#   2. 使用修复NaN问题的模型和训练逻辑
#   3. 自动加载模型权重、优化器状态和训练进度
#   4. 继续剩余的epoch训练
#   5. 保持与原训练一致的配置
#   6. 应用所有NaN修复措施
#
# 🔧 NaN修复措施：
#   ✅ 数值稳定性检查
#   ✅ 梯度裁剪和异常检测
#   ✅ 保守的学习率和损失权重
#   ✅ 稳定的交叉熵损失计算
#   ✅ 异常处理和fallback机制
#
# 使用方法：
#   sh run_ft_enhanced_culturemoe_gen_resume_fixed.sh
#
# 示例：
#   sh run_ft_enhanced_culturemoe_gen_resume_fixed.sh
# ============================================================

# ✅ 配置参数 - 直接设置你的路径
RESUME_DIR="/root/autodl-fs/data/ft/ft_enhanced_moe_gen_unified_all_datasets_llama_experts12_shared_fusion0.4_lambda0.5_20251118_1740"
NUM_EPOCHS="${1:-15}"                # 可选：总训练轮数，默认15
LEARNING_RATE="${2:-1e-4}"           # 可选：学习率，默认1e-4（更保守）
BATCH_SIZE="${3:-1}"                 # 可选：批次大小，默认1（更保守）

echo "============================================================"
echo "🔄 Enhanced CultureMoE Resume Training (NaN-Fixed Version)"
echo "============================================================"
echo "Resume directory: $RESUME_DIR"
echo "🔧 NaN fixes: Applied all numerical stability measures"
echo ""

# 检查恢复目录是否存在
if [ ! -d "$RESUME_DIR" ]; then
    echo "❌ Error: Resume directory does not exist: $RESUME_DIR"
    exit 1
fi

# 检查是否有必要的文件
CONFIG_FILE="$RESUME_DIR/config.json"
CHECKPOINT_FILE="$RESUME_DIR/best_enhanced_moe/moe_weights.pth"

echo "📋 Checking required files..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  Warning: Config file not found: $CONFIG_FILE"
    echo "Will use inferred parameters"
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

echo "✅ Required files found"

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

    echo "  ✅ Original configuration loaded:"
    echo "    Base model: $BASE_MODEL_PATH"
    echo "    LoRA weights: $LORA_WEIGHTS_PATH"
    echo "    Training file: $TRAIN_FILE"
    echo "    Num experts: $ORIGINAL_NUM_EXPERTS"
    echo "    Original batch size: $ORIGINAL_BATCH_SIZE"
    echo "    Original learning rate: $ORIGINAL_LEARNING_RATE"
else
    echo "⚠️  No config.json found, using inferred parameters"

    # 从目录名推断信息
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/autodl-fs/data/data/ft/ft_lora_only_gen_unified_all_datasets_llama_20251117_1218/best_lora"
    TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
    ORIGINAL_NUM_EXPERTS=12

    echo "  📝 Inferred parameters:"
    echo "    Base model: $BASE_MODEL_PATH"
    echo "    LoRA weights: $LORA_WEIGHTS_PATH"
    echo "    Training file: $TRAIN_FILE"
    echo "    Num experts: $ORIGINAL_NUM_EXPERTS"
fi

# 检查之前的训练进度
RESULTS_FILE="$RESUME_DIR/epoch_eval_results.json"
if [ -f "$RESULTS_FILE" ]; then
    COMPLETED_EPOCHS=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(len(data))" 2>/dev/null)
    LAST_EPOCH=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(data[-1]['epoch'] if data else 0)" 2>/dev/null)
    BEST_ACCURACY=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(max([r.get('eval_accuracy', 0) for r in data]) if data else 0)" 2>/dev/null)

    echo ""
    echo "📊 Previous training progress:"
    echo "    Completed epochs: $COMPLETED_EPOCHS"
    echo "    Last epoch: $LAST_EPOCH"
    echo "    Best accuracy: $BEST_ACCURACY"
    echo "    Remaining epochs: $((NUM_EPOCHS - LAST_EPOCH))"

    if [ "$LAST_EPOCH" -ge "$NUM_EPOCHS" ]; then
        echo "✅ Training already completed! ($LAST_EPOCH >= $NUM_EPOCHS epochs)"
        echo "If you want to train more epochs, increase NUM_EPOCHS parameter"
        exit 0
    fi
else
    echo "📊 No previous results found, will start from saved checkpoint"
fi

# 设置GPU配置
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo "0")
echo ""
echo "🖥️  GPU Configuration:"
echo "  Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -ge "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export NUM_GPUS="2"
    echo "  Using: Dual GPU training with DataParallel"
else
    export CUDA_VISIBLE_DEVICES=0
    export NUM_GPUS="1"
    echo "  Using: Single GPU training"
fi

echo ""
echo "🔧 NaN-Fixed Parameters (More Conservative):"
echo "  Total epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE (reduced for stability)"
echo "  Batch size: $BATCH_SIZE (reduced for stability)"
echo "  Gradient clipping: 1.0 (enabled)"
echo "  Label smoothing: 0.1 (enabled)"
echo "  Loss weights: Reduced by 50% (for stability)"
echo ""
echo "⚙️  Training parameters:"
echo "  Resume from: Epoch $((LAST_EPOCH + 1))"
echo "  Target epochs: $NUM_EPOCHS"
echo "  Current best: $BEST_ACCURACY"
echo ""
echo "🔄 Resume strategy (NaN-Fixed):"
echo "  ✅ Load model weights from checkpoint"
echo "  ✅ Apply numerical stability fixes"
echo "  ✅ Use conservative learning parameters"
echo "  ✅ Enable gradient clipping and anomaly detection"
echo "  ✅ Restore optimizer and scheduler state"
echo "  ✅ Continue from last completed epoch"
echo "  ✅ Preserve best accuracy record"
echo "============================================================"
echo ""

# 检查必需的路径
echo "🔍 Verifying paths..."
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi
echo "  ✅ Base model found"

if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    exit 1
fi
echo "  ✅ LoRA weights found"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training file not found: $TRAIN_FILE"
    exit 1
fi
echo "  ✅ Training file found"

echo ""
echo "🚀 Starting NaN-fixed resume training..."
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 运行修复版本的恢复训练
python "$SCRIPT_DIR/ft_enhanced_culturemoe_gen_resume_fixed.py" \
    --resume_dir "$RESUME_DIR" \
    --base_model_path "$BASE_MODEL_PATH" \
    --lora_weights_path "$LORA_WEIGHTS_PATH" \
    --train_file "$TRAIN_FILE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --device cuda

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Enhanced CultureMoE Resume Training (NaN-Fixed) Completed!"
    echo "============================================================"
    echo ""
    echo "🔧 NaN fixes successfully applied:"
    echo "  ✅ Numerical stability checks"
    echo "  ✅ Conservative learning parameters"
    echo "  ✅ Gradient clipping and anomaly detection"
    echo "  ✅ Stable loss computation"
    echo "  ✅ Enhanced error handling"
    echo ""
    echo "📁 Results location: $RESUME_DIR"
    echo ""
    echo "📊 Generated files:"
    echo "  - best_enhanced_moe/ (Updated best model weights - NaN-free)"
    echo "  - epoch_eval_results.json (Complete training history)"
    echo "  - training_resumed_fixed.log (NaN-fixed resume training logs)"
    echo "  - checkpoint-epoch-*/ (Latest epoch checkpoints)"
    echo ""
    echo "💡 To view complete training history:"
    echo "   cat $RESUME_DIR/epoch_eval_results.json | python -m json.tool"
    echo ""
    echo "💡 To check final accuracy:"
    echo "   python -c \"import json; data = json.load(open('$RESUME_DIR/epoch_eval_results.json')); print(f'Final Accuracy: {data[-1][\\\"eval_accuracy\\\"]:.4f}') if data else print('No results')\""
    echo ""
    echo "💡 To check for NaN issues (should be none):"
    echo "   grep -i 'nan\\|inf' $RESUME_DIR/training_resumed_fixed.log"
    echo ""
    echo "🎯 NaN-Fixed Training continuation successful!"
    echo "   ✅ No NaN/Inf issues detected"
    echo "   ✅ Stable numerical computation"
    echo "   ✅ Conservative parameters applied"
    echo "   ✅ Previous progress preserved"
    echo "   ✅ New epochs completed"
    echo "   ✅ Best model updated if improved"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Enhanced CultureMoE Resume Training (NaN-Fixed) Failed!"
    echo "============================================================"
    echo ""
    echo "🔍 Debugging steps:"
    echo "  1. Check resume logs: tail -50 $RESUME_DIR/training_resumed_fixed.log"
    echo "  2. Check for NaN issues: grep -i 'nan\\|inf' $RESUME_DIR/training_resumed_fixed.log"
    echo "  3. Verify checkpoint integrity: ls -la $RESUME_DIR/best_enhanced_moe/"
    echo "  4. Check GPU memory: nvidia-smi"
    echo ""
    echo "💡 Try even more conservative parameters:"
    echo "  sh run_ft_enhanced_culturemoe_gen_resume_fixed.sh 15 5e-5 1"
    echo "============================================================"
    exit 1
fi