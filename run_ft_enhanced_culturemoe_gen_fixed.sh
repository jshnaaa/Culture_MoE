#!/bin/bash

# ============================================================
# 🚀 ENHANCED CULTUREMOE TRAINING SCRIPT (FIXED VERSION)
# 使用增强的文化感知组件训练 CultureMoE 模型 - 修复NaN问题版本
# ============================================================
# 修复的 Enhanced CultureMoE 训练脚本
#
# ✅ 修复内容：
#   1. 数值稳定性增强：添加NaN/Inf检测和修复
#   2. 改进的损失计算：使用label smoothing和梯度裁剪
#   3. 更保守的学习率和权重初始化
#   4. 增强的异常处理和fallback机制
#   5. 更稳定的参数设置
#
# 使用方法：
#   sh run_ft_enhanced_culturemoe_gen_fixed.sh <BACKBONE> <DATA_ID> <USE_CULTURE_LOSS> <NUM_EXPERTS> <MOE_FUSION> <CULTURE_LOSS_WEIGHT>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 1=unified_all_datasets, 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 1)
#   USE_CULTURE_LOSS: True 或 False (默认 True)
#   NUM_EXPERTS: 专家数量 (默认 12)
#   MOE_FUSION: MoE 融合系数 (默认 0.3，更保守)
#   CULTURE_LOSS_WEIGHT: 文化损失权重 (默认 0.3，更保守)
#
# 示例：
#   # 修复版本训练（更稳定的参数）
#   sh run_ft_enhanced_culturemoe_gen_fixed.sh llama 1 True 12 0.3 0.3
#
#   # 消融实验：使用6个专家
#   sh run_ft_enhanced_culturemoe_gen_fixed.sh llama 1 True 6 0.3 0.3
# ============================================================

# ✅ 配置参数 - 使用更保守的默认值
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-1}"                   # 默认 unified_all_datasets (1)
USE_CULTURE_LOSS="${3:-True}"       # 默认使用文化损失
NUM_EXPERTS="${4:-12}"              # 默认 12 个专家
MOE_FUSION="${5:-0.3}"              # 默认 MoE 融合系数 0.3 (更保守)
LAMBDA="${6:-0.3}"                  # 默认文化损失权重 0.3 (更保守)
MARGIN="${7:-0.5}"                  # 默认 margin 0.5
LAMBDA_DIFF="${8:-0.5}"             # 默认 lambda_diff 0.5 (更保守)
USE_SHARED="${9:-True}"             # 默认使用共享专家
ROUTER_TEMP="${10:-1.5}"            # 默认 Router 温度参数 1.5 (更保守)
LOAD_BAL="${11:-0.0001}"            # 默认负载均衡权重 0.0001 (更小)
ENTROPY="${12:-0.001}"              # 默认熵正则化权重 0.001 (更小)
NUM_GPUS="${13:-1}"                 # 默认使用 1 个 GPU

# 根据 backbone 选择 base 模型路径和 LoRA 权重路径
if [ "$BACKBONE" = "qwen" ]; then
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    MODEL_NAME="Qwen 2.5-7B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_*/best_lora"
else
    BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    MODEL_NAME="LLaMA 3.1-8B-Instruct"
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_*/best_lora"
fi

# 根据 DATA_ID 选择数据集
case $DATA_ID in
    1)
        # unified_all_datasets
        DATASET_NAME="unified_all_datasets"
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets.json"
        DATASET_TAG="unified_all_datasets"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_qwen_20251111_1421/best_lora"
        else
            LORA_WEIGHTS_PATH="/autodl-fs/data/data/ft/ft_lora_only_gen_unified_all_datasets_llama_20251117_1218/best_lora"
        fi
        echo "Using unified_all_datasets dataset (enhanced format)"
        ;;
    2)
        # CulturalBench
        DATASET_NAME="CulturalBench"
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen.json"
        DATASET_TAG="CulturalBench"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_qwen_20251112_1228/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_llama_20251112_1141/best_lora"
        fi
        echo "Using CulturalBench dataset (enhanced format)"
        ;;
    3)
        # NormAD
        DATASET_NAME="NormAD"
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen.json"
        DATASET_TAG="normad"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_qwen_20251111_1204/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_llama_20251112_1335/best_lora"
        fi
        echo "Using NormAD dataset (enhanced format)"
        ;;
    4)
        # CultureLLM (默认)
        DATASET_NAME="CultureLLM"
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
        DATASET_TAG="cultureLLM"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-fs/data/ft/ft_lora_only_gen_cultureLLM_qwen_20251114_1301/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_20251112_1551/best_lora"
        fi
        echo "Using CultureLLM dataset (enhanced format)"
        ;;
    *)
        echo "❌ Error: Invalid DATA_ID=$DATA_ID. Must be 1, 2, 3, or 4."
        echo ""
        echo "DATA_ID options:"
        echo "  1 - unified_all_datasets"
        echo "  2 - CulturalBench"
        echo "  3 - NormAD"
        echo "  4 - CultureLLM (default)"
        exit 1
        ;;
esac

# 输出目录
if [ "$USE_SHARED" = "True" ] || [ "$USE_SHARED" = "true" ]; then
    SHARED_TAG="shared"
else
    SHARED_TAG="noshared"
fi
OUTPUT_DIR="/root/autodl-fs/data/ft/ft_enhanced_moe_gen_fixed_${DATASET_TAG}_${BACKBONE}_experts${NUM_EXPERTS}_${SHARED_TAG}_fusion${MOE_FUSION}_lambda${LAMBDA}_$(date +%Y%m%d_%H%M)"

# 设置 GPU - 明确的单卡/双卡逻辑
export NUM_GPUS="$NUM_GPUS"  # 传递给Python脚本

if [ "$NUM_GPUS" = "1" ]; then
    GPU_INFO="Single GPU"
    echo "🔧 GPU Configuration: Single GPU training"
elif [ "$NUM_GPUS" = "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    GPU_INFO="Dual GPUs (GPU 0,1)"
    echo "🔧 GPU Configuration: Dual GPU training with DataParallel"
else
    # 任何其他值都默认为单GPU
    export NUM_GPUS="1"
    GPU_INFO="Single GPU (GPU) - fallback"
    echo "🔧 GPU Configuration: Invalid NUM_GPUS=$NUM_GPUS, falling back to single GPU"
fi

echo "============================================================"
echo "🚀 Enhanced CultureMoE Training (FIXED VERSION - No NaN)"
echo "============================================================"
echo "Training Mode: FROZEN LoRA-FINETUNED MODEL + Enhanced MoE Training"
echo "Strategy: Preserve LoRA accuracy + Add advanced cultural specialization"
echo ""
echo "🔧 NaN Fixes Applied:"
echo "  ✅ Numerical stability checks for all tensors"
echo "  ✅ Stable cross-entropy loss with label smoothing"
echo "  ✅ Conservative parameter initialization"
echo "  ✅ Gradient clipping and anomaly detection"
echo "  ✅ Fallback mechanisms for failed computations"
echo "  ✅ Reduced learning rates and loss weights"
echo ""
echo "📊 Evaluation Strategy:"
echo "  - Evaluation interval: Every 3 epochs (Epoch 3, 6, 9, 12)"
echo "  - Best model saving: Automatically saves model with highest eval accuracy"
echo "  - Backup saving: Final epoch model saved as backup"
echo "  - Memory optimization: Enhanced for both LLaMA and Qwen models"
echo ""
echo "🖥️  GPU Configuration:"
echo "  - NUM_GPUS=1: Single GPU training (default, no DataParallel)"
echo "  - NUM_GPUS=2: Dual GPU training (DataParallel enabled)"
echo "  - Current setting: NUM_GPUS=$NUM_GPUS"
echo ""
echo "🔬 Cultural Awareness Components:"
echo "  1. Cultural Embedding Layer: 文化嵌入和上下文融合"
echo "  2. Cultural Aware Router: 多维度路由决策"
echo "  3. Culture Specific Experts: 文化条件专家处理"
echo "  4. Cultural Context Awareness: 文化线索和冲突检测"
echo "  5. Enhanced Cultural Loss: 多组件文化信息综合"
echo ""
echo "Model Architecture:"
echo "  Base Model + LoRA weights → Complete LoRA-finetuned model (FROZEN)"
echo "  + Enhanced MoE components → TRAINABLE"
echo ""
# 计算批次大小（用于显示）
if [ "$DATA_ID" = "2" ] || [ "$DATA_ID" = "3" ]; then
    if [ "$BACKBONE" = "llama" ]; then
        TRAIN_BATCH_SIZE=3
        EVAL_BATCH_SIZE=2
    else
        TRAIN_BATCH_SIZE=8
        EVAL_BATCH_SIZE=4
    fi
else
    if [ "$BACKBONE" = "llama" ]; then
        TRAIN_BATCH_SIZE=2
        EVAL_BATCH_SIZE=1
    else
        TRAIN_BATCH_SIZE=4
        EVAL_BATCH_SIZE=2
    fi
fi

echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset: $DATASET_NAME (ID: $DATA_ID)"
echo "Use culture loss: $USE_CULTURE_LOSS"
echo "Culture loss weight (lambda): $LAMBDA (reduced from 0.5)"
echo "Num experts: $NUM_EXPERTS"
echo "Use shared expert: $USE_SHARED"
echo "MoE fusion coefficient: $MOE_FUSION (reduced from 0.4)"
echo ""
echo "📦 Batch Size Configuration:"
echo "  Train batch size: $TRAIN_BATCH_SIZE"
echo "  Eval batch size: $EVAL_BATCH_SIZE"
echo "  Logic: DATA_ID=$DATA_ID + BACKBONE=$BACKBONE"
echo ""
echo "🔧 Fixed Learning Configuration:"
echo "  LoRA-finetuned model: FROZEN (0.0)"
echo "  Enhanced MoE components: 1e-4 (reduced from 2e-4)"
echo "  Cultural components: 1e-4 (reduced from 2e-4)"
echo "  All MoE multipliers: 0.8 (reduced from 1.0)"
echo ""
echo "🛡️  Anti-collapse mechanisms (Enhanced):"
echo "  Router temperature: $ROUTER_TEMP (reduced from 2.0)"
echo "  Load balance weight: $LOAD_BAL (reduced from 0.001)"
echo "  Entropy weight: $ENTROPY (reduced from 0.01)"
echo "  Gradient clipping: 1.0 (new)"
echo "  Label smoothing: 0.1 (new)"
echo ""
echo "GPUs: $GPU_INFO"
echo ""
echo "Components:"
echo "  Base model: $BASE_MODEL_PATH"
echo "  LoRA weights: $LORA_WEIGHTS_PATH"
echo "  → Merged LoRA-finetuned model: FROZEN"
echo "  → Enhanced MoE + Cultural components: TRAINABLE"
echo ""
echo "Train file: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 检查路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Train file not found: $TRAIN_FILE"
    exit 1
fi

# 检查 LoRA 权重
LORA_PATH=$(ls -d $LORA_WEIGHTS_PATH 2>/dev/null | head -1)
if [ -z "$LORA_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    echo ""
    echo "Please first run:"
    echo "  sh run_ft_lora_only_gen.sh $BACKBONE $DATA_ID"
    exit 1
fi

echo "Found LoRA weights: $LORA_PATH"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行训练 - 使用修复版本的脚本
echo "Starting enhanced CultureMoE training (fixed version)..."
echo ""

python ft_enhanced_culturemoe_gen_fixed.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --lora_weights_path "$LORA_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --use_culture_loss "$USE_CULTURE_LOSS" \
    --culture_loss_lambda "$LAMBDA" \
    --culture_loss_alpha "$MARGIN" \
    --culture_loss_beta "$LAMBDA_DIFF" \
    --router_temperature "$ROUTER_TEMP" \
    --load_balance_weight "$LOAD_BAL" \
    --entropy_weight "$ENTROPY" \
    --moe_fusion "$MOE_FUSION" \
    --freeze_base_model True \
    --num_epochs 12 \
    --num_experts "$NUM_EXPERTS" \
    --use_shared_experts "$USE_SHARED" \
    --shared_hidden_dim 4096 \
    --router_hidden_dim 2048 \
    --experts_hidden_dim 4096 \
    --moe_lora_rank 32 \
    --dropout 0.05 \
    --batch_size $(\
        if [ "$DATA_ID" = "2" ] || [ "$DATA_ID" = "3" ]; then\
            if [ "$BACKBONE" = "llama" ]; then echo "3"; else echo "8"; fi\
        else\
            if [ "$BACKBONE" = "llama" ]; then echo "2"; else echo "4"; fi\
        fi\
    ) \
    --eval_batch_size $(\
        if [ "$DATA_ID" = "2" ] || [ "$DATA_ID" = "3" ]; then\
            if [ "$BACKBONE" = "llama" ]; then echo "2"; else echo "4"; fi\
        else\
            if [ "$BACKBONE" = "llama" ]; then echo "1"; else echo "2"; fi\
        fi\
    ) \
    --learning_rate 1e-4 \
    --moe_lr_multiplier 0.8 \
    --router_lr_multiplier 0.8 \
    --shared_lr_multiplier 0.8 \
    --weight_decay 0.01 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2 \
    --eval_interval 3 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Enhanced CultureMoE Training (Fixed) completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "🔧 Fixes Applied:"
    echo "  ✅ Numerical stability checks for NaN/Inf detection"
    echo "  ✅ Stable cross-entropy loss with label smoothing"
    echo "  ✅ Conservative learning rates and loss weights"
    echo "  ✅ Enhanced gradient clipping and anomaly detection"
    echo "  ✅ Robust fallback mechanisms for error handling"
    echo ""
    echo "Files generated:"
    echo "  - best_enhanced_moe/ (Best Enhanced MoE weights - NaN-free)"
    echo "  - epoch_eval_results.json (Epoch-by-epoch results)"
    echo "  - generated_answers.json (Generated answers with cultural analysis)"
    echo "  - config.json (Training configuration)"
    echo "  - training.log (Detailed training logs with fix info)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/epoch_eval_results.json | python -m json.tool"
    echo ""
    echo "💡 To check for NaN issues:"
    echo "   grep -i 'nan\\|inf' $OUTPUT_DIR/training.log"
    echo ""
    echo "💡 To view accuracy:"
    echo "   python -c \"import json; data = json.load(open('$OUTPUT_DIR/epoch_eval_results.json')); print(f'Final Accuracy: {data[-1][\\\"eval_accuracy\\\"]:.4f}')\""
    echo ""
    echo "🔬 Usage Examples:"
    echo "   # Fixed version training (stable parameters):"
    echo "   sh run_ft_enhanced_culturemoe_gen_fixed.sh llama 1 True 12 0.3 0.3"
    echo ""
    echo "   # Ablation studies with fixed version:"
    echo "   - Try different expert counts: 6, 8, 12, 16"
    echo "   - Compare cultural vs non-cultural routing"
    echo "   - Analyze expert specialization patterns"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Enhanced CultureMoE Training (Fixed) failed!"
    echo "============================================================"
    echo ""
    echo "🔍 Debugging steps:"
    echo "  1. Check training logs for specific error messages"
    echo "  2. Verify all input paths are correct"
    echo "  3. Ensure sufficient GPU memory is available"
    echo "  4. Try with even more conservative parameters:"
    echo "     - Lower learning rate: 5e-5"
    echo "     - Smaller batch size"
    echo "     - Reduced number of experts"
    echo "============================================================"
    exit 1
fi