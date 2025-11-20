#!/bin/bash

# ============================================================
# 🚀 ENHANCED CULTUREMOE TRAINING SCRIPT
# 使用增强的文化感知组件训练 CultureMoE 模型
# 支持可配置的专家数量，便于消融实验
# ============================================================
# 增强的 CultureMoE 训练脚本
#
# ✅ 新增功能：
#   1. 文化嵌入层：显式文化表示和上下文融合
#   2. 文化感知路由器：多维度路由决策（内容+文化+亲和性+融合）
#   3. 文化特定专家：文化条件处理和适应机制
#   4. 文化上下文感知：文化线索检测、冲突检测、敏感性分析
#   5. 增强的文化损失：多组件文化信息综合
#   6. 可配置专家数量：支持消融实验
#
# 使用方法：
#   sh run_ft_enhanced_culturemoe_gen.sh <BACKBONE> <DATA_ID> <USE_CULTURE_LOSS> <NUM_EXPERTS> <MOE_FUSION> <CULTURE_LOSS_WEIGHT> <MARGIN> <LAMBDA_DIFF> <USE_SHARED> <ROUTER_TEMP> <LOAD_BAL> <ENTROPY> <NUM_GPUS> <USE_MASK>
#
# 参数说明：
#   BACKBONE: llama 或 qwen (默认 llama)
#   DATA_ID: 1=unified_all_datasets, 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
#   USE_CULTURE_LOSS: True 或 False (默认 True)
#   NUM_EXPERTS: 专家数量 (默认 12，支持消融实验)
#   MOE_FUSION: MoE 融合系数 (默认 0.4)
#   CULTURE_LOSS_WEIGHT: 文化损失权重 (默认 0.5)
#   USE_MASK: MASK机制开关 (默认 True，False为消融实验)
#
# 📦 批次大小自动配置：
#   DATA_ID=2或3 (CulturalBench/NormAD): llama=3, qwen=8 (训练) / llama=2, qwen=4 (评估)
#   DATA_ID=1或4 (unified/CultureLLM): llama=2, qwen=4 (训练) / llama=1, qwen=2 (评估)
#   MARGIN: 文化损失margin (默认 0.5)
#   LAMBDA_DIFF: 文化损失lambda_diff (默认 1.0)
#   USE_SHARED: 是否使用共享专家 (默认 True)
#   ROUTER_TEMP: 路由器温度 (默认 2.0)
#   LOAD_BAL: 负载均衡权重 (默认 0.001, 降低避免负损失)
#   ENTROPY: 熵正则化权重 (默认 0.01, 降低避免负损失)
#   NUM_GPUS: GPU数量 (默认 1，只有设置为2时才启用双GPU)
#   USE_MASK: MASK机制开关 (默认 True，设为 False 进行消融实验)
#
# 示例：
#   # 单GPU训练（默认）
#   sh run_ft_enhanced_culturemoe_gen.sh llama 4 True 12 0.4 0.5 0.5 1.0 True 2.0 0.001 0.01 1
#
#   # 双GPU训练
#   sh run_ft_enhanced_culturemoe_gen.sh llama 4 True 12 0.4 0.5 0.5 1.0 True 2.0 0.001 0.01 2
#
#   # 消融实验：使用6个专家（单GPU）
#   sh run_ft_enhanced_culturemoe_gen.sh llama 4 True 6 0.4 0.5 0.5 1.0 True 2.0 0.001 0.01 1
#
#   # 消融实验：使用24个专家（双GPU）
#   sh run_ft_enhanced_culturemoe_gen.sh llama 4 True 24 0.4 0.5 0.5 1.0 True 2.0 0.001 0.01 2
#
#   # MASK消融实验：关闭MASK机制（单GPU）
#   sh run_ft_enhanced_culturemoe_gen.sh llama 4 True 12 0.4 0.5 0.5 1.0 True 2.0 0.001 0.01 1 False
#
#   # MASK消融实验：关闭MASK机制（双GPU）
#   sh run_ft_enhanced_culturemoe_gen.sh llama 4 True 12 0.4 0.5 0.5 1.0 True 2.0 0.001 0.01 2 False
# ============================================================

# ✅ 配置参数
BACKBONE="${1:-llama}"              # 默认使用 llama
DATA_ID="${2:-4}"                   # 默认 CultureLLM (4)
USE_CULTURE_LOSS="${3:-True}"       # 默认使用文化损失
NUM_EXPERTS="${4:-12}"              # 默认 12 个专家（支持消融实验）
MOE_FUSION="${5:-0.4}"              # 默认 MoE 融合系数 0.4
LAMBDA="${6:-0.5}"                  # 默认文化损失权重 0.5
MARGIN="${7:-0.5}"                  # 默认 margin 0.5
LAMBDA_DIFF="${8:-1.0}"             # 默认 lambda_diff 1.0
USE_SHARED="${9:-True}"             # 默认使用共享专家
ROUTER_TEMP="${10:-2.0}"            # 默认 Router 温度参数 2.0
LOAD_BAL="${11:-0.001}"             # 默认负载均衡权重 0.001 (降低)
ENTROPY="${12:-0.01}"               # 默认熵正则化权重 0.01 (降低)
NUM_GPUS="${13:-1}"                 # 默认使用 1 个 GPU
USE_MASK="${14:-True}"              # 默认使用 MASK 机制 (True=共享专家使用instruction_mask, False=共享专家使用instruction)

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
    0)
        # unified_all_datasets
        DATASET_NAME="unified_all_datasets"
        TRAIN_FILE="/root/autodl-fs/unified_all_datasets_small.json"
        DATASET_TAG="unified_all_datasets_small"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_all_datasets_qwen_20251111_1421/best_lora"
        else
            LORA_WEIGHTS_PATH="/autodl-fs/data/data/ft/ft_lora_only_gen_unified_all_datasets_llama_20251117_1218/best_lora"
        fi
        echo "Using unified_all_datasets dataset (enhanced format)"
        ;;
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
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_20251114_1301/best_lora"
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

if [ "$USE_MASK" = "True" ] || [ "$USE_MASK" = "true" ]; then
    MASK_TAG="mask"
else
    MASK_TAG="nomask"
fi

OUTPUT_DIR="/root/autodl-fs/data/ft/ft_enhanced_moe_gen_${DATASET_TAG}_${BACKBONE}_experts${NUM_EXPERTS}_${SHARED_TAG}_${MASK_TAG}_fusion${MOE_FUSION}_lambda${LAMBDA}_$(date +%Y%m%d_%H%M)"

# 设置 GPU - 明确的单卡/双卡逻辑
export NUM_GPUS="$NUM_GPUS"  # 传递给Python脚本

if [ "$NUM_GPUS" = "1" ]; then
#    export CUDA_VISIBLE_DEVICES=0
    GPU_INFO="Single GPU"
    echo "🔧 GPU Configuration: Single GPU training"
elif [ "$NUM_GPUS" = "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    GPU_INFO="Dual GPUs (GPU 0,1)"
    echo "🔧 GPU Configuration: Dual GPU training with DataParallel"
else
    # 任何其他值都默认为单GPU
    export NUM_GPUS="1"
#    export CUDA_VISIBLE_DEVICES=0
    GPU_INFO="Single GPU (GPU) - fallback"
    echo "🔧 GPU Configuration: Invalid NUM_GPUS=$NUM_GPUS, falling back to single GPU"
fi

echo "============================================================"
echo "🚀 Enhanced CultureMoE Training with Cultural Awareness Components"
echo "============================================================"
echo "Training Mode: FROZEN LoRA-FINETUNED MODEL + Enhanced MoE Training"
echo "Strategy: Preserve LoRA accuracy + Add advanced cultural specialization"
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
echo "Culture loss weight (lambda): $LAMBDA"
echo "Num experts: $NUM_EXPERTS (configurable for ablation studies)"
echo "Use shared expert: $USE_SHARED"
echo "Use MASK mechanism: $USE_MASK (True=shared experts use instruction_mask, False=shared experts use instruction)"
echo "MoE fusion coefficient: $MOE_FUSION"
echo ""
echo "📦 Batch Size Configuration:"
echo "  Train batch size: $TRAIN_BATCH_SIZE"
echo "  Eval batch size: $EVAL_BATCH_SIZE"
echo "  Logic: DATA_ID=$DATA_ID + BACKBONE=$BACKBONE"
echo ""
echo "Enhanced Learning Configuration:"
echo "  LoRA-finetuned model: FROZEN (0.0)"
echo "  Enhanced MoE components: 2e-4"
echo "  Cultural components: 2e-4"
echo "  All MoE multipliers: 1.0"
echo ""
echo "Anti-collapse mechanisms:"
echo "  Router temperature: $ROUTER_TEMP"
echo "  Load balance weight: $LOAD_BAL"
echo "  Entropy weight: $ENTROPY"
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

# 运行训练
echo "Starting enhanced CultureMoE training..."
echo ""

python ft_enhanced_culturemoe_gen.py \
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
    --use_mask "$USE_MASK" \
    --freeze_base_model True \
    --num_epochs 12 \
    --num_experts "$NUM_EXPERTS" \
    --use_shared_experts "$USE_SHARED" \
    --shared_hidden_dim 4096 \
    --router_hidden_dim 2048 \
    --experts_hidden_dim 4096 \
    --moe_lora_rank 32 \
    --dropout 0.05 \
    --batch_size $(
        if [ "$DATA_ID" = "2" ] || [ "$DATA_ID" = "3" ]; then
            if [ "$BACKBONE" = "llama" ]; then echo "3"; else echo "8"; fi
        else
            if [ "$BACKBONE" = "llama" ]; then echo "2"; else echo "4"; fi
        fi
    ) \
    --eval_batch_size $(
        if [ "$DATA_ID" = "2" ] || [ "$DATA_ID" = "3" ]; then
            if [ "$BACKBONE" = "llama" ]; then echo "2"; else echo "4"; fi
        else
            if [ "$BACKBONE" = "llama" ]; then echo "1"; else echo "2"; fi
        fi
    ) \
    --learning_rate 2e-4 \
    --moe_lr_multiplier 1.0 \
    --router_lr_multiplier 1.0 \
    --shared_lr_multiplier 1.0 \
    --weight_decay 0.01 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2 \
    --eval_interval 3 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Enhanced CultureMoE Training completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - best_enhanced_moe/ (Best Enhanced MoE weights)"
    echo "  - epoch_eval_results.json (Epoch-by-epoch results)"
    echo "  - generated_answers.json (Generated answers with cultural analysis)"
    echo "  - config.json (Training configuration)"
    echo "  - training.log (Detailed training logs)"
    echo ""
    echo "💡 To view results:"
    echo "   cat $OUTPUT_DIR/epoch_eval_results.json | python -m json.tool"
    echo ""
    echo "💡 To view generated answers with cultural analysis:"
    echo "   cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -50"
    echo ""
    echo "💡 To view accuracy:"
    echo "   python -c \"import json; data = json.load(open('$OUTPUT_DIR/epoch_eval_results.json')); print(f'Final Accuracy: {data[-1][\\\"eval_accuracy\\\"]:.4f}')\""
    echo ""
    echo "🔬 Usage Examples:"
    echo "   # Single GPU training (default):"
    echo "   sh run_ft_enhanced_culturemoe_gen.sh llama 4 True 12 0.4 0.5"
    echo ""
    echo "   # Dual GPU training:"
    echo "   sh run_ft_enhanced_culturemoe_gen.sh llama 4 True 12 0.4 0.5 0.5 1.0 True 2.0 0.001 0.01 2"
    echo ""
    echo "   # Ablation studies:"
    echo "   - Try different expert counts: 6, 8, 12, 16, 24"
    echo "   - Compare cultural vs non-cultural routing"
    echo "   - Analyze expert specialization patterns"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Enhanced CultureMoE Training failed!"
    echo "============================================================"
    exit 1
fi