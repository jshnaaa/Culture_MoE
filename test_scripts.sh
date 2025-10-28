#!/bin/bash

# 测试脚本参数解析
# 用于验证训练脚本的参数是否正确

echo "=========================================="
echo "测试训练脚本参数解析"
echo "=========================================="
echo ""

# 测试函数
test_script() {
    local script_name=$1
    local backbone=$2
    local num_classes=$3

    echo "测试: $script_name $backbone $num_classes"
    echo "----------------------------------------"

    # 模拟脚本逻辑
    BACKBONE="${backbone:-llama}"
    NUM_CLASSES="${num_classes:-4}"

    # 数据集选择
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
            echo "❌ Error: Invalid num_classes=$NUM_CLASSES"
            return 1
            ;;
    esac

    # 模型路径选择
    if [ "$BACKBONE" = "qwen" ]; then
        MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
        MODEL_NAME="Qwen 2.5-7B-Instruct"
    else
        MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
        MODEL_NAME="LLaMA 3.1-8B-Instruct"
    fi

    echo "✅ Backbone: $BACKBONE"
    echo "✅ Model: $MODEL_NAME"
    echo "✅ Num classes: $NUM_CLASSES"
    echo "✅ Dataset: $TRAIN_FILE"
    echo ""
}

# 测试用例
echo "1. 测试默认参数（llama, 4）"
test_script "run_train_lora_only_flexible.sh" "" ""
echo ""

echo "2. 测试 llama + 2 分类"
test_script "run_train_lora_only_flexible.sh" "llama" "2"
echo ""

echo "3. 测试 qwen + 3 分类"
test_script "run_train_lora_only_flexible.sh" "qwen" "3"
echo ""

echo "4. 测试 qwen + 5 分类"
test_script "run_train_lora_only_flexible.sh" "qwen" "5"
echo ""

echo "5. 测试无效的 num_classes"
test_script "run_train_lora_only_flexible.sh" "llama" "6"
echo ""

echo "=========================================="
echo "测试完成"
echo "=========================================="

