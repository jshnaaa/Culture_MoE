#!/bin/bash

# 测试修复后的脚本
# 这个脚本只是验证参数解析是否正确，不会真正运行训练

echo "=========================================="
echo "测试 train_and_eval_lora_only.py 参数解析"
echo "=========================================="
echo ""

# 测试 1：检查 Python 文件语法
echo "测试 1: 检查 Python 文件语法..."
python -m py_compile train_and_eval_lora_only.py
if [ $? -eq 0 ]; then
    echo "✅ Python 文件语法正确"
else
    echo "❌ Python 文件语法错误"
    exit 1
fi
echo ""

# 测试 2：检查参数帮助信息
echo "测试 2: 检查参数帮助信息..."
python train_and_eval_lora_only.py --help | grep -q "save_model"
if [ $? -eq 0 ]; then
    echo "✅ --save_model 参数存在"
else
    echo "❌ --save_model 参数不存在"
    exit 1
fi
echo ""

# 测试 3：检查 num_classes 参数
echo "测试 3: 检查 num_classes 参数..."
python train_and_eval_lora_only.py --help | grep -q "num_classes"
if [ $? -eq 0 ]; then
    echo "✅ --num_classes 参数存在"
else
    echo "❌ --num_classes 参数不存在"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ 所有测试通过！"
echo "=========================================="
echo ""
echo "现在可以运行训练脚本："
echo "  sh run_train_lora_only.sh llama 3 true"
echo "  sh run_train_lora_only.sh qwen 5 false"

