#!/bin/bash

# ============================================================
# 自动修复 CultureMoE.py 的缩进错误
# ============================================================

echo "============================================================"
echo "修复 CultureMoE.py 缩进错误"
echo "============================================================"
echo ""

# 检查文件是否存在
if [ ! -f "src/llamafactory/model/CultureMoE.py" ]; then
    echo "❌ Error: File not found: src/llamafactory/model/CultureMoE.py"
    exit 1
fi

echo "📝 备份原文件..."
cp src/llamafactory/model/CultureMoE.py src/llamafactory/model/CultureMoE.py.backup
echo "✅ 备份完成: src/llamafactory/model/CultureMoE.py.backup"
echo ""

echo "🔧 安装 autopep8..."
pip install autopep8 -q
echo "✅ autopep8 安装完成"
echo ""

echo "🔧 修复缩进..."
autopep8 --in-place --aggressive --aggressive src/llamafactory/model/CultureMoE.py
echo "✅ 缩进修复完成"
echo ""

echo "🔍 验证修复..."
python -m py_compile src/llamafactory/model/CultureMoE.py

if [ $? -eq 0 ]; then
    echo "✅ 验证成功！文件没有语法错误"
    echo ""
    echo "============================================================"
    echo "✅ 修复完成！"
    echo "============================================================"
    echo ""
    echo "备份文件: src/llamafactory/model/CultureMoE.py.backup"
    echo "修复文件: src/llamafactory/model/CultureMoE.py"
    echo ""
    echo "💡 现在可以运行训练脚本了："
    echo "   bash run_ft_culturemoe_gen.sh qwen 3"
else
    echo "❌ 验证失败！仍然存在语法错误"
    echo ""
    echo "请手动检查文件或查看 INDENTATION_FIX_GUIDE.md"
    exit 1
fi

