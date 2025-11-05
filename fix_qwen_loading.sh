#!/bin/bash

# 快速修复 Qwen 模型加载问题
# 在 train_lora_only_gen.py 中添加 attn_implementation="eager"

echo "============================================================"
echo "Fixing Qwen Model Loading Issue"
echo "============================================================"
echo ""

TARGET_FILE="train_lora_only_gen.py"

# 检查文件是否存在
if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ Error: $TARGET_FILE not found"
    exit 1
fi

echo "Checking current file..."
if grep -q 'attn_implementation="eager"' "$TARGET_FILE"; then
    echo "✅ File already contains attn_implementation parameter"
    echo "   The fix is already applied!"
else
    echo "⚠️  File does not contain attn_implementation parameter"
    echo "   Applying fix..."

    # 备份原文件
    cp "$TARGET_FILE" "${TARGET_FILE}.backup"
    echo "   Backup created: ${TARGET_FILE}.backup"

    # 使用 sed 添加参数
    # 查找 trust_remote_code=True 这一行，在后面添加新参数
    sed -i '/trust_remote_code=True$/a\        attn_implementation="eager"  # 禁用 tensor parallel' "$TARGET_FILE"

    if [ $? -eq 0 ]; then
        echo "   ✅ Fix applied successfully!"
    else
        echo "   ❌ Failed to apply fix"
        echo "   Restoring backup..."
        mv "${TARGET_FILE}.backup" "$TARGET_FILE"
        exit 1
    fi
fi

echo ""
echo "Verifying fix..."
echo "Lines around the fix:"
grep -A 2 -B 2 'attn_implementation' "$TARGET_FILE" || echo "   (Parameter not found)"

echo ""
echo "============================================================"
echo "Fix Complete"
echo "============================================================"
echo ""
echo "You can now run:"
echo "  sh run_train_lora_only_gen.sh qwen"
echo ""

