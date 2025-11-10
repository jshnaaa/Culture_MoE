#!/bin/bash

# ============================================================
# 磁盘空间清理脚本
#
# 使用方法：
#   bash cleanup_disk_space.sh
#
# 功能：
#   1. 检查磁盘空间
#   2. 清理旧的训练输出
#   3. 清理缓存
#   4. 显示释放的空间
# ============================================================

echo "============================================================"
echo "磁盘空间清理工具"
echo "============================================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 步骤 1：检查当前磁盘使用情况
echo "📊 当前磁盘使用情况："
echo "============================================================"
df -h /root/
echo ""

# 步骤 2：显示训练输出目录大小
echo "📁 训练输出目录大小："
echo "============================================================"
du -sh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/ 2>/dev/null || echo "目录不存在"
echo ""

# 步骤 3：询问是否清理
read -p "是否清理旧的训练输出？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  清理旧的训练输出..."

    # 显示要删除的文件
    echo "以下目录将被删除："
    find /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/ -maxdepth 1 -type d -name "moe_*" -printf '%T@ %p\n' | sort -n | head -n -3 | cut -d' ' -f2-

    read -p "确认删除？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 删除除了最近 3 个外的所有目录
        find /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/ -maxdepth 1 -type d -name "moe_*" -printf '%T@ %p\n' | sort -n | head -n -3 | cut -d' ' -f2- | while read dir; do
            echo "删除: $dir"
            rm -rf "$dir"
        done
        echo -e "${GREEN}✅ 清理完成${NC}"
    else
        echo "取消清理"
    fi
else
    echo "跳过清理"
fi
echo ""

# 步骤 4：清理缓存
read -p "是否清理 Hugging Face 缓存？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  清理缓存..."
    rm -rf ~/.cache/huggingface/
    echo -e "${GREEN}✅ Hugging Face 缓存已清理${NC}"
fi
echo ""

# 步骤 5：清理 pip 缓存
read -p "是否清理 pip 缓存？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  清理 pip 缓存..."
    pip cache purge > /dev/null 2>&1
    echo -e "${GREEN}✅ pip 缓存已清理${NC}"
fi
echo ""

# 步骤 6：显示清理后的磁盘使用情况
echo "📊 清理后的磁盘使用情况："
echo "============================================================"
df -h /root/
echo ""

# 步骤 7：显示可用空间
AVAILABLE=$(df /root/ | awk 'NR==2 {print $4}')
echo "💾 可用空间: $AVAILABLE"
echo ""

# 检查是否有足够的空间
AVAILABLE_GB=$(df /root/ | awk 'NR==2 {print $4}' | sed 's/G//')
if (( $(echo "$AVAILABLE_GB > 20" | bc -l) )); then
    echo -e "${GREEN}✅ 磁盘空间充足，可以继续训练${NC}"
else
    echo -e "${YELLOW}⚠️  磁盘空间可能不足，建议继续清理${NC}"
fi
echo ""

echo "============================================================"
echo "清理完成！"
echo "============================================================"

