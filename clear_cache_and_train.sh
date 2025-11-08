#!/bin/bash

# 清理 Python 缓存
echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

echo "✅ Cache cleared"
echo ""

# 运行训练
echo "Starting training..."
bash run_train_culturemoe_from_base_gen.sh "$@"

