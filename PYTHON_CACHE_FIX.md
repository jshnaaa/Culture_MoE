# Python 缓存导致的代码修复未生效问题

## 🔍 问题描述

运行 `bash run_train_culturemoe_from_base_gen.sh llama 4` 时，报错：

```python
AttributeError: 'list' object has no attribute 'unsqueeze'
  File "CultureMoE.py", line 248, in compute_culture_loss
    culture_labels = culture_labels.unsqueeze(1)
```

但代码已经修复了（第 257 行使用的是 `culture_labels_expanded`）。

## 🔍 根本原因

**Python 缓存问题**：

1. **`.pyc` 文件缓存**：Python 编译的字节码文件没有更新
2. **`__pycache__` 目录**：包含旧版本的编译代码
3. **模块导入缓存**：Python 的 `sys.modules` 缓存了旧版本

## ✅ 解决方案

### 方案 1：清理缓存（推荐）

```bash
# 清理所有 __pycache__ 目录
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# 清理所有 .pyc 文件
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 清理所有 .pyo 文件
find . -type f -name "*.pyo" -delete 2>/dev/null || true
```

### 方案 2：使用提供的脚本

```bash
# 使用清理缓存脚本
bash clear_cache_and_train.sh llama 4
```

### 方案 3：在 Python 中清理缓存

```python
import sys
import importlib

# 清理模块缓存
if 'llamafactory.model.CultureMoE' in sys.modules:
    del sys.modules['llamafactory.model.CultureMoE']

# 重新导入
from llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
```

### 方案 4：使用 Python 的 `-B` 标志

```bash
# 不写入 .pyc 文件
python -B train_culturemoe_from_base_gen.py ...

# 或使用 PYTHONDONTWRITEBYTECODE
export PYTHONDONTWRITEBYTECODE=1
python train_culturemoe_from_base_gen.py ...
```

## 📊 缓存文件位置

### 典型的缓存位置

```
项目根目录/
├── src/
│   ├── llamafactory/
│   │   ├── __pycache__/          # ❌ 需要删除
│   │   │   └── *.pyc             # ❌ 需要删除
│   │   ├── model/
│   │   │   ├── __pycache__/      # ❌ 需要删除
│   │   │   ├── CultureMoE.py     # ✅ 源代码
│   │   │   └── CultureMoE.cpython-312.pyc  # ❌ 需要删除
│   │   └── ...
│   └── ...
└── ...
```

## 🔧 完整的清理脚本

### 创建 `clean_all.sh`

```bash
#!/bin/bash

echo "🧹 Cleaning Python cache..."

# 清理 __pycache__ 目录
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "  ✅ Removed __pycache__ directories"

# 清理 .pyc 文件
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  ✅ Removed .pyc files"

# 清理 .pyo 文件
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✅ Removed .pyo files"

# 清理 .egg-info 目录
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
echo "  ✅ Removed .egg-info directories"

# 清理 build 目录
rm -rf build/ 2>/dev/null || true
echo "  ✅ Removed build directory"

# 清理 dist 目录
rm -rf dist/ 2>/dev/null || true
echo "  ✅ Removed dist directory"

echo ""
echo "✅ Cache cleaning completed!"
```

### 使用方法

```bash
bash clean_all.sh
bash run_train_culturemoe_from_base_gen.sh llama 4
```

## 📝 预防措施

### 1. **添加到 `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
```

### 2. **在 CI/CD 中清理缓存**

```yaml
# GitHub Actions 示例
- name: Clean Python cache
  run: |
    find . -type d -name __pycache__ -exec rm -rf {} + || true
    find . -type f -name "*.pyc" -delete || true
```

### 3. **定期清理脚本**

```bash
# 在 setup.py 中添加
from setuptools import setup
import os
import shutil

# 清理缓存
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        shutil.rmtree(os.path.join(root, '__pycache__'))
```

## 🎯 快速修复步骤

### 步骤 1：清理缓存

```bash
cd /Users/yzl/ownCode/Culture_Moe
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
```

### 步骤 2：验证代码

```bash
# 检查代码是否正确
grep -n "culture_labels_expanded" src/llamafactory/model/CultureMoE.py
# 应该看到第 257 行有 culture_labels_expanded
```

### 步骤 3：重新运行训练

```bash
bash run_train_culturemoe_from_base_gen.sh llama 4
```

## 📊 常见的缓存问题

| 问题 | 症状 | 解决方案 |
|------|------|--------|
| **旧代码执行** | 修改后代码仍然执行旧逻辑 | 清理 `__pycache__` |
| **导入错误** | `ModuleNotFoundError` | 清理 `__pycache__` + 重启 Python |
| **属性错误** | `AttributeError` 在修复后仍然出现 | 清理 `.pyc` 文件 |
| **版本不匹配** | 代码版本与执行版本不一致 | 清理所有缓存 + 重新导入 |

## 🔍 调试技巧

### 1. **检查加载的模块**

```python
import sys
import llamafactory.model.CultureMoE as cm

print(f"Module file: {cm.__file__}")
print(f"Module cached: {'llamafactory.model.CultureMoE' in sys.modules}")
```

### 2. **查看源代码位置**

```python
import inspect
from llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel

print(inspect.getfile(LlamaSharedRouterExpertsModel))
```

### 3. **强制重新加载**

```python
import importlib
import llamafactory.model.CultureMoE

importlib.reload(llamafactory.model.CultureMoE)
```

## 总结

✅ **主要原因**：Python 缓存导致旧代码被执行

✅ **快速修复**：
```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
```

✅ **预防措施**：
- 添加 `__pycache__` 到 `.gitignore`
- 定期清理缓存
- 使用 `PYTHONDONTWRITEBYTECODE=1` 环境变量

✅ **验证修复**：
```bash
grep -n "culture_labels_expanded" src/llamafactory/model/CultureMoE.py
bash run_train_culturemoe_from_base_gen.sh llama 4
```

现在应该可以正常运行了！🎉

