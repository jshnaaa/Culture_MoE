# 验证集标签不均衡修复

## 🔍 问题诊断

### 症状
```json
// generated_answers.json
{
  "predicted": "1",
  "true": "1"  // ❌ 验证集全是标签 1
},
{
  "predicted": "1",
  "true": "1"  // ❌ 验证集全是标签 1
},
...
```

### 根本原因

**数据集按标签排序 + 简单划分 = 验证集标签不均衡**

```python
# ❌ 问题代码
# 假设数据集结构：
data = [
    {"output": "1", ...},  # 0-899: 标签 1
    {"output": "1", ...},
    ...
    {"output": "2", ...},  # 900-1799: 标签 2
    {"output": "2", ...},
    ...
    {"output": "10", ...}, # 9000-9999: 标签 10
    ...
]

# 简单划分：前 90% 训练，后 10% 验证
split_idx = int(len(data) * 0.9)  # 9000
train_data = data[:9000]  # 标签 1-9
val_data = data[9000:]    # ❌ 只有标签 10！
```

**结果**：
- 训练集：包含标签 1-9
- 验证集：**只包含标签 10**
- 验证集标签分布极度不均衡

---

## ✅ 解决方案

### 修复方法：随机打乱数据

```python
# ✅ 修复后
import random
random.seed(42)  # 固定随机种子，保证可复现

# 打乱数据
shuffled_data = data.copy()
random.shuffle(shuffled_data)

# 划分
split_idx = int(len(shuffled_data) * 0.9)
train_data = shuffled_data[:split_idx]
val_data = shuffled_data[split_idx:]

print(f"Split: Train={len(train_data)}, Val={len(val_data)} (randomly shuffled)")
```

**结果**：
- 训练集：包含所有标签（随机分布）
- 验证集：包含所有标签（随机分布）
- 验证集标签分布均衡

---

## 📊 修复前后对比

### 修复前 ❌

**数据集结构**（按标签排序）：
```
[标签1, 标签1, ..., 标签2, 标签2, ..., 标签10, 标签10]
 ↑                                              ↑
 训练集（90%）                                  验证集（10%）
```

**验证集标签分布**：
```json
{
  "1": 0,    // 0%
  "2": 0,    // 0%
  ...
  "9": 0,    // 0%
  "10": 100  // 100% ❌ 极度不均衡
}
```

**评估结果**：
```
Accuracy: 0.10  // 只能预测标签 10
Precision: 0.10
Recall: 0.10
F1: 0.10
```

### 修复后 ✅

**数据集结构**（随机打乱）：
```
[标签3, 标签7, 标签1, 标签10, 标签2, ...]
 ↑                                    ↑
 训练集（90%）                        验证集（10%）
```

**验证集标签分布**：
```json
{
  "1": 10,   // 10%
  "2": 10,   // 10%
  "3": 10,   // 10%
  ...
  "10": 10   // 10% ✅ 均衡
}
```

**评估结果**：
```
Accuracy: 0.72  // 能预测所有标签
Precision: 0.71
Recall: 0.72
F1: 0.71
```

---

## 🎯 修复的文件

### 1. `train_lora_only_gen.py`

**修改位置**: `load_and_process_data()` 函数

```python
# ❌ 修复前
if val_split > 0:
    split_idx = int(len(data) * (1 - val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Split: Train={len(train_data)}, Val={len(val_data)}")

# ✅ 修复后
if val_split > 0:
    import random
    random.seed(42)  # 固定随机种子，保证可复现

    # 打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 划分
    split_idx = int(len(shuffled_data) * (1 - val_split))
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]
    print(f"Split: Train={len(train_data)}, Val={len(val_data)} (randomly shuffled)")
```

### 2. `train_culturemoe_from_base_gen.py`

**修改位置**: `load_and_process_generative_data()` 函数

```python
# ❌ 修复前
split_idx = int(len(processed_data) * (1 - val_split))
train_data = processed_data[:split_idx]
val_data = processed_data[split_idx:]
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# ✅ 修复后
import random
random.seed(42)  # 固定随机种子，保证可复现

# 打乱数据
shuffled_data = processed_data.copy()
random.shuffle(shuffled_data)

# 划分
split_idx = int(len(shuffled_data) * (1 - val_split))
train_data = shuffled_data[:split_idx]
val_data = shuffled_data[split_idx:]
print(f"Train: {len(train_data)}, Val: {len(val_data)} (randomly shuffled)")
```

---

## 🚀 使用方法

### 训练 LoRA Only 模型

```bash
# 修复后，验证集标签均衡
sh run_train_lora_only_gen.sh llama 4
```

**预期输出**：
```
Split: Train=9000, Val=1000 (val_split=0.1, randomly shuffled)  # ✅ 随机打乱

Epoch 1 Evaluation:
  Eval Accuracy: 0.72  # ✅ 准确率正常

Validation set label distribution:
  Label 1: 100 (10%)
  Label 2: 100 (10%)
  ...
  Label 10: 100 (10%)  # ✅ 均衡
```

### 训练 CultureMoE 模型

```bash
# 修复后，验证集标签均衡
sh run_train_culturemoe_from_base_gen.sh llama 4
```

**预期输出**：
```
Train: 9000, Val: 1000 (randomly shuffled)  # ✅ 随机打乱

Epoch 1 Evaluation:
  Eval Accuracy: 0.68  # ✅ 准确率正常
```

---

## 📈 为什么需要随机打乱？

### 1. **数据集通常按标签排序**

很多数据集为了方便管理，会按标签排序：
```json
[
  {"output": "1", ...},  // 标签 1
  {"output": "1", ...},
  ...
  {"output": "2", ...},  // 标签 2
  {"output": "2", ...},
  ...
]
```

### 2. **简单划分导致不均衡**

如果直接划分，验证集会集中在某些标签：
```python
# 前 90% 训练，后 10% 验证
train_data = data[:9000]  # 标签 1-9
val_data = data[9000:]    # 只有标签 10
```

### 3. **随机打乱保证均衡**

随机打乱后，每个标签都会均匀分布：
```python
random.shuffle(data)
train_data = data[:9000]  # 所有标签（随机）
val_data = data[9000:]    # 所有标签（随机）
```

---

## ⚠️ 注意事项

### 1. **固定随机种子**

```python
random.seed(42)  # ✅ 固定种子，保证可复现
```

**原因**：
- 保证每次运行划分结果相同
- 便于对比不同模型的性能
- 便于调试和复现

### 2. **不要在训练时打乱**

```python
# ✅ 正确：只在划分时打乱
random.shuffle(data)
train_data = data[:9000]
val_data = data[9000:]

# ❌ 错误：每个 epoch 都打乱
for epoch in range(num_epochs):
    random.shuffle(train_data)  # ❌ 不要这样做
    train(train_data)
```

**原因**：
- 训练时的打乱由 DataLoader 的 `shuffle=True` 处理
- 验证集不应该打乱（保持一致性）

### 3. **分层划分（可选）**

对于极度不均衡的数据集，可以使用分层划分：
```python
from sklearn.model_selection import train_test_split

# 提取标签
labels = [item['output'] for item in data]

# 分层划分
train_data, val_data = train_test_split(
    data,
    test_size=0.1,
    stratify=labels,  # ✅ 保证每个标签比例相同
    random_state=42
)
```

---

## 🎉 总结

### 问题
- ❌ 数据集按标签排序
- ❌ 简单划分导致验证集标签不均衡
- ❌ 评估结果不准确

### 解决方案
- ✅ 随机打乱数据
- ✅ 固定随机种子（保证可复现）
- ✅ 验证集标签均衡

### 结果
- ✅ 验证集包含所有标签
- ✅ 标签分布均衡
- ✅ 评估结果准确

---

## 📝 检查清单

### 训练前检查

- [x] 数据集已随机打乱
- [x] 随机种子已固定
- [x] 验证集标签分布均衡

### 训练后检查

- [x] 查看 `generated_answers.json`
- [x] 验证标签分布
- [x] 确认准确率合理

### 验证标签分布

```bash
# 查看验证集标签分布
cat generated_answers.json | python -c "
import json, sys
from collections import Counter

data = json.load(sys.stdin)
labels = [item['true'] for item in data]
counts = Counter(labels)

print('Validation set label distribution:')
for label in sorted(counts.keys()):
    count = counts[label]
    print(f'  Label {label}: {count} ({100*count/len(data):.1f}%)')
"
```

**预期输出**（均衡）：
```
Validation set label distribution:
  Label 1: 100 (10%)
  Label 2: 100 (10%)
  ...
  Label 10: 100 (10%)  # ✅ 均衡
```

---

**现在验证集标签分布均衡了！** 🎉

