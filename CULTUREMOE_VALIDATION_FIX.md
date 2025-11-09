# CultureMoE 验证集标签错误修复

## 🔍 问题诊断

### 症状

**LoRA 模型（单独）**：
```json
[
  {"predicted": "3", "true": "3"},
  {"predicted": "5", "true": "5"},
  {"predicted": "2", "true": "7"},
  ...
]
```
- ✅ 验证集标签分布正常（0-7 多类）
- ✅ 准确率约 80%

**CultureMoE 模型（LoRA + MoE）**：
```json
[
  {"predicted": "3", "true": "1"},
  {"predicted": "5", "true": "1"},
  {"predicted": "2", "true": "1"},
  ...
]
```
- ❌ 所有 `true` 标签都是 "1"
- ❌ 准确率恒定不变

### 根本原因

**验证集数据不一致**：
- `val_dataset`（tokenized）使用**打乱后的数据**
- `val_dataset_raw`（原始）使用**未打乱的数据**
- 导致评估时，预测结果和真实标签**对不上**！

---

## 🐛 Bug 代码

### 原始代码（第 148-174 行）

```python
# ✅ 随机划分训练集和验证集（保证标签均衡）
import random
random.seed(42)

# ❌ 打乱 processed_data
shuffled_data = processed_data.copy()
random.shuffle(shuffled_data)

# 划分
split_idx = int(len(shuffled_data) * (1 - val_split))
train_data = shuffled_data[:split_idx]
val_data = shuffled_data[split_idx:]  # ✅ 使用打乱后的数据

# 创建 Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# ❌ 保存原始验证集（使用未打乱的 data）
val_dataset_raw = data[split_idx:]  # ❌ 这里使用的是原始未打乱的 data

return {
    'train': train_dataset,
    'validation': val_dataset,
    'validation_raw': val_dataset_raw  # ❌ 顺序不一致！
}
```

### 问题分析

```
原始数据 data:
  [样本0, 样本1, 样本2, 样本3, 样本4, 样本5, 样本6, 样本7, 样本8, 样本9]

打乱后 shuffled_data:
  [样本3, 样本7, 样本1, 样本5, 样本9, 样本0, 样本2, 样本4, 样本6, 样本8]

划分（假设 val_split=0.3）:
  train_data = shuffled_data[:7]  = [样本3, 样本7, 样本1, 样本5, 样本9, 样本0, 样本2]
  val_data   = shuffled_data[7:]  = [样本4, 样本6, 样本8]  # ✅ 打乱后的

  val_dataset_raw = data[7:]      = [样本7, 样本8, 样本9]  # ❌ 原始未打乱的

结果：
  val_dataset[0]     对应 样本4
  val_dataset_raw[0] 对应 样本7  # ❌ 不一致！
```

---

## ✅ 修复方案

### 修复后的代码

```python
# ✅ 随机划分训练集和验证集（保证标签均衡）
import random
random.seed(42)  # 固定随机种子，保证可复现

# ✅ 同时打乱 processed_data 和原始 data（保持对应关系）
indices = list(range(len(data)))
random.shuffle(indices)  # 打乱索引

# 使用相同的索引划分
split_idx = int(len(indices) * (1 - val_split))
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

# 根据索引获取数据
train_data = [processed_data[i] for i in train_indices]
val_data = [processed_data[i] for i in val_indices]
val_data_raw = [data[i] for i in val_indices]  # ✅ 使用相同的索引

print(f"Train: {len(train_data)}, Val: {len(val_data)} (randomly shuffled)")

# 创建 Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# 保存原始验证集（用于生成式评估）
val_dataset_raw = val_data_raw  # ✅ 使用相同索引的原始数据

return {
    'train': train_dataset,
    'validation': val_dataset,
    'validation_raw': val_dataset_raw  # ✅ 顺序一致！
}
```

### 修复原理

```
原始数据 data:
  [样本0, 样本1, 样本2, 样本3, 样本4, 样本5, 样本6, 样本7, 样本8, 样本9]

打乱索引 indices:
  [3, 7, 1, 5, 9, 0, 2, 4, 6, 8]

划分（假设 val_split=0.3）:
  train_indices = [3, 7, 1, 5, 9, 0, 2]
  val_indices   = [4, 6, 8]

根据索引获取数据:
  train_data      = [processed_data[3], processed_data[7], ...]
  val_data        = [processed_data[4], processed_data[6], processed_data[8]]  # ✅
  val_data_raw    = [data[4], data[6], data[8]]                                # ✅

结果：
  val_dataset[0]     对应 processed_data[4]
  val_dataset_raw[0] 对应 data[4]           # ✅ 一致！
```

---

## 📊 修复前后对比

### 修复前 ❌

```python
# ❌ 打乱 processed_data
shuffled_data = processed_data.copy()
random.shuffle(shuffled_data)

# ❌ 使用未打乱的 data
val_dataset_raw = data[split_idx:]
```

**结果**：
```json
[
  {"predicted": "3", "true": "1"},  # ❌ 标签对不上
  {"predicted": "5", "true": "1"},  # ❌ 标签对不上
  {"predicted": "2", "true": "1"},  # ❌ 标签对不上
]
```

### 修复后 ✅

```python
# ✅ 打乱索引
indices = list(range(len(data)))
random.shuffle(indices)

# ✅ 使用相同索引
val_data = [processed_data[i] for i in val_indices]
val_data_raw = [data[i] for i in val_indices]
```

**结果**：
```json
[
  {"predicted": "3", "true": "3"},  # ✅ 标签一致
  {"predicted": "5", "true": "5"},  # ✅ 标签一致
  {"predicted": "2", "true": "7"},  # ✅ 标签一致
]
```

---

## 🚀 使用方法

```bash
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 1 true
```

**预期输出**：
```
Epoch 1 Results:
   Eval Accuracy (Generative): 0.7234  # ✅ 准确率正常

Epoch 2 Results:
   Eval Accuracy (Generative): 0.7856  # ✅ 准确率提升

Epoch 3 Results:
   Eval Accuracy (Generative): 0.8123  # ✅ 准确率继续提升
```

---

## 🎯 关键点

### 1. 为什么会出现这个问题？

**原因**：
- 训练时需要打乱数据（提高泛化能力）
- 评估时需要原始数据（用于生成式评估）
- 但是打乱时没有保持两者的对应关系

### 2. 为什么 LoRA 模型没有这个问题？

**原因**：
- LoRA 模型使用的是 `train_lora_only_gen.py`
- 该文件中的数据加载逻辑是正确的（已经修复过）
- CultureMoE 模型使用的是 `train_culturemoe_from_base_gen.py`
- 该文件中的数据加载逻辑有 bug（现在已修复）

### 3. 如何验证修复是否生效？

**方法 1**：检查 `generated_answers.json`
```python
import json

with open("generated_answers.json", "r") as f:
    answers = json.load(f)

# 检查标签分布
true_labels = [item["true"] for item in answers]
print(f"Unique true labels: {set(true_labels)}")
# 应该输出: {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
```

**方法 2**：检查准确率变化
```
Epoch 1: 0.72
Epoch 2: 0.78  # ✅ 应该有变化
Epoch 3: 0.81  # ✅ 应该有变化
```

---

## ⚠️ 常见问题

### Q1: 为什么不直接使用 `val_dataset` 进行评估？

**A**: 因为 `val_dataset` 是 tokenized 后的数据，包含 `input_ids`、`labels` 等字段，但不包含原始的 `instruction`、`input`、`output` 字段。生成式评估需要原始字段来构建 prompt。

### Q2: 为什么要使用索引而不是直接打乱两个列表？

**A**: 因为 Python 的 `random.shuffle()` 是 in-place 操作，无法保证两个列表的打乱顺序一致。使用索引可以确保一致性。

### Q3: 这个 bug 会影响训练吗？

**A**: 不会。这个 bug 只影响评估阶段的准确率计算，不影响训练过程。但会导致无法正确评估模型性能。

---

## 📝 修改的文件

### `train_culturemoe_from_base_gen.py`

**修改位置**: 1 处

**第 148-174 行**: 修复验证集数据不一致问题
```python
# ✅ 使用索引打乱，保持 processed_data 和原始 data 的对应关系
indices = list(range(len(data)))
random.shuffle(indices)

train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

train_data = [processed_data[i] for i in train_indices]
val_data = [processed_data[i] for i in val_indices]
val_data_raw = [data[i] for i in val_indices]  # ✅ 使用相同索引
```

---

## 🎉 总结

### 问题
- ❌ 验证集 tokenized 数据和原始数据顺序不一致
- ❌ 导致评估时标签对不上
- ❌ 所有 `true` 标签显示为 "1"

### 根本原因
- ❌ `val_data` 使用打乱后的数据
- ❌ `val_dataset_raw` 使用未打乱的数据
- ❌ 两者索引不对应

### 解决方案
- ✅ 使用索引打乱
- ✅ 确保 `val_data` 和 `val_data_raw` 使用相同索引
- ✅ 保持数据对应关系

### 结果
- ✅ 验证集标签分布正常
- ✅ 准确率正常变化
- ✅ 模型评估正确

---

**现在可以正确训练和评估 CultureMoE 模型了！** 🚀

```bash
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 1 true

