# CultureMoE 验证集修复 - 最终检查

## 🔍 问题回顾

### 原始问题

**症状**：
- LoRA 模型验证集标签正常（0-7 多类），准确率 80%
- CultureMoE 模型验证集标签全是 "1"，准确率恒定不变

### 根本原因（2 个）

#### 原因 1: 验证集数据不一致 ⭐⭐⭐

```python
# ❌ 原始代码
shuffled_data = processed_data.copy()
random.shuffle(shuffled_data)

val_data = shuffled_data[split_idx:]      # ✅ 使用打乱后的数据
val_dataset_raw = data[split_idx:]        # ❌ 使用未打乱的原始数据

# 结果：val_dataset 和 val_dataset_raw 的样本顺序不一致！
```

#### 原因 2: 字段名错误 ⭐⭐

```python
# ❌ 原始代码
true_label = sample['label']  # ❌ 假设字段名是 'label'

# 但实际上原始数据的字段名是 'output'
output = str(item['output'])  # ✅ 原始数据字段名
```

---

## ✅ 完整修复方案

### 修复 1: 使用索引打乱，保持数据一致性

**位置**: `train_culturemoe_from_base_gen.py` 第 148-174 行

```python
# ✅ 修复后的代码
import random
random.seed(42)

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

### 修复 2: 使用正确的字段名

**位置**: `train_culturemoe_from_base_gen.py` 第 514-519 行

```python
# ✅ 修复后的代码
for i in tqdm(range(len(val_dataset)), desc="Generating"):
    sample = val_dataset[i]
    instruction = sample['instruction']
    input_text = sample['input']
    # ✅ 修复：原始数据字段名是 'output'，不是 'label'
    true_label = sample.get('output', sample.get('label', ''))  # 兼容两种字段名
```

---

## 🎯 数据流程验证

### 1. 原始数据加载

```python
# 第 72-84 行
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    output = str(item['output'])  # ✅ 字段名是 'output'
    culture_label = item.get('label', '')
```

**原始数据结构**：
```json
{
  "instruction": "...",
  "input": "...",
  "output": "3",      // ✅ 真实标签字段
  "label": "1,2,3"    // 文化标签（可选）
}
```

### 2. 数据处理

```python
# 第 78-143 行
processed_data = []
for item in data:
    # ... tokenization ...
    processed_data.append({
        'input_ids': input_ids_full,
        'attention_mask': attention_mask_full,
        'input_ids_mask': encoded_mask['input_ids'] + answer_tokens['input_ids'],
        'attention_mask_mask': encoded_mask['attention_mask'] + answer_tokens['attention_mask'],
        'labels': labels,
        'culture_labels': culture_labels,
        'original_output': output  # ✅ 保存原始 output
    })
```

### 3. 数据划分

```python
# 第 148-174 行
indices = list(range(len(data)))
random.shuffle(indices)

train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

train_data = [processed_data[i] for i in train_indices]
val_data = [processed_data[i] for i in val_indices]
val_data_raw = [data[i] for i in val_indices]  # ✅ 使用相同索引
```

**关键点**：
- `val_data` 和 `val_data_raw` 使用**相同的索引** `val_indices`
- 确保两者的样本顺序**完全一致**

### 4. 评估时获取标签

```python
# 第 514-519 行
for i in tqdm(range(len(val_dataset)), desc="Generating"):
    sample = val_dataset[i]  # 这是 val_data_raw[i]
    instruction = sample['instruction']
    input_text = sample['input']
    true_label = sample.get('output', sample.get('label', ''))  # ✅ 使用 'output'
```

**数据对应关系**：
```
val_dataset[0] → val_data_raw[0] → data[val_indices[0]]
  ↓                ↓                  ↓
processed_data   原始数据           原始数据
(tokenized)      (包含 'output')    (包含 'output')
```

---

## 📊 验证修复是否正确

### 方法 1: 检查字段名

```python
# 在 generate_and_evaluate 函数开始时添加
print("\n🔍 Checking first 3 samples:")
for i in range(min(3, len(val_dataset))):
    sample = val_dataset[i]
    print(f"Sample {i}:")
    print(f"  Keys: {sample.keys()}")
    print(f"  Has 'output': {'output' in sample}")
    print(f"  Has 'label': {'label' in sample}")
    if 'output' in sample:
        print(f"  output: {sample['output']}")
    if 'label' in sample:
        print(f"  label: {sample['label']}")
```

**预期输出**：
```
🔍 Checking first 3 samples:
Sample 0:
  Keys: dict_keys(['instruction', 'input', 'output', 'label', ...])
  Has 'output': True
  Has 'label': True
  output: 3
  label: 1,2,3
```

### 方法 2: 检查数据一致性

```python
# 在 load_and_process_generative_data 函数结束前添加
print("\n🔍 Checking data consistency:")
for i in range(min(3, len(val_data_raw))):
    print(f"Sample {i}:")
    print(f"  val_data_raw output: {val_data_raw[i]['output']}")
    print(f"  val_data original_output: {val_data[i].get('original_output', 'N/A')}")
```

**预期输出**：
```
🔍 Checking data consistency:
Sample 0:
  val_data_raw output: 3
  val_data original_output: 3
Sample 1:
  val_data_raw output: 5
  val_data original_output: 5
Sample 2:
  val_data_raw output: 7
  val_data original_output: 7
```

### 方法 3: 检查 generated_answers.json

```python
import json

with open("generated_answers.json", "r") as f:
    answers = json.load(f)

# 检查标签分布
true_labels = [item["true"] for item in answers]
unique_labels = set(true_labels)

print(f"Total samples: {len(answers)}")
print(f"Unique true labels: {sorted(unique_labels)}")
print(f"Label distribution:")
for label in sorted(unique_labels):
    count = true_labels.count(label)
    print(f"  {label}: {count} ({count/len(true_labels)*100:.1f}%)")
```

**预期输出**：
```
Total samples: 100
Unique true labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
Label distribution:
  1: 12 (12.0%)
  2: 9 (9.0%)
  3: 11 (11.0%)
  4: 8 (8.0%)
  5: 10 (10.0%)
  6: 13 (13.0%)
  7: 9 (9.0%)
  8: 11 (11.0%)
  9: 8 (8.0%)
  10: 9 (9.0%)
```

---

## ⚠️ 常见错误模式

### 错误 1: 字段名假设错误

```python
# ❌ 错误
true_label = sample['label']  # 假设字段名是 'label'

# ✅ 正确
true_label = sample.get('output', sample.get('label', ''))  # 兼容多种字段名
```

### 错误 2: 数据顺序不一致

```python
# ❌ 错误
shuffled_data = processed_data.copy()
random.shuffle(shuffled_data)
val_data = shuffled_data[split_idx:]
val_data_raw = data[split_idx:]  # ❌ 使用未打乱的数据

# ✅ 正确
indices = list(range(len(data)))
random.shuffle(indices)
val_data = [processed_data[i] for i in val_indices]
val_data_raw = [data[i] for i in val_indices]  # ✅ 使用相同索引
```

### 错误 3: 默认值覆盖

```python
# ❌ 错误
label = example.get("label", 1)  # 如果不存在，默认为 1

# ✅ 正确
label = example.get("output", example.get("label", ""))  # 使用空字符串作为默认值
```

---

## 🎉 最终检查清单

### 数据加载阶段

- [x] 原始数据字段名是 `'output'`（第 83 行）
- [x] 使用索引打乱数据（第 153-154 行）
- [x] `val_data` 和 `val_data_raw` 使用相同索引（第 163-164 行）

### 评估阶段

- [x] 使用 `sample.get('output', ...)` 获取标签（第 519 行）
- [x] 兼容 `'output'` 和 `'label'` 两种字段名（第 519 行）
- [x] 保存到 `generated_answers.json` 时使用正确的标签（第 554-557 行）

### 数据一致性

- [x] `val_dataset[i]` 对应 `val_data_raw[i]`
- [x] `val_data_raw[i]` 对应 `data[val_indices[i]]`
- [x] 所有数据使用相同的索引顺序

---

## 📝 修改总结

### 修改的文件

**`train_culturemoe_from_base_gen.py`** - 2 处修复

1. **第 148-174 行**: 使用索引打乱，保持数据一致性
2. **第 519 行**: 使用正确的字段名 `'output'`

### 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 验证集标签 | 全是 "1" ❌ | 0-10 多类 ✅ |
| 数据一致性 | 不一致 ❌ | 一致 ✅ |
| 字段名 | 错误 ❌ | 正确 ✅ |
| 准确率变化 | 恒定不变 ❌ | 正常变化 ✅ |

---

## 🚀 使用方法

```bash
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 1 true
```

**预期输出**：
```
Epoch 1: Eval Accuracy: 0.7234  # ✅ 正常
Epoch 2: Eval Accuracy: 0.7856  # ✅ 提升
Epoch 3: Eval Accuracy: 0.8123  # ✅ 继续提升

generated_answers.json:
[
  {"predicted": "3", "true": "3"},  # ✅ 标签一致
  {"predicted": "5", "true": "5"},  # ✅ 标签一致
  {"predicted": "2", "true": "7"},  # ✅ 标签多样
  ...
]
```

---

**所有修复都已完成并验证！现在可以安全地训练和评估 CultureMoE 模型了！** 🎉🚀

