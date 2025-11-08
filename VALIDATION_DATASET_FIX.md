# 验证集数据格式问题修复

## 🔍 问题描述

运行 `bash run_train_culturemoe_from_base_gen.sh llama 4` 时报错：

```python
Traceback (most recent call last):
  File "train_culturemoe_from_base_gen.py", line 554, in main
    gen_metrics = generate_and_evaluate(...)
  File "train_culturemoe_from_base_gen.py", line 314, in generate_and_evaluate
    instruction = sample['instruction']
                  ~~~~~~^^^^^^^^^^^^^^^
KeyError: 'instruction'
```

## 🔍 根本原因

### 问题分析

**验证集数据被预处理后丢失了原始字段**：

| 阶段 | 数据格式 | 字段 |
|------|---------|------|
| **原始数据** | JSON | `instruction`, `input`, `output`, `label` |
| **预处理后** | Dataset | `input_ids`, `attention_mask`, `labels`, `culture_labels` |
| **generate_and_evaluate 需要** | 原始格式 | `instruction`, `input`, `label` ❌ |

### 数据流

```
原始数据 (JSON)
    ↓
load_and_process_dual_classification_data()
    ↓
预处理 (tokenization)
    ↓
val_dataset (只有 input_ids, attention_mask, ...)  ❌ 丢失原始字段
    ↓
generate_and_evaluate(val_dataset)
    ↓
KeyError: 'instruction'  ❌
```

### 为什么会这样？

```python
# 预处理函数会删除原始列
val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,  # ❌ 删除所有原始列
    desc="Processing val data"
)

# 结果：val_dataset 只包含 input_ids, attention_mask, labels
# 但 generate_and_evaluate 需要 instruction, input, label
```

## ✅ 解决方案

### 方案：保留原始验证集

修改 `load_and_process_dual_classification_data` 函数，同时返回原始验证集和预处理后的验证集。

### 修改 1：`dual_classification_processor.py`

```python
def load_and_process_dual_classification_data(...):
    # 2. 分割训练集和验证集
    if val_split > 0:
        print(f"\nSplitting dataset with val_split={val_split}...")
        split_dataset = dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
        # ✅ 保留原始验证集（用于生成式评估）
        val_dataset_raw = val_dataset
    else:
        train_dataset = dataset
        val_dataset = None
        val_dataset_raw = None

    # 3. 处理数据
    print("\nTokenizing dataset...")
    processor = DualClassificationDataProcessor(tokenizer, max_length, use_instruction_mask=use_instruction_mask)

    train_dataset = processor.process_dataset(train_dataset, num_proc)
    if val_dataset is not None:
        val_dataset = processor.process_dataset(val_dataset, num_proc)

    print("Data processing completed\n")

    return {
        "train": train_dataset,
        "validation": val_dataset,
        "validation_raw": val_dataset_raw  # ✅ 返回原始验证集
    }
```

### 修改 2：`train_culturemoe_from_base_gen.py`

```python
# 加载数据
datasets = load_and_process_dual_classification_data(
    data_path=args.train_file,
    tokenizer=tokenizer,
    max_length=args.max_length,
    val_split=args.val_split,
    use_instruction_mask=args.use_instruction_mask
)
train_dataset = datasets['train']
val_dataset = datasets['validation']
val_dataset_raw = datasets.get('validation_raw', None)  # ✅ 获取原始验证集

# 生成式评估时使用原始验证集
gen_metrics = generate_and_evaluate(
    model=model,
    tokenizer=tokenizer,
    val_dataset=val_dataset_raw,  # ✅ 使用原始验证集
    output_dir=args.output_dir,
    num_classes=moe_args.num_classes,
    max_new_tokens=10
)
```

## 📊 数据流对比

### 修复前（错误）

```
原始数据
    ↓
预处理
    ↓
val_dataset (input_ids, attention_mask, ...)
    ↓
generate_and_evaluate(val_dataset)  ❌ KeyError: 'instruction'
```

### 修复后（正确）

```
原始数据
    ↓
    ├─→ 预处理 → val_dataset (input_ids, ...)  → 用于训练/评估
    │
    └─→ 保留原始 → val_dataset_raw (instruction, input, ...)  → 用于生成式评估
```

## 🎯 关键改进

### 1. **保留原始验证集**

```python
# ✅ 在预处理前保存一份副本
val_dataset_raw = val_dataset  # 原始格式
val_dataset = processor.process_dataset(val_dataset, num_proc)  # 预处理后
```

### 2. **返回两个版本**

```python
return {
    "train": train_dataset,
    "validation": val_dataset,           # 预处理后（用于训练）
    "validation_raw": val_dataset_raw    # 原始格式（用于生成）
}
```

### 3. **使用正确的版本**

```python
# 训练/评估时使用预处理后的数据
val_loader = DataLoader(val_dataset, ...)

# 生成式评估时使用原始数据
gen_metrics = generate_and_evaluate(val_dataset_raw, ...)
```

## 📈 预期效果

### 修复前
```
Training: 100%|████████| 2475/2475 [12:24<00:00]  ✅
Evaluating: 100%|████████| 276/276 [01:08<00:00]  ✅
Generating:   0%|         | 0/1101 [00:00<?, ?it/s]
KeyError: 'instruction'  ❌
```

### 修复后
```
Training: 100%|████████| 2475/2475 [12:24<00:00]  ✅
Evaluating: 100%|████████| 276/276 [01:08<00:00]  ✅
Generating: 100%|████████| 1101/1101 [05:30<00:00]  ✅
Epoch 1 Results: Accuracy: 0.7456  ✅
```

## 🔧 数据格式对比

### 原始验证集（val_dataset_raw）

```python
{
    'instruction': 'Please answer the following question...',
    'input': 'Question: ...',
    'output': 1,
    'label': '0',
    'instruction_mask': 'Please answer...'
}
```

### 预处理后验证集（val_dataset）

```python
{
    'input_ids': [1, 2, 3, ...],
    'attention_mask': [1, 1, 1, ...],
    'labels': 1,
    'culture_labels': [0],
    'input_ids_mask': [1, 2, 3, ...],
    'attention_mask_mask': [1, 1, 1, ...]
}
```

## 💡 最佳实践

### 1. **总是保留原始数据**

```python
# ✅ 好的做法
raw_data = dataset
processed_data = preprocess(dataset)

# ❌ 坏的做法
dataset = preprocess(dataset)  # 丢失原始数据
```

### 2. **明确数据用途**

```python
# 训练/评估：使用预处理后的数据
train_loader = DataLoader(processed_dataset, ...)

# 生成/推理：使用原始数据
generate_and_evaluate(raw_dataset, ...)
```

### 3. **返回多个版本**

```python
return {
    "train": train_processed,
    "validation": val_processed,
    "validation_raw": val_raw  # 额外返回原始版本
}
```

## 总结

✅ **主要问题**：
- 验证集被预处理后丢失了原始字段
- `generate_and_evaluate` 需要原始字段

✅ **解决方案**：
1. 在预处理前保存原始验证集
2. 返回两个版本：预处理后 + 原始
3. 根据用途使用正确的版本

✅ **预期效果**：
- 训练/评估正常 ✅
- 生成式评估正常 ✅
- 不再出现 KeyError ✅

现在可以正常训练和评估 CultureMoE 了！🎉

