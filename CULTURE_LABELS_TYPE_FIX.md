# culture_labels 类型转换修复

## 🔍 问题描述

运行 `sh run_train_culturemoe_from_base_gen.sh llama 4` 时报错：

```python
Traceback (most recent call last):
  File "train_culturemoe_from_base_gen.py", line 714, in <module>
    main()
  File "train_culturemoe_from_base_gen.py", line 537, in main
    train_metrics = train_epoch(...)
  File "train_culturemoe_from_base_gen.py", line 171, in train_epoch
    outputs = model(...)
  File "CultureMoE.py", line 220, in forward
    culture_loss = self.compute_culture_loss(expert_weights, culture_labels)
  File "CultureMoE.py", line 248, in compute_culture_loss
    culture_labels = culture_labels.unsqueeze(1)  # [B, 1]
                     ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'list' object has no attribute 'unsqueeze'
```

## 🔍 根本原因

### 问题分析

1. **训练代码传递的类型**：`culture_labels` 是 Python list

```python
# train_culturemoe_from_base_gen.py
culture_labels = batch['culture_labels']  # ❌ list 类型
outputs = model(..., culture_labels=culture_labels, ...)
```

2. **CultureMoE 期望的类型**：torch.Tensor

```python
# CultureMoE.py (修复前)
def compute_culture_loss(self, expert_weights, culture_labels):
    culture_labels = culture_labels.unsqueeze(1)  # ❌ list 没有 unsqueeze 方法
```

3. **错误原因**：尝试在 list 上调用 tensor 方法 → `AttributeError`

## ✅ 解决方案

### 修改 `compute_culture_loss` 方法

#### 修复前

```python
def compute_culture_loss(self, expert_weights, culture_labels):
    """
    计算文化损失：鼓励相同文化的样本使用相似的专家

    Args:
        expert_weights: [B, E] 专家权重
        culture_labels: [B] 文化标签

    Returns:
        culture_loss: 标量
    """
    device = expert_weights.device
    batch_size = expert_weights.size(0)

    # ❌ 直接调用 unsqueeze，假设 culture_labels 是 tensor
    culture_labels = culture_labels.unsqueeze(1)  # [B, 1]
    ...
```

#### 修复后

```python
def compute_culture_loss(self, expert_weights, culture_labels):
    """
    计算文化损失：鼓励相同文化的样本使用相似的专家

    Args:
        expert_weights: [B, E] 专家权重
        culture_labels: [B] 文化标签（可以是 list 或 tensor）

    Returns:
        culture_loss: 标量
    """
    device = expert_weights.device
    dtype = expert_weights.dtype

    # ✅ 将 culture_labels 转换为 tensor（如果还不是）
    if not isinstance(culture_labels, torch.Tensor):
        culture_labels = torch.tensor(culture_labels, device=device, dtype=torch.long)
    else:
        # 确保在正确的设备上
        culture_labels = culture_labels.to(device=device, dtype=torch.long)

    batch_size = expert_weights.size(0)

    # 计算样本对之间的文化相似度（相同文化为1，不同文化为0）
    culture_labels_expanded = culture_labels.unsqueeze(1)  # [B, 1]
    culture_similarity = (culture_labels_expanded == culture_labels_expanded.t()).float()  # [B, B]

    # 计算专家权重之间的余弦相似度
    expert_weights_norm = torch.nn.functional.normalize(expert_weights, p=2, dim=1)
    expert_similarity = torch.mm(expert_weights_norm, expert_weights_norm.t())  # [B, B]

    # 文化损失：相同文化的样本应该有相似的专家权重
    culture_loss = torch.nn.functional.mse_loss(
        expert_similarity * culture_similarity,
        culture_similarity
    )

    return culture_loss
```

## 📊 关键改进

### 1. **类型检查**

```python
# ✅ 检查是否是 tensor
if not isinstance(culture_labels, torch.Tensor):
    culture_labels = torch.tensor(culture_labels, device=device, dtype=torch.long)
```

### 2. **设备转移**

```python
# ✅ 确保在正确的设备上
culture_labels = culture_labels.to(device=device, dtype=torch.long)
```

### 3. **变量重命名**

```python
# ✅ 避免覆盖原始变量
culture_labels_expanded = culture_labels.unsqueeze(1)  # [B, 1]
culture_similarity = (culture_labels_expanded == culture_labels_expanded.t()).float()
```

## 🎯 支持的输入类型

### 1. **Python List**

```python
culture_labels = [0, 1, 2, 0, 1]  # ✅ 自动转换为 tensor
```

### 2. **NumPy Array**

```python
import numpy as np
culture_labels = np.array([0, 1, 2, 0, 1])  # ✅ 自动转换为 tensor
```

### 3. **PyTorch Tensor**

```python
import torch
culture_labels = torch.tensor([0, 1, 2, 0, 1])  # ✅ 直接使用
```

### 4. **不同设备的 Tensor**

```python
culture_labels = torch.tensor([0, 1, 2, 0, 1], device='cpu')  # ✅ 自动转移到正确设备
```

## 📈 预期效果

### 修复前

```
AttributeError: 'list' object has no attribute 'unsqueeze'  ❌
```

### 修复后

```
Training: 100%|████████████| 2475/2475 [30:45<00:00,  1.34it/s]
Loss: 1.2345, Culture Loss: 0.0234, Accuracy: 0.7234  ✅
```

## 🔧 完整的数据流

### 训练数据准备

```python
# train_culturemoe_from_base_gen.py
batch = {
    'input_ids': tensor([...]),
    'attention_mask': tensor([...]),
    'labels': tensor([...]),
    'culture_labels': [0, 1, 2, 0, 1]  # ❌ list
}

outputs = model(
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    labels=batch['labels'],
    culture_labels=batch['culture_labels']  # ❌ 传递 list
)
```

### CultureMoE 处理

```python
# CultureMoE.py
def forward(self, ..., culture_labels=None, ...):
    ...
    if use_culture_loss and culture_labels is not None:
        # ✅ 自动转换 list 到 tensor
        culture_loss = self.compute_culture_loss(expert_weights, culture_labels)
```

### compute_culture_loss 处理

```python
def compute_culture_loss(self, expert_weights, culture_labels):
    # ✅ 检查类型并转换
    if not isinstance(culture_labels, torch.Tensor):
        culture_labels = torch.tensor(culture_labels, device=device, dtype=torch.long)

    # ✅ 现在可以安全地使用 tensor 方法
    culture_labels_expanded = culture_labels.unsqueeze(1)
    ...
```

## 📝 最佳实践

### 1. **始终检查类型**

```python
def process_input(data):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return data
```

### 2. **确保设备一致**

```python
def ensure_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return torch.tensor(data, device=device)
```

### 3. **指定数据类型**

```python
# ✅ 明确指定 dtype
culture_labels = torch.tensor(culture_labels, dtype=torch.long)
```

## 🔍 调试技巧

### 1. **打印类型信息**

```python
print(f"Type: {type(culture_labels)}")
print(f"Is tensor: {isinstance(culture_labels, torch.Tensor)}")
```

### 2. **打印形状信息**

```python
if isinstance(culture_labels, torch.Tensor):
    print(f"Shape: {culture_labels.shape}")
    print(f"Device: {culture_labels.device}")
    print(f"Dtype: {culture_labels.dtype}")
```

### 3. **添加断言**

```python
assert isinstance(culture_labels, torch.Tensor), \
    f"Expected tensor, got {type(culture_labels)}"
```

## 总结

✅ **主要修复**：
1. 添加类型检查（list vs tensor）
2. 自动转换 list 到 tensor
3. 确保设备一致性
4. 指定正确的数据类型

✅ **支持的输入**：
- Python list ✅
- NumPy array ✅
- PyTorch tensor ✅
- 不同设备的 tensor ✅

✅ **预期效果**：
- 训练代码可以正常运行
- 支持灵活的输入类型
- 自动处理设备转移

现在可以正常训练 CultureMoE 了！🎉

