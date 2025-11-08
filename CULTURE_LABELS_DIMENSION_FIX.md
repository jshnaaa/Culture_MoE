# culture_labels 维度错误修复

## 🔍 问题描述

训练正常，但在评估时报错：

```python
Traceback (most recent call last):
  File "train_culturemoe_from_base_gen.py", line 543, in main
    val_metrics = evaluate(...)
  File "train_culturemoe_from_base_gen.py", line 242, in evaluate
    outputs = model(...)
  File "CultureMoE.py", line 262, in compute_culture_loss
    culture_labels_1 = culture_labels.unsqueeze(1)  # [B, 1]
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

## 🔍 根本原因

### 问题分析

**`culture_labels` 的维度不一致**：

| 阶段 | 期望维度 | 实际维度 | 问题 |
|------|---------|---------|------|
| **训练** | `[B]` (1D) | `[B]` (1D) | ✅ 正常 |
| **评估** | `[B]` (1D) | `[]` (0D) 或 `[B, 1]` (2D) | ❌ 错误 |

**为什么会这样？**

1. **训练时**：`culture_labels` 是一个 list，转换为 1D tensor `[B]`
2. **评估时**：`culture_labels` 可能是：
   - 0D tensor（标量）：`tensor(5)` → 无法 `unsqueeze(1)`
   - 2D tensor：`tensor([[5]])` → 已经有 2 个维度，再 `unsqueeze(1)` 会出错

### 错误示例

```python
# 0D tensor（标量）
culture_labels = torch.tensor(5)  # shape: []
culture_labels.unsqueeze(1)  # ❌ IndexError: Dimension out of range

# 2D tensor
culture_labels = torch.tensor([[5]])  # shape: [1, 1]
culture_labels.unsqueeze(1)  # ❌ IndexError: Dimension out of range
```

## ✅ 解决方案

### 修复代码

```python
def compute_culture_loss(self, expert_weights, culture_labels):
    device = expert_weights.device
    dtype = expert_weights.dtype

    # ✅ 将 culture_labels 转换为 tensor（如果还不是）
    if not isinstance(culture_labels, torch.Tensor):
        culture_labels = torch.tensor(culture_labels, device=device, dtype=torch.long)
    else:
        # 确保在正确的设备上
        culture_labels = culture_labels.to(device=device, dtype=torch.long)

    # ✅ 确保 culture_labels 是 1D [B]
    if culture_labels.dim() > 1:
        culture_labels = culture_labels.squeeze()

    # ✅ 如果 culture_labels 是 0D（标量），转换为 1D
    if culture_labels.dim() == 0:
        culture_labels = culture_labels.unsqueeze(0)

    batch_size = expert_weights.size(0)

    # ✅ 验证 batch size 一致性
    if culture_labels.size(0) != batch_size:
        raise ValueError(f"culture_labels size ({culture_labels.size(0)}) does not match batch size ({batch_size})")

    # 计算样本对之间的文化相似度（相同文化为1，不同文化为0）
    culture_labels_1 = culture_labels.unsqueeze(1)  # [B, 1]
    culture_labels_2 = culture_labels.unsqueeze(0)  # [1, B]
    culture_similarity = (culture_labels_1 == culture_labels_2).float()  # [B, B]

    # ... 后续代码
```

## 📊 维度处理流程

### 处理不同维度的 culture_labels

```python
# 情况 1：0D tensor（标量）
culture_labels = torch.tensor(5)  # shape: []
if culture_labels.dim() == 0:
    culture_labels = culture_labels.unsqueeze(0)  # shape: [1]

# 情况 2：1D tensor（正常）
culture_labels = torch.tensor([1, 2, 3])  # shape: [3]
# 不需要处理

# 情况 3：2D tensor
culture_labels = torch.tensor([[1], [2], [3]])  # shape: [3, 1]
if culture_labels.dim() > 1:
    culture_labels = culture_labels.squeeze()  # shape: [3]

# 情况 4：3D tensor
culture_labels = torch.tensor([[[1]], [[2]], [[3]]])  # shape: [3, 1, 1]
if culture_labels.dim() > 1:
    culture_labels = culture_labels.squeeze()  # shape: [3]
```

## 🎯 完整的维度处理逻辑

```python
def normalize_culture_labels(culture_labels, device, batch_size):
    """
    标准化 culture_labels 为 1D tensor [B]

    Args:
        culture_labels: 可以是 list, 0D tensor, 1D tensor, 2D tensor, ...
        device: 目标设备
        batch_size: 期望的 batch size

    Returns:
        culture_labels: 1D tensor [B]
    """
    # Step 1: 转换为 tensor
    if not isinstance(culture_labels, torch.Tensor):
        culture_labels = torch.tensor(culture_labels, device=device, dtype=torch.long)
    else:
        culture_labels = culture_labels.to(device=device, dtype=torch.long)

    # Step 2: 处理多余的维度
    if culture_labels.dim() > 1:
        culture_labels = culture_labels.squeeze()

    # Step 3: 处理 0D tensor
    if culture_labels.dim() == 0:
        culture_labels = culture_labels.unsqueeze(0)

    # Step 4: 验证 batch size
    if culture_labels.size(0) != batch_size:
        raise ValueError(f"culture_labels size ({culture_labels.size(0)}) does not match batch size ({batch_size})")

    return culture_labels
```

## 📈 测试用例

```python
import torch

def test_normalize_culture_labels():
    batch_size = 4
    device = 'cuda'

    # 测试 1：0D tensor
    labels = torch.tensor(5)
    result = normalize_culture_labels(labels, device, 1)
    assert result.shape == (1,), f"Expected (1,), got {result.shape}"

    # 测试 2：1D tensor
    labels = torch.tensor([1, 2, 3, 4])
    result = normalize_culture_labels(labels, device, 4)
    assert result.shape == (4,), f"Expected (4,), got {result.shape}"

    # 测试 3：2D tensor
    labels = torch.tensor([[1], [2], [3], [4]])
    result = normalize_culture_labels(labels, device, 4)
    assert result.shape == (4,), f"Expected (4,), got {result.shape}"

    # 测试 4：list
    labels = [1, 2, 3, 4]
    result = normalize_culture_labels(labels, device, 4)
    assert result.shape == (4,), f"Expected (4,), got {result.shape}"

    print("✅ All tests passed!")

test_normalize_culture_labels()
```

## 🔧 调试技巧

### 1. **打印维度信息**

```python
def compute_culture_loss(self, expert_weights, culture_labels):
    print(f"culture_labels type: {type(culture_labels)}")
    if isinstance(culture_labels, torch.Tensor):
        print(f"culture_labels shape: {culture_labels.shape}")
        print(f"culture_labels dim: {culture_labels.dim()}")
    else:
        print(f"culture_labels value: {culture_labels}")
```

### 2. **添加断言**

```python
# 在 unsqueeze 之前添加断言
assert culture_labels.dim() == 1, f"Expected 1D tensor, got {culture_labels.dim()}D"
assert culture_labels.size(0) == batch_size, f"Expected size {batch_size}, got {culture_labels.size(0)}"
```

### 3. **使用 try-except**

```python
try:
    culture_labels_1 = culture_labels.unsqueeze(1)
except IndexError as e:
    print(f"Error: {e}")
    print(f"culture_labels shape: {culture_labels.shape}")
    print(f"culture_labels dim: {culture_labels.dim()}")
    raise
```

## 📊 常见的维度问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| **0D tensor** | 单个标量 | `unsqueeze(0)` |
| **2D tensor** | 多余的维度 | `squeeze()` |
| **3D+ tensor** | 多余的维度 | `squeeze()` |
| **batch size 不匹配** | 数据处理错误 | 验证并报错 |

## 总结

✅ **主要修复**：
1. 添加 0D tensor 处理（`unsqueeze(0)`）
2. 添加 batch size 验证
3. 确保 `culture_labels` 始终是 1D tensor `[B]`

✅ **处理流程**：
```
输入 → 转换为 tensor → 处理多余维度 → 处理 0D → 验证 batch size → 输出 1D [B]
```

✅ **预期效果**：
- 训练正常 ✅
- 评估正常 ✅
- 不再出现维度错误 ✅

现在应该可以正常训练和评估了！🎉

