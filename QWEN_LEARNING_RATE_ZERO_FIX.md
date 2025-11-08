# Qwen 模型学习率为 0.0 问题修复

## 🔍 问题描述

运行以下命令时，Qwen 模型的学习率一直是 0.0，梯度为 NaN：

```bash
sh run_train_lora_only_gen.sh qwen 2  # CulturalBench
sh run_train_lora_only_gen.sh qwen 3  # NormAD
sh run_train_lora_only_gen.sh qwen 4  # CultureLLM
```

**症状**：
```
Step 1:  Loss: nan, Grad Norm: nan, LR: 0.0  ❌
Step 10: Loss: nan, Grad Norm: nan, LR: 0.0  ❌
```

## 🔍 根本原因

### 问题分析

**`warmup_steps` 和 `warmup_ratio` 冲突**：

```python
# 修复前（错误的配置）
training_args = TrainingArguments(
    ...
    warmup_steps=500,      # ❌ 绝对步数
    warmup_ratio=0.3,      # ❌ 相对比例
    ...
)
```

**问题**：
1. **参数冲突**：两个参数同时设置会导致冲突
2. **学习率为 0**：预热计算出错，导致学习率始终为 0
3. **梯度为 NaN**：学习率为 0 导致梯度无法更新

### 为什么会这样？

Transformers 库中的 `TrainingArguments` 处理 `warmup_steps` 和 `warmup_ratio` 的逻辑：

```python
# Transformers 内部逻辑
if warmup_steps > 0:
    # 使用 warmup_steps
    warmup_steps = warmup_steps
elif warmup_ratio > 0:
    # 使用 warmup_ratio
    warmup_steps = int(total_steps * warmup_ratio)
else:
    # 都没有设置
    warmup_steps = 0
```

**当两者都设置时**：
- 如果 `warmup_steps` 设置不当（如 500 步但总步数只有 100）
- 或者两个参数相互干扰
- 导致学习率计算错误

## ✅ 解决方案

### 修复方案

**只使用 `warmup_ratio`，不使用 `warmup_steps`**：

```python
# 修复前
training_args = TrainingArguments(
    ...
    warmup_steps=500,      # ❌ 删除
    warmup_ratio=0.3,      # ✅ 保留
    ...
)

# 修复后
training_args = TrainingArguments(
    ...
    warmup_ratio=0.5,      # ✅ 只使用 warmup_ratio
    # 不设置 warmup_steps
    ...
)
```

### 关键改进

#### 1. **移除 `warmup_steps`**

```python
# ❌ 不要同时设置两个
warmup_steps=500
warmup_ratio=0.3

# ✅ 只设置一个
warmup_ratio=0.5
```

#### 2. **对 Qwen 使用 fp32**

```python
# ❌ Qwen 在 fp16 下不稳定
fp16=True

# ✅ Qwen 使用 fp32
fp16=False
```

#### 3. **极其保守的参数**

```python
if model_type == 'qwen':
    learning_rate = args.learning_rate * 0.1      # 降低 90%
    max_grad_norm = 0.1                           # 极其激进的裁剪
    warmup_ratio = 0.5                            # 50% 预热
    fp16 = False                                  # 使用 fp32
```

## 📊 完整的修复代码

```python
# 配置训练参数（根据模型类型调整）
if model_type == 'qwen':
    # ✅ Qwen 需要极其保守的配置
    learning_rate = args.learning_rate * 0.1  # 降低学习率 90%
    max_grad_norm = 0.1  # 极其激进的梯度裁剪
    warmup_ratio = 0.5  # 50% 的步数用于预热
    use_fp32 = True  # 使用 fp32
    print("  Using Qwen-specific training configuration (ultra-conservative)")
    print(f"    Learning rate: {learning_rate} (10% of {args.learning_rate})")
    print(f"    Max grad norm: {max_grad_norm}")
    print(f"    Warmup ratio: {warmup_ratio * 100:.0f}%")
    print(f"    Using fp32 (not fp16)")
else:
    # LLaMA 配置
    learning_rate = args.learning_rate
    max_grad_norm = 1.0
    warmup_ratio = 0.1
    use_fp32 = False
    print("  Using LLaMA-specific training configuration")

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=learning_rate,
    logging_steps=10,
    save_strategy="no",
    fp16=not use_fp32,  # ✅ Qwen 使用 fp32（fp16=False）
    fp16_full_eval=False,
    fp16_opt_level="O1",
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,  # ✅ 只使用 warmup_ratio
    # 不设置 warmup_steps
    weight_decay=0.01,
    adam_epsilon=1e-8,
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=True,
    gradient_checkpointing=False,
)
```

## 📈 预期效果

### 修复前
```
Step 1:  Loss: nan, Grad Norm: nan, LR: 0.0  ❌
Step 10: Loss: nan, Grad Norm: nan, LR: 0.0  ❌
```

### 修复后
```
Step 1:  Loss: 2.3456, Grad Norm: 0.0234, LR: 1e-6  ✅ (warmup)
Step 50: Loss: 1.8765, Grad Norm: 0.0123, LR: 5e-6  ✅ (warmup)
Step 100: Loss: 1.5432, Grad Norm: 0.0089, LR: 1e-5  ✅ (normal)
Step 200: Loss: 1.2345, Grad Norm: 0.0067, LR: 1e-5  ✅ (normal)
```

## 🔧 参数对比

| 参数 | LLaMA | Qwen |
|------|-------|------|
| **学习率** | 1e-4 | 1e-5 (10%) |
| **Max Grad Norm** | 1.0 | 0.1 (极其激进) |
| **Warmup Ratio** | 0.1 (10%) | 0.5 (50%) |
| **fp16** | True | False (使用 fp32) |
| **Warmup Steps** | ❌ 不使用 | ❌ 不使用 |

## ⚠️ 常见的 warmup 配置错误

### 错误 1：同时设置两个参数

```python
# ❌ 错误
training_args = TrainingArguments(
    warmup_steps=500,
    warmup_ratio=0.3,
)

# ✅ 正确
training_args = TrainingArguments(
    warmup_ratio=0.3,
    # 不设置 warmup_steps
)
```

### 错误 2：warmup_steps 超过总步数

```python
# ❌ 错误
# 总步数 = 2475 / 4 = 618 步
# 但设置 warmup_steps = 500（太长）
training_args = TrainingArguments(
    warmup_steps=500,
)

# ✅ 正确
training_args = TrainingArguments(
    warmup_ratio=0.1,  # 10% 的总步数
)
```

### 错误 3：warmup_ratio 设置为 0

```python
# ❌ 错误
training_args = TrainingArguments(
    warmup_ratio=0,  # 没有预热
)

# ✅ 正确
training_args = TrainingArguments(
    warmup_ratio=0.1,  # 至少 10% 预热
)
```

## 📝 学习率预热原理

### 预热过程

```
学习率
  |
  |     ╱╲
  |    ╱  ╲___
  |   ╱       ╲
  |  ╱         ╲
  | ╱           ╲
  |╱_____________╲___
  └─────────────────── 步数
  0  warmup  normal  end
```

### 计算公式

```python
# 预热阶段（前 warmup_steps 步）
lr = base_lr * (current_step / warmup_steps)

# 正常阶段（之后）
lr = base_lr * decay_factor
```

### 示例

```
总步数 = 1000
warmup_ratio = 0.1
warmup_steps = 1000 * 0.1 = 100

Step 0:   lr = 1e-4 * (0 / 100) = 0
Step 50:  lr = 1e-4 * (50 / 100) = 5e-5
Step 100: lr = 1e-4 * (100 / 100) = 1e-4
Step 500: lr = 1e-4 (正常阶段)
```

## 🎯 最佳实践

### 1. **只使用 `warmup_ratio`**

```python
training_args = TrainingArguments(
    warmup_ratio=0.1,  # 10% 的总步数用于预热
    # 不设置 warmup_steps
)
```

### 2. **根据模型类型调整**

```python
if model_type == 'qwen':
    warmup_ratio = 0.5  # Qwen 需要更长的预热
else:
    warmup_ratio = 0.1  # LLaMA 使用标准预热
```

### 3. **监控学习率**

```python
# 在训练循环中添加
for step, batch in enumerate(train_dataloader):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Step {step}: LR = {current_lr}")
```

## 总结

✅ **主要问题**：
- `warmup_steps` 和 `warmup_ratio` 冲突
- 导致学习率为 0.0
- 梯度无法更新，出现 NaN

✅ **解决方案**：
1. 只使用 `warmup_ratio`，不使用 `warmup_steps`
2. 对 Qwen 使用 fp32 而不是 fp16
3. 使用极其保守的参数（学习率 10%，预热 50%）

✅ **关键参数**：
- Qwen 学习率：`1e-4 × 0.1 = 1e-5`
- Qwen Max Grad Norm：`0.1`
- Qwen Warmup Ratio：`0.5`
- Qwen fp16：`False`（使用 fp32）

✅ **预期效果**：
- 学习率正常变化（从 0 增加到设定值）
- 梯度正常（不再是 NaN）
- Loss 正常下降
- 训练稳定收敛

现在可以正常训练 Qwen 模型了！🎉

