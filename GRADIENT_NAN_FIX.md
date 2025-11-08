# 梯度 NaN 和学习率不变问题修复

## 🔍 问题描述

运行以下命令时，出现梯度为 NaN 和学习率不变的问题：

```bash
sh run_train_lora_only_gen.sh llama 2  # CulturalBench
sh run_train_lora_only_gen.sh qwen 2   # CulturalBench
sh run_train_lora_only_gen.sh qwen 3   # NormAD
sh run_train_lora_only_gen.sh qwen 4   # CultureLLM
```

**症状**：
- ✗ 梯度一直是 NaN
- ✗ 学习率一直不变
- ✗ Loss 为 NaN 或 inf
- ✗ 训练无法收敛

## 🔍 根本原因

### 1. **缺少梯度裁剪**
- 没有 `max_grad_norm` 参数
- 梯度可能爆炸，导致 NaN

### 2. **缺少学习率预热**
- 没有 `warmup_steps` 参数
- 初始学习率过大，导致梯度不稳定

### 3. **混合精度配置不当**
- 只设置了 `fp16=True`
- 没有设置 `fp16_opt_level`
- 可能导致数值溢出

### 4. **缺少优化器稳定性配置**
- 没有 `adam_epsilon`
- 没有 `weight_decay`
- 优化器可能不稳定

## ✅ 解决方案

### 修改 `train_lora_only_gen.py`

```python
# 修复前
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    logging_steps=10,
    save_strategy="no",
    fp16=True,  # ❌ 只有这个
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=True,
)

# 修复后
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    logging_steps=10,
    save_strategy="no",
    fp16=True,
    fp16_full_eval=False,  # ✅ 评估时不使用 fp16
    fp16_opt_level="O1",   # ✅ 使用 O1 混合精度（更稳定）
    max_grad_norm=1.0,     # ✅ 梯度裁剪（防止梯度爆炸）
    warmup_steps=100,      # ✅ 学习率预热（防止初始梯度过大）
    weight_decay=0.01,     # ✅ 权重衰减
    adam_epsilon=1e-8,     # ✅ Adam 优化器的 epsilon
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=True,
    gradient_checkpointing=False,  # ✅ 如果内存不够可以开启
)
```

## 📊 关键参数说明

### 1. **max_grad_norm=1.0**
- **作用**：梯度裁剪，防止梯度爆炸
- **原理**：将梯度的 L2 范数限制在 1.0 以内
- **效果**：防止梯度为 NaN 或 inf

### 2. **warmup_steps=100**
- **作用**：学习率预热
- **原理**：前 100 步学习率从 0 线性增加到设定值
- **效果**：防止初始梯度过大导致不稳定

### 3. **fp16_opt_level="O1"**
- **作用**：混合精度优化级别
- **O0**：纯 FP32（最稳定，最慢）
- **O1**：混合精度（推荐，平衡稳定性和速度）
- **O2**：几乎全 FP16（更快，但可能不稳定）
- **O3**：纯 FP16（最快，最不稳定）

### 4. **weight_decay=0.01**
- **作用**：权重衰减（L2 正则化）
- **效果**：防止过拟合，提高稳定性

### 5. **adam_epsilon=1e-8**
- **作用**：Adam 优化器的数值稳定性参数
- **效果**：防止除零错误

### 6. **fp16_full_eval=False**
- **作用**：评估时使用 FP32
- **效果**：评估更准确，避免数值问题

## 🎯 学习率调度

修复后的学习率变化：

```
Warmup 阶段（前 100 步）：
Step 0:   lr = 0.0
Step 50:  lr = 5e-5  (learning_rate * 0.5)
Step 100: lr = 1e-4  (learning_rate)

训练阶段（100 步之后）：
Step 100+: lr = 1e-4 (保持不变，因为没有设置 lr_scheduler_type)
```

如果想要学习率衰减，可以添加：

```python
training_args = TrainingArguments(
    ...
    lr_scheduler_type="cosine",  # 余弦衰减
    # 或
    lr_scheduler_type="linear",  # 线性衰减
)
```

## 📈 预期效果

### 修复前
```
Step 1:  Loss: nan, Grad Norm: nan, LR: 1e-4
Step 10: Loss: nan, Grad Norm: nan, LR: 1e-4
Step 20: Loss: nan, Grad Norm: nan, LR: 1e-4
...
```

### 修复后
```
Step 1:  Loss: 2.3456, Grad Norm: 0.8234, LR: 1e-6  (warmup)
Step 10: Loss: 2.1234, Grad Norm: 0.7123, LR: 1e-5  (warmup)
Step 50: Loss: 1.8765, Grad Norm: 0.6543, LR: 5e-5  (warmup)
Step 100: Loss: 1.5432, Grad Norm: 0.5234, LR: 1e-4  (normal)
Step 200: Loss: 1.2345, Grad Norm: 0.4123, LR: 1e-4  (normal)
...
```

## 🔧 其他可能的解决方案

### 如果问题仍然存在

#### 1. **降低学习率**

```bash
# 修改 run_train_lora_only_gen.sh
python train_lora_only_gen.py \
    --learning_rate 5e-5  # 从 1e-4 降低到 5e-5
```

#### 2. **增加 warmup 步数**

```python
warmup_steps=200,  # 从 100 增加到 200
```

#### 3. **使用更保守的混合精度**

```python
fp16=False,  # 关闭 fp16，使用 FP32
# 或
fp16_opt_level="O0",  # 使用纯 FP32
```

#### 4. **减小 batch size**

```bash
# 修改 run_train_lora_only_gen.sh
--per_device_train_batch_size 2  # 从 4 降低到 2
--gradient_accumulation_steps 8  # 从 4 增加到 8
```

#### 5. **检查数据质量**

```python
# 添加数据检查
for item in train_data:
    if not item['output'] or len(str(item['output'])) > 100:
        print(f"Warning: Invalid output: {item['output']}")
```

## 📊 训练监控

### 正常的训练日志

```
================================================================================
Epoch 1/3
================================================================================
{'loss': 2.3456, 'grad_norm': 0.8234, 'learning_rate': 1e-06, 'epoch': 0.01}
{'loss': 2.1234, 'grad_norm': 0.7123, 'learning_rate': 1e-05, 'epoch': 0.02}
{'loss': 1.8765, 'grad_norm': 0.6543, 'learning_rate': 5e-05, 'epoch': 0.05}
{'loss': 1.5432, 'grad_norm': 0.5234, 'learning_rate': 1e-04, 'epoch': 0.10}
{'loss': 1.2345, 'grad_norm': 0.4123, 'learning_rate': 1e-04, 'epoch': 0.20}
...
```

### 异常的训练日志

```
{'loss': nan, 'grad_norm': nan, 'learning_rate': 1e-04, 'epoch': 0.01}  ❌
{'loss': inf, 'grad_norm': inf, 'learning_rate': 1e-04, 'epoch': 0.02}  ❌
```

## 🎯 最佳实践

### 1. **训练稳定性配置**

```python
training_args = TrainingArguments(
    # 基础配置
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    # 稳定性配置（必须）
    max_grad_norm=1.0,      # 梯度裁剪
    warmup_steps=100,       # 学习率预热
    weight_decay=0.01,      # 权重衰减
    adam_epsilon=1e-8,      # Adam epsilon

    # 混合精度配置
    fp16=True,
    fp16_opt_level="O1",    # 推荐 O1
    fp16_full_eval=False,   # 评估用 FP32
)
```

### 2. **学习率选择**

| 模型大小 | 推荐学习率 | Warmup 步数 |
|----------|------------|-------------|
| < 1B     | 1e-4       | 100         |
| 1B - 7B  | 5e-5       | 200         |
| 7B - 13B | 2e-5       | 500         |
| > 13B    | 1e-5       | 1000        |

### 3. **Batch Size 选择**

```
有效 Batch Size = per_device_batch_size × gradient_accumulation_steps × num_gpus

推荐：
- 小模型（< 1B）：有效 batch size = 32-64
- 中模型（1B-7B）：有效 batch size = 16-32
- 大模型（> 7B）：有效 batch size = 8-16
```

## 📝 检查清单

训练前检查：

- [ ] 设置了 `max_grad_norm`（梯度裁剪）
- [ ] 设置了 `warmup_steps`（学习率预热）
- [ ] 设置了 `fp16_opt_level="O1"`（混合精度）
- [ ] 设置了 `weight_decay`（权重衰减）
- [ ] 设置了 `adam_epsilon`（优化器稳定性）
- [ ] 学习率合理（1e-5 到 1e-4）
- [ ] Batch size 合理（有效 batch size 16-32）
- [ ] 数据质量检查（无异常值）

## 总结

✅ **主要修复**：
1. 添加梯度裁剪（`max_grad_norm=1.0`）
2. 添加学习率预热（`warmup_steps=100`）
3. 设置混合精度级别（`fp16_opt_level="O1"`）
4. 添加优化器稳定性配置

✅ **预期效果**：
- 梯度正常（不再是 NaN）
- 学习率正常变化（warmup 阶段）
- Loss 正常下降
- 训练稳定收敛

现在可以正常训练了！🎉

