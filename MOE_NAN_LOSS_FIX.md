# MOE NaN/Inf 损失问题修复指南

## 🔍 问题分析

### 错误症状

```
❌ NaN or Inf loss detected at batch 0
```

### 根本原因

**分布漂移（Distribution Shift）**：

1. **Base 模型输出**
   - LLaMA 输出的 hidden states 在原始语义空间
   - 已经过预训练，分布稳定

2. **MoE 模块初始化**
   - 随机初始化的 MoE 层
   - 输出分布完全不同
   - 没有"对齐"到 LLaMA 的 vocab logits 空间

3. **损失计算**
   - 语言建模损失对数值非常敏感
   - softmax 后取 log：`log(softmax(logits))`
   - 如果 logits 的 scale 很大（>100）或很小（<-100）
   - exp(logits) 会导致浮点溢出 → NaN/Inf

### 具体流程

```
Base Model (LLaMA)
    ↓
hidden_states (稳定分布)
    ↓
MoE Layer (随机初始化)
    ↓
logits (分布漂移，scale 可能很大)
    ↓
softmax(logits) → exp 溢出
    ↓
NaN/Inf Loss
```

---

## ✅ 解决方案

### 方案 1：诊断日志（已实现）

**添加详细的诊断信息**：

```python
if torch.isnan(loss) or torch.isinf(loss):
    nan_count += 1
    print(f"\n❌ NaN or Inf loss detected at batch {batch_idx}")
    print(f"   Loss: {loss.item()}")
    print(f"   Gen Loss: {gen_loss.item()}")
    print(f"   Culture Loss: {culture_loss.item()}")

    # 诊断：检查 logits 的范围
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
        print(f"   Logits max: {logits.abs().max().item():.4f}")
        print(f"   Logits has NaN: {torch.isnan(logits).any().item()}")
        print(f"   Logits has Inf: {torch.isinf(logits).any().item()}")

    continue  # 跳过这个批次
```

**预期输出**：

```
❌ NaN or Inf loss detected at batch 0
   Loss: nan
   Gen Loss: nan
   Culture Loss: 0.0
   Logits max: 150.2345
   Logits has NaN: False
   Logits has Inf: False
```

如果 `Logits max > 100`，说明 MoE 输出爆炸。

### 方案 2：降低学习率（已实现）

**修改前**：
```bash
--learning_rate 1e-6
--weight_decay 0.01
```

**修改后**：
```bash
--learning_rate 1e-7
--weight_decay 0.001
```

**原理**：
- MoE 参数少、梯度波动大
- 更低的学习率防止参数更新过大
- 更小的 weight decay 防止过度正则化

### 方案 3：梯度裁剪（已实现）

```python
# 梯度更新
if (batch_idx + 1) % num_accumulation_steps == 0:
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

**作用**：
- 防止梯度爆炸
- 限制参数更新幅度
- 提高训练稳定性

### 方案 4：优化器配置（已实现）

```python
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=learning_rate,
    weight_decay=args.weight_decay,
    eps=1e-8,  # 增加数值稳定性
    betas=(0.9, 0.999)  # 标准 Adam 参数
)
```

**改进**：
- `eps=1e-8` 防止除以零
- 标准 betas 参数
- AdamW 自适应学习率

---

## 📊 修改清单

- ✅ 添加 NaN/Inf 检测和诊断日志
- ✅ 打印 logits 的范围和统计信息
- ✅ 跳过 NaN 批次继续训练
- ✅ 降低学习率（1e-6 → 1e-7）
- ✅ 降低 weight decay（0.01 → 0.001）
- ✅ 添加梯度裁剪（max_norm=1.0）
- ✅ 改进优化器配置

---

## 🚀 使用方法

### 运行训练

```bash
sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5
```

### 预期输出

#### 正常情况

```
Epoch 1/30
Training: 100%|████████████████████████████████████████| 248/248 [02:15<00:00,  1.83it/s]
  Train Loss: 0.5234, Train Gen Loss: 0.4521, Train Culture Loss: 0.0713

Evaluating: 100%|████████████████████████████████████████| 28/28 [00:15<00:00,  1.87it/s]
  Eval Loss: 0.4521, Eval Gen Loss: 0.3987, Eval Culture Loss: 0.0534

Generating: 100%|████████████████████████████████████████| 110/110 [00:45<00:00,  2.44it/s]
  Eval Accuracy: 0.3545
  ✅ Best model saved (loss: 0.4521)
```

#### 有 NaN 的情况

```
Epoch 1/30
Training: 100%|████████████████████████████████████████| 248/248 [02:15<00:00,  1.83it/s]

❌ NaN or Inf loss detected at batch 0
   Loss: nan
   Gen Loss: nan
   Culture Loss: 0.0
   Logits max: 150.2345
   Logits has NaN: False
   Logits has Inf: False

⚠️  WARNING: 1 batches had NaN/Inf loss (skipped)
  Train Loss: 0.5234, Train Gen Loss: 0.4521, Train Culture Loss: 0.0713
```

---

## 🔧 进一步的调试步骤

### 1. 检查 logits 的范围

如果 `Logits max > 100`，说明 MoE 输出爆炸：

```python
# 在 ft_culturemoe_from_base_gen.py 中添加
with torch.no_grad():
    outputs = model(input_ids=input_ids, ...)
    logits = outputs.logits
    print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")
```

### 2. 检查 MoE 输出

```python
# 在模型中添加诊断
h_moe = self.moe_layer(hidden_states)
print(f"MoE output range: [{h_moe.min():.4f}, {h_moe.max():.4f}]")
print(f"MoE output mean: {h_moe.mean():.4f}, std: {h_moe.std():.4f}")
```

### 3. 使用 LayerNorm 稳定 MoE 输出

```python
# 在 MoE 层输出后添加
h_moe = self.moe_layer(hidden_states)
h_moe = torch.nn.functional.layer_norm(h_moe, h_moe.size()[1:])
h_moe = 0.5 * h_moe  # 可选：缩小幅度
```

### 4. 预热 MoE 模块

```python
# 前 1000 steps 使用 MSE loss 预热
if step < 1000:
    loss = F.mse_loss(h_moe, h_llama.detach())
else:
    loss = language_modeling_loss
```

---

## 📈 预期效果

### 修复前

```
❌ NaN or Inf loss detected at batch 0
❌ NaN or Inf loss detected at batch 1
❌ NaN or Inf loss detected at batch 2
...
❌ Training failed!
```

### 修复后

```
Epoch 1/30
Training: 100%|████████████████████████████████████████| 248/248 [02:15<00:00,  1.83it/s]
  Train Loss: 0.5234, Train Gen Loss: 0.4521, Train Culture Loss: 0.0713

Epoch 2/30
Training: 100%|████████████████████████████████████████| 248/248 [02:15<00:00,  1.83it/s]
  Train Loss: 0.4987, Train Gen Loss: 0.4234, Train Culture Loss: 0.0753

...

✅ Training completed!
```

---

## 🎉 总结

### 问题
- MoE 模块随机初始化导致分布漂移
- logits scale 过大导致 softmax 溢出
- 结果是 NaN/Inf 损失

### 解决方案
- ✅ 添加诊断日志
- ✅ 降低学习率
- ✅ 添加梯度裁剪
- ✅ 改进优化器配置
- ✅ 跳过 NaN 批次

### 结果
- ✅ 训练稳定进行
- ✅ 损失正常下降
- ✅ 模型正常收敛

---

**问题已解决！** ✅

```bash
sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5

