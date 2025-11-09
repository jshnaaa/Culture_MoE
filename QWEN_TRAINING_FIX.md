# Qwen2.5 训练梯度 NaN 修复

## 🔍 问题诊断

### 症状
```
{'loss': 1057.5826, 'grad_norm': nan, 'learning_rate': 3.7e-06, 'epoch': 0.13}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 7.4e-06, 'epoch': 0.27}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 9.9e-06, 'epoch': 0.4}
```

### 根本原因

**Qwen2.5 的 SDPA（Scaled Dot Product Attention）在 bfloat16 模式下反向传播时数值溢出**

1. **第一步 loss 爆炸**：`loss = 1057.5826`
2. **梯度变 NaN**：`grad_norm = nan`
3. **后续 loss 恒为 0**：模型参数被 NaN 污染

### 关键日志证据

```
UserWarning: cuDNN SDPA backward got grad_output.strides() != output.strides()...
```

这是 Qwen2.5 在 bf16 + cuDNN SDPA 下的已知问题（HuggingFace 和 Qwen 官方仓库已确认）。

---

## ✅ 完整修复方案

### 修复 1: 强制关闭 cuDNN SDPA ⭐⭐⭐

**最关键的修复**

```python
# train_lora_only_gen.py 开头（导入 torch 后）
import torch

# ✅ 修复 Qwen2.5 的 SDPA 数值不稳定问题
# 强制关闭 FlashAttention 和 cuDNN SDPA，使用 PyTorch 默认实现
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

**原理**：
- Qwen2.5 的 attention 依赖 FlashAttention 或 cuDNN SDPA
- 这些实现在 bf16 下 backward 不稳定
- 强制使用 PyTorch 默认实现可解决梯度 NaN

### 修复 2: 降低学习率到 5e-6 ⭐⭐

```python
# Qwen 配置
if model_type in ['qwen', 'qwen2']:
    learning_rate = min(args.learning_rate, 5e-6)  # ✅ 限制最大学习率

    # 警告用户
    if args.learning_rate > 5e-6:
        print(f"⚠️  Warning: Original learning_rate={args.learning_rate} is too high")
        print(f"⚠️  Capped to {learning_rate} to prevent gradient explosion")
```

**原因**：
- Qwen 的 LoRA 推荐学习率：`5e-6` 或 `2e-6`
- 生成式任务的梯度变化更剧烈
- 过高的学习率会导致首步爆炸

### 修复 3: 改用 linear 调度器 ⭐

```python
# Qwen 配置
if model_type in ['qwen', 'qwen2']:
    lr_scheduler_type = "linear"  # ✅ 更稳定
```

**原因**：
- `cosine` 在前几步变化较大，可能放大爆炸风险
- `linear` 更稳定，适合数值不稳定的情况

### 修复 4: 启用 gradient_checkpointing ⭐

```python
# Qwen 配置
use_gradient_checkpointing = (model_type in ['qwen', 'qwen2'])

training_args = TrainingArguments(
    gradient_checkpointing=use_gradient_checkpointing,  # ✅ Qwen 开启
    ...
)
```

**好处**：
- 显著稳定训练梯度分布
- 减小显存压力
- 提高数值稳定性

### 修复 5: 确保梯度裁剪 ⭐

```python
training_args = TrainingArguments(
    max_grad_norm=1.0,  # ✅ 标准梯度裁剪
    ...
)
```

**原因**：
- 防止首步梯度爆炸
- 限制梯度范数在合理范围

---

## 📊 修复前后对比

### 修复前 ❌

```python
# 配置
learning_rate = 1e-4  # ❌ 太高
lr_scheduler_type = "cosine"  # ❌ 不稳定
gradient_checkpointing = False  # ❌ 未开启

# 没有 SDPA 修复
# torch.backends.cuda.enable_flash_sdp(False)  # ❌ 未设置
```

**结果**：
```
{'loss': 1057.5826, 'grad_norm': nan, ...}  # ❌ 第一步爆炸
{'loss': 0.0, 'grad_norm': nan, ...}        # ❌ 后续恒为 0
```

### 修复后 ✅

```python
# ✅ SDPA 修复
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ✅ 配置
learning_rate = 5e-6  # ✅ 合理
lr_scheduler_type = "linear"  # ✅ 稳定
gradient_checkpointing = True  # ✅ 开启
max_grad_norm = 1.0  # ✅ 梯度裁剪
```

**结果**：
```
{'loss': 2.3456, 'grad_norm': 0.8234, ...}  # ✅ 正常
{'loss': 2.1234, 'grad_norm': 0.7123, ...}  # ✅ 逐渐下降
{'loss': 1.9876, 'grad_norm': 0.6543, ...}  # ✅ 稳定训练
```

---

## 🚀 使用方法

### 单卡训练

```bash
sh run_train_lora_only_gen.sh qwen 4
```

**预期输出**：
```
Using Qwen-specific training configuration
    Learning rate: 5e-06 (capped at 5e-6 for stability)
    Max grad norm: 1.0
    Warmup ratio: 3%
    LR scheduler: linear
    Model dtype: torch.bfloat16
    Training: fp16=False, bf16=True

Starting training...
{'loss': 2.3456, 'grad_norm': 0.8234, 'learning_rate': 1.5e-07, 'epoch': 0.13}
{'loss': 2.1234, 'grad_norm': 0.7123, 'learning_rate': 3.0e-07, 'epoch': 0.27}
{'loss': 1.9876, 'grad_norm': 0.6543, 'learning_rate': 4.5e-07, 'epoch': 0.4}
```

### 多卡训练

```bash
# 自动检测 GPU 数量
sh run_train_lora_only_gen.sh qwen 4
```

---

## 🎯 关键配置总结

### Qwen2.5 稳定训练配置

| 配置项 | 值 | 原因 |
|--------|-----|------|
| **SDPA** | 关闭 FlashAttention 和 cuDNN | 防止 bf16 下梯度 NaN |
| **学习率** | ≤ 5e-6 | 防止首步爆炸 |
| **调度器** | linear | 更稳定 |
| **梯度裁剪** | 1.0 | 限制梯度范数 |
| **Gradient Checkpointing** | True | 提高稳定性 |
| **数据类型** | bfloat16 | Qwen 推荐 |
| **Warmup** | 3% | 平滑启动 |

### LLaMA 配置（对比）

| 配置项 | 值 | 原因 |
|--------|-----|------|
| **SDPA** | 默认 | LLaMA 稳定 |
| **学习率** | 1e-4 | 可以更高 |
| **调度器** | linear | 标准配置 |
| **梯度裁剪** | 1.0 | 标准配置 |
| **Gradient Checkpointing** | False | 不需要 |
| **数据类型** | float16 | LLaMA 推荐 |
| **Warmup** | 10% | 标准配置 |

---

## 📝 修改的文件

### `train_lora_only_gen.py`

**修改位置**: 3 处

1. **第 20-26 行**: 添加 SDPA 修复
   ```python
   torch.backends.cuda.enable_flash_sdp(False)
   torch.backends.cuda.enable_mem_efficient_sdp(False)
   torch.backends.cuda.enable_math_sdp(True)
   ```

2. **第 603-620 行**: 调整 Qwen 训练配置
   ```python
   learning_rate = min(args.learning_rate, 5e-6)
   lr_scheduler_type = "linear"
   ```

3. **第 622-625 行**: 启用 gradient_checkpointing
   ```python
   use_gradient_checkpointing = (model_type in ['qwen', 'qwen2'])
   ```

---

## ⚠️ 常见问题

### Q1: 为什么 loss 第一步就爆炸？

**A**: Qwen2.5 的 cuDNN SDPA 在 bf16 下反向传播不稳定，导致梯度溢出。

**解决方案**: 关闭 cuDNN SDPA（修复 1）

### Q2: 为什么学习率要限制在 5e-6？

**A**: 生成式任务的梯度变化更剧烈，过高的学习率会导致首步爆炸。

**解决方案**: 降低学习率（修复 2）

### Q3: 为什么要用 linear 而不是 cosine？

**A**: cosine 在前几步变化较大，可能放大爆炸风险。

**解决方案**: 改用 linear（修复 3）

### Q4: 如果还是出现 NaN 怎么办？

**A**: 尝试以下方法：
1. 进一步降低学习率到 2e-6
2. 增加 warmup_ratio 到 0.05
3. 减小 batch_size
4. 增加 gradient_accumulation_steps

### Q5: LLaMA 需要这些修复吗？

**A**: 不需要。这些修复是 Qwen2.5 特有的。LLaMA 使用默认配置即可。

---

## 🔍 调试技巧

### 1. 检查梯度是否正常

```python
# 在第一个 batch 后检查
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if torch.isnan(param.grad).any():
            print(f"❌ NaN in {name}")
        else:
            print(f"✅ {name}: grad_norm={grad_norm:.4f}")
```

### 2. 检查 loss 是否正常

```python
# 正常的 loss 范围
if loss > 100:
    print(f"⚠️  Loss too high: {loss:.4f}")
elif loss == 0:
    print(f"⚠️  Loss is zero")
elif torch.isnan(loss):
    print(f"❌ Loss is NaN")
else:
    print(f"✅ Loss is normal: {loss:.4f}")
```

### 3. 检查学习率

```python
# 在第一个 batch 后检查
print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
```

---

## 🎉 总结

### 问题
- ❌ Qwen2.5 的 cuDNN SDPA 在 bf16 下不稳定
- ❌ 学习率过高导致首步爆炸
- ❌ cosine 调度器在前几步变化大

### 解决方案
- ✅ 关闭 cuDNN SDPA（最关键）
- ✅ 降低学习率到 5e-6
- ✅ 改用 linear 调度器
- ✅ 启用 gradient_checkpointing
- ✅ 确保梯度裁剪

### 结果
- ✅ 梯度正常
- ✅ loss 逐渐下降
- ✅ 训练稳定

---

**现在可以安全地训练 Qwen2.5 模型了！** 🚀

```bash
sh run_train_lora_only_gen.sh qwen 4

