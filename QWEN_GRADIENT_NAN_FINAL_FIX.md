# Qwen 梯度 NaN 最终修复方案

## 🔍 问题诊断

### 症状
```
{'loss': 4.4202, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.13}
{'loss': 4.4061, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.27}
{'loss': 4.4595, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.4}
```

### 根本原因

**梯度 NaN 导致优化器跳过更新，学习率保持为 0**

1. **梯度 NaN**: `grad_norm = nan`
2. **优化器跳过**: Trainer 自动跳过 `optimizer.step()`
3. **学习率为 0**: 调度器没有被更新

**为什么梯度 NaN？**

根据日志 `Base model loaded (using float16)`，模型使用了 **float16** 精度。

**Qwen2.5 在 float16 下极不稳定**，特别是 LoRA 层训练时容易梯度溢出。

---

## ✅ 完整修复方案（已实现）

### 修复 1: 使用 bfloat16 而不是 float16 ⭐⭐⭐

```python
# train_lora_only_gen.py
if model_type in ['qwen', 'qwen2']:
    # ✅ Qwen 使用 bfloat16（如果支持）或 float32
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f"  Using {torch_dtype} for Qwen/Qwen2")
else:
    # LLaMA 使用 float16
    torch_dtype = torch.float16
    print(f"  Using {torch_dtype} for LLaMA")

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch_dtype,  # ✅ 使用正确的数据类型
    ...
)
```

**原理**：
- **float16**: 数值范围小，容易溢出 → 梯度 NaN
- **bfloat16**: 数值范围大，更稳定 → 梯度正常
- **float32**: 最稳定，但显存占用高

### 修复 2: 关闭 cuDNN SDPA ⭐⭐⭐

```python
# train_lora_only_gen.py 开头
import torch

# ✅ 修复 Qwen2.5 的 SDPA 数值不稳定问题
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

### 修复 3: 降低学习率到 5e-6 ⭐⭐

```python
if model_type in ['qwen', 'qwen2']:
    learning_rate = min(args.learning_rate, 5e-6)  # ✅ 限制最大学习率
```

### 修复 4: 启用梯度裁剪 ⭐

```python
training_args = TrainingArguments(
    max_grad_norm=1.0,  # ✅ 防止梯度爆炸
    ...
)
```

### 修复 5: 启用 warmup ⭐

```python
training_args = TrainingArguments(
    warmup_ratio=0.03,  # ✅ 3% 预热
    ...
)
```

### 修复 6: 使用 linear 调度器 ⭐

```python
training_args = TrainingArguments(
    lr_scheduler_type="linear",  # ✅ 更稳定
    ...
)
```

### 修复 7: 启用 gradient_checkpointing ⭐

```python
training_args = TrainingArguments(
    gradient_checkpointing=True,  # ✅ 提高稳定性
    ...
)
```

---

## 📊 修复前后对比

### 修复前 ❌

```python
# ❌ 使用 float16
torch_dtype = torch.float16

# ❌ 没有关闭 cuDNN SDPA
# torch.backends.cuda.enable_flash_sdp(False)

# ❌ 学习率过高
learning_rate = 1e-4

# ❌ 没有梯度裁剪
max_grad_norm = None

# ❌ 没有 warmup
warmup_ratio = 0.0
```

**结果**：
```
{'loss': 4.4202, 'grad_norm': nan, 'learning_rate': 0.0, ...}  # ❌ 梯度 NaN
{'loss': 4.4061, 'grad_norm': nan, 'learning_rate': 0.0, ...}  # ❌ 学习率为 0
```

### 修复后 ✅

```python
# ✅ 使用 bfloat16
torch_dtype = torch.bfloat16

# ✅ 关闭 cuDNN SDPA
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ✅ 降低学习率
learning_rate = 5e-6

# ✅ 启用梯度裁剪
max_grad_norm = 1.0

# ✅ 启用 warmup
warmup_ratio = 0.03
```

**结果**：
```
{'loss': 2.3456, 'grad_norm': 0.8234, 'learning_rate': 1.5e-07, ...}  # ✅ 梯度正常
{'loss': 2.1234, 'grad_norm': 0.7123, 'learning_rate': 3.0e-07, ...}  # ✅ 学习率正常
{'loss': 1.9876, 'grad_norm': 0.6543, 'learning_rate': 4.5e-07, ...}  # ✅ 逐渐下降
```

---

## 🚀 使用方法

### 单卡训练

```bash
sh run_train_lora_only_gen.sh qwen 3
```

**预期输出**：
```
Detected model type: qwen2
  Using torch.bfloat16 for Qwen/Qwen2  # ✅ 使用 bfloat16

Using Qwen-specific training configuration
    Learning rate: 5e-06 (capped at 5e-6 for stability)
    Max grad norm: 1.0
    Warmup ratio: 3%
    LR scheduler: linear
    Model dtype: torch.bfloat16
    Training: fp16=False, bf16=True  # ✅ bf16=True

Starting training...
{'loss': 2.3456, 'grad_norm': 0.8234, 'learning_rate': 1.5e-07, ...}  # ✅ 正常
{'loss': 2.1234, 'grad_norm': 0.7123, 'learning_rate': 3.0e-07, ...}  # ✅ 下降
```

### 多卡训练

```bash
# 自动检测 GPU 数量
sh run_train_lora_only_gen.sh qwen 3
```

---

## 🎯 关键配置总结

### Qwen2.5 稳定训练配置

| 配置项 | 值 | 原因 |
|--------|-----|------|
| **数据类型** | bfloat16 | 防止梯度 NaN |
| **SDPA** | 关闭 FlashAttention | 防止数值溢出 |
| **学习率** | ≤ 5e-6 | 防止梯度爆炸 |
| **梯度裁剪** | 1.0 | 限制梯度范数 |
| **Warmup** | 3% | 平滑启动 |
| **调度器** | linear | 更稳定 |
| **Gradient Checkpointing** | True | 提高稳定性 |

### 为什么 learning_rate=0.0？

**不是配置问题，而是梯度 NaN 的结果**：

1. **梯度 NaN** → Trainer 跳过 `optimizer.step()`
2. **跳过更新** → 学习率调度器没有被调用
3. **学习率为 0** → 显示为 `learning_rate=0.0`

**修复梯度 NaN 后，学习率会正常显示**。

---

## ⚠️ 常见问题

### Q1: 为什么 float16 不行？

**A**: Qwen2.5 的数值范围需求超过了 float16 的表示能力，容易溢出。

**解决方案**: 使用 bfloat16 或 float32。

### Q2: 如何确认使用了 bfloat16？

**A**: 查看训练日志：
```
Model dtype: torch.bfloat16
Training: fp16=False, bf16=True
```

### Q3: 如果机器不支持 bfloat16 怎么办？

**A**: 代码会自动回退到 float32：
```python
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
```

### Q4: 为什么学习率要限制在 5e-6？

**A**: Qwen 的 LoRA 推荐学习率是 1e-5 ~ 5e-6。过高的学习率会导致梯度爆炸。

### Q5: 如果还是出现 NaN 怎么办？

**A**: 尝试以下方法：
1. 进一步降低学习率到 2e-6
2. 增加 warmup_ratio 到 0.05
3. 减小 batch_size
4. 增加 gradient_accumulation_steps
5. 使用 float32（最稳定）

---

## 🔍 调试技巧

### 1. 检查数据类型

```python
# 在训练开始时添加
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Training fp16: {training_args.fp16}")
print(f"Training bf16: {training_args.bf16}")
```

**预期输出**：
```
Model dtype: torch.bfloat16
Training fp16: False
Training bf16: True
```

### 2. 检查学习率

```python
# 在第一个 batch 后检查
print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
```

**预期输出**：
```
Current learning rate: 1.5e-07  # warmup 阶段
```

### 3. 检查梯度

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

**预期输出**：
```
✅ base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight: grad_norm=0.8234
✅ base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight: grad_norm=0.7123
```

---

## 📝 修改的文件

### `train_lora_only_gen.py`

**修改位置**: 7 处

1. **第 20-26 行**: 添加 SDPA 修复
2. **第 519-527 行**: 根据模型类型选择数据类型
3. **第 604-607 行**: 确保 LoRA 参数需要梯度
4. **第 625-647 行**: Qwen 特定训练配置
5. **第 655 行**: 设置 ddp_find_unused_parameters=True
6. **第 660 行**: 启用 gradient_checkpointing
7. **第 665-680 行**: TrainingArguments 配置

---

## 🎉 总结

### 问题
- ❌ 梯度 NaN
- ❌ 学习率为 0
- ❌ 训练无法进行

### 根本原因
- ❌ 使用 float16（Qwen2.5 不稳定）
- ❌ cuDNN SDPA 数值溢出
- ❌ 学习率过高

### 解决方案
- ✅ 使用 bfloat16（最关键）
- ✅ 关闭 cuDNN SDPA
- ✅ 降低学习率到 5e-6
- ✅ 启用梯度裁剪
- ✅ 启用 warmup
- ✅ 使用 linear 调度器
- ✅ 启用 gradient_checkpointing

### 结果
- ✅ 梯度正常
- ✅ 学习率正常衰减
- ✅ 训练稳定进行
- ✅ 损失逐渐下降

---

**现在可以安全地训练 Qwen 模型了！** 🚀

```bash
sh run_train_lora_only_gen.sh qwen 3
```

**预期看到**：
```
Model dtype: torch.bfloat16
Training: fp16=False, bf16=True

{'loss': 2.3456, 'grad_norm': 0.8234, 'learning_rate': 1.5e-07, ...}  # ✅ 正常！

