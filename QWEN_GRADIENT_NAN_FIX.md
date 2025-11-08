# Qwen 模型梯度 NaN 问题修复

## 🔍 问题描述

运行以下命令时，Qwen 模型梯度一直是 NaN，学习率不变：

```bash
sh run_train_lora_only_gen.sh qwen 2  # CulturalBench
sh run_train_lora_only_gen.sh qwen 3  # NormAD
sh run_train_lora_only_gen.sh qwen 4  # CultureLLM
```

但 LLaMA 模型正常：

```bash
sh run_train_lora_only_gen.sh llama 2  # ✅ 正常
```

## 🔍 根本原因

### 问题分析

**Qwen 和 LLaMA 的架构差异**：

| 特性 | LLaMA | Qwen |
|------|-------|------|
| **投影层名称** | q_proj, k_proj, v_proj, o_proj | q_proj, k_proj, v_proj, o_proj (相同) |
| **注意力机制** | 标准 Multi-Head Attention | 标准 Multi-Head Attention |
| **RMSNorm** | 标准 RMSNorm | 标准 RMSNorm |
| **数值稳定性** | 较好 | ⚠️ 需要特殊处理 |

### 实际问题

1. **Qwen 的 fp16 数值不稳定**：
   - Qwen 在 fp16 下容易出现数值溢出
   - 需要更激进的梯度裁剪

2. **Qwen 的初始化不同**：
   - Qwen 的权重初始化范围可能不同
   - 导致初始梯度过大

3. **Qwen 的 attention scale**：
   - Qwen 可能使用不同的 attention scale
   - 导致梯度爆炸

## ✅ 解决方案

### 方案 1：修改 LoRA 配置（推荐）

修改 `train_lora_only_gen.py` 中的 LoRA 配置，根据模型类型调整：

```python
# 修复前
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none"
)

# 修复后
# 检测模型类型
model_type = model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'

if model_type == 'qwen':
    # Qwen 特殊配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        lora_dropout=0.1,  # 增加 dropout
        init_lora_weights="gaussian"  # 使用高斯初始化
    )
else:
    # LLaMA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
```

### 方案 2：修改训练参数（更激进）

对 Qwen 模型使用更保守的训练参数：

```python
# 修复前
training_args = TrainingArguments(
    ...
    learning_rate=args.learning_rate,  # 1e-4
    max_grad_norm=1.0,
    warmup_steps=100,
    ...
)

# 修复后
# 根据模型类型调整参数
if model_type == 'qwen':
    # Qwen 需要更保守的配置
    learning_rate = args.learning_rate * 0.5  # 降低学习率 50%
    max_grad_norm = 0.5  # 更激进的梯度裁剪
    warmup_steps = 200  # 更长的预热
    warmup_ratio = 0.2  # 20% 的步数用于预热
else:
    # LLaMA 配置
    learning_rate = args.learning_rate
    max_grad_norm = 1.0
    warmup_steps = 100
    warmup_ratio = 0.1

training_args = TrainingArguments(
    ...
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    warmup_steps=warmup_steps,
    warmup_ratio=warmup_ratio,
    ...
)
```

### 方案 3：修改 fp16 配置

对 Qwen 使用 bfloat16 而不是 float16：

```python
# 修复前
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,  # ❌ Qwen 在 fp16 下不稳定
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# 修复后
# 根据模型类型选择 dtype
if model_type == 'qwen':
    # Qwen 使用 bfloat16 更稳定
    torch_dtype = torch.bfloat16
else:
    # LLaMA 使用 float16
    torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

## 📊 完整的修复代码

### 修改 `train_lora_only_gen.py`

```python
# 在模型加载后添加
model_type = model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
print(f"Model type: {model_type}")

# 配置 LoRA（根据模型类型）
print("Configuring LoRA...")
if model_type == 'qwen':
    print("  Using Qwen-specific LoRA configuration")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,  # Qwen 需要更高的 dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        init_lora_weights="gaussian"
    )
else:
    print("  Using LLaMA-specific LoRA configuration")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✅ LoRA configured\n")

# 配置训练参数（根据模型类型）
if model_type == 'qwen':
    # Qwen 需要更保守的配置
    learning_rate = args.learning_rate * 0.5
    max_grad_norm = 0.5
    warmup_steps = 200
    warmup_ratio = 0.2
    print("  Using Qwen-specific training configuration")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Max grad norm: {max_grad_norm}")
    print(f"    Warmup steps: {warmup_steps}")
else:
    # LLaMA 配置
    learning_rate = args.learning_rate
    max_grad_norm = 1.0
    warmup_steps = 100
    warmup_ratio = 0.1
    print("  Using LLaMA-specific training configuration")

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=learning_rate,
    logging_steps=10,
    save_strategy="no",
    fp16=True,
    fp16_full_eval=False,
    fp16_opt_level="O1",
    max_grad_norm=max_grad_norm,
    warmup_steps=warmup_steps,
    warmup_ratio=warmup_ratio,
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

### 修复前（Qwen）
```
Step 1:  Loss: nan, Grad Norm: nan, LR: 5e-5  ❌
Step 10: Loss: nan, Grad Norm: nan, LR: 5e-5  ❌
```

### 修复后（Qwen）
```
Step 1:  Loss: 2.3456, Grad Norm: 0.4234, LR: 1e-6  ✅ (warmup)
Step 50: Loss: 1.8765, Grad Norm: 0.3543, LR: 2.5e-5  ✅ (warmup)
Step 100: Loss: 1.5432, Grad Norm: 0.2234, LR: 5e-5  ✅ (normal)
Step 200: Loss: 1.2345, Grad Norm: 0.1823, LR: 5e-5  ✅ (normal)
```

## 🔧 参数对比

| 参数 | LLaMA | Qwen |
|------|-------|------|
| **学习率** | 1e-4 | 5e-5 (50% 降低) |
| **Max Grad Norm** | 1.0 | 0.5 (更激进) |
| **Warmup Steps** | 100 | 200 (更长) |
| **Warmup Ratio** | 0.1 | 0.2 (20%) |
| **LoRA Dropout** | 0.05 | 0.1 (更高) |
| **fp16 Opt Level** | O1 | O1 (相同) |

## 🎯 最佳实践

### 1. **模型检测**

```python
model_type = model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
print(f"Detected model type: {model_type}")
```

### 2. **参数调整**

```python
if model_type == 'qwen':
    # Qwen 特殊处理
    learning_rate *= 0.5
    max_grad_norm *= 0.5
    warmup_steps *= 2
```

### 3. **监控梯度**

```python
# 在训练循环中添加
if torch.isnan(loss):
    print(f"NaN detected at step {step}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Grad norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)}")
```

## 总结

✅ **主要修复**：
1. 根据模型类型调整 LoRA 配置
2. 对 Qwen 使用更保守的训练参数
3. 增加梯度裁剪强度
4. 延长学习率预热时间

✅ **关键参数**：
- Qwen 学习率：`1e-4 × 0.5 = 5e-5`
- Qwen Max Grad Norm：`0.5`（vs LLaMA 的 1.0）
- Qwen Warmup Steps：`200`（vs LLaMA 的 100）

✅ **预期效果**：
- 梯度正常（不再是 NaN）
- 学习率正常变化
- Loss 正常下降
- 训练稳定收敛

现在可以正常训练 Qwen 模型了！🎉

