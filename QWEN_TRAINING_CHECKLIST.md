# Qwen 训练问题完整检查清单

## 🔍 逐步排查建议

根据社区经验和你的补充，以下是完整的排查步骤：

---

## ✅ Step 1: 检查 LoRA 是否真正注入

### 问题描述

> 若几乎所有参数都是 frozen → 表示 LoRA 没被注入成功。这种情况通常导致 loss 恒为初始值 ≈ 13。

### 检查方法

```python
for n, p in model.named_parameters():
    if p.requires_grad:
        print(n)
```

**预期输出**：
```
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight
base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight
...
```

**应当能看到**：
- `lora_A` 层 ✅
- `lora_B` 层 ✅

### 当前代码检查

```python
# train_lora_only_gen.py (line 543-545)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # ✅ 已有
print("✅ LoRA configured\n")
```

**状态**：✅ 已有 `print_trainable_parameters()`

### 建议增强

```python
# 在 line 545 之后添加
print("\n" + "="*80)
print("LoRA Injection Check")
print("="*80)

trainable_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)

print(f"Total trainable parameters: {len(trainable_params)}")

if trainable_params:
    print("\nFirst 10 trainable parameters:")
    for name in trainable_params[:10]:
        print(f"   {name}")

    # 检查是否包含 lora_A 和 lora_B
    has_lora_a = any('lora_A' in name for name in trainable_params)
    has_lora_b = any('lora_B' in name for name in trainable_params)

    if has_lora_a and has_lora_b:
        print("\n✅ LoRA injection successful (found lora_A and lora_B)")
    else:
        print("\n⚠️  Warning: LoRA layers found but no lora_A/lora_B")
else:
    print("\n❌ No trainable parameters found!")
    print("   LoRA was not injected successfully.")
    raise RuntimeError("LoRA injection failed")

print("="*80)
```

---

## ✅ Step 2: 确认标签是否正确

### 问题描述

> 如果 labels 全是 -100，说明训练过程中没有有效的监督信号。这时 loss 会恒定在 13~14 左右。

### 检查方法

```python
print(batch["input_ids"][0][:50])
print(batch["labels"][0][:50])
```

**预期输出**：
```
input_ids: [151644, 151645, ..., 151643, 151643]
labels: [-100, -100, ..., 1, 2, 3, 4]  # 前面是 -100，后面是真实标签
```

### 当前代码检查

```python
# train_lora_only_gen.py (line 604-630)
def custom_data_collator(features):
    """自定义数据整理器，正确处理 labels（支持多卡训练）"""
    ...
    labels = labels + [-100] * padding_length  # ✅ padding 的 labels 用 -100
    ...
```

**状态**：✅ 已正确处理

### 建议增强

在训练开始前添加调试信息：

```python
# 在 Trainer 创建后，训练前添加（line 660 之后）
print("\n" + "="*80)
print("First Batch Debug")
print("="*80)

# 获取第一个 batch
train_dataloader = trainer.get_train_dataloader()
for batch in train_dataloader:
    print(f"Batch keys: {batch.keys()}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"\nFirst sample:")
    print(f"   input_ids[:20]: {batch['input_ids'][0][:20].tolist()}")
    print(f"   labels[:20]: {batch['labels'][0][:20].tolist()}")

    # 统计 labels 中的 -100
    num_ignore = (batch['labels'][0] == -100).sum().item()
    num_total = batch['labels'][0].shape[0]
    num_valid = num_total - num_ignore

    print(f"\nLabel statistics:")
    print(f"   Total tokens: {num_total}")
    print(f"   Ignore tokens (-100): {num_ignore} ({num_ignore/num_total*100:.1f}%)")
    print(f"   Valid tokens: {num_valid} ({num_valid/num_total*100:.1f}%)")

    if num_valid == 0:
        print("\n❌ All labels are -100! No supervision signal!")
        raise RuntimeError("All labels are -100")
    else:
        print(f"\n✅ Found {num_valid} valid labels")

    break

print("="*80)
```

### 修复方法（如果需要）

```python
# 确保这些配置正确
tokenizer.padding_side = "right"  # ✅ 右侧 padding
tokenizer.pad_token = tokenizer.eos_token  # ✅ 已有
model.config.pad_token_id = model.config.eos_token_id  # ✅ 已有
```

---

## ✅ Step 3: 检查优化器学习率

### 问题描述

> 如果恒为 0，说明 scheduler 初始化错误。

### 检查方法

```python
print(optimizer.param_groups[0]['lr'])
```

**预期输出**：
```
5e-05  # 或其他非零值
```

### 当前代码检查

```python
# train_lora_only_gen.py (line 580-600)
training_args = TrainingArguments(
    ...
    learning_rate=learning_rate,  # ✅ 已设置
    warmup_ratio=warmup_ratio,    # ✅ 已设置
    ...
)
```

**状态**：✅ 已正确配置

### 建议增强

添加学习率监控：

```python
# 创建一个 Callback 监控学习率
class LRMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            lr = kwargs.get('optimizer').param_groups[0]['lr']
            print(f"Step {state.global_step}: LR = {lr:.2e}")

# 在创建 Trainer 时添加
trainer = Trainer(
    ...
    callbacks=[epoch_callback, LRMonitorCallback()]  # ✅ 添加 LR 监控
)
```

### 推荐配置

```python
# 在 TrainingArguments 中
if model_type == 'qwen':
    training_args = TrainingArguments(
        ...
        lr_scheduler_type="cosine",  # ✅ 使用 cosine
        warmup_ratio=0.03,           # ✅ 3% 预热（更短）
        learning_rate=2e-4,          # ✅ 更高的学习率
        ...
    )
```

---

## ✅ Step 4: 测试梯度是否为 NaN

### 问题描述

> 若在前几步中梯度即为 NaN，说明 Qwen 模型输出 logit 出现无穷大。

### 检查方法

```python
for n, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f"NaN gradient in {n}")
```

### 建议增强

在训练循环中添加梯度检查：

```python
# 创建一个 Callback 检查梯度
class GradientCheckCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step <= 10:  # 只检查前 10 步
            nan_gradients = []
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    nan_gradients.append(name)

            if nan_gradients:
                print(f"\n❌ Step {state.global_step}: Found NaN gradients in {len(nan_gradients)} parameters")
                for name in nan_gradients[:5]:
                    print(f"   {name}")
                raise RuntimeError("NaN gradients detected")
            else:
                if state.global_step % 5 == 0:
                    print(f"✅ Step {state.global_step}: No NaN gradients")

# 在创建 Trainer 时添加
trainer = Trainer(
    ...
    callbacks=[epoch_callback, GradientCheckCallback()]  # ✅ 添加梯度检查
)
```

### 修复方法（如果出现 NaN）

```python
# 1. 使用更激进的梯度裁剪
training_args = TrainingArguments(
    ...
    max_grad_norm=0.3,  # ✅ 更小的值
    ...
)

# 2. 禁用 autocast（如果使用 fp16）
# 在 forward 中暂时禁用
with torch.cuda.amp.autocast(enabled=False):
    outputs = model(**inputs)
```

---

## ✅ Step 5: 重新确认 config 与 checkpoint 一致

### 问题描述

> 有时候加载 Qwen2.5 checkpoint 时会自动添加 rope_scaling，而 config.json 未更新，导致位置编码错位引发溢出。

### 检查方法

```python
print(model.config.rope_scaling)
print(model.config.model_type)
```

**预期输出（Qwen 2.5）**：
```
rope_scaling: {'type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
model_type: qwen2
```

### 当前代码检查

```python
# train_lora_only_gen.py (line 520-540)
# ❌ 没有检查 rope_scaling
```

**状态**：⚠️ 需要添加

### 建议增强

在模型配置检查中添加：

```python
# 在 line 540 之后添加
print("\n" + "="*80)
print("Model Config Check")
print("="*80)
print(f"Model type: {model.config.model_type}")
print(f"RoPE scaling: {getattr(model.config, 'rope_scaling', None)}")
print(f"Max position embeddings: {getattr(model.config, 'max_position_embeddings', None)}")

# 检查 rope_scaling
rope_scaling = getattr(model.config, 'rope_scaling', None)
if rope_scaling is not None:
    print(f"\n⚠️  RoPE scaling is enabled: {rope_scaling}")
    print("   This may cause position encoding issues if not configured correctly.")

    # 验证 rope_scaling 配置
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get('type', 'unknown')
        rope_factor = rope_scaling.get('factor', 1.0)
        print(f"   RoPE type: {rope_type}")
        print(f"   RoPE factor: {rope_factor}")
else:
    print("\n✅ No RoPE scaling")

print("="*80)
```

---

## ⚙️ 推荐的稳定训练设置（Qwen2.5 + LoRA + FP32）

### 完整配置

```python
# train_lora_only_gen.py

if model_type == 'qwen':
    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # ✅ lora_r
        lora_alpha=32,          # ✅ lora_alpha (更大的 alpha)
        lora_dropout=0.05,      # ✅ lora_dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # ✅ target_modules
        bias="none",
        init_lora_weights="gaussian"
    )

    # 训练配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # ✅ 学习率配置
        learning_rate=2e-4,           # ✅ 更高的学习率
        lr_scheduler_type="cosine",   # ✅ cosine scheduler
        warmup_ratio=0.03,            # ✅ 3% 预热

        # ✅ 梯度配置
        max_grad_norm=1.0,            # ✅ 标准梯度裁剪
        gradient_checkpointing=False, # ✅ 关闭 gradient checkpointing

        # ✅ 精度配置
        fp16=False,                   # ✅ 不使用 fp16
        bf16=False,                   # ✅ 不使用 bf16
        # 使用 fp32（默认）

        # 其他配置
        logging_steps=10,
        save_strategy="no",
        weight_decay=0.01,
        adam_epsilon=1e-8,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
    )
```

### 命令行参数

```bash
python train_lora_only_gen.py \
    --model_name_or_path /path/to/qwen \
    --data_path /path/to/data.json \
    --output_dir /path/to/output \
    --learning_rate 2e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

---

## 📊 完整检查清单

| 步骤 | 检查项 | 当前状态 | 建议 |
|------|--------|---------|------|
| **Step 1** | LoRA 注入 | ✅ 有 `print_trainable_parameters()` | 添加详细检查 |
| **Step 2** | 标签正确性 | ✅ 正确处理 -100 | 添加第一个 batch 调试 |
| **Step 3** | 学习率 | ✅ 正确配置 | 添加 LR 监控 |
| **Step 4** | 梯度 NaN | ⚠️ 没有检查 | 添加梯度检查 Callback |
| **Step 5** | Config 一致性 | ⚠️ 没有检查 rope_scaling | 添加 config 检查 |

---

## 🎯 推荐的修改

### 1. 添加诊断脚本

使用 `diagnose_qwen_training.py` 进行诊断：

```bash
python diagnose_qwen_training.py \
    --model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct \
    --data_path /root/autodl-fs/cultureLLM_merge_gen.json
```

### 2. 修改 train_lora_only_gen.py

添加以下增强：

1. **LoRA 注入检查**（line 545 之后）
2. **第一个 Batch 调试**（line 660 之后）
3. **学习率监控 Callback**
4. **梯度检查 Callback**
5. **Config 检查**（line 540 之后）

### 3. 使用推荐的训练配置

```python
if model_type == 'qwen':
    learning_rate = 2e-4          # ✅ 更高
    max_grad_norm = 1.0           # ✅ 标准值
    warmup_ratio = 0.03           # ✅ 更短
    use_fp32 = True               # ✅ 使用 fp32
    lr_scheduler_type = "cosine"  # ✅ cosine
    lora_alpha = 32               # ✅ 更大的 alpha
```

---

## 总结

✅ **已正确配置**：
1. LoRA 配置（target_modules）✅
2. Tokenizer pad_token ✅
3. Model config pad_token_id ✅
4. Labels padding (-100) ✅
5. Gradient checkpointing (False) ✅

⚠️ **需要添加**：
6. LoRA 注入详细检查
7. 第一个 Batch 调试信息
8. 学习率监控
9. 梯度 NaN 检查
10. RoPE Scaling 检查

🎯 **推荐操作**：
1. 运行 `diagnose_qwen_training.py` 诊断
2. 添加建议的增强代码
3. 使用推荐的训练配置
4. 重新训练并监控

现在可以开始完整的诊断和修复了！🔍

