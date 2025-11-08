# Qwen 关键配置检查（梯度 NaN 和 Loss 不变）

## 🔍 关键配置检查清单

根据你的补充，以下是 4 个可能导致 Qwen 梯度 NaN 和 Loss 不变的关键配置：

### 1. ⚠️ RMS Norm Epsilon 不匹配
### 2. ✅ Gradient Checkpointing 配置
### 3. ⚠️ Pad Token ID 配置
### 4. ⚠️ Model Config Pad Token ID

---

## 1. ⚠️ RMS Norm Epsilon 检查

### 问题描述

> Qwen2.5 默认使用 `rms_norm_eps=1e-6` 而不是 LLaMA 的 `1e-5`。

### 当前配置

```python
# train_lora_only_gen.py (line 504-509)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
    # ❌ 没有显式设置 rms_norm_eps
)
```

### 验证结果

**⚠️ 可能存在问题！**

**原因**：
- **Qwen 2.5** 使用 `rms_norm_eps=1e-6`
- **LLaMA** 使用 `rms_norm_eps=1e-5`
- 如果配置不匹配，可能导致数值不稳定

### 如何验证

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct",
    trust_remote_code=True
)

print("RMS Norm Eps:", getattr(config, 'rms_norm_eps', None))
print("Layer Norm Eps:", getattr(config, 'layer_norm_eps', None))
```

**预期输出（Qwen 2.5）**：
```
RMS Norm Eps: 1e-06
Layer Norm Eps: None
```

**预期输出（LLaMA）**：
```
RMS Norm Eps: 1e-05
Layer Norm Eps: None
```

### 修复方案

**方案 1：使用模型默认配置（推荐）**

```python
# train_lora_only_gen.py
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
    # ✅ 不设置 rms_norm_eps，使用模型默认配置
)
```

**方案 2：显式设置 RMS Norm Eps**

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True
)

# ✅ 确保 rms_norm_eps 配置正确
if model_type == 'qwen':
    config.rms_norm_eps = 1e-6  # Qwen 2.5 默认值
else:
    config.rms_norm_eps = 1e-5  # LLaMA 默认值

print(f"Using RMS Norm Eps: {config.rms_norm_eps}")

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

**方案 3：使用更大的 Epsilon（调试用）**

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True
)

# ✅ 使用更大的 epsilon 提高数值稳定性（仅用于调试）
config.rms_norm_eps = 1e-5  # 使用 LLaMA 的值

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

---

## 2. ✅ Gradient Checkpointing 检查

### 问题描述

> 如果你在 Trainer 参数里启用了 gradient checkpointing 或 zero-init 参数不一致，会出现梯度爆炸 → NaN。

### 当前配置

```python
# train_lora_only_gen.py (line 599)
training_args = TrainingArguments(
    ...
    gradient_checkpointing=False,  # ✅ 已关闭
    ...
)
```

### 验证结果

**✅ 当前配置是正确的！**

**原因**：
- `gradient_checkpointing=False` ✅
- 没有启用梯度检查点

### Gradient Checkpointing 说明

| 配置 | 内存使用 | 训练速度 | 数值稳定性 |
|------|---------|---------|-----------|
| **False** | 高 | 快 | ✅ 稳定 |
| **True** | 低 | 慢 | ⚠️ 可能不稳定 |

**Gradient Checkpointing 的作用**：
- 节省内存：不保存中间激活值
- 降低速度：需要重新计算前向传播
- 可能不稳定：对于某些模型（如 Qwen）可能导致梯度异常

### 建议配置

```python
if model_type == 'qwen':
    # ✅ Qwen 关闭 gradient checkpointing
    gradient_checkpointing = False
    print("  Gradient checkpointing: False (for stability)")
else:
    # LLaMA 可以开启（如果内存不够）
    gradient_checkpointing = False  # 或 True
    print(f"  Gradient checkpointing: {gradient_checkpointing}")

training_args = TrainingArguments(
    ...
    gradient_checkpointing=gradient_checkpointing,
    ...
)
```

### 调试建议

如果梯度仍然是 NaN，尝试：

```python
# ✅ 同时关闭梯度累积（调试用）
training_args = TrainingArguments(
    ...
    gradient_accumulation_steps=1,  # 关闭梯度累积
    gradient_checkpointing=False,   # 关闭梯度检查点
    ...
)
```

---

## 3. ⚠️ Pad Token ID 配置检查

### 问题描述

> Qwen2.5 的 `pad_token_id` 默认为 None，若没显式指定，会导致 loss 计算溢出或梯度 NaN。

### 当前配置

```python
# train_lora_only_gen.py (line 498-499)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# ❌ 但没有设置 model.config.pad_token_id
```

### 验证结果

**⚠️ 配置不完整！**

**原因**：
1. ✅ 设置了 `tokenizer.pad_token = tokenizer.eos_token`
2. ❌ 但没有设置 `model.config.pad_token_id = model.config.eos_token_id`

### 为什么需要两个配置？

```python
# Tokenizer 的 pad_token
tokenizer.pad_token = tokenizer.eos_token
# 用于：tokenizer 编码时使用

# Model Config 的 pad_token_id
model.config.pad_token_id = model.config.eos_token_id
# 用于：模型计算 loss 时忽略 padding
```

**如果只设置 tokenizer.pad_token**：
- Tokenizer 会正确添加 padding
- 但模型不知道哪些是 padding
- Loss 计算时会包含 padding，导致梯度异常

### 如何验证

```python
print("Tokenizer pad_token:", tokenizer.pad_token)
print("Tokenizer pad_token_id:", tokenizer.pad_token_id)
print("Model config pad_token_id:", model.config.pad_token_id)
print("Model config eos_token_id:", model.config.eos_token_id)
```

**预期输出（正确配置）**：
```
Tokenizer pad_token: <|endoftext|>
Tokenizer pad_token_id: 151643
Model config pad_token_id: 151643  # ✅ 应该相同
Model config eos_token_id: 151643
```

**预期输出（错误配置）**：
```
Tokenizer pad_token: <|endoftext|>
Tokenizer pad_token_id: 151643
Model config pad_token_id: None  # ❌ 是 None
Model config eos_token_id: 151643
```

### 修复方案

**方案 1：同时设置 tokenizer 和 model config（推荐）**

```python
# train_lora_only_gen.py (line 498-500)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ✅ 同时设置 model config
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

print(f"✅ Tokenizer pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"✅ Model config pad_token_id: {model.config.pad_token_id}")
```

**方案 2：在模型加载后立即设置**

```python
# train_lora_only_gen.py

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# ✅ 立即设置 pad_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
    print(f"✅ Set model.config.pad_token_id = {model.config.pad_token_id}")

# 验证配置
assert tokenizer.pad_token_id == model.config.pad_token_id, \
    f"Tokenizer pad_token_id ({tokenizer.pad_token_id}) != Model pad_token_id ({model.config.pad_token_id})"
```

**方案 3：在 LoRA 配置后设置**

```python
# train_lora_only_gen.py

# 配置 LoRA
model = get_peft_model(model, lora_config)

# ✅ 确保 pad_token_id 配置正确
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
    print(f"✅ Set model.config.pad_token_id = {model.config.pad_token_id}")

# 验证配置
print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
print(f"Model config pad_token_id: {model.config.pad_token_id}")
assert tokenizer.pad_token_id == model.config.pad_token_id
```

---

## 4. ⚠️ Loss 计算中的 Pad Token 处理

### 问题描述

> 否则在生成任务中会出现 CrossEntropy 计算全为 NaN 或 loss≈13 不变。

### Loss = 13 的含义

对于生成式任务：
```python
# 假设词表大小为 151643（Qwen 2.5）
random_loss = log(151643) ≈ 11.9

# 如果模型输出完全随机
# loss ≈ 12-14 是正常的
```

**Loss = 13 说明**：
- 模型输出完全随机
- 没有学习到任何东西
- 可能是 pad_token_id 配置错误导致

### 当前配置

```python
# train_lora_only_gen.py (line 604-630)
def custom_data_collator(features):
    """自定义数据整理器，正确处理 labels（支持多卡训练）"""
    import torch

    # 获取最大长度
    max_length = max(len(f["input_ids"]) for f in features)

    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for feature in features:
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        labels = feature["labels"]

        # Padding
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length  # ✅ padding 的 labels 用 -100

        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
        batch["labels"].append(labels)

    # 转换为 tensor
    batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    return batch
```

### 验证结果

**✅ 当前配置是正确的！**

**原因**：
- Padding 的 labels 使用 `-100` ✅
- `-100` 会被 CrossEntropyLoss 忽略 ✅

### Loss 计算原理

```python
# PyTorch CrossEntropyLoss
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

# 如果 labels 包含 -100，这些位置会被忽略
# 例如：
labels = [1, 2, 3, -100, -100]  # 前 3 个是真实标签，后 2 个是 padding
# loss 只计算前 3 个位置
```

### 如何验证

添加调试代码：

```python
# 在训练循环中添加
for step, batch in enumerate(train_dataloader):
    if step == 0:
        print("\n" + "="*80)
        print("First Batch Debug Info")
        print("="*80)
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        print(f"input_ids[0]: {batch['input_ids'][0]}")
        print(f"labels[0]: {batch['labels'][0]}")
        print(f"Number of -100 in labels[0]: {(batch['labels'][0] == -100).sum()}")
        print(f"Number of pad_token_id in input_ids[0]: {(batch['input_ids'][0] == tokenizer.pad_token_id).sum()}")
        print("="*80)

    # 训练...
```

**预期输出**：
```
First Batch Debug Info
================================================================================
input_ids shape: torch.Size([4, 512])
labels shape: torch.Size([4, 512])
input_ids[0]: tensor([151644, 151645, ..., 151643, 151643])  # 最后是 pad_token_id
labels[0]: tensor([151644, 151645, ..., -100, -100])  # 最后是 -100
Number of -100 in labels[0]: 100
Number of pad_token_id in input_ids[0]: 100
================================================================================
```

---

## 📊 完整诊断清单

| 配置项 | 状态 | 说明 |
|--------|------|------|
| **1. RMS Norm Eps** | ⚠️ 需要验证 | 没有显式设置，可能不匹配 |
| **2. Gradient Checkpointing** | ✅ 正确 | 已关闭（False） |
| **3. Tokenizer pad_token** | ✅ 正确 | 已设置为 eos_token |
| **4. Model config pad_token_id** | ❌ 缺失 | 没有设置 |
| **5. Labels padding** | ✅ 正确 | 使用 -100 |

---

## 🎯 推荐的修复步骤

### Step 1：验证 RMS Norm Eps

```bash
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct', trust_remote_code=True)
print('RMS Norm Eps:', getattr(config, 'rms_norm_eps', None))
"
```

### Step 2：验证 Pad Token ID

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct', trust_remote_code=True)
print('Tokenizer pad_token:', tokenizer.pad_token)
print('Tokenizer pad_token_id:', tokenizer.pad_token_id)
print('Model config pad_token_id:', model.config.pad_token_id)
print('Model config eos_token_id:', model.config.eos_token_id)
"
```

### Step 3：修改 train_lora_only_gen.py

在 `train_lora_only_gen.py` 中添加以下代码：

```python
# 在模型加载后（line 513 之后）
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
    print(f"✅ Set model.config.pad_token_id = {model.config.pad_token_id}")

# 验证配置
print("\n" + "="*80)
print("Model Configuration Check")
print("="*80)
print(f"Model type: {model.config.model_type}")
print(f"RMS Norm Eps: {getattr(model.config, 'rms_norm_eps', None)}")
print(f"Tokenizer pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"Model config pad_token_id: {model.config.pad_token_id}")
print(f"Model config eos_token_id: {model.config.eos_token_id}")
assert tokenizer.pad_token_id == model.config.pad_token_id, \
    f"Tokenizer pad_token_id ({tokenizer.pad_token_id}) != Model pad_token_id ({model.config.pad_token_id})"
print("✅ All configurations are correct")
print("="*80)
```

### Step 4：添加第一个 Batch 的调试信息

```python
# 在 Trainer 创建后，训练前添加
print("\n" + "="*80)
print("First Batch Debug")
print("="*80)

# 获取第一个 batch
for batch in train_loader:
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"input_ids[0][:10]: {batch['input_ids'][0][:10]}")
    print(f"labels[0][:10]: {batch['labels'][0][:10]}")
    print(f"Number of -100 in labels[0]: {(batch['labels'][0] == -100).sum()}")
    print(f"Number of pad_token_id in input_ids[0]: {(batch['input_ids'][0] == tokenizer.pad_token_id).sum()}")
    break

print("="*80)
```

---

## 📝 完整的修复代码

### 修改 train_lora_only_gen.py

```python
# 在 line 513 之后添加
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
    print(f"✅ Set model.config.pad_token_id = {model.config.pad_token_id}")

# 验证配置
print("\n" + "="*80)
print("Model Configuration Check")
print("="*80)
print(f"Model type: {model.config.model_type}")
print(f"RMS Norm Eps: {getattr(model.config, 'rms_norm_eps', None)}")
print(f"Tokenizer pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"Model config pad_token_id: {model.config.pad_token_id}")
print(f"Model config eos_token_id: {model.config.eos_token_id}")

# 断言验证
assert tokenizer.pad_token_id == model.config.pad_token_id, \
    f"Tokenizer pad_token_id ({tokenizer.pad_token_id}) != Model pad_token_id ({model.config.pad_token_id})"

print("✅ All configurations are correct")
print("="*80)
```

---

## 总结

✅ **已验证正确**：
1. Gradient Checkpointing：已关闭 ✅
2. Tokenizer pad_token：已设置 ✅
3. Labels padding：使用 -100 ✅

❌ **需要修复**：
4. **Model config pad_token_id**：没有设置 ❌（**最关键！**）

⚠️ **需要验证**：
5. RMS Norm Eps：没有显式设置

🎯 **推荐操作**：
1. **立即修复**：添加 `model.config.pad_token_id = model.config.eos_token_id`
2. 验证 RMS Norm Eps
3. 添加调试日志
4. 重新训练

**这个配置缺失很可能是导致 Loss = 13 不变的主要原因！** 🔥

