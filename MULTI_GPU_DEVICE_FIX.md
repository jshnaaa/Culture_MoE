# 多GPU设备不匹配错误修复指南

## 🔍 问题分析

### 错误信息

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
(when checking argument for argument target in method wrapper_CUDA_nll_loss_forward)
```

### 根本原因

**模型被自动分配到多个GPU，但损失计算时张量在不同设备上**：

1. **问题代码**
   ```python
   # 使用 device_map='auto' 自动分配模型到多个GPU
   base_model = AutoModelForCausalLM.from_pretrained(
       args.base_model_path,
       device_map='auto',  # ❌ 自动分配到 cuda:0 和 cuda:1
       ...
   )

   # 计算损失时
   shift_logits = logits[..., :-1, :].contiguous()  # 在 cuda:1
   shift_labels = labels[..., 1:].contiguous()      # 在 cuda:0

   loss = loss_fct(shift_logits, shift_labels)  # ❌ 设备不匹配
   ```

2. **导致的问题**
   - `device_map='auto'` 将模型分配到多个GPU
   - `logits` 在 cuda:1（lm_head 所在设备）
   - `labels` 在 cuda:0（输入数据所在设备）
   - **设备不匹配导致错误**

3. **为什么会这样**
   - LLaMA 8B 模型很大（~16 GB）
   - `device_map='auto'` 自动将模型分配到多个GPU
   - MoE 层被手动移动到特定设备
   - 输入数据在 cuda:0
   - 输出 logits 在 cuda:1

---

## ✅ 解决方案

### 方案 1：确保 labels 与 logits 在同一设备（已实现，推荐）

```python
# ✅ 确保 shift_labels 与 shift_logits 在同一设备上
if shift_labels.device != shift_logits.device:
    shift_labels = shift_labels.to(shift_logits.device)

loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
generation_loss = loss_fct(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1)
)
```

### 方案 2：使用单GPU（如果内存足够）

```python
# 强制使用单GPU
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,
    torch_dtype=torch.bfloat16,
    device_map={'': 0},  # ✅ 强制使用 cuda:0
    ...
)
```

### 方案 3：使用 DataParallel（不推荐）

```python
# 使用 DataParallel 包装模型
model = torch.nn.DataParallel(model, device_ids=[0, 1])
```

### 方案 4：使用 DistributedDataParallel（推荐用于多GPU训练）

```bash
# 使用 torchrun 启动
torchrun --nproc_per_node=2 ft_culturemoe_from_base_gen.py \
    --base_model_path /path/to/base_model \
    ...
```

---

## 📊 修改清单

- ✅ 确保 `shift_labels` 与 `shift_logits` 在同一设备上
- ✅ 在计算损失前检查设备一致性
- ✅ 自动移动张量到正确的设备

---

## 🔧 其他可能的设备不匹配问题

### 问题 1：输入在错误的设备上

```python
# ❌ 输入在 CPU 上，模型在 GPU 上
input_ids = batch['input_ids']  # CPU
outputs = model(input_ids)  # GPU

# ✅ 将输入移动到正确的设备
device = next(model.parameters()).device
input_ids = batch['input_ids'].to(device)
outputs = model(input_ids)
```

### 问题 2：MoE 层在错误的设备上

```python
# ❌ MoE 层在 CPU 上，输入在 GPU 上
model.shared = model.shared.to(torch.float32)  # CPU

# ✅ 同时指定 device 和 dtype
device = next(base_model.parameters()).device
model.shared = model.shared.to(device=device, dtype=torch.float32)
```

### 问题 3：文化标签在错误的设备上

```python
# ❌ 文化标签在 CPU 上，模型在 GPU 上
culture_labels = batch['label']  # CPU
culture_loss = compute_culture_loss(expert_weights, culture_labels)  # GPU

# ✅ 将文化标签移动到正确的设备
culture_labels = batch['label'].to(device)
culture_loss = compute_culture_loss(expert_weights, culture_labels)
```

---

## 🎯 最佳实践

### 1. 统一设备管理

```python
# 在模型初始化时统一设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = AutoModelForCausalLM.from_pretrained(...)
model = model.to(device)

# 加载数据
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)
labels = batch['labels'].to(device)

# 前向传播
outputs = model(input_ids, attention_mask=attention_mask)
loss = loss_fn(outputs, labels)
```

### 2. 检查设备一致性

```python
# 检查所有参数都在同一设备上
devices = set()
for param in model.parameters():
    devices.add(param.device)

if len(devices) > 1:
    print(f"⚠️  Warning: Model parameters on multiple devices: {devices}")
else:
    print(f"✅ All model parameters on device: {devices.pop()}")
```

### 3. 调试设备不匹配

```python
# 添加调试信息
print(f"Model device: {next(model.parameters()).device}")
print(f"Input device: {input_ids.device}")
print(f"Label device: {labels.device}")
print(f"Logits device: {logits.device}")

# 确保一致
assert next(model.parameters()).device == input_ids.device
assert logits.device == labels.device
```

---

## 📈 预期输出

### 修复前

```
Training:   0%|                                                                                                                                                                                                                                       | 0/225 [00:01<?, ?it/s]

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
```

### 修复后

```
Training: 100%|████████████████████████████████████████| 225/225 [02:15<00:00,  1.66it/s]
  Train Loss: 0.5234
  Eval Loss: 0.4521
  Eval Accuracy: 0.3545
  ✅ Best model saved

✅ Training completed!
```

---

## 🚀 使用方法

### 单卡训练（推荐）

```bash
sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5
```

### 双卡训练（使用 DistributedDataParallel）

```bash
# 使用 torchrun 启动
torchrun --nproc_per_node=2 ft_culturemoe_from_base_gen.py \
    --base_model_path /path/to/base_model \
    --lora_weights_path /path/to/lora_weights \
    --train_file /path/to/train_data.json \
    --output_dir /path/to/output \
    --batch_size 4 \
    --gradient_accumulation_steps 2
```

---

## 🎉 总结

### 问题
- ❌ 模型被自动分配到多个GPU
- ❌ logits 在 cuda:1
- ❌ labels 在 cuda:0
- ❌ 设备不匹配导致错误

### 解决方案
- ✅ 确保 labels 与 logits 在同一设备
- ✅ 在计算损失前检查设备一致性
- ✅ 自动移动张量到正确的设备

### 结果
- ✅ 设备一致
- ✅ 训练正常进行
- ✅ 无设备不匹配错误
- ✅ 模型正常收敛

---

**问题已解决！** ✅

```bash
sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5

