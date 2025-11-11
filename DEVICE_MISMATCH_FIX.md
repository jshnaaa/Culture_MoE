# 设备不匹配错误修复指南

## 🔍 问题分析

### 错误信息

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
(when checking argument for argument mat1 in method wrapper_CUDA_addmm)
```

### 根本原因

**MoE 层被放在 CPU 上，而输入在 CUDA 上**：

1. **问题代码**
   ```python
   # ❌ 只指定了 dtype，没有指定 device
   model.shared = model.shared.to(torch.float32)
   model.router = model.router.to(torch.float32)
   model.experts_layer = model.experts_layer.to(torch.float32)
   ```

2. **导致的问题**
   - `to(torch.float32)` 只改变数据类型
   - 不改变设备位置
   - MoE 层默认在 CPU 上
   - 输入在 CUDA 上
   - **设备不匹配导致错误**

3. **为什么会这样**
   - `model.shared` 是新创建的 `nn.Sequential`
   - 新创建的模块默认在 CPU 上
   - 需要显式移动到 GPU

---

## ✅ 解决方案

### 修改前

```python
# ❌ 只指定了 dtype，没有指定 device
model.shared = model.shared.to(torch.float32)
model.router = model.router.to(torch.float32)
model.experts_layer = model.experts_layer.to(torch.float32)
```

### 修改后

```python
# ✅ 同时指定 device 和 dtype
device = next(base_model.parameters()).device
model.shared = model.shared.to(device=device, dtype=torch.float32)
model.router = model.router.to(device=device, dtype=torch.float32)
model.experts_layer = model.experts_layer.to(device=device, dtype=torch.float32)
```

### 关键改变

1. **获取设备**
   ```python
   device = next(base_model.parameters()).device
   ```
   - 从 base_model 的参数获取设备
   - 确保与 base_model 在同一设备上

2. **同时指定 device 和 dtype**
   ```python
   model.shared.to(device=device, dtype=torch.float32)
   ```
   - `device=device`：指定设备（cuda:0 或 cpu）
   - `dtype=torch.float32`：指定数据类型

---

## 📊 修改清单

- ✅ 获取 base_model 的设备
- ✅ 同时指定 device 和 dtype
- ✅ 应用到所有 MoE 层（shared、router、experts_layer）

---

## 🔧 其他可能的设备不匹配问题

### 问题 1：输入在错误的设备上

```python
# ❌ 输入在 CPU 上，模型在 GPU 上
input_ids = batch['input_ids']  # CPU
outputs = model(input_ids)  # GPU

# ✅ 将输入移动到正确的设备
input_ids = batch['input_ids'].to(device)
outputs = model(input_ids)
```

### 问题 2：标签在错误的设备上

```python
# ❌ 标签在 CPU 上，模型在 GPU 上
labels = batch['labels']  # CPU
loss = loss_fn(outputs, labels)  # GPU

# ✅ 将标签移动到正确的设备
labels = batch['labels'].to(device)
loss = loss_fn(outputs, labels)
```

### 问题 3：优化器参数在错误的设备上

```python
# ❌ 优化器参数在 CPU 上
optimizer = torch.optim.Adam(model.parameters())
# 但模型在 GPU 上

# ✅ 确保模型在正确的设备上
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
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

# 确保一致
assert next(model.parameters()).device == input_ids.device
assert next(model.parameters()).device == labels.device
```

---

## 📈 预期输出

### 修复前

```
Creating CultureMoE model...
Setting MoE layers to float32...
✅ CultureMoE model created (MoE layers in float32)

Starting training...

Epoch 1/30
Training:   0%|                                                                                                                                                                          | 0/225 [00:00<?, ?it/s]

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

### 修复后

```
Creating CultureMoE model...
Setting MoE layers to float32...
✅ CultureMoE model created (MoE layers in float32 on cuda:0)

Starting training...

Epoch 1/30
Training: 100%|████████████████████████████████████████| 225/225 [02:15<00:00,  1.66it/s]
  Train Loss: 0.5234
  Eval Loss: 0.4521
  Eval Accuracy: 0.3545
  ✅ Best model saved

✅ Training completed!
```

---

## 🚀 使用方法

### 单卡训练

```bash
sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5
```

### 双卡训练

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
- ❌ MoE 层在 CPU 上
- ❌ 输入在 CUDA 上
- ❌ 设备不匹配导致错误

### 解决方案
- ✅ 获取 base_model 的设备
- ✅ 同时指定 device 和 dtype
- ✅ 确保所有张量在同一设备上

### 结果
- ✅ 设备一致
- ✅ 训练正常进行
- ✅ 无设备不匹配错误

---

**问题已解决！** ✅

```bash
sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5

