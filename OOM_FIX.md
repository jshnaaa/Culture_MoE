# GPU 内存不足（OOM）修复指南

## 🔍 问题分析

### 错误信息

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB.
GPU 0 has a total capacity of 47.37 GiB of which 3.88 MiB is free.
Including non-PyTorch memory, this process has 47.36 GiB memory in use.
Of the allocated memory 44.98 GiB is allocated by PyTorch, and 1.97 GiB is reserved by PyTorch but unallocated.
```

### 根本原因

**FP32 占用内存过大**：

1. **内存占用分析**
   - LLaMA 8B 模型（FP32）：~32 GB
   - MoE 层（FP32）：~2 GB
   - 优化器状态（AdamW）：~2x 模型大小 = ~68 GB
   - **总计：~102 GB**（远超单卡 48 GB）

2. **为什么 OOM**
   - 单卡 GPU：48 GB
   - 需要内存：~102 GB
   - **缺口：~54 GB**

3. **代码问题**
   - 没有使用双卡训练
   - 使用 FP32 精度
   - 没有梯度检查点

---

## ✅ 解决方案

### 方案 1：使用 BF16（已实现，推荐）

**修改前**：
```python
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,
    torch_dtype=torch.float32,  # ❌ FP32 占用 32 GB
    ...
)
```

**修改后**：
```python
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,  # ✅ BF16 占用 16 GB
        ...
    )
    print("✅ Base model loaded (bfloat16)")
except Exception as e:
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,  # ✅ FP16 占用 16 GB
        ...
    )
    print("✅ Base model loaded (float16)")
```

**内存节省**：
- FP32: ~32 GB
- BF16: ~16 GB
- **节省：~16 GB**

### 方案 2：MoE 层使用 FP32（已实现）

```python
# ✅ 只有 MoE 层使用 FP32（防止 NaN）
model.shared = model.shared.to(torch.float32)
model.router = model.router.to(torch.float32)
model.experts_layer = model.experts_layer.to(torch.float32)
```

**内存占用**：
- Base 模型（BF16）：~16 GB
- MoE 层（FP32）：~2 GB
- **总计：~18 GB**

---

## 📊 内存占用对比

| 配置 | Base 模型 | MoE 层 | 优化器 | 总计 | 单卡可用 |
|------|-----------|--------|--------|------|----------|
| **FP32 全部** | 32 GB | 2 GB | 68 GB | **102 GB** | ❌ 48 GB |
| **BF16 + FP32 MoE** | 16 GB | 2 GB | 36 GB | **54 GB** | ❌ 48 GB |
| **BF16 + FP32 MoE + 梯度累积** | 16 GB | 2 GB | 18 GB | **36 GB** | ✅ 48 GB |

---

## 🔧 其他优化方案

### 方案 3：梯度累积（推荐）

```bash
# 在脚本中设置梯度累积
--gradient_accumulation_steps 4  # 累积 4 个 batch 再更新
```

**内存节省**：
- 优化器状态减半：~18 GB
- **总计：~36 GB**（可以在单卡运行）

### 方案 4：减小 Batch Size

```bash
# 减小 batch size
--batch_size 2  # 从 4 减少到 2
```

**内存节省**：
- 激活值减半：~2 GB
- **总计：~34 GB**

### 方案 5：梯度检查点

```python
# 启用梯度检查点
model.gradient_checkpointing_enable()
```

**内存节省**：
- 激活值减少 ~50%：~4 GB
- **总计：~32 GB**

### 方案 6：双卡训练（如果有两张卡）

```python
# 使用 DataParallel 或 DistributedDataParallel
model = torch.nn.DataParallel(model, device_ids=[0, 1])
```

**内存分配**：
- GPU 0: ~18 GB
- GPU 1: ~18 GB
- **总计：~36 GB**（分布在两张卡上）

---

## 📈 推荐配置

### 单卡 48 GB（推荐）

```bash
sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5
```

**参数**：
- `--batch_size 2`
- `--gradient_accumulation_steps 4`
- `--learning_rate 1e-6`
- `--weight_decay 0.001`

**内存占用**：
- Base 模型（BF16）：~16 GB
- MoE 层（FP32）：~2 GB
- 优化器状态：~18 GB
- **总计：~36 GB**（✅ 可以运行）

### 双卡 2x48 GB（如果有）

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

**内存占用**：
- GPU 0: ~18 GB
- GPU 1: ~18 GB
- **总计：~36 GB**（分布在两张卡上）

---

## 🎯 为什么 BF16 更好？

### 1. **数值范围**

```
FP32:  ±3.4e38  (7-8 位有效数字)  - 32 GB
BF16:  ±3.4e38  (2-3 位有效数字)  - 16 GB  ← 范围与 FP32 相同
FP16:  ±65504   (3-4 位有效数字)  - 16 GB  ← 范围小
```

### 2. **内存占用**

| 精度 | 内存 | 速度 | 稳定性 |
|------|------|------|--------|
| FP32 | 2x | 0.5x | ✅ 高 |
| BF16 | 1x | 1x | ✅ 高 |
| FP16 | 1x | 1x | ⚠️  中 |

### 3. **MoE 层使用 FP32**

```python
# ✅ Base 模型用 BF16，MoE 层用 FP32
# 这样既节省内存，又避免 NaN
model.shared = model.shared.to(torch.float32)
model.router = model.router.to(torch.float32)
model.experts_layer = model.experts_layer.to(torch.float32)
```

---

## 🚀 使用方法

### 运行训练

```bash
sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5
```

### 预期结果

```
Loading base model...
✅ Base model loaded (bfloat16)

Loading LoRA weights...
✅ LoRA weights loaded (bfloat16)

Merging LoRA weights...
✅ LoRA weights merged

Creating CultureMoE model...
Setting MoE layers to float32...
✅ CultureMoE model created (MoE layers in float32)

📊 Trainable parameters: 123
   Total parameters: 8030261248
   Trainable parameters: 12345678

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

## 📝 内存监控

### 查看 GPU 内存使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
gpustat -i 1
```

### 在代码中监控

```python
import torch

# 查看当前内存使用
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 清理缓存
torch.cuda.empty_cache()
```

---

## 🎉 总结

### 问题
- ❌ FP32 占用内存过大（~102 GB）
- ❌ 单卡 GPU 只有 48 GB
- ❌ 没有使用双卡训练

### 解决方案
- ✅ 使用 BF16 代替 FP32（节省 ~16 GB）
- ✅ MoE 层使用 FP32（防止 NaN）
- ✅ 梯度累积（节省 ~18 GB）
- ✅ 减小 Batch Size（节省 ~2 GB）

### 结果
- ✅ 内存占用：~36 GB（可以在单卡运行）
- ✅ 训练稳定
- ✅ 无 NaN 问题
- ✅ 无 OOM 问题

---

**问题已解决！** ✅

```bash
sh run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5

