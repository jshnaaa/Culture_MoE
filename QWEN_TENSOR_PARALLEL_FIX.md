# Qwen 模型 Tensor Parallel 错误修复

## ❌ 问题

运行 `sh run_train_lora_only_gen.sh qwen` 时报错：

```
OSError: tensor parallel is only supported for `torch>=2.5`.
```

## 🔍 问题分析

### 根本原因

Qwen 模型默认尝试使用 **tensor parallel** 功能来加速推理，但这个功能需要 PyTorch >= 2.5。

当前环境的 PyTorch 版本 < 2.5，导致加载失败。

### 什么是 Tensor Parallel？

Tensor Parallel 是一种模型并行技术，可以将模型的不同部分分布到多个 GPU 上，用于：
- 加速大模型推理
- 减少单个 GPU 的内存占用

但这个功能：
- 需要 PyTorch >= 2.5
- 主要用于推理，训练时通常不需要

## ✅ 修复方案

### 方案 1：禁用 Tensor Parallel（推荐）

在加载模型时，使用标准的注意力实现：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    # ✅ 禁用 tensor parallel
    attn_implementation="eager"  # 使用标准注意力实现
)
```

**优点**：
- ✅ 不需要升级 PyTorch
- ✅ 兼容性好
- ✅ 训练时性能差异不大

**缺点**：
- ❌ 推理时可能稍慢（但训练时影响不大）

### 方案 2：升级 PyTorch（不推荐）

```bash
# 升级到 PyTorch 2.5+
pip install torch>=2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**优点**：
- ✅ 支持最新功能

**缺点**：
- ❌ 可能破坏现有环境
- ❌ 其他依赖可能不兼容
- ❌ 训练时不需要 tensor parallel

## 🎯 attn_implementation 参数说明

### 可选值

1. **`"eager"`**（推荐用于训练）
   - 标准的 PyTorch 注意力实现
   - 兼容性最好
   - 不需要特殊依赖

2. **`"flash_attention_2"`**
   - 使用 Flash Attention 2
   - 需要安装 `flash-attn`
   - 训练时更快，内存更少

3. **`"sdpa"`**（Scaled Dot Product Attention）
   - PyTorch 2.0+ 的优化实现
   - 自动选择最优实现
   - 需要 PyTorch >= 2.0

4. **`None`**（默认）
   - 自动选择
   - Qwen 默认尝试使用 tensor parallel
   - 可能导致错误

### 推荐配置

```python
# 训练时（推荐）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager"  # 标准实现，兼容性好
)

# 如果安装了 Flash Attention（更快）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2"  # 需要 flash-attn
)

# PyTorch 2.0+（自动优化）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="sdpa"  # 需要 PyTorch >= 2.0
)
```

## 🚀 立即使用

### 重新运行训练

```bash
sh run_train_lora_only_gen.sh qwen
```

### 预期输出

```
Loading base model...
✅ Base model loaded

Configuring LoRA...
trainable params: 4,194,304 || all params: 7,615,616,000 || trainable%: 0.0551
✅ LoRA configured

Loading data...
✅ Loaded 11001 samples

Starting training...
  0%|          | 0/2061 [00:00<?, ?it/s]
  1%|▏         | 10/2061 [00:05<18:30, 1.85it/s]
```

## 📝 其他模型的类似问题

### LLaMA 模型

LLaMA 模型通常不会有这个问题，但如果遇到，同样使用：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager"
)
```

### Mistral 模型

Mistral 模型可能也会尝试使用 tensor parallel：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager"
)
```

## 🔍 检查 PyTorch 版本

```python
import torch
print(f"PyTorch version: {torch.__version__}")

# 检查是否支持 tensor parallel
if torch.__version__ >= "2.5.0":
    print("✅ Supports tensor parallel")
else:
    print("❌ Does not support tensor parallel")
    print("   Use attn_implementation='eager'")
```

## ⚠️ 注意事项

### 1. Flash Attention

如果想使用 Flash Attention 2（更快）：

```bash
# 安装 flash-attn
pip install flash-attn --no-build-isolation

# 使用
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2"
)
```

**注意**：
- 需要 CUDA 11.6+
- 需要 GPU 支持（A100, A6000, RTX 3090+）
- 编译时间较长

### 2. 多卡训练

使用 `attn_implementation="eager"` 不影响多卡训练：

```bash
# 双卡训练正常工作
torchrun --nproc_per_node=2 train_lora_only_gen.py ...
```

### 3. 性能影响

对于训练：
- `eager` vs `flash_attention_2`：速度差异约 10-20%
- `eager` vs `tensor_parallel`：训练时差异很小

对于推理：
- `tensor_parallel` 主要用于推理加速
- 训练时不需要

## 🎉 总结

### 问题
- Qwen 模型尝试使用 tensor parallel
- 需要 PyTorch >= 2.5
- 当前版本不支持

### 修复
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager"  # ✅ 使用标准实现
)
```

### 优点
- ✅ 不需要升级 PyTorch
- ✅ 兼容性好
- ✅ 训练性能差异不大

### 验证
```bash
sh run_train_lora_only_gen.sh qwen
```

现在应该能正常训练 Qwen 模型了！🎉

