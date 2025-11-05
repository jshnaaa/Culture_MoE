# 多卡训练设备不匹配问题修复

## ❌ 问题

运行 `sh run_train_lora_only_gen.sh qwen` 进行双卡训练时报错：

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
```

## 🔍 问题分析

### 根本原因

在多卡训练（DDP）中，数据整理器（data collator）创建的张量没有正确处理设备分配，导致：
- 某些张量在 `cuda:0`
- 某些张量在 `cuda:1`
- 计算时出现设备不匹配

### 具体问题

```python
# 错误的代码
batch = {k: torch.tensor(v) for k, v in batch.items()}
# 问题：没有指定 dtype，可能导致类型不一致
# 问题：在多卡训练中，Trainer 需要自动处理设备分配
```

## ✅ 修复方案

### 1. 修改数据整理器

```python
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
        labels = labels + [-100] * padding_length

        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
        batch["labels"].append(labels)

    # ✅ 修复：明确指定 dtype，让 Trainer 自动处理设备
    batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    return batch
```

**关键点**：
- ✅ 明确指定 `dtype=torch.long`
- ✅ 不手动指定设备（`.to(device)`）
- ✅ 让 Trainer 自动处理设备分配

### 2. 添加多卡训练配置

```python
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    logging_steps=10,
    save_steps=args.save_steps,
    save_total_limit=3,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
    # ✅ 多卡训练配置
    ddp_find_unused_parameters=False,  # 加速训练
    dataloader_pin_memory=True,        # 加速数据加载
)
```

**关键点**：
- ✅ `ddp_find_unused_parameters=False`：加速 DDP 训练
- ✅ `dataloader_pin_memory=True`：加速数据加载

### 3. 使用 torchrun 启动

Shell 脚本已经正确配置：

```bash
# 检测 GPU 数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

if [ $NUM_GPUS -gt 1 ]; then
    # 使用 torchrun 进行多卡训练
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_lora_only_gen.py \
        --model_name_or_path $BASE_MODEL_PATH \
        ...
else
    # 单卡训练
    python train_lora_only_gen.py \
        --model_name_or_path $BASE_MODEL_PATH \
        ...
fi
```

## 🎯 多卡训练最佳实践

### 1. 数据整理器

**❌ 错误做法**：
```python
# 手动指定设备
batch = {k: torch.tensor(v).to('cuda:0') for k, v in batch.items()}
```

**✅ 正确做法**：
```python
# 让 Trainer 自动处理
batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
```

### 2. 模型加载

**❌ 错误做法**：
```python
# 手动移动到设备
model = model.to('cuda:0')
```

**✅ 正确做法**：
```python
# 使用 device_map="auto" 或让 Trainer 处理
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配设备
)
```

### 3. TrainingArguments

```python
training_args = TrainingArguments(
    # 基本配置
    output_dir=output_dir,
    per_device_train_batch_size=4,

    # 多卡训练配置
    ddp_find_unused_parameters=False,  # 加速 DDP
    dataloader_pin_memory=True,        # 加速数据加载

    # 混合精度训练
    fp16=True,  # 或 bf16=True
)
```

### 4. 启动命令

```bash
# 使用 torchrun（推荐）
torchrun --nproc_per_node=2 train.py

# 或使用 accelerate
accelerate launch --num_processes=2 train.py

# 或使用 torch.distributed.launch（旧版）
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

## 🚀 立即使用

### 重新运行训练

```bash
sh run_train_lora_only_gen.sh qwen
```

### 预期输出

```
Detected 2 GPUs
Using multi-GPU training with 2 GPUs

Loading model...
✅ Model loaded

Loading data...
✅ Loaded 11001 samples

Starting training...
  0%|          | 0/2061 [00:00<?, ?it/s]
  1%|▏         | 10/2061 [00:05<18:30, 1.85it/s]  ← 正常训练
  2%|▏         | 20/2061 [00:10<18:25, 1.85it/s]
...
```

## ⚠️ 注意事项

### 1. 批次大小

多卡训练时，**总批次大小 = per_device_batch_size × num_gpus × gradient_accumulation_steps**

```bash
# 示例：2 卡训练
per_device_batch_size = 4
num_gpus = 2
gradient_accumulation_steps = 4

# 总批次大小 = 4 × 2 × 4 = 32
```

### 2. 学习率调整

多卡训练时，可能需要调整学习率：

```python
# 单卡学习率
lr_single = 1e-4

# 多卡学习率（线性缩放）
lr_multi = lr_single * num_gpus  # 2e-4 for 2 GPUs
```

### 3. 保存模型

多卡训练时，只有主进程（rank 0）保存模型：

```python
# Trainer 自动处理
trainer.save_model(output_dir)  # 只在 rank 0 保存
```

### 4. 日志输出

多卡训练时，每个进程都会输出日志，可能看到重复信息：

```bash
# 只显示主进程日志
export TRANSFORMERS_VERBOSITY=error
```

## 🔍 验证修复

### 1. 检查 GPU 使用

```bash
# 持续监控
watch -n 1 nvidia-smi

# 应该看到两个 GPU 都在使用
```

### 2. 检查训练速度

```bash
# 多卡训练应该比单卡快接近 2 倍
# 单卡：~1.0 it/s
# 双卡：~1.8-1.9 it/s（考虑通信开销）
```

### 3. 检查模型输出

```bash
# 训练完成后，检查输出目录
ls -la /path/to/output/

# 应该看到：
# - adapter_config.json
# - adapter_model.bin
# - tokenizer files
```

## 🎉 总结

### 问题
- 数据整理器没有正确处理多卡训练的设备分配
- 张量分布在不同设备上

### 修复
- ✅ 明确指定 `dtype=torch.long`
- ✅ 不手动指定设备
- ✅ 让 Trainer 自动处理设备分配
- ✅ 添加多卡训练配置

### 验证
```bash
sh run_train_lora_only_gen.sh qwen
```

现在应该能正常进行双卡训练了！🎉

