# MASK_USE 和 LORA_USE 参数功能说明

## 📋 功能概述

新增两个参数用于 CultureMoE 训练的消融实验：
- **`MASK_USE`**: 控制是否使用数据集中的 `instruction_mask` 字段
- **`LORA_USE`**: 控制是否从合并后的 LoRA 模型开始训练

## 🎯 参数说明

### MASK_USE 参数

| 参数值 | 说明 | 双路输入 |
|--------|------|---------|
| **`true`** (默认) | 使用 `instruction_mask` 字段 | 路1: `instruction` + `input`<br>路2: `instruction_mask` + `input` |
| **`false`** | 不使用 `instruction_mask` 字段 | 路1: `instruction` + `input`<br>路2: `instruction` + `input` |

### LORA_USE 参数

| 参数值 | 说明 | 起始模型 |
|--------|------|---------|
| **`true`** (默认) | 从合并后的 LoRA 模型开始 | `{backbone}_merge_{num_classes}` |
| **`false`** | 从 Base 模型开始 | `Meta-{Llama/Qwen}-{version}` |

## 🔧 修改的文件

### 1. Shell 脚本
- **文件**: `run_train_culturemoe_from_merged.sh`
- **修改**:
  - 添加第7个参数 `MASK_USE="${7:-true}"`
  - 在训练命令中添加 `--use_instruction_mask $MASK_USE`
  - 在输出信息中显示 `Use instruction_mask: $MASK_USE`

### 2. Python 训练脚本
- **文件**: `train_culturemoe_from_merged.py`
- **修改**:
  - 添加参数: `--use_instruction_mask`
  - 在数据加载时传递: `use_instruction_mask=args.use_instruction_mask`
  - 打印信息: `Using instruction_mask: {args.use_instruction_mask}`

### 3. 数据处理器
- **文件**: `src/llamafactory/data/dual_classification_processor.py`
- **修改**:
  - `DualClassificationDataProcessor.__init__()`: 添加 `use_instruction_mask` 参数
  - `preprocess_function()`: 根据 `use_instruction_mask` 决定使用哪个字段
  - `load_and_process_dual_classification_data()`: 添加并传递 `use_instruction_mask` 参数

## 🚀 使用方法

### 基本用法

```bash
# 使用 instruction_mask（默认）
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 true

# 不使用 instruction_mask（两路都用 instruction）
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 false
```

### 参数顺序

```bash
sh run_train_culturemoe_from_merged.sh \
    <BACKBONE>           # 1. llama/qwen
    <NUM_CLASSES>        # 2. 2/3/4/5
    <USE_CULTURE_LOSS>   # 3. True/False
    <NUM_EXPERTS>        # 4. 6 (默认)
    <SAVE_MODEL>         # 5. true/false
    <NUM_GPUS>           # 6. 1/2
    <MASK_USE>           # 7. true/false (新增)
    <LORA_USE>           # 8. true/false (新增)
```

### 完整示例

```bash
# 完整模型：使用 LoRA + instruction_mask
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 true true

# 消融实验 1：不使用 instruction_mask
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 false true

# 消融实验 2：不使用 LoRA（从 Base 开始）
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 true false

# 消融实验 3：两者都不使用
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 false false
```

## 📊 实际效果

### MASK_USE=true (默认)

```python
# 数据样本
{
    "instruction": "You are a helpful assistant.",
    "instruction_mask": "You are a neutral assistant.",
    "input": "What is your opinion?",
    "output": 1
}

# 双路输入
路1: "You are a helpful assistant.\nWhat is your opinion?"
路2: "You are a neutral assistant.\nWhat is your opinion?"  # ✅ 使用 instruction_mask
```

### MASK_USE=false

```python
# 数据样本
{
    "instruction": "You are a helpful assistant.",
    "instruction_mask": "You are a neutral assistant.",
    "input": "What is your opinion?",
    "output": 1
}

# 双路输入
路1: "You are a helpful assistant.\nWhat is your opinion?"
路2: "You are a helpful assistant.\nWhat is your opinion?"  # ✅ 也使用 instruction
```

## 🎯 应用场景

### MASK_USE 参数

#### 使用 instruction_mask (MASK_USE=true)

**适用于**：
- 需要对比不同 instruction 的影响
- 消融实验：测试 instruction 的作用
- 文化维度分析：不同文化背景的 instruction

**示例**：
```bash
# 对比有无文化背景的 instruction
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 true
```

### 不使用 instruction_mask (MASK_USE=false)

**适用于**：
- 简化模型：两路输入相同
- 基线实验：排除 instruction_mask 的影响
- 消融实验：测试双路架构本身的作用

**示例**：
```bash
# 消融实验：测试双路架构（输入相同）
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 false
```

## 📝 代码示例

### 数据处理逻辑

```python
# 在 DualClassificationDataProcessor.preprocess_function() 中

# 构建第二组输入文本
for i in range(batch_size):
    if self.use_instruction_mask:
        # MASK_USE=true: 使用 instruction_mask
        instruction_mask = examples["instruction_mask"][i]
    else:
        # MASK_USE=false: 使用 instruction
        instruction_mask = examples["instruction"][i]

    input_text = examples.get("input", [""] * batch_size)[i]

    if input_text and input_text.strip():
        text = f"{instruction_mask}\n{input_text}"
    else:
        text = instruction_mask

    texts_mask.append(text)
```

## 🔬 实验建议

### 消融实验设计

```bash
# 实验 1: 完整模型（使用 instruction_mask）
sh run_train_culturemoe_from_merged.sh llama 2 True 6 true 1 true

# 实验 2: 不使用 instruction_mask
sh run_train_culturemoe_from_merged.sh llama 2 True 6 true 1 false

# 对比结果，分析 instruction_mask 的作用
```

### 预期结果

| 实验 | MASK_USE | 预期效果 |
|------|----------|---------|
| **实验 1** | true | 模型能学习到不同 instruction 的差异 |
| **实验 2** | false | 模型只学习双路架构本身的作用 |

## ✅ 验证方法

### 1. 检查训练日志

```bash
# 查看是否正确使用参数
grep "Using instruction_mask" <log_file>

# 应该看到：
# ✅ Using instruction_mask: True
# 或
# ✅ Using instruction_mask: False
```

### 2. 检查数据处理

```python
# 在训练开始时，会打印数据样本
# 检查两路输入是否符合预期
```

## 🎉 总结

- ✅ 添加了 `MASK_USE` 参数（第7个参数）
- ✅ 默认值为 `true`（保持原有行为）
- ✅ 支持消融实验和对比分析
- ✅ 代码向后兼容（不传参数时使用默认值）

**立即使用**：
```bash
# 使用 instruction_mask
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 true

# 不使用 instruction_mask
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 false

