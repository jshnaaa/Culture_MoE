# 标准语言建模损失微调 - 实现总结

## 📋 概述

为了支持使用标准语言建模损失微调模型，我创建了 4 个新的脚本和文件。

---

## 📁 新增文件

### 1. LoRA Only 微调脚本

#### `ft_lora_only_from_components.py`

**功能**：使用标准语言建模损失微调 LoRA Only 模型

**关键特性**：
- ✅ 标准语言建模损失：`Loss = -Σ log P(token_t | token_<t)`
- ✅ 9:1 训练/验证集划分
- ✅ 每个 epoch 生成答案并评估准确率
- ✅ 保存最好的模型权重
- ✅ 详细的训练日志和结果

**数据格式**：
```json
{
    "text": "### Question: ... ### Answer: 2",
    "label": "0"
}
```

**输出**：
```
output_dir/
├── best_lora/                    # 最好的 LoRA 权重
├── epoch_eval_results.json       # 每个 epoch 的结果
├── generated_answers.json        # 生成的答案
└── config.json                   # 配置
```

#### `run_ft_lora_only_from_components.sh`

**功能**：LoRA Only 微调的 Shell 脚本

**使用方法**：
```bash
sh run_ft_lora_only_from_components.sh <BACKBONE> <DATA_ID>

# 参数
BACKBONE: llama 或 qwen (默认 llama)
DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)

# 示例
sh run_ft_lora_only_from_components.sh llama 4
sh run_ft_lora_only_from_components.sh qwen 4
```

**训练参数**：
- num_epochs: 6
- batch_size: 4
- learning_rate: 2e-4
- lora_r: 64
- lora_alpha: 16
- lora_dropout: 0.1

---

### 2. CultureMoE 微调脚本

#### `ft_culturemoe_from_base_gen.py`

**功能**：使用标准语言建模损失 + 文化专注性损失微调 CultureMoE 模型

**关键特性**：
- ✅ 标准语言建模损失 + 文化专注性损失
- ✅ 支持消融实验（调整 culture_loss_lambda）
- ✅ 9:1 训练/验证集划分
- ✅ 每个 epoch 生成答案并评估准确率
- ✅ 保存最好的模型权重
- ✅ 详细的训练日志和结果

**损失函数**：
```
Total Loss = Generation Loss + λ * Culture Loss

其中：
- Generation Loss = -Σ log P(token_t | token_<t)
- Culture Loss = 基于文化标签的分类损失
- λ = culture_loss_lambda（文化损失权重）
```

**数据格式**：
```json
{
    "text": "### Question: ... ### Answer: 2",
    "text_mask": "### Question: ... ### Answer: 2",
    "label": "0"
}
```

**输出**：
```
output_dir/
├── best_moe/                     # 最好的 MOE 权重
├── epoch_eval_results.json       # 每个 epoch 的结果
├── generated_answers.json        # 生成的答案
└── config.json                   # 配置
```

#### `run_ft_culturemoe_from_base_gen.sh`

**功能**：CultureMoE 微调的 Shell 脚本

**使用方法**：
```bash
sh run_ft_culturemoe_from_base_gen.sh <BACKBONE> <DATA_ID> <USE_CULTURE_LOSS> <NUM_EXPERTS> <NUM_GPUS> <CULTURE_LOSS_WEIGHT>

# 参数
BACKBONE: llama 或 qwen (默认 llama)
DATA_ID: 2=CulturalBench, 3=NormAD, 4=CultureLLM (默认 4)
USE_CULTURE_LOSS: True 或 False (默认 True)
NUM_EXPERTS: 专家数量 (默认 6)
NUM_GPUS: GPU 数量 (默认 2)
CULTURE_LOSS_WEIGHT: 文化损失权重 (默认 0.5)

# 示例
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.1
```

**训练参数**：
- num_epochs: 30
- batch_size: 4
- learning_rate: 1e-6
- culture_loss_lambda: 0.5（可调）
- num_experts: 6
- shared_hidden_dim: 4096
- router_hidden_dim: 2048
- experts_hidden_dim: 4096
- moe_lora_rank: 32

---

## 🔄 工作流程

### 第一步：微调 LoRA Only 模型

```bash
# 使用 CultureLLM 数据集
sh run_ft_lora_only_from_components.sh llama 4

# 等待训练完成（约 1-2 小时）
# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool
```

### 第二步：微调 CultureMoE 模型

```bash
# 基础配置
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5

# 等待训练完成（约 3-4 小时）
# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/epoch_eval_results.json | python -m json.tool
```

### 第三步：消融实验

```bash
# 测试不同的文化损失权重
for weight in 0.1 0.3 0.5 0.7 0.9; do
    sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 $weight
done

# 比较结果
python -c "
import json
import os

weights = [0.1, 0.3, 0.5, 0.7, 0.9]
for weight in weights:
    # 找到对应的输出目录
    # 加载结果并比较
    pass
"
```

---

## 📊 输出格式

### Epoch 结果

```json
{
    "epoch": 1,
    "train_loss": 0.4471,
    "train_gen_loss": 0.5323,
    "train_culture_loss": 0.0,
    "eval_loss": 0.5380,
    "eval_gen_loss": 0.5380,
    "eval_culture_loss": 0.0,
    "eval_accuracy": 0.4487,
    "correct": 45,
    "total": 100
}
```

### 生成的答案

```json
{
    "text": "### Question: ... ### Answer: 2",
    "true_label": "0",
    "generated_text": "The answer is 2",
    "predicted_answer": "2",
    "correct": true
}
```

---

## 🎯 关键改进

### 与原有脚本的区别

| 方面 | 原有脚本 | 新脚本 |
|------|---------|--------|
| 损失函数 | 分类损失 | 标准语言建模损失 |
| 数据格式 | instruction/input/output | text/text_mask |
| 评估方式 | 分类准确率 | 生成答案 + 准确率 |
| 消融实验 | 不支持 | 支持（culture_loss_lambda） |
| 输出 | 基本结果 | 详细结果 + 生成答案 |

### 新增功能

✅ 标准语言建模损失函数
✅ 支持 9:1 训练/验证集划分
✅ 每个 epoch 生成答案并评估
✅ 保存最好的模型权重
✅ 详细的训练日志
✅ 支持消融实验
✅ 完整的结果输出

---

## 🔧 技术细节

### 数据加载

```python
class CultureLLMDataset(Dataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        labels = input_ids.clone()  # 用于语言建模损失

        return {
            'input_ids': input_ids,
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels,
            'text': text,
            'label': item.get('label', '')
        }
```

### 损失计算

```python
# LoRA Only
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
loss = outputs.loss  # 标准语言建模损失

# CultureMoE
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    input_ids_mask=input_ids_mask,
    attention_mask_mask=attention_mask_mask,
    labels=labels,
    culture_labels=culture_labels,
    use_culture_loss=True,
    culture_loss_lambda=0.5
)
total_loss = outputs['loss']  # 生成损失 + 文化损失
```

### 答案提取

```python
import re

def extract_answer_from_text(text: str) -> str:
    # 查找 "Answer: " 后面的数字
    match = re.search(r'Answer:\s*(\d+)', text)
    if match:
        return match.group(1)
    return ""
```

---

## 📚 文档

### 新增文档

1. **FT_STANDARD_LM_LOSS_GUIDE.md** - 完整的微调指南
2. **FT_QUICK_START.md** - 快速开始指南
3. **FT_IMPLEMENTATION_SUMMARY.md** - 实现总结（本文件）

---

## ✅ 验证清单

- ✅ 创建 `ft_lora_only_from_components.py`
- ✅ 创建 `run_ft_lora_only_from_components.sh`
- ✅ 创建 `ft_culturemoe_from_base_gen.py`
- ✅ 创建 `run_ft_culturemoe_from_base_gen.sh`
- ✅ 支持标准语言建模损失
- ✅ 支持 9:1 训练/验证集划分
- ✅ 每个 epoch 生成答案并评估
- ✅ 保存最好的模型权重
- ✅ 详细的训练日志和结果
- ✅ 支持消融实验
- ✅ 完整的文档

---

## 🚀 快速开始

### 1. 微调 LoRA Only

```bash
sh run_ft_lora_only_from_components.sh llama 4
```

### 2. 微调 CultureMoE

```bash
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5
```

### 3. 查看结果

```bash
# LoRA Only
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool

# CultureMoE
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/epoch_eval_results.json | python -m json.tool
```

---

## 🎉 总结

### 新增功能

✅ 使用标准语言建模损失微调 LoRA Only 模型
✅ 使用标准语言建模损失 + 文化专注性损失微调 CultureMoE 模型
✅ 支持消融实验（调整 culture_loss_lambda）
✅ 每个 epoch 生成答案并评估准确率
✅ 保存详细的训练结果和生成的答案

### 关键改进

- 使用标准的语言建模损失函数
- 支持 9:1 的训练/验证集划分
- 每个 epoch 进行完整的评估
- 保存最好的模型权重
- 详细的日志和结果输出

---

**现在可以开始使用标准语言建模损失微调模型了！** 🚀

