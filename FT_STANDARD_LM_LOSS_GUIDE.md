# 使用标准语言建模损失微调指南

## 📋 概述

本指南介绍如何使用标准语言建模损失（Standard Language Modeling Loss）微调 LoRA Only 和 CultureMoE 模型。

### 新增脚本

1. **ft_lora_only_from_components.py** - LoRA Only 微调脚本
2. **run_ft_lora_only_from_components.sh** - LoRA Only 微调脚本（Shell）
3. **ft_culturemoe_from_base_gen.py** - CultureMoE 微调脚本
4. **run_ft_culturemoe_from_base_gen.sh** - CultureMoE 微调脚本（Shell）

### 数据格式

```json
{
    "text": "### Question: Give me the answer from 1 to 4: Do you agree with A university education is more important for a boy than for a girl? 1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree. You can only choose one option.\n ### Answer: 2",
    "text_mask": "### Question: Give me the answer from 1 to 4: Do you agree with A university education is more important for a boy than for a girl? 1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree. You can only choose one option.\n ### Answer: 2",
    "label": "0"
}
```

---

## 🚀 快速开始

### 1. 微调 LoRA Only 模型

```bash
# 使用 LLaMA + CultureLLM 数据集
sh run_ft_lora_only_from_components.sh llama 4

# 使用 Qwen + CultureLLM 数据集
sh run_ft_lora_only_from_components.sh qwen 4

# 使用其他数据集
sh run_ft_lora_only_from_components.sh llama 2  # CulturalBench
sh run_ft_lora_only_from_components.sh llama 3  # NormAD
```

### 2. 微调 CultureMoE 模型

```bash
# 基础训练（默认参数）
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5

# 消融实验：低文化损失权重
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.1

# 消融实验：高文化损失权重
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.8

# 消融实验：不使用文化损失
sh run_ft_culturemoe_from_base_gen.sh llama 4 False 6 2 0.0
```

---

## 📊 损失函数说明

### LoRA Only 模型

**损失函数**：标准语言建模损失

```
Loss = -Σ log P(token_t | token_<t)
```

- 对每个 token 计算交叉熵损失
- 使用所有 token 的平均损失

### CultureMoE 模型

**损失函数**：标准语言建模损失 + 文化专注性损失

```
Total Loss = Generation Loss + λ * Culture Loss

其中：
- Generation Loss = -Σ log P(token_t | token_<t)
- Culture Loss = 基于文化标签的分类损失
- λ = culture_loss_lambda（文化损失权重）
```

---

## 🔧 参数说明

### LoRA Only 微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_epochs | 6 | 训练轮数 |
| batch_size | 4 | 批大小 |
| learning_rate | 2e-4 | 学习率 |
| lora_r | 64 | LoRA 秩 |
| lora_alpha | 16 | LoRA alpha |
| lora_dropout | 0.1 | LoRA dropout |
| max_length | 512 | 最大序列长度 |
| val_split | 0.1 | 验证集比例（9:1） |

### CultureMoE 微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_epochs | 30 | 训练轮数 |
| batch_size | 4 | 批大小 |
| learning_rate | 1e-6 | 学习率 |
| use_culture_loss | True | 是否使用文化损失 |
| culture_loss_lambda | 0.5 | 文化损失权重 |
| num_experts | 6 | 专家数量 |
| shared_hidden_dim | 4096 | 共享层隐藏维度 |
| router_hidden_dim | 2048 | 路由器隐藏维度 |
| experts_hidden_dim | 4096 | 专家隐藏维度 |
| moe_lora_rank | 32 | MOE LoRA 秩 |

---

## 📈 输出文件

### LoRA Only 模型输出

```
output_dir/
├── best_lora/                    # 最好的 LoRA 权重
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── epoch_eval_results.json       # 每个 epoch 的评估结果
├── generated_answers.json        # 验证集上生成的答案
└── config.json                   # 训练配置
```

### CultureMoE 模型输出

```
output_dir/
├── best_moe/                     # 最好的 MOE 权重
│   └── pytorch_model.bin
├── epoch_eval_results.json       # 每个 epoch 的评估结果
├── generated_answers.json        # 验证集上生成的答案
└── config.json                   # 训练配置
```

---

## 📊 评估指标

### 每个 Epoch 的输出

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

### 指标说明

| 指标 | 说明 |
|------|------|
| train_loss | 训练总损失 |
| train_gen_loss | 训练生成损失 |
| train_culture_loss | 训练文化损失 |
| eval_loss | 验证总损失 |
| eval_gen_loss | 验证生成损失 |
| eval_culture_loss | 验证文化损失 |
| eval_accuracy | 验证准确率（通过生成答案计算） |
| correct | 验证集上正确的样本数 |
| total | 验证集总样本数 |

---

## 🔍 生成的答案格式

```json
{
    "text": "### Question: ... ### Answer: 2",
    "true_label": "0",
    "generated_text": "The answer is 2",
    "predicted_answer": "2",
    "correct": true
}
```

### 答案提取逻辑

使用正则表达式从生成的文本中提取答案：

```python
import re
match = re.search(r'Answer:\s*(\d+)', text)
if match:
    answer = match.group(1)
```

---

## 💡 常见问题

### Q1：LoRA Only 和 CultureMoE 的区别是什么？

**A**：
- **LoRA Only**：只使用标准语言建模损失，模型更简单
- **CultureMoE**：使用标准语言建模损失 + 文化专注性损失，模型更复杂但性能更好

### Q2：如何选择 culture_loss_lambda？

**A**：
- `0.0` - 不使用文化损失（基准）
- `0.1` - 低权重（验证文化损失的必要性）
- `0.5` - 中等权重（推荐）
- `0.8` - 高权重（强调文化对齐）
- `1.0` - 仅文化损失（极端情况）

### Q3：如何快速测试？

**A**：
```bash
# 修改脚本中的参数
# 减少 num_epochs（例如 2 而不是 6）
# 减少 batch_size（例如 2 而不是 4）
# 使用较小的数据集进行测试
```

### Q4：如何比较不同的模型？

**A**：
```bash
# 运行多个实验
sh run_ft_lora_only_from_components.sh llama 4
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5

# 比较结果
python -c "
import json

# LoRA Only
with open('/path/to/lora_output/epoch_eval_results.json') as f:
    lora_results = json.load(f)
    print(f'LoRA Only - Final Accuracy: {lora_results[-1][\"eval_accuracy\"]:.4f}')

# CultureMoE
with open('/path/to/moe_output/epoch_eval_results.json') as f:
    moe_results = json.load(f)
    print(f'CultureMoE - Final Accuracy: {moe_results[-1][\"eval_accuracy\"]:.4f}')
"
```

---

## 🎯 推荐的工作流程

### 第一步：微调 LoRA Only 模型

```bash
# 使用 CultureLLM 数据集
sh run_ft_lora_only_from_components.sh llama 4

# 等待训练完成，查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool
```

### 第二步：微调 CultureMoE 模型

```bash
# 基础配置
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5

# 等待训练完成，查看结果
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

for weight in [0.1, 0.3, 0.5, 0.7, 0.9]:
    # 找到对应的输出目录
    # 加载结果并比较
    pass
"
```

---

## ✅ 验证修复

### 检查 1：脚本是否存在

```bash
ls -la ft_lora_only_from_components.py
ls -la ft_culturemoe_from_base_gen.py
ls -la run_ft_lora_only_from_components.sh
ls -la run_ft_culturemoe_from_base_gen.sh
```

### 检查 2：运行测试

```bash
# 测试 LoRA Only
sh run_ft_lora_only_from_components.sh llama 4

# 测试 CultureMoE
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5
```

### 检查 3：查看输出

```bash
# 查看 LoRA Only 结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool | head -50

# 查看 CultureMoE 结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/epoch_eval_results.json | python -m json.tool | head -50
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

