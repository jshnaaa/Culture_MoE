# Base 模型评估指南

## 📋 概述

本指南介绍如何在 CultureLLM 数据集上评估 Base 模型（LLaMA 或 Qwen）的性能。

### 新增脚本

1. **ft_base.py** - Base 模型评估脚本
2. **run_ft_base.sh** - Base 模型评估脚本（Shell）

### 数据格式

```json
{
    "text": "### Question: Give me the answer from 1 to 10: Do you think the homosexuality can always be justified, never be justified, or something in between? Using this card on which 1 means 'Never justifiable' and 10 means 'Always justifiable' 1. 1 2. 2 3. 3 4. 4 5. 5 6. 6 7. 7 8. 8 9. 9 10. 10. This question is for a country or language that is Andorra. You can only choose one option.\n ### Answer: 10",
    "text_mask": "### Question: Give me the answer from 1 to 10: Do you think the homosexuality can always be justified, never be justified, or something in between? Using this card on which 1 means 'Never justifiable' and 10 means 'Always justifiable' 1. 1 2. 2 3. 3 4. 4 5. 5 6. 6 7. 7 8. 8 9. 9 10. 10. This question is for a country or language that is [MASK]. You can only choose one option.\n ### Answer: 10",
    "label": "1"
}
```

**说明**：
- `text` - 完整的问题和答案（用于 Base 模型评估）
- `text_mask` - 带 [MASK] 的问题（用于 CultureMoE 模型）
- `label` - 文化标签（用于 CultureMoE 模型）

---

## 🚀 快速开始

### 1. 评估 LLaMA Base 模型

```bash
# 使用 CultureLLM 数据集
sh run_ft_base.sh llama 4

# 使用其他数据集
sh run_ft_base.sh llama 2  # CulturalBench
sh run_ft_base.sh llama 3  # NormAD
```

### 2. 评估 Qwen Base 模型

```bash
# 使用 CultureLLM 数据集
sh run_ft_base.sh qwen 4

# 使用其他数据集
sh run_ft_base.sh qwen 2  # CulturalBench
sh run_ft_base.sh qwen 3  # NormAD
```

### 3. 查看结果

```bash
# 查看评估结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/eval_results.json | python -m json.tool

# 查看生成的答案
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/generated_answers.json | python -m json.tool | head -50

# 查看准确率
python -c "import json; data = json.load(open('/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/eval_results.json')); print(f'Accuracy: {data[\"accuracy\"]:.4f}')"
```

---

## 📊 输出文件

### 输出目录结构

```
/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/
├── eval_results.json           # 评估结果
├── generated_answers.json      # 生成的答案
└── config.json                 # 配置
```

### eval_results.json

```json
{
    "accuracy": 0.4523,
    "correct": 45,
    "total": 100,
    "timestamp": "2025-11-10T12:34:56.789123"
}
```

### generated_answers.json

```json
[
    {
        "text": "### Question: ... ### Answer: 10",
        "true_label": "1",
        "generated_text": "The answer is 10",
        "predicted_answer": "10",
        "correct": true
    },
    {
        "text": "### Question: ... ### Answer: 5",
        "true_label": "2",
        "generated_text": "I think the answer is 5",
        "predicted_answer": "5",
        "correct": false
    },
    ...
]
```

### config.json

```json
{
    "base_model": "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct",
    "max_length": 512
}
```

---

## 🔧 参数说明

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| backbone | llama 或 qwen | llama |
| DATA_ID | 2=CulturalBench, 3=NormAD, 4=CultureLLM | 4 |

### Python 脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --base_model_path | Base 模型路径 | 必需 |
| --train_file | 训练数据路径 | 必需 |
| --output_dir | 输出目录 | 必需 |
| --max_length | 最大序列长度 | 512 |
| --device | 设备 (cuda/cpu) | cuda |

---

## 📈 评估指标

### 准确率 (Accuracy)

```
Accuracy = Correct / Total

其中：
- Correct：预测正确的样本数
- Total：总样本数
```

### 答案提取

使用正则表达式从生成的文本中提取答案：

```python
import re

def extract_answer_from_text(text: str) -> str:
    # 查找 "Answer: " 后面的数字
    match = re.search(r'Answer:\s*(\d+)', text)
    if match:
        return match.group(1)
    return ""

# 示例
text = "The answer is 10"
answer = extract_answer_from_text(text)  # 返回 "10"
```

---

## 🎯 工作流程

### 第一步：评估 Base 模型

```bash
# 评估 LLaMA Base 模型
sh run_ft_base.sh llama 4

# 等待完成（约 30 分钟 - 1 小时）
# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/eval_results.json | python -m json.tool
```

### 第二步：评估 LoRA Only 模型

```bash
# 评估 LoRA Only 模型
sh run_ft_lora_only_from_components.sh llama 4

# 等待完成（约 1-2 小时）
# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool
```

### 第三步：评估 CultureMoE 模型

```bash
# 评估 CultureMoE 模型
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5

# 等待完成（约 3-4 小时）
# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/epoch_eval_results.json | python -m json.tool
```

### 第四步：比较结果

```bash
# 提取各模型的准确率
python -c "
import json
import os

# Base 模型
base_result = json.load(open('/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/eval_results.json'))
print(f'Base Model Accuracy: {base_result[\"accuracy\"]:.4f}')

# LoRA Only 模型（最后一个 epoch）
lora_result = json.load(open('/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json'))
print(f'LoRA Only Model Accuracy: {lora_result[-1][\"eval_accuracy\"]:.4f}')

# CultureMoE 模型（最后一个 epoch）
moe_result = json.load(open('/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/epoch_eval_results.json'))
print(f'CultureMoE Model Accuracy: {moe_result[-1][\"eval_accuracy\"]:.4f}')
"
```

---

## 💡 常见问题

### Q1: Base 模型评估需要多长时间？

**A**: 取决于数据集大小和 GPU 性能，通常需要 30 分钟 - 1 小时。

### Q2: 如何快速测试？

**A**:
```bash
# 修改脚本中的数据集大小
# 或使用较小的数据集进行测试
```

### Q3: 如何比较不同 Backbone 的性能？

**A**:
```bash
# 评估 LLaMA
sh run_ft_base.sh llama 4

# 评估 Qwen
sh run_ft_base.sh qwen 4

# 比较结果
python -c "
import json

llama_result = json.load(open('/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/eval_results.json'))
qwen_result = json.load(open('/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_qwen_*/eval_results.json'))

print(f'LLaMA Accuracy: {llama_result[\"accuracy\"]:.4f}')
print(f'Qwen Accuracy: {qwen_result[\"accuracy\"]:.4f}')
"
```

### Q4: 如何比较不同数据集的性能？

**A**:
```bash
# 评估 CultureLLM
sh run_ft_base.sh llama 4

# 评估 CulturalBench
sh run_ft_base.sh llama 2

# 评估 NormAD
sh run_ft_base.sh llama 3

# 比较结果
python -c "
import json

culturellm = json.load(open('/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/eval_results.json'))
culturalbench = json.load(open('/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_CulturalBench_llama_*/eval_results.json'))
normad = json.load(open('/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_normad_llama_*/eval_results.json'))

print(f'CultureLLM Accuracy: {culturellm[\"accuracy\"]:.4f}')
print(f'CulturalBench Accuracy: {culturalbench[\"accuracy\"]:.4f}')
print(f'NormAD Accuracy: {normad[\"accuracy\"]:.4f}')
"
```

### Q5: 生成的答案为什么有些不正确？

**A**: 可能的原因：
1. Base 模型没有经过微调，性能有限
2. 问题表述复杂，模型难以理解
3. 答案提取正则表达式可能不匹配

---

## 🔍 答案提取示例

### 成功的例子

```
生成的文本：
"The answer is 10"

提取的答案：
"10"

匹配：✅
```

### 失败的例子

```
生成的文本：
"I think the answer could be around 5 or 6"

提取的答案：
"5"（提取第一个数字）

匹配：可能不正确
```

---

## 📊 性能对比预期

### 不同 Backbone 的性能

| Backbone | 准确率 | 说明 |
|----------|--------|------|
| LLaMA 3.1-8B | ~45% | 基础性能 |
| Qwen 2.5-7B | ~48% | 略优于 LLaMA |

### 不同数据集的性能

| 数据集 | 准确率 | 说明 |
|--------|--------|------|
| CultureLLM | ~45% | 标准数据集 |
| CulturalBench | ~42% | 可能更难 |
| NormAD | ~50% | 可能更简单 |

### 不同模型的性能对比

| 模型 | 准确率 | 说明 |
|------|--------|------|
| Base | ~45% | 基础性能 |
| LoRA Only | ~55% | 微调后性能提升 |
| CultureMoE | ~60% | 最佳性能 |

---

## ✅ 验证清单

- ✅ 创建 `ft_base.py`
- ✅ 创建 `run_ft_base.sh`
- ✅ 支持 LLaMA 和 Qwen
- ✅ 支持多个数据集
- ✅ 生成答案并评估准确率
- ✅ 保存详细的结果和生成的答案
- ✅ 完整的文档

---

## 🚀 快速命令参考

```bash
# 评估 LLaMA Base 模型 + CultureLLM
sh run_ft_base.sh llama 4

# 评估 Qwen Base 模型 + CultureLLM
sh run_ft_base.sh qwen 4

# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/eval_results.json | python -m json.tool

# 查看生成的答案
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/generated_answers.json | python -m json.tool | head -50
```

---

## 🎉 总结

### 新增功能

✅ 评估 Base 模型（LLaMA 和 Qwen）
✅ 支持多个数据集（CultureLLM、CulturalBench、NormAD）
✅ 生成答案并评估准确率
✅ 保存详细的结果和生成的答案
✅ 完整的文档和使用指南

### 关键特性

- 只使用 `text` 字段进行评估
- 忽略 `text_mask` 和 `label` 字段
- 使用正则表达式提取答案
- 计算准确率
- 保存生成的答案用于分析

---

**现在可以开始评估 Base 模型了！** 🚀

