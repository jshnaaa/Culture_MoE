# 生成式模型评估输出格式说明

## 📋 概述

生成式模型评估时，会强制模型只生成数字（选项编号），并保存详细的答案文件，方便查看和分析。

## 🎯 核心功能

### 1. 强制只生成数字

```python
# 方法1：只生成 1 个 token
outputs = model.generate(
    **inputs,
    max_new_tokens=1,  # 只生成 1 个 token
    min_new_tokens=1,
    do_sample=False,   # 贪婪解码
    num_beams=1
)

# 方法2：从 logits 中选择最可能的数字
digit_token_ids = [tokenizer.encode(str(i))[0] for i in range(num_classes)]
digit_logits = [logits[tid] for tid in digit_token_ids]
predicted_class = argmax(digit_logits)
```

### 2. 多层提取策略

```python
# 策略1：从生成的 token ID 提取
if generated_token_id in digit_token_ids:
    predicted_class = digit_token_ids.index(generated_token_id)

# 策略2：从 logits 提取
digit_logits = [logits[tid] for tid in digit_token_ids]
predicted_class = argmax(digit_logits)

# 策略3：从原始文本提取
match = re.search(r'([0-9])', raw_answer)
if match:
    predicted_class = int(match.group(1))

# 策略4：默认值（中间类别）
predicted_class = num_classes // 2
```

## 📁 输出文件格式

### 1. eval_metrics.json - 总体指标

```json
{
  "model_type": "generative_lora",
  "base_model": "/path/to/base/model",
  "lora_path": "/path/to/lora",
  "num_classes": 5,
  "test_samples": 1000,
  "metrics": {
    "accuracy": 0.8523,
    "precision": 0.8456,
    "recall": 0.8501,
    "f1": 0.8478
  },
  "confusion_matrix": [
    [180, 10, 5, 3, 2],
    [8, 185, 12, 4, 1],
    [5, 10, 190, 8, 2],
    [3, 5, 10, 175, 7],
    [2, 1, 3, 8, 186]
  ],
  "timestamp": "2024-11-05 14:30:00"
}
```

### 2. generated_answers.json - 详细答案

```json
[
  {
    "instruction": "You are a helpful assistant...",
    "input": "Question: How important is work-life balance?",
    "true_label": 1,
    "predicted_label": 1,
    "raw_answer": "1",
    "correct": true
  },
  {
    "instruction": "You are a helpful assistant...",
    "input": "Question: How important is high income?",
    "true_label": 2,
    "predicted_label": 3,
    "raw_answer": "3",
    "correct": false
  },
  ...
]
```

**字段说明**：
- `instruction`: 系统指令
- `input`: 输入问题
- `true_label`: 真实标签（整数）
- `predicted_label`: 预测标签（整数）
- `raw_answer`: 模型生成的原始答案（字符串）
- `correct`: 是否预测正确（布尔值）

### 3. error_cases.json - 错误案例

```json
[
  {
    "instruction": "...",
    "input": "...",
    "true_label": 2,
    "predicted_label": 3,
    "raw_answer": "3",
    "correct": false
  },
  ...
]
```

**用途**：
- 只包含预测错误的样本
- 方便分析模型的错误模式
- 用于改进 prompt 或训练数据

### 4. predictions.txt - 简洁格式

```
Index	True	Pred	Correct	Raw_Answer
0	1	1	✓	1
1	2	3	✗	3
2	0	0	✓	0
3	4	4	✓	4
4	3	2	✗	2
...
```

**用途**：
- 快速浏览预测结果
- 易于用 Excel 或文本编辑器打开
- 方便统计和分析

## 🚀 使用方法

### 基本用法

```bash
# LLaMA 模型
sh run_eval_lora_only_gen.sh llama

# Qwen 模型
sh run_eval_lora_only_gen.sh qwen
```

### 完整命令

```bash
python eval_lora_only_gen.py \
    --model_path /path/to/base/model \
    --lora_path /path/to/lora \
    --test_file /path/to/test.json \
    --output_dir /path/to/output \
    --num_classes 5 \
    --device cuda:0 \
    --save_answers
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 基座模型路径 | 必需 |
| `--lora_path` | LoRA 权重路径 | 必需 |
| `--test_file` | 测试数据文件 | 必需 |
| `--output_dir` | 输出目录 | 必需 |
| `--num_classes` | 类别数量 | 5 |
| `--device` | 设备 | cuda:0 |
| `--save_answers` | 保存详细答案 | True |

## 📊 输出示例

### 控制台输出

```
================================================================================
Evaluating Generative LoRA Model
================================================================================
Base model: /root/autodl-tmp/.../Meta-Llama-3.1-8B-Instruct
LoRA path: /root/autodl-tmp/.../lora_only_gen_llama
Test file: /root/autodl-fs/wvs_gen_test.json
Num classes: 5
Output dir: /root/autodl-tmp/.../eval_output/lora_gen_llama_20241105_1430
================================================================================

1. Loading tokenizer...
   ✅ Tokenizer loaded

2. Loading model...
   ✅ Model loaded

3. Loading test data...
   ✅ Loaded 1000 samples

4. Generating predictions...
Evaluating: 100%|████████████████████| 1000/1000 [05:30<00:00,  3.03it/s]

================================================================================
Evaluation Results
================================================================================

📊 Overall Metrics:
   Accuracy:   0.8523
   Precision:  0.8456
   Recall:     0.8501
   F1:         0.8478

📊 Classification Report:
              precision    recall  f1-score   support

   class_0       0.9000    0.9000    0.9000       200
   class_1       0.8810    0.8810    0.8810       210
   class_2       0.8636    0.8837    0.8735       215
   class_3       0.8750    0.8750    0.8750       200
   class_4       0.9302    0.9302    0.9302       175

    accuracy                         0.8523      1000
   macro avg     0.8456    0.8501    0.8478      1000
weighted avg    0.8523    0.8523    0.8523      1000

📊 Confusion Matrix:
           Predicted
             0    1    2    3    4
Actual   0  180   10    5    3    2
Actual   1    8  185   12    4    1
Actual   2    5   10  190    8    2
Actual   3    3    5   10  175    7
Actual   4    2    1    3    8  186

5. Saving results...
   ✅ Metrics saved to: .../eval_metrics.json
   ✅ Answers saved to: .../generated_answers.json
   ✅ Error cases saved to: .../error_cases.json
      Total errors: 148/1000 (14.8%)
   ✅ Predictions saved to: .../predictions.txt

================================================================================
✅ Evaluation completed!
================================================================================
```

### 查看结果

```bash
# 查看总体指标
cat eval_metrics.json | jq '.metrics'

# 查看前 10 个答案
head -10 generated_answers.json | jq '.'

# 查看错误案例
cat error_cases.json | jq '.[] | {true: .true_label, pred: .predicted_label, answer: .raw_answer}'

# 查看预测文件
head -20 predictions.txt
```

## 🔍 分析方法

### 1. 统计答案分布

```bash
# 统计每个类别的预测数量
cat predictions.txt | awk '{print $3}' | tail -n +2 | sort | uniq -c

# 输出：
#  180 0
#  185 1
#  190 2
#  175 3
#  186 4
```

### 2. 查看特定类别的错误

```bash
# 查看类别 2 的错误案例
cat error_cases.json | jq '.[] | select(.true_label == 2)'
```

### 3. 分析原始答案格式

```bash
# 查看是否有非数字答案
cat generated_answers.json | jq -r '.[] | .raw_answer' | grep -v '^[0-9]$'

# 如果有输出，说明有些答案不是纯数字
```

### 4. 计算每个类别的准确率

```python
import json
import numpy as np

with open('generated_answers.json') as f:
    answers = json.load(f)

# 按类别统计
from collections import defaultdict
class_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

for ans in answers:
    true_label = ans['true_label']
    class_stats[true_label]['total'] += 1
    if ans['correct']:
        class_stats[true_label]['correct'] += 1

# 打印每个类别的准确率
for label in sorted(class_stats.keys()):
    stats = class_stats[label]
    acc = stats['correct'] / stats['total']
    print(f"Class {label}: {acc:.4f} ({stats['correct']}/{stats['total']})")
```

## 🎯 优势

### 1. 强制数字输出
- ✅ 避免生成多余内容
- ✅ 易于提取和计算指标
- ✅ 与分类模型输出格式一致

### 2. 详细的答案文件
- ✅ 保存所有预测结果
- ✅ 包含原始答案和提取结果
- ✅ 方便错误分析

### 3. 多种输出格式
- ✅ JSON 格式（机器可读）
- ✅ TXT 格式（人类可读）
- ✅ 错误案例单独保存

### 4. 完整的评估指标
- ✅ 准确率、精确率、召回率、F1
- ✅ 混淆矩阵
- ✅ 分类报告

## 🔧 故障排除

### 问题 1：生成的不是数字

**现象**：
```json
{
  "raw_answer": "The answer is 2",
  "predicted_label": 2
}
```

**解决**：
- 提取逻辑会自动处理
- 如果提取失败，会使用默认值
- 检查 `error_cases.json` 中的 `raw_answer` 字段

### 问题 2：准确率很低

**可能原因**：
1. 模型训练不充分
2. Prompt 不够清晰
3. 测试数据与训练数据分布不同

**解决**：
1. 检查 `generated_answers.json`，看模型是否理解任务
2. 改进 Prompt
3. 增加训练数据或训练轮数

### 问题 3：某些类别准确率特别低

**分析**：
```bash
# 查看混淆矩阵
cat eval_metrics.json | jq '.confusion_matrix'

# 查看该类别的错误案例
cat error_cases.json | jq '.[] | select(.true_label == 2)'
```

## 🎉 总结

生成式模型评估现在可以：
1. ✅ 强制只生成数字
2. ✅ 保存详细的答案文件
3. ✅ 提供多种输出格式
4. ✅ 方便查看和分析结果

**立即使用**：
```bash
sh run_eval_lora_only_gen.sh llama

