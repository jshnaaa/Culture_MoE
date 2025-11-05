# 生成式模型：强制只输出数字

## ✅ 已更新的文件

- **`eval_base_gen.py`** - Base 模型生成式评估

## 🎯 核心功能

### 1. 强制只生成数字（选项编号）

```python
def generate_answer(model, tokenizer, instruction, input_text, num_classes=5):
    # 构建 prompt - 明确要求只输出数字
    full_input = f"{instruction}\n{input_text}\n\nPlease answer with ONLY ONE NUMBER (0 to {num_classes-1}).\nYour answer:"

    # 严格限制生成长度
    outputs = model.generate(
        **inputs,
        max_new_tokens=3,      # 只生成 3 个 token（足够一个数字）
        min_new_tokens=1,
        do_sample=False,       # 贪婪解码
        num_beams=1            # 禁用 beam search
    )

    # 只保留第一个字符
    raw_answer = raw_answer.split()[0]  # 取第一个词
    if len(raw_answer) > 1:
        raw_answer = raw_answer[0]  # 只取第一个字符

    return raw_answer, predicted_label
```

### 2. 多层标签提取策略

```python
def extract_label(answer, num_classes=5):
    # 策略1：直接转换为整数
    try:
        label = int(answer)
        if 0 <= label < num_classes:
            return label
    except ValueError:
        pass

    # 策略2：正则匹配第一个数字
    match = re.search(r'(\d+)', answer)
    if match:
        label = int(match.group(1))
        if 0 <= label < num_classes:
            return label

    # 策略3：默认返回中间类别
    return num_classes // 2
```

### 3. 保存详细答案文件

```python
# 每个样本保存：
{
    "instruction": "...",
    "input": "...",
    "true_label": 1,
    "culture_label": 1,
    "predicted_label": 1,
    "raw_answer": "1",
    "correct": true,
    "culture_correct": true
}
```

## 📁 输出文件

### 1. `generated_answers.json` - 详细答案

```json
[
  {
    "instruction": "You are a helpful assistant...",
    "input": "Question: How important is work-life balance?",
    "true_label": 1,
    "culture_label": 1,
    "predicted_label": 1,
    "raw_answer": "1",
    "correct": true,
    "culture_correct": true
  },
  {
    "instruction": "You are a helpful assistant...",
    "input": "Question: How important is high income?",
    "true_label": 2,
    "culture_label": 2,
    "predicted_label": 3,
    "raw_answer": "3",
    "correct": false,
    "culture_correct": false
  }
]
```

**字段说明**：
- `instruction`: 系统指令
- `input`: 输入问题
- `true_label`: 真实标签（主任务）
- `culture_label`: 文化标签
- `predicted_label`: 预测标签（整数）
- `raw_answer`: 模型生成的原始答案（字符串）
- `correct`: 主任务是否正确
- `culture_correct`: 文化任务是否正确

### 2. `evaluation_results.json` - 评估指标

```json
{
  "main_task": {
    "accuracy": 0.8523,
    "precision": 0.8456,
    "recall": 0.8501,
    "f1": 0.8478,
    "predictions": [1, 3, 2, 0, ...],
    "labels": [1, 2, 2, 0, ...]
  },
  "culture_task": {
    "accuracy": 0.8612,
    "precision": 0.8534,
    "recall": 0.8589,
    "f1": 0.8561,
    "predictions": [1, 3, 2, 0, ...],
    "labels": [1, 2, 2, 0, ...]
  },
  "num_samples": 1000,
  "failed_extractions": 15,
  "failed_rate": 0.015
}
```

### 3. `evaluation_summary.json` - 摘要

```json
{
  "evaluation_time": "2024-11-05 14:30:00",
  "model_path": "/path/to/model",
  "test_file": "/path/to/test.json",
  "num_samples": 1000,
  "main_task": {
    "accuracy": 0.8523,
    "precision": 0.8456,
    "recall": 0.8501,
    "f1": 0.8478
  },
  "culture_task": {
    "accuracy": 0.8612,
    "precision": 0.8534,
    "recall": 0.8589,
    "f1": 0.8561
  }
}
```

## 🚀 使用方法

### 评估 Base 模型

```bash
python eval_base_gen.py \
    --model_path /path/to/base/model \
    --test_file /path/to/test.json \
    --output_dir /path/to/output \
    --num_classes 5 \
    --save_answers
```

### 使用 Shell 脚本

```bash
# 评估 LLaMA base 模型
sh run_eval_base_gen.sh llama 11

# 评估 Qwen base 模型
sh run_eval_base_gen.sh qwen 11
```

## 📊 输出示例

```
Running evaluation...
Evaluating: 100%|████████████| 1000/1000 [05:23<00:00,  3.09it/s]

⚠️  Warning: 15/1000 samples failed to extract valid labels
   Failed rate: 1.50%

✅ Saved 1000 detailed answers to: /path/to/output/generated_answers.json

================================================================================
Evaluation Results
================================================================================

📊 Main Task (Classification):
   Accuracy:  0.8523
   Precision: 0.8456
   Recall:    0.8501
   F1:        0.8478

📊 Culture Task:
   Accuracy:  0.8612
   Precision: 0.8534
   Recall:    0.8589
   F1:        0.8561

📋 Detailed Classification Report (Main Task):
              precision    recall  f1-score   support

           0       0.85      0.83      0.84       200
           1       0.86      0.88      0.87       200
           2       0.84      0.85      0.84       200
           3       0.83      0.84      0.84       200
           4       0.85      0.83      0.84       200

    accuracy                           0.85      1000
   macro avg       0.85      0.85      0.85      1000
weighted avg       0.85      0.85      0.85      1000
```

## 🎯 优势

### 1. 强制数字输出
- ✅ 明确的 prompt 指令
- ✅ 严格限制生成长度（max_new_tokens=3）
- ✅ 只保留第一个字符

### 2. 多层提取策略
- ✅ 直接转换整数
- ✅ 正则匹配数字
- ✅ 默认值兜底

### 3. 详细答案保存
- ✅ 每个样本的原始答案
- ✅ 预测标签和真实标签
- ✅ 是否正确的标记
- ✅ 方便人工检查

### 4. 失败率统计
- ✅ 统计提取失败的样本数
- ✅ 计算失败率
- ✅ 打印警告信息

## 📝 数据格式要求

### 输入数据格式

```json
[
  {
    "instruction": "You are a helpful assistant...",
    "input": "Question: ...",
    "output": 1,  # 整数：主任务标签
    "label": 1    # 整数：文化任务标签（可选）
  }
]
```

### 类别编号

- **num_classes=2**: 标签范围 [0, 1]
- **num_classes=5**: 标签范围 [0, 1, 2, 3, 4]
- **num_classes=11**: 标签范围 [0, 1, 2, ..., 10]

## ⚠️ 注意事项

### 1. 提取失败处理

如果模型生成的不是数字（如 "The answer is 1"），会：
1. 尝试提取第一个数字
2. 如果失败，使用默认值（中间类别）
3. 统计失败率

### 2. 默认值选择

```python
default_label = num_classes // 2

# 示例：
# num_classes=2  → default=1
# num_classes=5  → default=2
# num_classes=11 → default=5
```

### 3. 查看详细答案

```bash
# 查看前 10 个样本
cat /path/to/output/generated_answers.json | jq '.[:10]'

# 查看错误样本
cat /path/to/output/generated_answers.json | jq '.[] | select(.correct == false)'

# 统计正确率
cat /path/to/output/generated_answers.json | jq '[.[] | select(.correct == true)] | length'
```

## 🎉 总结

现在生成式模型评估：
1. ✅ 强制只输出数字（选项编号）
2. ✅ 多层提取策略确保鲁棒性
3. ✅ 保存详细答案方便检查
4. ✅ 统计失败率
5. ✅ 方便计算损失和准确率

立即使用：
```bash
sh run_eval_base_gen.sh llama 11

