## 📊 Base LLaMA 评估脚本对比

### 问题背景

WVS 数据集的标签**不统一**：
- **类别数不同**：有的是 4 分类，有的是 5 分类
- **标签名称不同**：
  - `1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree`
  - `1. Very frequently 2. Quite frequently 3. Not frequently 4. Not at all frequently`
  - `1. Definitely should have the right 2. probably should have the right ...`
  - `1. Strongly agree 2. agree 3. Neither agree nor disagree 4. Disagree 5. Disagree strongly`

---

## 🔧 两种评估方案

### 方案 1：固定标签（`eval_base_llama.py`）

**适用场景**：
- ✅ 标签数量固定（如全部是 2 分类或全部是 5 分类）
- ✅ 标签名称统一（如全部使用 "strongly_disagree", "disagree" 等）

**限制**：
- ❌ **不适用于 WVS 数据集**（标签不统一）
- ❌ 需要预先指定 `--num_classes`
- ❌ 使用固定的 token 映射

**使用方法**：
```bash
# 二分类
bash run_eval_base_llama.sh

# 五分类
bash run_eval_base_llama_4class.sh
```

---

### 方案 2：灵活标签（`eval_base_llama_flexible.py`）✅ 推荐

**适用场景**：
- ✅ **标签数量不固定**（同一数据集中有 4 分类和 5 分类）
- ✅ **标签名称不统一**（不同问题有不同的标签文本）
- ✅ **WVS 数据集**

**特点**：
- ✅ 自动从 `instruction` 中提取标签选项
- ✅ 每个样本独立处理
- ✅ 支持任意标签名称

**使用方法**：
```bash
bash run_eval_base_llama_flexible.sh
```

---

## 📝 数据格式要求

### 方案 1（固定标签）

```json
{
  "instruction": "Question: Do you agree?",
  "input": "...",
  "output": 0,
  "label": "0"
}
```

**要求**：
- `output` 必须是 0-based index（0, 1, 2, ...）
- 所有样本的类别数必须相同

---

### 方案 2（灵活标签）

```json
{
  "instruction": "Question: ... Options: 1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree",
  "input": "...",
  "output": 0,
  "label": "0"
}
```

**要求**：
- `instruction` 中必须包含标签选项（格式：`1. 标签1 2. 标签2 ...`）
- `output` 是 0-based index，对应标签选项的位置

**支持的格式**：
```
1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree
1. Very frequently 2. Quite frequently 3. Not frequently 4. Not at all frequently
1. Definitely should have the right 2. probably should have the right 3. probably should not have the right 4. Definitely should not have the right
1. Strongly agree 2. agree 3. Neither agree nor disagree 4. Disagree 5. Disagree strongly
```

---

## 🔍 工作原理对比

### 方案 1：固定 Token 映射

```python
# 预定义的映射
if num_classes == 2:
    class_tokens = ["no", "yes"]
elif num_classes == 5:
    class_tokens = ["strongly_disagree", "disagree", "neutral", "agree", "strongly_agree"]

# 获取 token ID
token_ids = [tokenizer.encode(token)[0] for token in class_tokens]

# 提取 logits
logits = [model_logits[tid] for tid in token_ids]
```

**问题**：
- ❌ 如果实际标签是 "Very frequently"，但映射中只有 "strongly_disagree"，会失败
- ❌ 无法处理不同数量的类别

---

### 方案 2：动态提取标签

```python
# 从 instruction 中提取标签
instruction = "Options: 1. Very frequently 2. Quite frequently 3. Not frequently 4. Not at all frequently"
labels = extract_label_info_from_instruction(instruction)
# 结果：["Very frequently", "Quite frequently", "Not frequently", "Not at all frequently"]

# 动态获取 token ID
token_ids = [tokenizer.encode(label)[0] for label in labels]

# 提取 logits
logits = [model_logits[tid] for tid in token_ids]
```

**优点**：
- ✅ 使用实际的标签文本
- ✅ 每个样本独立处理
- ✅ 支持任意标签名称和数量

---

## 📊 输出对比

### 方案 1：固定标签

```
📊 Overall Metrics:
   Accuracy:   0.6500
   Precision:  0.6800
   Recall:     0.6200
   F1:         0.6485

📊 Classification Report:
                          precision    recall  f1-score   support

strongly_disagree (0)       0.6500    0.7000    0.6742       100
        disagree (1)       0.7200    0.6800    0.6995       100
         neutral (2)       0.7000    0.7200    0.7099       100
           agree (3)       0.7500    0.7200    0.7347       100
  strongly_agree (4)       0.8000    0.8200    0.8099       100
```

---

### 方案 2：灵活标签

```
📊 Overall Metrics:
   Accuracy: 0.6500

📊 Accuracy by Number of Classes:
   4-class: 0.6800 (2500 samples)
   5-class: 0.6200 (1500 samples)
```

**说明**：
- 按类别数分组统计
- 更适合分析不同类型问题的表现

---

## 🎯 使用建议

### 对于 WVS 数据集

**推荐使用方案 2**（灵活标签）：

```bash
bash run_eval_base_llama_flexible.sh
```

**原因**：
1. ✅ WVS 数据集标签不统一
2. ✅ 自动适配不同的标签名称
3. ✅ 无需手动指定类别数

---

### 对于统一标签的数据集

**可以使用方案 1**（固定标签）：

```bash
# 二分类（CulturalBench）
bash run_eval_base_llama.sh

# 五分类（统一的 Likert 量表）
bash run_eval_base_llama_4class.sh
```

**原因**：
1. ✅ 标签统一，更简单
2. ✅ 可以生成详细的分类报告
3. ✅ 混淆矩阵更直观

---

## 🔧 如何选择

| 数据集特征 | 推荐方案 | 脚本 |
|-----------|---------|------|
| 标签数量固定 + 名称统一 | 方案 1 | `eval_base_llama.py` |
| 标签数量不固定 | 方案 2 | `eval_base_llama_flexible.py` |
| 标签名称不统一 | 方案 2 | `eval_base_llama_flexible.py` |
| **WVS 数据集** | **方案 2** | **`eval_base_llama_flexible.py`** |
| CulturalBench | 方案 1 | `eval_base_llama.py` |

---

## 📝 总结

### 方案 1（固定标签）

**优点**：
- ✅ 简单直接
- ✅ 详细的分类报告
- ✅ 混淆矩阵可视化

**缺点**：
- ❌ 不适用于标签不统一的数据集
- ❌ 需要预先知道类别数

---

### 方案 2（灵活标签）

**优点**：
- ✅ 自动适配不同标签
- ✅ 支持混合类别数
- ✅ 适用于 WVS 数据集

**缺点**：
- ⚠️ 输出相对简单（只有总体准确率和分组准确率）
- ⚠️ 需要 instruction 中包含标签信息

---

**对于你的 WVS 数据集，请使用方案 2（灵活标签）！** 🚀

