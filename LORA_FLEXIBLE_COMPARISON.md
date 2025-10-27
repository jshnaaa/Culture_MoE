## 📊 LoRA Only 训练脚本对比

### 问题背景

WVS 数据集的标签**不统一**：
- **类别数不同**：有的是 4 分类，有的是 5 分类
- **标签名称不同**：
  - `1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree`
  - `1. Very frequently 2. Quite frequently 3. Not frequently 4. Not at all frequently`
  - `1. Definitely should have the right 2. probably should have the right ...`
  - `1. Strongly agree 2. agree 3. Neither agree nor disagree 4. Disagree 5. Disagree strongly`

---

## 🔧 两种训练方案

### 方案 1：固定标签（`train_and_eval_lora_only.py`）

**适用场景**：
- ✅ 标签数量固定（如全部是 2 分类）
- ✅ 标签名称统一

**限制**：
- ❌ **不适用于 WVS 数据集**（标签不统一）
- ❌ 固定为 2 分类
- ❌ 使用 binary 平均计算指标

**使用方法**：
```bash
bash run_train_lora_only_ddp.sh
```

---

### 方案 2：灵活标签（`train_and_eval_lora_only_flexible.py`）✅ 推荐

**适用场景**：
- ✅ **标签数量不固定**（同一数据集中有 4 分类和 5 分类）
- ✅ **标签名称不统一**（不同问题有不同的标签文本）
- ✅ **WVS 数据集**

**特点**：
- ✅ 自动从 `input` 或 `instruction` 中提取标签选项
- ✅ 自动确定主要的类别数（使用最常见的类别数）
- ✅ 使用 macro 平均计算指标（适用于多分类）
- ✅ 支持任意标签名称

**使用方法**：
```bash
bash run_train_lora_only_flexible.sh
```

---

## 📝 数据格式要求

### 方案 1（固定标签）

```json
{
  "instruction": "Question: Do you agree?",
  "input": "...",
  "output": 0
}
```

**要求**：
- `output` 必须是 0 或 1（二分类）
- 所有样本的类别数必须相同

---

### 方案 2（灵活标签）

```json
{
  "instruction": "Question: ...",
  "input": "1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree. You can only choose one option.",
  "output": 0
}
```

**要求**：
- `input` 或 `instruction` 中必须包含标签选项（格式：`1. 标签1 2. 标签2 ...`）
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

### 方案 1：固定 2 分类

```python
class BinaryClassificationModel(torch.nn.Module):
    def __init__(self, llama_model, num_classes=2):
        # 固定为 2 分类
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.config.hidden_size),
            torch.nn.Linear(self.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 2)  # 固定输出 2 个类别
        )

def compute_metrics(eval_pred):
    # 使用 binary 平均
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )
```

**问题**：
- ❌ 无法处理 4 分类或 5 分类数据
- ❌ 指标计算不适用于多分类

---

### 方案 2：动态多分类

```python
# 1. 自动提取标签
instruction = "Options: 1. Very frequently 2. Quite frequently 3. Not frequently 4. Not at all frequently"
labels = extract_label_info_from_text(instruction)
# 结果：["Very frequently", "Quite frequently", "Not frequently", "Not at all frequently"]
num_classes = len(labels)  # 4

# 2. 动态创建分类模型
class FlexibleClassificationModel(torch.nn.Module):
    def __init__(self, llama_model, num_classes=4):
        # 动态设置类别数
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.config.hidden_size),
            torch.nn.Linear(self.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_classes)  # 动态输出 num_classes 个类别
        )

# 3. 使用 macro 平均
def compute_metrics(eval_pred):
    # 使用 macro 平均（适用于多分类）
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
```

**优点**：
- ✅ 自动适配不同的类别数
- ✅ 使用实际的标签文本
- ✅ 指标计算适用于多分类

---

## 📊 输出对比

### 方案 1：固定标签

```
2. Loading data from /root/autodl-fs/CulturalBench_Hard_merge.json...
   Data format: instruction/input/output
Loading data from /root/autodl-fs/CulturalBench_Hard_merge.json...
Loaded 1000 samples
Train: 900, Val: 100
   ✅ Train dataset size: 900
   ✅ Validation dataset size: 100

5. Creating classification model...
   ✅ Classification model created (2 classes, using float32)

Final Evaluation Results
============================================================
   Accuracy:   0.8500
   Precision:  0.8600
   Recall:     0.8400
   F1:         0.8499
============================================================
```

---

### 方案 2：灵活标签

```
2. Loading data from /root/autodl-fs/wvs_all_llama_merge.json...
   Data format: instruction/input/output (flexible labels)
Loading data from /root/autodl-fs/wvs_all_llama_merge.json...
Loaded 4000 samples

Label distribution:
  4-class: 2500 samples (62.5%)
  5-class: 1500 samples (37.5%)

Using 4 classes for model (most common)
Train: 3600, Val: 400
   ✅ Train dataset size: 3600
   ✅ Validation dataset size: 400
   ✅ Number of classes: 4

5. Creating classification model...
   ✅ Classification model created (4 classes, using float32)

Final Evaluation Results
============================================================
   Accuracy:   0.6500
   Precision:  0.6800
   Recall:     0.6200
   F1:         0.6485
============================================================
```

**说明**：
- 自动检测标签分布
- 使用最常见的类别数（4 分类）
- 使用 macro 平均计算指标

---

## 🎯 使用建议

### 对于 WVS 数据集

**推荐使用方案 2**（灵活标签）：

```bash
bash run_train_lora_only_flexible.sh
```

**原因**：
1. ✅ WVS 数据集标签不统一
2. ✅ 自动适配不同的标签名称和类别数
3. ✅ 使用 macro 平均，更适合多分类

---

### 对于统一标签的数据集

**可以使用方案 1**（固定标签）：

```bash
bash run_train_lora_only_ddp.sh
```

**原因**：
1. ✅ 标签统一，更简单
2. ✅ 二分类任务
3. ✅ 使用 binary 平均，更精确

---

## 🔧 如何选择

| 数据集特征 | 推荐方案 | 脚本 |
|-----------|---------|------|
| 标签数量固定 + 二分类 | 方案 1 | `train_and_eval_lora_only.py` |
| 标签数量不固定 | 方案 2 | `train_and_eval_lora_only_flexible.py` |
| 标签名称不统一 | 方案 2 | `train_and_eval_lora_only_flexible.py` |
| **WVS 数据集** | **方案 2** | **`train_and_eval_lora_only_flexible.py`** |
| CulturalBench（二分类） | 方案 1 | `train_and_eval_lora_only.py` |

---

## 📝 总结

### 方案 1（固定标签）

**优点**：
- ✅ 简单直接
- ✅ 适用于二分类任务
- ✅ 使用 binary 平均，更精确

**缺点**：
- ❌ 不适用于标签不统一的数据集
- ❌ 固定为 2 分类

---

### 方案 2（灵活标签）

**优点**：
- ✅ 自动适配不同标签
- ✅ 支持混合类别数
- ✅ 适用于 WVS 数据集
- ✅ 使用 macro 平均，适合多分类

**缺点**：
- ⚠️ 需要 input 或 instruction 中包含标签信息
- ⚠️ 使用最常见的类别数（可能不适合所有样本）

---

## 🚀 快速开始

### 训练 WVS 数据集

```bash
# 使用灵活标签版本
bash run_train_lora_only_flexible.sh
```

### 训练 CulturalBench（二分类）

```bash
# 使用固定标签版本
bash run_train_lora_only_ddp.sh
```

---

**对于你的 WVS 数据集，请使用方案 2（灵活标签）！** 🚀

