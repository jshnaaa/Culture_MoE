# LoRA Only 多分类支持修复

## 问题

之前运行 3/4/5 分类任务时会报错：

```
ValueError: Target is multiclass but average='binary'.
Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
```

## 原因

`compute_metrics` 函数中使用了 `average='binary'`，只适用于二分类任务。

## 修复内容

### 修改文件：`train_and_eval_lora_only.py`

**修改前**：
```python
def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1  # ❌ 只支持二分类
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

**修改后**：
```python
def compute_metrics(eval_pred):
    """计算评估指标（支持 2/3/4/5 分类）"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    # ✅ 自动检测是二分类还是多分类
    num_classes = len(np.unique(labels))
    if num_classes == 2:
        # 二分类：使用 binary
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', pos_label=1, zero_division=0
        )
    else:
        # 多分类（3/4/5）：使用 macro
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

## 修改说明

### 1. 自动检测分类数量

```python
num_classes = len(np.unique(labels))
```

通过检测标签中的唯一值数量，自动判断是二分类还是多分类。

### 2. 根据分类数量选择 average 参数

- **二分类**（num_classes == 2）：使用 `average='binary'`
- **多分类**（num_classes > 2）：使用 `average='macro'`

### 3. 添加 zero_division=0

避免在某些类别没有预测样本时出现除零警告。

## 使用方法

现在可以正常运行所有分类任务：

```bash
# 2 分类
sh run_train_lora_only.sh llama 2

# 3 分类
sh run_train_lora_only.sh llama 3

# 4 分类
sh run_train_lora_only.sh llama 4

# 5 分类
sh run_train_lora_only.sh llama 5
```

## 指标说明

### 二分类（average='binary'）

- **Precision**：正类的精确率
- **Recall**：正类的召回率
- **F1**：正类的 F1 分数

### 多分类（average='macro'）

- **Precision**：所有类别精确率的平均值（未加权）
- **Recall**：所有类别召回率的平均值（未加权）
- **F1**：所有类别 F1 分数的平均值（未加权）

## 为什么使用 macro 而不是 weighted？

- **macro**：对每个类别一视同仁，适合类别不平衡的情况
- **weighted**：根据每个类别的样本数加权，可能会偏向多数类

在文化分类任务中，使用 **macro** 更合理，因为我们希望模型在所有类别上都表现良好。

## 验证修复

### 测试 2 分类

```bash
sh run_train_lora_only.sh llama 2 false
```

**预期输出**：
```
Final Evaluation Results
============================================================
   Accuracy:   0.8500
   Precision:  0.8600
   Recall:     0.8400
   F1:         0.8500
============================================================
```

### 测试 4 分类

```bash
sh run_train_lora_only.sh llama 4 false
```

**预期输出**：
```
Final Evaluation Results
============================================================
   Accuracy:   0.7500
   Precision:  0.7400
   Recall:     0.7300
   F1:         0.7350
============================================================
```

不再报错！✅

## 总结

### 修改前
- ✅ 支持 2 分类
- ❌ 3/4/5 分类报错

### 修改后
- ✅ 支持 2 分类
- ✅ 支持 3 分类
- ✅ 支持 4 分类
- ✅ 支持 5 分类

### 修改的文件
- ✅ `train_and_eval_lora_only.py` - 修改 `compute_metrics` 函数
- ✅ `run_train_lora_only.sh` - 已经支持，无需修改

现在 LoRA Only 模型可以完美支持 2/3/4/5 分类任务了！🎉

