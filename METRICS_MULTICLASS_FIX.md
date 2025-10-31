# CultureMoE 多分类指标修复

## 问题

运行 4 分类或 5 分类训练时报错：

```
ValueError: Number of classes, 4, does not match size of target_names, 3.
Try specifying the labels parameter
```

## 原因

`src/llamafactory/train/classification/metrics.py` 中的 `compute_classification_metrics` 函数只支持 2 和 3 分类，不支持 4 和 5 分类。

## 修复内容

### 1. 支持 2/3/4/5 分类

**修改前**：
```python
if num_classes == 2:
    class_labels = [0, 1]
    target_names = ["no (0)", "yes (1)"]
else:  # num_classes == 3
    class_labels = [0, 1, 2]
    target_names = ["no (0)", "neutral (1)", "yes (2)"]
```

**修改后**：
```python
class_labels = list(range(num_classes))

if num_classes == 2:
    target_names = ["no (0)", "yes (1)"]
elif num_classes == 3:
    target_names = ["no (0)", "neutral (1)", "yes (2)"]
elif num_classes == 4:
    target_names = ["strongly_disagree (0)", "disagree (1)", "agree (2)", "strongly_agree (3)"]
elif num_classes == 5:
    target_names = ["strongly_disagree (0)", "disagree (1)", "neutral (2)", "agree (3)", "strongly_agree (4)"]
else:
    target_names = [f"class_{i} ({i})" for i in range(num_classes)]
```

### 2. 添加 labels 参数

**修改前**：
```python
print(classification_report(
    labels, preds,
    target_names=target_names,
    digits=4,
    zero_division=0
))
```

**修改后**：
```python
print(classification_report(
    labels, preds,
    labels=class_labels,  # ✅ 显式指定所有类别标签
    target_names=target_names,
    digits=4,
    zero_division=0
))
```

### 3. 支持 4/5 分类的混淆矩阵打印

**新增**：
```python
elif num_classes == 4:
    print("              SD   D    A    SA")
    for i, row in enumerate(cm):
        label_name = label_names[i][:4]
        print(f"Actual {label_name:4s} {row[0]:3d}  {row[1]:3d}  {row[2]:3d}  {row[3]:3d}")
elif num_classes == 5:
    print("              SD   D    N    A    SA")
    for i, row in enumerate(cm):
        label_name = label_names[i][:4]
        print(f"Actual {label_name:4s} {row[0]:3d}  {row[1]:3d}  {row[2]:3d}  {row[3]:3d}  {row[4]:3d}")
```

### 4. 通用的指标返回

**修改前**：
```python
# 硬编码每个类别的指标名称
if num_classes == 2:
    result.update({
        "precision_no": precision[0],
        "precision_yes": precision[1],
        ...
    })
else:  # num_classes == 3
    result.update({
        "precision_no": precision[0],
        "precision_neutral": precision[1],
        "precision_yes": precision[2],
        ...
    })
```

**修改后**：
```python
# 通用方式，支持任意分类数
for i in range(num_classes):
    if i < len(precision):
        result[f"precision_class_{i}"] = precision[i]
        result[f"recall_class_{i}"] = recall[i]
        result[f"f1_class_{i}"] = f1[i]
```

### 5. 添加数据集类别检查

**新增**：
```python
# 检查数据集中实际的类别数
actual_num_classes = len(np.unique(labels))
if actual_num_classes != num_classes:
    print(f"\n⚠️  Warning: Expected {num_classes} classes, but found {actual_num_classes} unique classes in the dataset")
    print(f"   Unique labels in data: {sorted(np.unique(labels).tolist())}")
    print(f"   Missing classes: {sorted(set(range(num_classes)) - set(np.unique(labels).tolist()))}")
```

## 使用方法

现在可以正常运行所有分类任务：

```bash
# 2 分类
sh run_train_ddp_lora_dual.sh llama 2 True false

# 3 分类
sh run_train_ddp_lora_dual.sh llama 3 True false

# 4 分类 ✅ 现在可以正常运行
sh run_train_ddp_lora_dual.sh llama 4 True false

# 5 分类 ✅ 现在可以正常运行
sh run_train_ddp_lora_dual.sh llama 5 True false
```

## 输出示例

### 4 分类

```
==================================================
Classification Report:
==================================================
                          precision    recall  f1-score   support

strongly_disagree (0)       0.7500    0.7200    0.7347       100
         disagree (1)       0.7800    0.7600    0.7699       120
            agree (2)       0.7600    0.7800    0.7699       110
   strongly_agree (3)       0.7900    0.8000    0.7950       90

             accuracy                           0.7675       420
            macro avg       0.7700    0.7650    0.7674       420
         weighted avg       0.7680    0.7675    0.7677       420

Confusion Matrix:
              SD   D    A    SA
Actual stro   72   15   10    3
Actual disa   12   91   15    2
Actual agre    8   12   86    4
Actual stro    2    5   11   72
==================================================
```

### 5 分类

```
==================================================
Classification Report:
==================================================
                          precision    recall  f1-score   support

strongly_disagree (0)       0.7200    0.7000    0.7099       100
         disagree (1)       0.7500    0.7300    0.7399       120
          neutral (2)       0.7100    0.7200    0.7150       90
            agree (3)       0.7600    0.7800    0.7699       110
   strongly_agree (4)       0.7800    0.7900    0.7850       80

             accuracy                           0.7440       500
            macro avg       0.7440    0.7440    0.7439       500
         weighted avg       0.7445    0.7440    0.7442       500

Confusion Matrix:
              SD   D    N    A    SA
Actual stro   70   18    8    3    1
Actual disa   15   88   12    4    1
Actual neut    8   10   65    6    1
Actual agre    4    5    8   86    7
Actual stro    1    2    3    6   68
==================================================
```

## 返回的指标

### 主要指标（所有分类任务）

```python
{
    "accuracy": 0.7675,
    "precision": 0.7700,  # macro 平均
    "recall": 0.7650,     # macro 平均
    "f1": 0.7674,         # macro 平均
    "precision_macro": 0.7700,
    "recall_macro": 0.7650,
    "f1_macro": 0.7674,
    "precision_weighted": 0.7680,
    "recall_weighted": 0.7675,
    "f1_weighted": 0.7677,
}
```

### 每个类别的指标

```python
{
    # 类别 0
    "precision_class_0": 0.7500,
    "recall_class_0": 0.7200,
    "f1_class_0": 0.7347,

    # 类别 1
    "precision_class_1": 0.7800,
    "recall_class_1": 0.7600,
    "f1_class_1": 0.7699,

    # ... 其他类别
}
```

## 数据集类别不平衡警告

如果数据集中某些类别缺失（如 5 分类数据集实际只有 4 个类别），会显示警告：

```
⚠️  Warning: Expected 5 classes, but found 4 unique classes in the dataset
   Unique labels in data: [0, 1, 2, 3]
   Missing classes: [4]
```

这是正常的，指标计算会使用 `zero_division=0` 来处理缺失类别。

## 总结

### 修复前
- ✅ 支持 2 分类
- ✅ 支持 3 分类
- ❌ 4 分类报错
- ❌ 5 分类报错

### 修复后
- ✅ 支持 2 分类
- ✅ 支持 3 分类
- ✅ 支持 4 分类
- ✅ 支持 5 分类
- ✅ 支持任意分类数（通用实现）

现在可以正常运行所有分类任务了！🎉

