# eval_base_llama.py 自动推断 num_classes 修复

## ❌ 问题

运行 `sh run_eval_base_llama.sh qwen 22` 时，数据集是**二分类**（只有类别 0 和 1），但混淆矩阵显示 **22 个类别**（0-21），导致评估结果完全错误：

```
📊 Confusion Matrix:
           C0  C1  C2  C3  ... C21
Actual C0  481 1649 1164 48  ... 0
Actual C1  176 711  370  17  ... 0
Actual C2  0   0    0    0   ... 0
...
Actual C21 0   0    0    0   ... 0
```

## 🔍 问题分析

### 根本原因

和 `eval_base_gen.py` 一样，`22` 只是**数据集标识符**，不是实际的分类数量：

```bash
sh run_eval_base_llama.sh qwen 22
# 22 = CulturalBench_Hard_merge_rp.json（数据集标识符）
# 不是 num_classes=22
```

但之前的代码错误地将它当作 `num_classes` 传递：

```bash
# 错误的做法
python eval_base_llama.py --num_classes 22  # ❌ 22 不是分类数量！
```

### 数据集标识符说明

| 标识符 | 数据集文件 | 实际 num_classes |
|--------|-----------|------------------|
| 21 | CulturalBench_Hard_merge.json | 2（二分类） |
| 31 | normad_ed_merge.json | 3（三分类） |
| 41 | wvs_all_llama_merge_4.json | 4（四分类） |
| 51 | wvs_all_llama_merge_5.json | 5（五分类） |
| 22 | CulturalBench_Hard_merge_rp.json | 2（二分类） |
| 32 | normad_ed_merge_rp.json | 3（三分类） |
| 42 | wvs_all_llama_merge_4_rp.json | 4（四分类） |
| 52 | wvs_all_llama_merge_5_rp.json | 5（五分类） |

**说明**：
- 第一位数字：分类数量（2/3/4/5）
- 第二位数字：数据集类型（1=普通，2=role-play）
- 但这只是标识符，不是实际的 `num_classes`

## ✅ 修复方案

### 1. Shell 脚本改名

```bash
# 修复前
NUM_CLASSES="${2:-21}"  # 误导性

# 修复后
DATASET_ID="${2:-21}"   # 明确是数据集标识符
```

### 2. 不传递 num_classes

```bash
# 修复前
python eval_base_llama.py \
    --num_classes $NUM_CLASSES \  # ❌ 错误传递
    ...

# 修复后
python eval_base_llama.py \
    # 不传递 num_classes，让脚本自动推断
    ...
```

### 3. Python 脚本自动推断

```python
# 从数据中自动推断 num_classes
if args.num_classes is None:
    print("Auto-inferring num_classes from data...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    unique_labels = set(int(item['output']) for item in test_data)
    args.num_classes = max(unique_labels) + 1
    print(f"✅ Auto-inferred num_classes: {args.num_classes}")
    print(f"   Unique labels in data: {sorted(unique_labels)}")
else:
    print(f"✅ Using specified num_classes: {args.num_classes}")
```

## 🚀 立即使用

### 重新运行评估

```bash
sh run_eval_base_llama.sh qwen 22
```

### 预期输出

**修复前**：
```
Evaluating Base Qwen 2.5-7B-Instruct Model (22-class)  ❌
Num classes: 22

📊 Confusion Matrix:
           C0  C1  C2  ... C21  ← 22 个类别
Actual C0  481 1649 ...  0
Actual C1  176 711  ...  0
Actual C2  0   0    ...  0     ← 空类别
...
```

**修复后**：
```
Evaluating Base Qwen 2.5-7B-Instruct Model
Dataset ID: 22

Auto-inferring num_classes from data...
✅ Auto-inferred num_classes: 2  ← 正确！
   Unique labels in data: [0, 1]

📊 Confusion Matrix:
           C0  C1  ← 只有 2 个类别
Actual C0  481 1649
Actual C1  176 711

Classification Report:
           precision  recall  f1-score  support
no (0)     0.73      0.13    0.23      3582
yes (1)    0.30      0.54    0.39      1326
```

## 📊 修复前后对比

### 修复前

```bash
sh run_eval_base_llama.sh qwen 22

# Shell 传递
python eval_base_llama.py --num_classes 22  # ❌

# Python 使用
num_classes = 22  # 但数据只有 2 个类别

# 结果
混淆矩阵：22x22（大部分为空）
分类报告：22 个类别（20 个类别的 support=0）
准确率：错误（因为类别数不对）
```

### 修复后

```bash
sh run_eval_base_llama.sh qwen 22

# Shell 不传递 num_classes

# Python 自动推断
unique_labels = {0, 1}
num_classes = 2  # 自动推断

# 结果
混淆矩阵：2x2（正确）
分类报告：2 个类别（正确）
准确率：正确
```

## 🎯 所有数据集的正确 num_classes

| 数据集标识符 | 数据集文件 | 自动推断的 num_classes |
|-------------|-----------|----------------------|
| 21 | CulturalBench_Hard_merge.json | 2 |
| 31 | normad_ed_merge.json | 3 |
| 41 | wvs_all_llama_merge_4.json | 4 |
| 51 | wvs_all_llama_merge_5.json | 5 |
| 221 | wvs_2_merge.json | 2 |
| 2221 | wvs_2c_merge.json | 2 |
| 441 | wvs_4_merge.json | 4 |
| 22 | CulturalBench_Hard_merge_rp.json | 2 |
| 32 | normad_ed_merge_rp.json | 3 |
| 42 | wvs_all_llama_merge_4_rp.json | 4 |
| 52 | wvs_all_llama_merge_5_rp.json | 5 |
| 222 | wvs_2_merge_rp.json | 2 |
| 2222 | wvs_2c_merge_rp.json | 2 |
| 442 | wvs_4_merge_rp.json | 4 |

## ⚠️ 注意事项

### 1. 标签必须从 0 开始

如果标签从 1 开始（如 [1, 2]）：
```python
unique_labels = {1, 2}
num_classes = max(unique_labels) + 1  # 2 + 1 = 3 ❌ 错误！
```

解决方法：
- 手动指定 `--num_classes 2`
- 或修改数据，让标签从 0 开始

### 2. 标签必须连续

如果标签不连续（如 [0, 2]）：
```python
unique_labels = {0, 2}
num_classes = max(unique_labels) + 1  # 2 + 1 = 3
# 但实际只有 2 个类别
```

解决方法：
- 手动指定 `--num_classes 2`
- 或修改数据，让标签连续

### 3. 手动指定（如果需要）

如果自动推断不正确，可以手动指定：

```bash
python eval_base_llama.py \
    --model_path /path/to/model \
    --test_file /path/to/test.json \
    --num_classes 2  # 手动指定
```

## 🎉 总结

### 问题
- 数据集标识符（22）被错误地当作 `num_classes`
- 导致混淆矩阵和分类报告完全错误

### 修复
- ✅ Shell 脚本使用 `DATASET_ID` 而不是 `NUM_CLASSES`
- ✅ 不传递 `--num_classes` 参数
- ✅ Python 脚本从数据中自动推断
- ✅ 支持手动指定（如果需要）

### 验证
```bash
sh run_eval_base_llama.sh qwen 22
```

现在应该能看到正确的二分类评估结果了！🎉

