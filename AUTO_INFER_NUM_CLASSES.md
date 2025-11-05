# 自动推断 num_classes 修复

## ❌ 之前的问题

Shell 脚本中的 `11`、`21`、`12`、`22` 只是**数据集标识符**，不是实际的分类数量：

```bash
sh run_eval_base_gen.sh llama 11
# 11 = cultureLLM_merge_gen.json（数据集标识符）
# 不是 num_classes=11
```

但之前的代码错误地将它当作 `num_classes` 传递给 Python 脚本：

```bash
# 错误的做法
python eval_base_gen.py --num_classes 11  # ❌ 11 不是分类数量！
```

导致：
- 模型生成 `'5'`
- 检查 `0 <= 5 < 11` → True
- 但实际数据可能只有 5 个类别 [0, 1, 2, 3, 4]
- 或者有其他数量的类别

## ✅ 修复方案

### 1. Shell 脚本改名

```bash
# 修复前
NUM_CLASSES="${2:-11}"  # 误导性的变量名

# 修复后
DATASET_ID="${2:-11}"   # 明确这只是数据集标识符
```

### 2. 不传递 num_classes

```bash
# 修复前
python eval_base_gen.py \
    --num_classes $NUM_CLASSES \  # ❌ 错误传递
    ...

# 修复后
python eval_base_gen.py \
    # 不传递 num_classes，让脚本自动推断
    ...
```

### 3. Python 脚本自动推断

```python
# 从数据中自动推断 num_classes
unique_labels = set(int(item['output']) for item in test_data)
inferred_num_classes = max(unique_labels) + 1  # 假设标签从 0 开始

# 如果用户指定了 num_classes，使用用户指定的；否则使用推断的
if args.num_classes is None:
    args.num_classes = inferred_num_classes
    print(f"✅ Auto-inferred num_classes: {args.num_classes}")
else:
    print(f"✅ Using specified num_classes: {args.num_classes}")

print(f"   Unique labels in data: {sorted(unique_labels)}")
```

## 📊 数据集标识符说明

| 标识符 | 数据集文件 | 实际 num_classes |
|--------|-----------|------------------|
| 11 | cultureLLM_merge_gen.json | 自动推断（如 5） |
| 21 | wvs_merge_gen.json | 自动推断（如 4） |
| 12 | cultureLLM_merge_rp_gen.json | 自动推断（如 5） |
| 22 | wvs_merge_rp_gen.json | 自动推断（如 4） |

**说明**：
- `11`、`21` 等只是用来区分不同的数据集
- 不代表实际的分类数量
- 实际的 `num_classes` 从数据文件中自动推断

## 🎯 自动推断逻辑

```python
# 示例数据
test_data = [
    {"instruction": "...", "input": "...", "output": 0},
    {"instruction": "...", "input": "...", "output": 1},
    {"instruction": "...", "input": "...", "output": 2},
    {"instruction": "...", "input": "...", "output": 3},
    {"instruction": "...", "input": "...", "output": 4},
]

# 提取所有唯一标签
unique_labels = {0, 1, 2, 3, 4}

# 推断 num_classes（假设标签从 0 开始）
num_classes = max(unique_labels) + 1  # 4 + 1 = 5
```

**假设**：
- 标签从 0 开始
- 标签连续（0, 1, 2, ..., n-1）

**如果标签不连续**（如 [0, 2, 4]）：
- 推断的 `num_classes = 5`（0 到 4）
- 但实际只有 3 个类别
- 这种情况下，用户需要手动指定 `--num_classes 3`

## 🚀 使用方法

### 自动推断（推荐）

```bash
# 不指定 num_classes，自动从数据推断
sh run_eval_base_gen.sh llama 11
```

输出：
```
✅ Loaded 11001 test samples
✅ Auto-inferred num_classes: 5
   Unique labels in data: [0, 1, 2, 3, 4]
```

### 手动指定

```bash
# 如果需要手动指定
python eval_base_gen.py \
    --model_path /path/to/model \
    --test_file /path/to/test.json \
    --output_dir /path/to/output \
    --num_classes 5  # 手动指定
```

输出：
```
✅ Loaded 11001 test samples
✅ Using specified num_classes: 5
   Unique labels in data: [0, 1, 2, 3, 4]
```

## 📝 修复前后对比

### 修复前

```bash
sh run_eval_base_gen.sh llama 11

# Shell 传递
python eval_base_gen.py --num_classes 11  # ❌ 错误！

# Python 使用
num_classes = 11  # 但数据只有 5 个类别

# 结果
答案 '5': 0 <= 5 < 11 → True ✅（但实际应该是 False）
答案 '6': 0 <= 6 < 11 → True ✅（但实际应该是 False）
...
```

### 修复后

```bash
sh run_eval_base_gen.sh llama 11

# Shell 不传递 num_classes

# Python 自动推断
unique_labels = {0, 1, 2, 3, 4}
num_classes = 5  # 自动推断

# 结果
答案 '0': 0 <= 0 < 5 → True ✅
答案 '4': 0 <= 4 < 5 → True ✅
答案 '5': 0 <= 5 < 5 → False ❌（正确！）
```

## ⚠️ 注意事项

### 1. 标签必须从 0 开始

如果标签从 1 开始（如 [1, 2, 3, 4, 5]）：
```python
unique_labels = {1, 2, 3, 4, 5}
num_classes = max(unique_labels) + 1  # 5 + 1 = 6 ❌ 错误！
```

解决方法：
- 手动指定 `--num_classes 5`
- 或修改推断逻辑：`num_classes = len(unique_labels)`

### 2. 标签不连续

如果标签不连续（如 [0, 2, 4]）：
```python
unique_labels = {0, 2, 4}
num_classes = max(unique_labels) + 1  # 4 + 1 = 5
# 但实际只有 3 个类别
```

解决方法：
- 手动指定 `--num_classes 3`
- 或使用 `num_classes = len(unique_labels)`

### 3. 检查推断结果

运行时会打印推断的 `num_classes`：
```
✅ Auto-inferred num_classes: 5
   Unique labels in data: [0, 1, 2, 3, 4]
```

如果不正确，手动指定：
```bash
python eval_base_gen.py --num_classes 5
```

## 🎉 总结

### 问题
- 数据集标识符（11, 21）被错误地当作 `num_classes`
- 导致标签提取错误

### 修复
- Shell 脚本不再传递 `num_classes`
- Python 脚本从数据中自动推断
- 支持手动指定（如果需要）

### 优势
- ✅ 自动适应不同数据集
- ✅ 避免手动配置错误
- ✅ 支持手动覆盖
- ✅ 打印推断结果供验证

立即使用：
```bash
sh run_eval_base_gen.sh llama 11

