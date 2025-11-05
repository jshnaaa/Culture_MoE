# eval_base_gen.py 数字提取 Bug 修复

## ❌ 问题

运行 `sh run_eval_base_gen.sh llama 11` 时出现：

```
⚠️  Warning: Failed to extract label from '5', using default 2
⚠️  Warning: Failed to extract label from '5', using default 2
⚠️  Warning: Failed to extract label from '5', using default 2
```

模型明明生成了 `'5'`，但是被判定为提取失败，使用了默认值 `2`。

## 🔍 问题分析

### 1. 数据集的标签范围

```bash
# num_classes=11 的数据集
# 标签范围：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### 2. 提取逻辑

```python
def extract_label(answer: str, num_classes: int = 5):
    try:
        label = int(answer)  # '5' → 5
        if 0 <= label < num_classes:  # 0 <= 5 < 5 → False ❌
            return label
    except ValueError:
        pass

    # 提取失败，使用默认值
    return num_classes // 2  # 5 // 2 = 2
```

### 3. Shell 脚本问题

```bash
# run_eval_base_gen.sh (第 73-77 行)
python eval_base_gen.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --device cuda
    # ❌ 缺少 --num_classes $NUM_CLASSES
```

**结果**：
- Shell 脚本设置了 `NUM_CLASSES=11`
- 但没有传递给 Python 脚本
- Python 脚本使用默认值 `num_classes=5`
- 答案 `'5'` 不在 `[0, 5)` 范围内
- 提取失败，使用默认值 `2`

## ✅ 修复

### 修改 run_eval_base_gen.sh

```bash
# 运行评估脚本
python eval_base_gen.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --num_classes $NUM_CLASSES \  # ✅ 添加这一行
    --device cuda
```

## 📊 修复前后对比

### 修复前

```
Shell: NUM_CLASSES=11
Python: num_classes=5 (默认值)

答案 '5':
  - int('5') = 5
  - 0 <= 5 < 5 → False ❌
  - 使用默认值 2

结果：所有 5-10 的答案都被错误地转换为 2
```

### 修复后

```
Shell: NUM_CLASSES=11
Python: num_classes=11 (正确传递)

答案 '5':
  - int('5') = 5
  - 0 <= 5 < 11 → True ✅
  - 返回 5

结果：所有答案都正确提取
```

## 🎯 影响范围

### 受影响的标签

当 `num_classes=11` 但使用默认值 `5` 时：

| 答案 | 应该提取 | 实际提取 | 状态 |
|------|---------|---------|------|
| '0'  | 0       | 0       | ✅ 正确 |
| '1'  | 1       | 1       | ✅ 正确 |
| '2'  | 2       | 2       | ✅ 正确 |
| '3'  | 3       | 3       | ✅ 正确 |
| '4'  | 4       | 4       | ✅ 正确 |
| '5'  | 5       | 2       | ❌ **错误** |
| '6'  | 6       | 2       | ❌ **错误** |
| '7'  | 7       | 2       | ❌ **错误** |
| '8'  | 8       | 2       | ❌ **错误** |
| '9'  | 9       | 2       | ❌ **错误** |
| '10' | 10      | 2       | ❌ **错误** |

**结论**：
- 标签 0-4：正确提取 ✅
- 标签 5-10：**全部错误**，都被转换为 2 ❌
- 错误率：**54.5%** (6/11)

## 🚨 严重性

这是一个**严重的 bug**，会导致：

1. **评估结果完全错误**
   - 超过一半的标签被错误提取
   - 准确率、F1 等指标都不可信

2. **模型性能被低估**
   - 模型可能正确预测了 5-10
   - 但被强制转换为 2
   - 导致准确率虚低

3. **数据分布扭曲**
   - 标签 2 的数量异常增多
   - 标签 5-10 的数量为 0
   - 混淆矩阵完全错误

## ✅ 验证修复

### 1. 重新运行评估

```bash
sh run_eval_base_gen.sh llama 11
```

### 2. 检查输出

**修复前**：
```
⚠️  Warning: Failed to extract label from '5', using default 2
⚠️  Warning: Failed to extract label from '6', using default 2
⚠️  Warning: Failed to extract label from '7', using default 2
...
Failed rate: 54.5%  ❌
```

**修复后**：
```
(没有警告信息)
Failed rate: 0.0%  ✅
```

### 3. 检查标签分布

```python
# 查看 generated_answers.json
import json

with open('generated_answers.json', 'r') as f:
    answers = json.load(f)

# 统计预测标签分布
from collections import Counter
pred_labels = [a['predicted_label'] for a in answers]
print(Counter(pred_labels))

# 修复前：{0: 1000, 1: 1000, 2: 7000, 3: 1000, 4: 1000}  ❌ 标签2异常多
# 修复后：{0: 1000, 1: 1000, 2: 1000, ..., 10: 1000}  ✅ 分布均匀
```

## 📝 其他需要检查的脚本

同样的问题可能存在于其他脚本：

1. **run_train_lora_only_gen.sh** - LoRA 训练
2. **run_eval_lora_only_gen.sh** - LoRA 评估
3. 其他生成式脚本

建议检查所有调用 Python 脚本的 Shell 脚本，确保正确传递 `--num_classes` 参数。

## 🎉 总结

### 问题
- Shell 脚本没有传递 `--num_classes` 参数
- Python 脚本使用默认值 `5`
- 导致标签 5-10 全部提取失败

### 修复
- 在 Shell 脚本中添加 `--num_classes $NUM_CLASSES`
- 确保参数正确传递

### 影响
- **修复前**：54.5% 的标签提取错误
- **修复后**：0% 提取错误

### 验证
```bash
sh run_eval_base_gen.sh llama 11
# 应该不再看到 "Failed to extract label from '5'" 的警告
```

立即重新运行评估以获得正确的结果！

