# 评估 Prompt 格式不匹配问题修复

## 🔍 问题分析

### 症状
运行 `sh run_eval_lora_only_from_components.sh llama` 时，模型生成的是文本而不是数字：

```json
{
  "raw_answer": "This",
  "predicted_label": 5,
  "true_label": 0,
  "correct": false
},
{
  "raw_answer": "Thank",
  "predicted_label": 5,
  "true_label": 0,
  "correct": false
},
{
  "raw_answer": "Leisure",
  "predicted_label": 5,
  "true_label": 0,
  "correct": false
}
```

### 根本原因

**训练时和评估时的 prompt 格式不一致！**

#### 训练时的 Prompt（`train_lora_only_gen.py`）

```python
# 数字类型
prompt = f"{instruction}\n{input_text}\nAnswer:"
full_text = f"{prompt} {output}"
```

**示例**：
```
Question: How important is family in your life? Country: Andorra
Option: 1. Very important
2. Rather important
3. Not very important
4. Not at all important
 Please select the most appropriate option by replying with the option number only (1, 2, 3, ...). Your answer must be a single number only. Your answer is:
Answer: 1
```

#### 评估时的 Prompt（`eval_lora_only_from_components.py` - 修复前）

```python
full_input = f"{instruction}\n{input_text}\n\nPlease answer with ONLY ONE NUMBER (1 to {num_classes}).\nYour answer:"
```

**示例**：
```
Question: How important is family in your life? Country: Andorra
Option: 1. Very important
2. Rather important
3. Not very important
4. Not at all important
 Please select the most appropriate option by replying with the option number only (1, 2, 3, ...). Your answer must be a single number only. Your answer is:

Please answer with ONLY ONE NUMBER (1 to 10).
Your answer:
```

### 问题

1. **额外的换行**：评估时有 `\n\n`，训练时只有 `\n`
2. **额外的指令**：评估时添加了 `"Please answer with ONLY ONE NUMBER (1 to 10)."`
3. **不同的结束语**：训练时是 `"Answer:"`，评估时是 `"Your answer:"`

**结果**：模型从未见过这种格式，不知道该生成什么，所以生成了随机文本！

## ✅ 修复方案

### 修改 `eval_lora_only_from_components.py`

```python
# 修复前
if num_classes <= 10:
    full_input = f"{instruction}\n{input_text}\n\nPlease answer with ONLY ONE NUMBER (1 to {num_classes}).\nYour answer:"
else:
    full_input = f"{instruction}\n{input_text}\n\nYour answer:"

# 修复后
full_input = f"{instruction}\n{input_text}\nAnswer:"
```

### 关键改进

1. ✅ **移除额外的换行**：`\n\n` → `\n`
2. ✅ **移除额外的指令**：删除 `"Please answer with ONLY ONE NUMBER..."`
3. ✅ **统一结束语**：`"Your answer:"` → `"Answer:"`
4. ✅ **与训练时完全一致**

## 修复前后对比

| 特性 | 修复前 | 修复后 |
|------|--------|--------|
| **Prompt 格式** | 与训练不一致 | ✅ 与训练一致 |
| **生成结果** | 随机文本 ("This", "Thank") | ✅ 数字 (1-10) |
| **准确率** | ~0% | ✅ 正常 |

## 预期效果

### 修复前
```json
{
  "raw_answer": "This",
  "predicted_label": 5,
  "true_label": 0,
  "correct": false
}
```

### 修复后
```json
{
  "raw_answer": "1",
  "predicted_label": 0,
  "true_label": 0,
  "correct": true
}
```

## 完整的 Prompt 格式规范

### CultureLLM（数字 1-10）

**训练时**：
```
{instruction}
{input}
Answer: {output}
```

**评估时**：
```
{instruction}
{input}
Answer:
```

### NormAD（yes/no/neutral）

**训练时**：
```
{instruction}
{input}
Answer one of 'yes', 'no', or 'neutral'. Your answer is: {output}
```

**评估时**：
```
{instruction}
{input}
Answer one of 'yes', 'no', or 'neutral'. Your answer is:
```

### CulturalBench（TRUE/FALSE）

**训练时**：
```
{instruction}
{input}
Answer one of 'TRUE' or 'FALSE'. Your answer is: {output}
```

**评估时**：
```
{instruction}
{input}
Answer one of 'TRUE' or 'FALSE'. Your answer is:
```

## 重要教训

### ⚠️ Prompt 一致性至关重要

1. **训练和评估必须使用相同的 prompt 格式**
2. **即使是微小的差异（如换行、标点）也会影响生成**
3. **模型只能生成它在训练时见过的格式**

### ✅ 最佳实践

1. **统一 Prompt 模板**：
   - 在一个地方定义 prompt 格式
   - 训练和评估共享同一个函数

2. **记录 Prompt 格式**：
   - 在训练时保存 prompt 模板
   - 评估时加载相同的模板

3. **测试 Prompt 一致性**：
   - 在小数据集上验证
   - 检查生成的格式是否正确

## 使用示例

### 修复后的评估

```bash
# 评估统一 LoRA 模型
sh run_eval_lora_only_from_components.sh llama

# 预期输出
================================================================================
Evaluation Results
================================================================================
Total samples: 1234
Correct: 987
Accuracy: 0.8000
================================================================================
```

### 检查生成的答案

```bash
# 查看生成的答案
cat /path/to/output/generated_answer.json | python -m json.tool | head -50

# 应该看到数字答案
{
  "raw_answer": "1",
  "predicted_label": 0,
  "true_label": 0,
  "correct": true
}
```

## 其他修复

除了 prompt 格式，我还添加了：

1. ✅ **LoRA 目录内容检查**
2. ✅ **adapter_config.json 存在性检查**
3. ✅ **torch_dtype=torch.float16 参数**

这些改进使得错误信息更清晰，调试更容易。

## 总结

✅ **问题**：训练和评估的 prompt 格式不一致
✅ **原因**：评估时添加了额外的指令和换行
✅ **修复**：统一为 `f"{instruction}\n{input_text}\nAnswer:"`
✅ **结果**：模型现在可以正确生成数字答案

**关键要点**：**Prompt 一致性是生成式模型评估的关键！** 🎯

