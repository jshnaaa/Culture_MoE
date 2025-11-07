# 修复：从生成文本中提取数字

## 问题

生成的答案包含很多无关内容，而不是简单的数字：

```json
{
  "predicted": "2._REF_1_2_",
  "true": "1"
},
{
  "predicted": "1.İsim:Do you agree with One",
  "true": "1"
},
{
  "predicted": "3.5 →2.5 →2 →",
  "true": "2"
}
```

## 原因

1. 模型生成了额外的文本
2. 没有对生成的文本进行后处理

## 解决方案

在 `generate_and_evaluate()` 函数中添加数字提取逻辑：

```python
# 对于数字类型，只提取第一个数字
if output_type == "number":
    # 尝试从生成的文本中提取第一个数字
    import re
    numbers = re.findall(r'\d+', generated_text)
    if numbers:
        generated_text = numbers[0]
    else:
        # 如果没有找到数字，保留原始文本
        pass
```

## 效果

修复后，`generated_answers.json` 应该变成：

```json
{
  "predicted": "2",
  "true": "1"
},
{
  "predicted": "1",
  "true": "1"
},
{
  "predicted": "3",
  "true": "2"
}
```

## 其他改进

同时添加了：

1. **禁用 temperature 和 top_p**：
   ```python
   temperature=None,  # 禁用 temperature
   top_p=None,  # 禁用 top_p
   ```
   这样可以避免警告信息。

2. **正则表达式提取数字**：
   - 使用 `re.findall(r'\d+', generated_text)` 提取所有数字
   - 只取第一个数字作为答案

## 适用场景

这个修复适用于：
- ✅ CultureLLM 数据集（答案是 1-4）
- ✅ 任何数字分类任务
- ✅ 多选题（答案是数字）

对于文本类型（yes/no/neutral），不会进行数字提取。

## 测试

重新运行训练：

```bash
sh run_train_lora_only_gen.sh llama 4
```

现在 `generated_answers.json` 中的 `predicted` 字段应该只包含数字了！

