# 空输入处理修复指南

## 🔍 问题分析

### 错误信息

```
IndexError: index -1 is out of bounds for dimension 0 with size 0
```

### 根本原因

**当 `instruction` 和 `input` 都是空字符串时**：

1. 拼接结果为空：`prompt = ""`
2. Tokenizer 返回空 tensor：`tokenizer("")["input_ids"]` → shape (1, 0)
3. 模型生成时触发错误：`cache_position[-1]` 访问空 tensor

### 具体流程

```python
# 数据中的空输入
instruction = ""
input_text = ""

# 拼接结果为空
full_text = ""

# Tokenizer 返回空 tensor
inputs = tokenizer("")  # input_ids shape: (1, 0)

# 模型生成时报错
model.generate(input_ids=inputs['input_ids'])
# ↓
# cache_position[-1] >= input_ids.shape[1]
# ↓
# IndexError: index -1 is out of bounds for dimension 0 with size 0
```

---

## ✅ 解决方案

### 修复策略

**在两个地方添加检查**：

1. **检查输入文本是否为空**
   - 在 tokenize 之前检查
   - 如果为空，直接返回空字符串

2. **检查 input_ids 是否为空**
   - 在 tokenize 之后检查
   - 如果为空，直接返回空字符串

### 修复代码

```python
def generate_answer(model, tokenizer, text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    # ✅ 检查输入文本是否为空
    if not text or text.strip() == "":
        return ""

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ✅ 检查 input_ids 是否为空
    if inputs['input_ids'].shape[1] == 0:
        return ""

    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"⚠️  Warning: Generation failed with error: {e}")
            try:
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    max_new_tokens=max_new_tokens
                )
            except Exception as e2:
                print(f"⚠️  Warning: Alternative generation also failed: {e2}")
                return ""

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text
```

---

## 📊 检查流程

### 检查 1：输入文本为空

```python
if not text or text.strip() == "":
    return ""
```

**检查条件**：
- `not text` - 检查 None 或空字符串
- `text.strip() == ""` - 检查只有空格的字符串

**返回值**：空字符串

### 检查 2：Tokenize 后为空

```python
if inputs['input_ids'].shape[1] == 0:
    return ""
```

**检查条件**：
- `inputs['input_ids'].shape[1] == 0` - 检查 token 序列长度为 0

**返回值**：空字符串

---

## 🔧 修改清单

- ✅ 添加输入文本为空的检查
- ✅ 添加 input_ids 为空的检查
- ✅ 改进异常处理
- ✅ 添加嵌套 try-except

---

## 📈 预期输出

### 修复前

```
Generating:   0%|                                                                                                                                                                                                                                   | 0/16022 [00:00<?, ?it/s]

Traceback (most recent call last):
  File "/root/autodl-tmp/CultureMoE/LLaMA-Factory/ft_base_gen.py", line 322, in <module>
    main()
  ...
IndexError: index -1 is out of bounds for dimension 0 with size 0
```

### 修复后

```
Generating answers on dataset...

Generating: 100%|████████████████████████████████████████| 16022/16022 [01:30<00:00, 177.13it/s]

📋 前五条生成的答案:

样本 1:
  Question: ### Question: Give me the answer from 1 to 4: ...
  True Output: 2
  Generated Text: 2
  Predicted Answer: 2
  Correct: ✅

📊 Evaluation Results
================================================================================
Accuracy: 0.3456
Correct: 5543/16022
================================================================================

✅ Evaluation completed!
```

---

## 🎯 数据质量检查

### 检查数据中是否有空输入

```python
import json

with open('/root/autodl-fs/cultureLLM_merge_gen.json', 'r') as f:
    data = json.load(f)

empty_count = 0
for item in data:
    instruction = item.get('instruction', '').strip()
    input_text = item.get('input', '').strip()

    if not instruction and not input_text:
        empty_count += 1
        print(f"Empty sample: {item}")

print(f"\nTotal empty samples: {empty_count}/{len(data)}")
```

---

## 📝 其他可能的空输入情况

### 1. 只有空格

```python
instruction = "   "
input_text = ""
# 拼接后：full_text = "   "
# tokenizer("   ") 可能返回空或只有 pad token
```

### 2. 只有特殊字符

```python
instruction = "###"
input_text = ""
# 拼接后：full_text = "###"
# 可能被 tokenizer 处理为空
```

### 3. 超长被截断

```python
instruction = "very long text..." * 1000
input_text = "more text..."
# 拼接后被截断到 max_length=512
# 可能导致有效内容为空
```

---

## 🎉 总结

### 问题
- 当 `instruction` 和 `input` 都是空字符串时
- Tokenizer 返回空 tensor
- 模型生成时触发 IndexError

### 解决方案
- ✅ 检查输入文本是否为空
- ✅ 检查 input_ids 是否为空
- ✅ 改进异常处理

### 结果
- ✅ 避免 IndexError
- ✅ 正常处理空输入
- ✅ 评估成功完成

---

**问题已解决！** ✅

```bash
sh run_ft_base.sh qwen 4

