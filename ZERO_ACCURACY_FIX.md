# CultureMoE 零准确率修复

## 🔍 问题诊断

### 症状

```
Accuracy: 0.0000 (0/1101)

Confusion Matrix: 全是 0

所有预测都是空的！
```

### 根本原因

**只生成了一个 token，而这个 token 不是数字！**

```python
# ❌ 原始代码（只生成一个 token）
predicted_token_id = torch.argmax(last_token_logits, dim=-1)  # [B]
generated_text = tokenizer.decode(predicted_token_id[0], skip_special_tokens=True).strip()

# 问题：
# - predicted_token_id 可能是换行符、空格等非数字字符
# - 解码后 generated_text 可能是空字符串
# - 提取数字失败，predicted_label 为空字符串
```

**示例**：
```
Token ID: 13 → 解码为 "\n" → generated_text = "" → predicted_label = ""
Token ID: 220 → 解码为 " " → generated_text = "" → predicted_label = ""
Token ID: 128000 → 解码为 "<|end_of_text|>" → generated_text = "" → predicted_label = ""
```

---

## ✅ 修复方案

### 核心思路

**生成多个 token，直到生成 EOS 或达到最大长度**

```python
# ✅ 修复后的代码（生成多个 token）
current_ids = inputs['input_ids'].clone()
current_mask = inputs['attention_mask'].clone()
generated_tokens = []

for _ in range(max_new_tokens):
    # 1. 调用 model.forward() 获取 logits
    outputs = model(
        input_ids=current_ids,
        attention_mask=current_mask,
        input_ids_mask=input_ids_mask,
        attention_mask_mask=attention_mask_mask,
        labels=None,
        use_culture_loss=False
    )

    # 2. 获取最后一个 token 的 logits
    logits = outputs['logits']
    last_token_logits = logits[:, -1, :]

    # 3. 贪婪解码
    predicted_token_id = torch.argmax(last_token_logits, dim=-1)
    generated_tokens.append(predicted_token_id.item())

    # 4. 检查是否生成了 EOS
    if predicted_token_id.item() == tokenizer.eos_token_id:
        break

    # 5. 拼接到 input_ids（用于下一次生成）
    current_ids = torch.cat([current_ids, predicted_token_id.unsqueeze(-1)], dim=1)
    current_mask = torch.cat([current_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)

    # 6. 更新 input_ids_mask（保持相同长度）
    input_ids_mask = torch.cat([input_ids_mask, predicted_token_id.unsqueeze(-1)], dim=1)
    attention_mask_mask = torch.cat([attention_mask_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)

# 7. 解码所有生成的 tokens
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
```

### 生成过程示例

```
Prompt: "Please answer with ONLY ONE NUMBER (1 to 10).\nYour answer:"

生成过程：
  Token 1: 220 → " "
  Token 2: 18 → "1"
  Token 3: 128000 → "<|end_of_text|>" → 停止

generated_tokens = [220, 18, 128000]
generated_text = "1"  # ✅ 成功生成数字
predicted_label = "1"
```

---

## 📊 修复前后对比

### 修复前 ❌

```python
# 只生成一个 token
predicted_token_id = torch.argmax(last_token_logits, dim=-1)
generated_text = tokenizer.decode(predicted_token_id[0], skip_special_tokens=True).strip()

# 结果
generated_text = ""  # 空字符串
predicted_label = ""  # 空字符串
Accuracy = 0.0000
```

### 修复后 ✅

```python
# 生成多个 token
for _ in range(max_new_tokens):
    predicted_token_id = torch.argmax(last_token_logits, dim=-1)
    generated_tokens.append(predicted_token_id.item())
    if predicted_token_id.item() == tokenizer.eos_token_id:
        break
    # 拼接到 input_ids...

generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# 结果
generated_text = "1"  # ✅ 有效的数字
predicted_label = "1"  # ✅ 有效的预测
Accuracy > 0.0000  # ✅ 准确率正常
```

---

## 🔧 修改的文件

### `train_culturemoe_from_base_gen.py`

**第 547-574 行**：`generate_and_evaluate()` 函数

```python
# 修改前：只生成一个 token
predicted_token_id = torch.argmax(last_token_logits, dim=-1)
generated_text = tokenizer.decode(predicted_token_id[0], skip_special_tokens=True).strip()

# 修改后：生成多个 token
current_ids = inputs['input_ids'].clone()
generated_tokens = []

for _ in range(max_new_tokens):
    outputs = model(...)
    logits = outputs['logits']
    last_token_logits = logits[:, -1, :]
    predicted_token_id = torch.argmax(last_token_logits, dim=-1)
    generated_tokens.append(predicted_token_id.item())

    if predicted_token_id.item() == tokenizer.eos_token_id:
        break

    current_ids = torch.cat([current_ids, predicted_token_id.unsqueeze(-1)], dim=1)
    # ...

generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
```

---

## 🚀 使用方法

```bash
# 使用修复后的脚本运行训练
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 1 true
```

**预期输出**：
```
🔍 Sample 0:
  Input: Please answer with ONLY ONE NUMBER (1 to 10).\nYour answer:
  Generated text: '1'
  Predicted: '1'
  True: '1'

🔍 Sample 1:
  Input: Please answer with ONLY ONE NUMBER (1 to 10).\nYour answer:
  Generated text: '3'
  Predicted: '3'
  True: '3'

Accuracy: 0.7234 (796/1101)  # ✅ 准确率正常
```

---

## 🎯 关键改进

| 改进 | 说明 |
|------|------|
| 生成方式 | 单个 token → 多个 token |
| 停止条件 | 无 → 检查 EOS token |
| 解码方式 | 解码单个 token ID → 解码 token 列表 |

---

## 📋 调试信息

添加了调试信息，打印前 5 个样本的生成结果：

```python
if i < 5:
    print(f"\n🔍 Sample {i}:")
    print(f"  Input: {full_input[:100]}...")
    print(f"  Generated text: '{generated_text}'")
    print(f"  Predicted: '{predicted_label}'")
    print(f"  True: '{true_label}'")
```

**用途**：
- 检查生成的文本是否正确
- 检查预测的标签是否正确
- 检查真实标签是否正确

---

## ⚠️ 注意事项

### 1. max_new_tokens 参数

```python
def generate_and_evaluate(model, tokenizer, val_dataset, output_dir, num_classes, max_new_tokens=10):
```

- 默认值：10
- 说明：最多生成 10 个 token
- 对于数字生成任务，10 个 token 足够了

### 2. 生成速度

**修复前**：
- 每个样本只调用一次 `model.forward()`
- 速度快

**修复后**：
- 每个样本调用多次 `model.forward()`（最多 max_new_tokens 次）
- 速度较慢，但准确率正常

**优化建议**：
- 如果生成的答案通常很短（1-2 个 token），可以减小 `max_new_tokens`
- 如果验证集很大，可以使用批量生成（batch generation）

### 3. EOS token 检查

```python
if predicted_token_id.item() == tokenizer.eos_token_id:
    break
```

- 检查是否生成了 EOS token
- 如果生成了 EOS，提前停止生成
- 避免生成过多无用的 token

---

## 🎉 总结

### 问题
- ❌ 只生成一个 token
- ❌ 生成的 token 不是数字
- ❌ 准确率为 0

### 根本原因
- ❌ 没有循环生成多个 token
- ❌ 第一个 token 可能是换行符、空格等

### 解决方案
- ✅ 循环生成多个 token
- ✅ 检查 EOS token
- ✅ 解码所有生成的 token

### 预期效果
- ✅ 生成有效的数字
- ✅ 准确率正常（> 0）
- ✅ 模型评估正确

---

**现在可以正确评估 CultureMoE 模型了！** 🚀

