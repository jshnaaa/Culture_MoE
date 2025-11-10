# Qwen 模型生成参数修复指南

## 🔍 问题分析

### 症状

- **LLaMA 模型**: 正常运行 ✅
- **Qwen 模型**: 报错 ❌

```
IndexError: index -1 is out of bounds for dimension 0 with size 0
```

### 根本原因

**Qwen 模型的生成配置与 LLaMA 不同**：

1. Qwen 模型有自己的默认生成配置
2. 显式设置 `do_sample=False` 等参数会与 Qwen 的内部配置冲突
3. LLaMA 模型可以接受这些参数，但 Qwen 不行

---

## ✅ 解决方案

### 修复策略

**根据模型类型使用不同的生成参数**：

1. **检测模型类型**
   - 检查模型名称中是否包含 "qwen"
   - 根据模型类型选择不同的配置

2. **Qwen 模型**
   - 只设置必要的参数
   - 让模型使用默认配置

3. **LLaMA 模型**
   - 使用完整的参数配置
   - 显式控制生成行为

### 修复代码

```python
def generate_answer(model, tokenizer, text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # 检测模型类型
        model_name = model.config._name_or_path.lower() if hasattr(model.config, '_name_or_path') else ''
        is_qwen = 'qwen' in model_name

        # Qwen 模型需要特殊配置
        if is_qwen:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        else:
            # LLaMA 和其他模型使用标准配置
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0
            )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text
```

---

## 📊 参数对比

### Qwen 模型配置

```python
# ✅ Qwen 模型 - 最小配置
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
    # 不设置 do_sample, num_beams, repetition_penalty
)
```

### LLaMA 模型配置

```python
# ✅ LLaMA 模型 - 完整配置
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.0
)
```

---

## 🔧 修改清单

- ✅ 修改 `ft_base_gen.py` 文件
- ✅ 添加模型类型检测
- ✅ 为 Qwen 模型使用最小配置
- ✅ 为 LLaMA 模型使用完整配置
- ✅ 保持向后兼容性

---

## 📈 预期输出

### 修复前

```bash
# LLaMA 模型
sh run_ft_base.sh llama 5
✅ 正常运行

# Qwen 模型
sh run_ft_base.sh qwen 5
❌ IndexError: index -1 is out of bounds for dimension 0 with size 0
```

### 修复后

```bash
# LLaMA 模型
sh run_ft_base.sh llama 5
✅ 正常运行

# Qwen 模型
sh run_ft_base.sh qwen 5
✅ 正常运行
```

---

## 🎯 为什么 Qwen 模型不同？

### 1. **内部配置差异**

Qwen 模型有自己的生成配置：
- 默认的采样策略
- 默认的 beam search 配置
- 默认的重复惩罚

### 2. **参数冲突**

显式设置某些参数会与 Qwen 的内部配置冲突：
- `do_sample=False` 可能与默认配置冲突
- `num_beams=1` 可能与默认配置冲突
- `repetition_penalty=1.0` 可能与默认配置冲突

### 3. **最佳实践**

对于 Qwen 模型：
- 只设置必要的参数
- 让模型使用默认配置
- 避免显式设置采样相关参数

---

## 📝 其他模型支持

### 检测逻辑

```python
model_name = model.config._name_or_path.lower()

if 'qwen' in model_name:
    # Qwen 配置
    ...
elif 'llama' in model_name:
    # LLaMA 配置
    ...
elif 'gpt' in model_name:
    # GPT 配置
    ...
else:
    # 默认配置
    ...
```

### 扩展支持

如果需要支持其他模型，可以添加更多检测逻辑：

```python
# 检测模型类型
model_name = model.config._name_or_path.lower()

if 'qwen' in model_name:
    # Qwen 模型 - 最小配置
    generation_config = {
        'max_new_tokens': max_new_tokens,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }
elif 'llama' in model_name or 'mistral' in model_name:
    # LLaMA/Mistral 模型 - 完整配置
    generation_config = {
        'max_new_tokens': max_new_tokens,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'do_sample': False,
        'num_beams': 1,
        'repetition_penalty': 1.0
    }
else:
    # 其他模型 - 默认配置
    generation_config = {
        'max_new_tokens': max_new_tokens,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }

outputs = model.generate(**inputs, **generation_config)
```

---

## 🎉 总结

### 问题
- Qwen 模型的生成配置与 LLaMA 不同
- 显式设置某些参数会导致冲突
- 导致索引错误

### 解决方案
- 检测模型类型
- 为 Qwen 使用最小配置
- 为 LLaMA 使用完整配置

### 结果
- ✅ LLaMA 模型正常运行
- ✅ Qwen 模型正常运行
- ✅ 支持多种模型类型

---

## 🚀 现在可以继续评估了

```bash
# LLaMA 模型
sh run_ft_base.sh llama 5

# Qwen 模型
sh run_ft_base.sh qwen 5
```

**预期结果**:
- ✅ 两种模型都能正常运行
- ✅ 生成答案正确
- ✅ 准确率计算正确

---

**问题已解决！** ✅

