# LoRA 空输出问题 - 快速修复

## 🚨 问题

LoRA 微调后模型生成空输出：

```json
{
    "generated_text": "",
    "predicted_answer": "",
    "correct": false
}
```

## 🔍 原因

**数据格式和训练任务类型不匹配**

- 输入包含完整答案
- 模型不知道要生成什么
- 推理时输入包含答案，模型不会生成

## ✅ 快速修复（3 步）

### 步骤 1：转换数据格式

```bash
# 转换 CultureLLM 格式 → Alpaca 格式
python convert_culturellm_to_alpaca.py \
    --input_file /root/autodl-fs/cultureLLM_merge_gen.json \
    --output_file /root/autodl-fs/cultureLLM_merge_gen_alpaca.json
```

### 步骤 2：使用 Alpaca 格式微调

```bash
# 使用 LLaMA
sh run_ft_lora_alpaca.sh llama 4

# 使用 Qwen
sh run_ft_lora_alpaca.sh qwen 4
```

### 步骤 3：验证修复

```bash
# 查看生成的答案
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_alpaca_cultureLLM_llama_*/generated_answers.json | python -m json.tool | head -20
```

## 📊 数据格式对比

### ❌ 原始格式（导致空输出）

```json
{
    "text": "### Question: ... ### Answer: 2"
}
```

### ✅ Alpaca 格式（正确）

```json
{
    "instruction": "Give me the answer from 1 to 4: ...",
    "input": "",
    "output": "2"
}
```

## 🎯 关键改进

| 方面 | 原始方式 | 改进方式 |
|------|---------|---------|
| 数据格式 | CultureLLM | Alpaca |
| 训练输入 | 包含答案 | 只有 instruction |
| 推理输入 | 包含答案 | 只有 instruction |
| 生成结果 | 空字符串 | 正确答案 |

## 📈 预期结果

### 修复前

```json
{
    "generated_text": "",
    "predicted_answer": "",
    "correct": false
}
```

### 修复后

```json
{
    "generated_text": "2",
    "predicted_answer": "2",
    "correct": true
}
```

## 🚀 完整命令

```bash
# 1. 转换数据
python convert_culturellm_to_alpaca.py \
    --input_file /root/autodl-fs/cultureLLM_merge_gen.json \
    --output_file /root/autodl-fs/cultureLLM_merge_gen_alpaca.json

# 2. 微调模型
sh run_ft_lora_alpaca.sh llama 4

# 3. 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_alpaca_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool
```

## 📚 详细文档

查看 `FT_LORA_ALPACA_FIX_GUIDE.md` 获取完整的文档。

---

**现在可以正确微调 LoRA 模型了！** 🚀

