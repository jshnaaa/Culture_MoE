# Base 模型评估 - 快速开始

## 🚀 30 秒快速开始

### 1. 评估 LLaMA Base 模型

```bash
sh run_ft_base.sh llama 4
```

### 2. 评估 Qwen Base 模型

```bash
sh run_ft_base.sh qwen 4
```

### 3. 查看结果

```bash
# 查看评估结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/eval_results.json | python -m json.tool

# 查看生成的答案
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_base_cultureLLM_llama_*/generated_answers.json | python -m json.tool | head -50
```

---

## 📊 数据格式

```json
{
    "text": "### Question: ... ### Answer: 10",
    "text_mask": "...",
    "label": "1"
}
```

**说明**：只使用 `text` 字段，忽略 `text_mask` 和 `label`

---

## 🎯 常用命令

### 评估不同 Backbone

```bash
# LLaMA
sh run_ft_base.sh llama 4

# Qwen
sh run_ft_base.sh qwen 4
```

### 评估不同数据集

```bash
# CultureLLM (默认)
sh run_ft_base.sh llama 4

# CulturalBench
sh run_ft_base.sh llama 2

# NormAD
sh run_ft_base.sh llama 3
```

---

## 📈 输出文件

```
output_dir/
├── eval_results.json           # 评估结果
├── generated_answers.json      # 生成的答案
└── config.json                 # 配置
```

---

## 📊 关键指标

| 指标 | 说明 |
|------|------|
| accuracy | 准确率 |
| correct | 正确的样本数 |
| total | 总样本数 |

---

## 💡 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| backbone | llama 或 qwen | llama |
| DATA_ID | 2=CulturalBench, 3=NormAD, 4=CultureLLM | 4 |

---

## 🔧 直接运行 Python

```bash
python ft_base.py \
    --base_model_path /path/to/base_model \
    --train_file /path/to/train_data.json \
    --output_dir /path/to/output \
    --max_length 512 \
    --device cuda
```

---

## ✅ 验证

```bash
# 检查脚本
ls -la ft_base.py run_ft_base.sh

# 查看帮助
python ft_base.py --help
```

---

## 📚 详细文档

查看 `FT_BASE_EVALUATION_GUIDE.md` 获取完整的文档。

---

**现在可以开始评估 Base 模型了！** 🚀

