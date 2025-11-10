# 标准语言建模损失微调 - 快速开始

## 🚀 30 秒快速开始

### 1. 微调 LoRA Only 模型

```bash
sh run_ft_lora_only_from_components.sh llama 4
```

### 2. 微调 CultureMoE 模型

```bash
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5
```

### 3. 查看结果

```bash
# LoRA Only 结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool

# CultureMoE 结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/epoch_eval_results.json | python -m json.tool
```

---

## 📊 数据格式

```json
{
    "text": "### Question: ... ### Answer: 2",
    "text_mask": "### Question: ... ### Answer: 2",
    "label": "0"
}
```

---

## 🎯 常用命令

### LoRA Only

```bash
# LLaMA + CultureLLM
sh run_ft_lora_only_from_components.sh llama 4

# Qwen + CultureLLM
sh run_ft_lora_only_from_components.sh qwen 4

# 其他数据集
sh run_ft_lora_only_from_components.sh llama 2  # CulturalBench
sh run_ft_lora_only_from_components.sh llama 3  # NormAD
```

### CultureMoE

```bash
# 基础训练
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5

# 消融实验：低文化损失
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.1

# 消融实验：高文化损失
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.8

# 消融实验：无文化损失
sh run_ft_culturemoe_from_base_gen.sh llama 4 False 6 2 0.0
```

---

## 📈 输出文件

```
output_dir/
├── best_lora/ 或 best_moe/       # 最好的权重
├── epoch_eval_results.json       # 每个 epoch 的结果
├── generated_answers.json        # 生成的答案
└── config.json                   # 配置
```

---

## 📊 关键指标

| 指标 | 说明 |
|------|------|
| train_loss | 训练总损失 |
| eval_loss | 验证总损失 |
| eval_accuracy | 验证准确率 |

---

## 💡 参数说明

### LoRA Only

| 参数 | 默认值 |
|------|--------|
| num_epochs | 6 |
| batch_size | 4 |
| learning_rate | 2e-4 |
| lora_r | 64 |

### CultureMoE

| 参数 | 默认值 |
|------|--------|
| num_epochs | 30 |
| batch_size | 4 |
| learning_rate | 1e-6 |
| culture_loss_lambda | 0.5 |
| num_experts | 6 |

---

## 🔧 直接运行 Python

### LoRA Only

```bash
python ft_lora_only_from_components.py \
    --base_model_path /path/to/base_model \
    --train_file /path/to/train_data.json \
    --output_dir /path/to/output \
    --num_epochs 6
```

### CultureMoE

```bash
python ft_culturemoe_from_base_gen.py \
    --base_model_path /path/to/base_model \
    --lora_weights_path /path/to/lora_weights \
    --train_file /path/to/train_data.json \
    --output_dir /path/to/output \
    --use_culture_loss True \
    --culture_loss_lambda 0.5
```

---

## ✅ 验证

```bash
# 检查脚本
ls -la ft_*.py run_ft_*.sh

# 查看帮助
python ft_lora_only_from_components.py --help
python ft_culturemoe_from_base_gen.py --help
```

---

## 📚 详细文档

查看 `FT_STANDARD_LM_LOSS_GUIDE.md` 获取完整的文档。

---

**现在可以开始微调了！** 🚀

