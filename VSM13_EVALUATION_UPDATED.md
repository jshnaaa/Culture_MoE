# VSM13 评估脚本 - 更新版本

## ✅ 更新内容

现在支持三种模型类型的评估：
1. **base**：直接加载 Base 模型
2. **lora_only**：从 Base 模型 + LoRA 权重还原完整模型
3. **culturemoe**：直接加载 CultureMoE 模型

---

## 🚀 使用方法

### 方法 1：使用 Shell 脚本（推荐）

```bash
# 评估 Base 模型 (Qwen)
sh run_eval_vsm13.sh base qwen

# 评估 LoRA Only 模型 (Qwen)
sh run_eval_vsm13.sh lora_only qwen

# 评估 CultureMoE 模型 (Qwen)
sh run_eval_vsm13.sh culturemoe qwen

# 评估 LoRA Only 模型 (LLaMA)
sh run_eval_vsm13.sh lora_only llama

# 评估 Base 模型 (LLaMA)
sh run_eval_vsm13.sh base llama
```

### 方法 2：直接使用 Python 脚本

#### 评估 Base 模型

```bash
python eval_vsm13.py \
    --model_type base \
    --backbone qwen \
    --data_path /root/autodl-fs/vsm13.json \
    --output_dir /path/to/output \
    --device cuda
```

#### 评估 LoRA Only 模型

```bash
python eval_vsm13.py \
    --model_type lora_only \
    --backbone qwen \
    --base_model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct \
    --lora_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora \
    --data_path /root/autodl-fs/vsm13.json \
    --output_dir /path/to/output \
    --device cuda
```

#### 评估 CultureMoE 模型

```bash
python eval_vsm13.py \
    --model_type culturemoe \
    --backbone qwen \
    --base_model_path /path/to/culturemoe_model \
    --data_path /root/autodl-fs/vsm13.json \
    --output_dir /path/to/output \
    --device cuda
```

---

## 📋 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model_type` | 模型类型 | `base`, `lora_only`, `culturemoe` |
| `--backbone` | 基座模型 | `qwen`, `llama` |
| `--data_path` | VSM13 数据集路径 | `/root/autodl-fs/vsm13.json` |
| `--output_dir` | 输出目录 | `/path/to/output` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_model_path` | Base 模型路径（不提供则自动推断） | 根据 backbone 自动推断 |
| `--lora_weights_path` | LoRA 权重路径（仅用于 lora_only） | 根据 backbone 自动推断 |
| `--device` | 设备 | `cuda` |

---

## 🔄 模型加载流程

### Base 模型

```
Base 模型路径
    ↓
加载 Base 模型
    ↓
评估
```

### LoRA Only 模型

```
Base 模型路径 + LoRA 权重路径
    ↓
加载 Base 模型
    ↓
加载 LoRA 权重
    ↓
合并 LoRA 权重到 Base 模型
    ↓
评估
```

### CultureMoE 模型

```
CultureMoE 模型路径
    ↓
加载 CultureMoE 模型
    ↓
评估
```

---

## 📊 自动路径推断

### 当 backbone = qwen 时

| 模型类型 | Base 模型路径 | LoRA 权重路径 |
|---------|--------------|-------------|
| base | `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct` | - |
| lora_only | `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct` | `/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora` |
| culturemoe | `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct` | - |

### 当 backbone = llama 时

| 模型类型 | Base 模型路径 | LoRA 权重路径 |
|---------|--------------|-------------|
| base | `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct` | - |
| lora_only | `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct` | `/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora` |
| culturemoe | `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct` | - |

---

## 🎯 快速开始

### 1. 评估 Base 模型

```bash
# Qwen
sh run_eval_vsm13.sh base qwen

# LLaMA
sh run_eval_vsm13.sh base llama
```

### 2. 评估 LoRA Only 模型

```bash
# Qwen
sh run_eval_vsm13.sh lora_only qwen

# LLaMA
sh run_eval_vsm13.sh lora_only llama
```

### 3. 评估 CultureMoE 模型

```bash
# Qwen
sh run_eval_vsm13.sh culturemoe qwen

# LLaMA
sh run_eval_vsm13.sh culturemoe llama
```

### 4. 查看结果

```bash
# 查看详细结果
cat /path/to/output/vsm13_results.json | python -m json.tool

# 查看平均欧式距离
python -c "import json; data = json.load(open('/path/to/output/vsm13_results.json')); print(f'Average Distance: {data[\"average_euclidean_distance\"]:.2f}')"
```

---

## 📈 对比分析

### 比较三种模型

```bash
# 1. 评估 Base 模型
sh run_eval_vsm13.sh base qwen
# 输出目录：eval_vsm13_base_qwen_YYYYMMDD_HHMM

# 2. 评估 LoRA Only 模型
sh run_eval_vsm13.sh lora_only qwen
# 输出目录：eval_vsm13_lora_only_qwen_YYYYMMDD_HHMM

# 3. 评估 CultureMoE 模型
sh run_eval_vsm13.sh culturemoe qwen
# 输出目录：eval_vsm13_culturemoe_qwen_YYYYMMDD_HHMM

# 4. 比较结果
python -c "
import json
import os

models = ['base', 'lora_only', 'culturemoe']
for model in models:
    # 找到最新的输出目录
    dirs = [d for d in os.listdir('/root/autodl-tmp/CultureMoE/Culture_Alignment') if d.startswith(f'eval_vsm13_{model}_qwen')]
    if dirs:
        latest_dir = sorted(dirs)[-1]
        path = f'/root/autodl-tmp/CultureMoE/Culture_Alignment/{latest_dir}/vsm13_results.json'
        data = json.load(open(path))
        print(f'{model}: Average Distance = {data[\"average_euclidean_distance\"]:.2f}')
"
```

---

## 🔍 预期输出

### 控制台输出

```
================================================================================
Loading Model
================================================================================
Model type: lora_only
Backbone: qwen
Base model path: /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct
LoRA weights path: /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora
================================================================================

Loading tokenizer from /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora...
✅ Tokenizer loaded

Loading base model from /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct...
✅ Base model loaded

Loading LoRA weights from /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora...
✅ LoRA weights loaded

Merging LoRA weights into base model...
✅ LoRA weights merged

✅ Model ready for evaluation

================================================================================
Evaluating China...
Generating answers for China: 100%|██████████| 13/13 [00:45<00:00,  3.46s/it]

================================================================================
VSM13 Evaluation Results
================================================================================

📊 Dimension Scores by Country:
...

📊 Euclidean Distances:
...
Average              6.94

✅ Results saved to /path/to/output/vsm13_results.json
```

---

## 💡 最佳实践

### 1. 批量评估

```bash
#!/bin/bash

# 评估所有模型
for model_type in base lora_only culturemoe; do
    for backbone in qwen llama; do
        echo "Evaluating $model_type with $backbone..."
        sh run_eval_vsm13.sh $model_type $backbone
    done
done
```

### 2. 结果对比

```bash
#!/bin/bash

# 提取所有结果的平均距离
echo "Model Type | Backbone | Average Distance"
echo "-----------|----------|------------------"

for model_type in base lora_only culturemoe; do
    for backbone in qwen llama; do
        dirs=$(ls -d /root/autodl-tmp/CultureMoE/Culture_Alignment/eval_vsm13_${model_type}_${backbone}_* 2>/dev/null | sort -r | head -1)
        if [ -n "$dirs" ]; then
            distance=$(python -c "import json; data = json.load(open('$dirs/vsm13_results.json')); print(f'{data[\"average_euclidean_distance\"]:.2f}')")
            printf "%-10s | %-8s | %s\n" "$model_type" "$backbone" "$distance"
        fi
    done
done
```

---

## 🎉 总结

### ✅ 功能

1. **三种模型类型**
   - Base：直接加载
   - LoRA Only：从 Base + LoRA 还原
   - CultureMoE：直接加载

2. **自动路径推断**
   - 根据 backbone 自动推断模型路径
   - 支持自定义路径

3. **完整的评估流程**
   - 文化提示词添加
   - 答案生成和提取
   - 分数计算
   - 距离计算

### 📊 输出

- **vsm13_results.json**：详细的评估结果
- **控制台输出**：可视化的结果展示

### 🚀 使用

```bash
# 快速开始
sh run_eval_vsm13.sh lora_only qwen
```

---

**现在可以开始评估模型了！** 🚀

