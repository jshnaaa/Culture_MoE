# VSM24 评估完整指南

## 📋 概述

VSM24 评估脚本已完全更新，支持：
- ✅ 24 个问题（VSM24 完整版本）
- ✅ 10 个国家
- ✅ 完整的霍夫斯泰德公式计算
- ✅ 生成式模型评估

---

## 🌍 10 个国家及其霍夫斯泰德分数

| 国家 | PDI | IDV | MAS | UAI | LTO | IVR |
|------|-----|-----|-----|-----|-----|-----|
| China | 80 | 43 | 66 | 30 | 77 | 24 |
| South Korea | 60 | 58 | 39 | 85 | 86 | 29 |
| Turkey | 66 | 46 | 45 | 85 | 35 | 49 |
| Saudi Arabia | 72 | 48 | 43 | 64 | 27 | 14 |
| Bangladesh | 80 | 5 | 55 | 60 | 38 | 20 |
| Germany | 35 | 79 | 66 | 65 | 57 | 40 |
| Portugal | 63 | 59 | 31 | 99 | 42 | 33 |
| Spain | 57 | 67 | 42 | 86 | 47 | 44 |
| England | 35 | 76 | 66 | 35 | 60 | 69 |
| Greece | 60 | 59 | 57 | 100 | 51 | 50 |

---

## 📊 VSM24 数据集格式

每个问题的格式如下：

```json
{
  "instruction": "Please read the following questions and options, and answer with the number of the option you think is correct. Give me the answer from 1 to 5: [问题内容] 1. [选项1] 2. [选项2] 3. [选项3] 4. [选项4] 5. [选项5] You can only choose one option.",
  "instruction_mask": "Please read the following questions and options, and answer with the number of the option you think is correct. Give me the answer from 1 to 5: [问题内容] 1. [选项1] 2. [选项2] 3. [选项3] 4. [选项4] 5. [选项5] You can only choose one option.",
  "input": "Please select the most appropriate option by replying with the option number only (1, 2, 3, ...). Your answer must be a single number only. Your answer is:",
  "output": "",
  "label": ""
}
```

**字段说明**：
- `instruction`：完整的问题指令
- `instruction_mask`：指令掩码版本（通常与 instruction 相同）
- `input`：输入提示词，引导模型生成数字答案
- `output`：**模型生成的答案（1-5）**
- `label`：原始标签（通常为空）

---

## 🎯 霍夫斯泰德维度计算公式

### 完整的 VSM24 公式

#### 1️⃣ PDI（权力距离）
```
PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 3
```

#### 2️⃣ IDV（个人主义）
```
IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + 3
```

#### 3️⃣ MAS（男性气质）
```
MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + 3
```

#### 4️⃣ UAI（不确定性规避）
```
UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + 3
```

#### 5️⃣ LTO（长期导向）
```
LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + 3
```

#### 6️⃣ IVR（放纵指数）
```
IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + 3
```

**说明**：
- μQn：第 n 题的平均答案（1-5）
- 所有常量都是 3
- 最终分数限制在 0-100 范围内

---

## 📝 问题编号映射

| 0-based 索引 | 问题编号 | 0-based 索引 | 问题编号 |
|-------------|---------|-------------|---------|
| 0 | Q1 | 12 | Q13 |
| 1 | Q2 | 13 | Q14 |
| 2 | Q3 | 14 | Q15 |
| 3 | Q4 | 15 | Q16 |
| 4 | Q5 | 16 | Q17 |
| 5 | Q6 | 17 | Q18 |
| 6 | Q7 | 18 | Q19 |
| 7 | Q8 | 19 | Q20 |
| 8 | Q9 | 20 | Q21 |
| 9 | Q10 | 21 | Q22 |
| 10 | Q11 | 22 | Q23 |
| 11 | Q12 | 23 | Q24 |

---

## 🚀 使用方法

### 方法 1：使用 Shell 脚本（推荐）

```bash
# 评估 LoRA Only 模型 (Qwen)
sh run_eval_vsm13.sh lora_only qwen

# 评估 Base 模型 (Qwen)
sh run_eval_vsm13.sh base qwen

# 评估 CultureMoE 模型 (Qwen)
sh run_eval_vsm13.sh culturemoe qwen

# 评估 LoRA Only 模型 (LLaMA)
sh run_eval_vsm13.sh lora_only llama
```

### 方法 2：直接使用 Python 脚本

```bash
# 评估 LoRA Only 模型
python eval_vsm13.py \
    --model_type lora_only \
    --backbone qwen \
    --data_path /root/autodl-fs/vsm24.json \
    --output_dir /path/to/output \
    --device cuda

# 评估 Base 模型
python eval_vsm13.py \
    --model_type base \
    --backbone qwen \
    --data_path /root/autodl-fs/vsm24.json \
    --output_dir /path/to/output \
    --device cuda
```

---

## 📁 输出文件结构

```
output_dir/
├── vsm13_results.json                    # 主要结果（汇总）
├── vsm13_detailed_scores.json            # 详细的分数和距离报告
├── vsm13_all_generated.json              # 所有国家的生成数据汇总
├── vsm13_generated_China.json            # 中国的生成数据
├── vsm13_generated_South Korea.json      # 韩国的生成数据
├── vsm13_generated_Turkey.json           # 土耳其的生成数据
├── vsm13_generated_Saudi Arabia.json     # 沙特阿拉伯的生成数据
├── vsm13_generated_Bangladesh.json       # 孟加拉国的生成数据
├── vsm13_generated_Germany.json          # 德国的生成数据
├── vsm13_generated_Portugal.json         # 葡萄牙的生成数据
├── vsm13_generated_Spain.json            # 西班牙的生成数据
├── vsm13_generated_England.json          # 英国的生成数据
└── vsm13_generated_Greece.json           # 希腊的生成数据
```

---

## 📄 输出文件详细说明

### 1️⃣ vsm13_results.json（主要结果）

```json
{
  "country_dimension_scores": {
    "China": {
      "PDI": 75.0,
      "IDV": 40.0,
      "MAS": 65.0,
      "UAI": 28.0,
      "LTO": 75.0,
      "IVR": 22.0
    }
  },
  "hofstede_scores": {
    "China": {
      "PDI": 80,
      "IDV": 43,
      "MAS": 66,
      "UAI": 30,
      "LTO": 77,
      "IVR": 24
    }
  },
  "euclidean_distances": {
    "China": 5.57
  },
  "average_euclidean_distance": 8.92
}
```

---

### 2️⃣ vsm13_detailed_scores.json（详细分数和距离）

```json
{
  "China": {
    "dimension_scores": {
      "PDI": 75.0,
      "IDV": 40.0,
      "MAS": 65.0,
      "UAI": 28.0,
      "LTO": 75.0,
      "IVR": 22.0
    },
    "hofstede_scores": {
      "PDI": 80,
      "IDV": 43,
      "MAS": 66,
      "UAI": 30,
      "LTO": 77,
      "IVR": 24
    },
    "euclidean_distance": 5.57
  }
}
```

---

### 3️⃣ vsm13_generated_[Country].json（单个国家的生成数据）

```json
[
  {
    "instruction": "Please read the following questions and options...",
    "instruction_mask": "Please read the following questions and options...",
    "input": "Please select the most appropriate option...",
    "output": "2",
    "label": "",
    "generated_text": "2",
    "question_idx": 0
  },
  {
    "instruction": "Please read the following questions and options...",
    "instruction_mask": "Please read the following questions and options...",
    "input": "Please select the most appropriate option...",
    "output": "3",
    "label": "",
    "generated_text": "The answer is 3",
    "question_idx": 1
  }
]
```

**字段说明**：
- `instruction`：原始指令
- `instruction_mask`：指令掩码版本
- `input`：输入提示词
- `output`：**模型生成的答案**
- `label`：原始标签
- `generated_text`：模型生成的完整文本
- `question_idx`：问题索引

---

## 🔍 文化提示词

对于每个国家，模型会收到带有文化背景的提示词：

```
You are a [country_adj] culture chatbot that knows [country_adj] culture very well. [instruction] This question is for a country or language that is [country].
```

**例子**：
```
You are a Chinese culture chatbot that knows Chinese culture very well. Please read the following questions and options, and answer with the number of the option you think is correct. Give me the answer from 1 to 5: [问题内容] This question is for a country or language that is China.
```

---

## 📊 维度含义

| 维度 | 英文 | 说明 |
|------|------|------|
| PDI | Power Distance Index | 权力距离：社会对不平等的接受程度 |
| IDV | Individualism | 个人主义：个人与集体的关系 |
| MAS | Masculinity | 男性气质：竞争与合作的平衡 |
| UAI | Uncertainty Avoidance Index | 不确定性规避：对不确定性的容忍度 |
| LTO | Long-Term Orientation | 长期导向：对长期目标的重视 |
| IVR | Indulgence vs. Restraint | 放纵指数：对欲望的满足程度 |

---

## 🎯 快速开始

### 1. 准备数据集

确保 VSM24 数据集包含 24 个问题，格式如上所示。

### 2. 运行评估

```bash
sh run_eval_vsm13.sh lora_only qwen
```

### 3. 查看结果

```bash
# 查看主要结果
cat output_dir/vsm13_results.json | python -m json.tool

# 查看详细分数
cat output_dir/vsm13_detailed_scores.json | python -m json.tool

# 查看特定国家的生成数据
cat output_dir/vsm13_generated_China.json | python -m json.tool | head -50
```

### 4. 提取关键指标

```bash
# 平均欧式距离
python -c "import json; data = json.load(open('output_dir/vsm13_results.json')); print(f'Average Distance: {data[\"average_euclidean_distance\"]:.2f}')"

# 特定国家的距离
python -c "import json; data = json.load(open('output_dir/vsm13_results.json')); print(f'China Distance: {data[\"euclidean_distances\"][\"China\"]:.2f}')"

# 特定国家的维度分数
python -c "import json; data = json.load(open('output_dir/vsm13_results.json')); scores = data['country_dimension_scores']['China']; print(f'China PDI: {scores[\"PDI\"]:.2f}')"
```

---

## 💡 计算示例

### 假设某国家的答案

```
Q1=2, Q2=3, Q3=4, Q4=2, Q5=3, Q6=4,
Q7=3, Q8=2, Q9=4, Q10=3, Q11=2, Q12=4,
Q13=3, Q14=2, Q15=4, Q16=3, Q17=2, Q18=4,
Q19=3, Q20=2, Q21=4, Q22=3, Q23=2, Q24=4
```

### 计算 PDI

```
PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 3
    = 35(3 − 3) + 25(2 − 2) + 3
    = 0 + 0 + 3
    = 3
```

### 计算 IDV

```
IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + 3
    = 35(2 − 2) + 35(4 − 4) + 3
    = 0 + 0 + 3
    = 3
```

### 计算 MAS

```
MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + 3
    = 35(3 − 4) + 25(2 − 3) + 3
    = -35 - 25 + 3
    = -57
    → max(0, min(100, -57)) = 0
```

---

## 🎉 总结

### ✅ 功能

1. **24 个问题**：完整的 VSM24 数据集
2. **10 个国家**：中国、韩国、土耳其、沙特阿拉伯、孟加拉国、德国、葡萄牙、西班牙、英国、希腊
3. **完整的霍夫斯泰德公式**：6 个维度的准确计算
4. **生成式评估**：模型生成答案并填充到 output 字段
5. **文化提示词**：为每个国家添加文化背景

### 📊 输出

- **vsm13_results.json**：主要结果汇总
- **vsm13_detailed_scores.json**：详细分数和距离
- **vsm13_generated_[Country].json**：每个国家的生成数据（10 个文件）
- **vsm13_all_generated.json**：所有国家的生成数据汇总

### 🚀 使用

```bash
# 运行评估
sh run_eval_vsm13.sh lora_only qwen

# 查看结果
cat output_dir/vsm13_results.json | python -m json.tool
```

---

**现在可以开始 VSM24 评估了！** 🚀

