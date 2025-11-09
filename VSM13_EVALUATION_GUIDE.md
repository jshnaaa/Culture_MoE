# VSM13 评估指南

## 📋 概述

VSM13 评估脚本用于评估 LoRA Only 和 CultureMoE 模型在 VSM13 数据集上的表现。

**主要功能**：
- ✅ 为每个国家添加文化提示词
- ✅ 使用模型生成答案（1-5）
- ✅ 提取模型生成的数字答案
- ✅ 计算每个国家在每个维度上的分数
- ✅ 计算与霍夫斯泰德标准分数的欧式距离

---

## 🎯 核心功能

### 1️⃣ 文化提示词添加

对于每个国家，在指令前后添加文化提示词：

```python
# 原始指令
instruction = "Please read the following questions and options..."

# 添加文化提示词
instruction_with_country = f"You are a {country_adj} culture chatbot that knows {country_adj} culture very well. {instruction}"
instruction_with_country += f" This question is for a country or language that is {country}."

# 例如，对于中国：
# "You are a Chinese culture chatbot that knows Chinese culture very well. Please read the following questions and options... This question is for a country or language that is China."
```

**10 个国家及其形容词**：
- China → Chinese
- India → Indian
- Japan → Japanese
- Brazil → Brazilian
- Germany → German
- France → French
- United States → American
- Mexico → Mexican
- Russia → Russian
- South Korea → Korean

---

### 2️⃣ 答案生成和提取

**生成答案**：
```python
# 构建完整输入
full_input = f"{instruction_with_country}\n{input_text}"

# 使用模型生成答案
outputs = model.generate(
    input_ids,
    max_new_tokens=10,
    do_sample=False
)

# 解码生成的文本
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
```

**提取数字答案**：
```python
# 从生成的文本中提取数字
# 例如：生成文本 "The answer is 3" → 提取 3
# 例如：生成文本 "I think the answer is 2" → 提取 2

def extract_number_from_text(text: str, min_val: int = 1, max_val: int = 5) -> int:
    numbers = re.findall(r'\d+', text)
    for num_str in numbers:
        num = int(num_str)
        if min_val <= num <= max_val:
            return num
    return -1
```

---

### 3️⃣ 维度分数计算

**VSM13 问题到维度的映射**：

| 问题索引 | 维度 | 问题内容 |
|---------|------|---------|
| 0 | PDI | 时间充足 |
| 1 | PDI | 尊重的老板 |
| 2 | IDV | 与同事的关系 |
| 3 | IDV | 工作安全 |
| 4 | MAS | 工作成就 |
| 5 | MAS | 晋升机会 |
| 6 | UAI | 工作安全 |
| 7 | UAI | 公司规则 |
| 8 | LTO | 长期就业 |
| 9 | LTO | 谦虚 |
| 10 | IVR | 生活质量 |
| 11 | IVR | 个人成就 |
| 12 | PDI | 参与决策 |

**分数计算**：
```python
# 1. 对于每个维度，找到所有属于该维度的问题
dimension_questions = [q_idx for q_idx, dim in QUESTION_TO_DIMENSION.items() if dim == dimension]

# 2. 计算该维度的平均答案
avg_answer = np.mean([answers for q_idx in dimension_questions])

# 3. 将答案从 1-5 转换为 0-100 的分数
score = (avg_answer - 1) / 4 * 100
```

**转换公式**：
- 答案 1 → 分数 0
- 答案 2 → 分数 25
- 答案 3 → 分数 50
- 答案 4 → 分数 75
- 答案 5 → 分数 100

---

### 4️⃣ 欧式距离计算

**公式**：
```
distance = sqrt(sum((predicted_score - hofstede_score)^2 for each dimension))
```

**例子**：
```
中国：
  预测分数：PDI=75, IDV=20, MAS=65, UAI=30, LTO=85, IVR=25
  标准分数：PDI=80, IDV=20, MAS=66, UAI=30, LTO=87, IVR=24

  距离 = sqrt((75-80)^2 + (20-20)^2 + (65-66)^2 + (30-30)^2 + (85-87)^2 + (25-24)^2)
       = sqrt(25 + 0 + 1 + 0 + 4 + 1)
       = sqrt(31)
       ≈ 5.57
```

---

## 🚀 使用方法

### 方法 1：使用 Shell 脚本（推荐）

```bash
# 评估 LoRA Only 模型
sh run_eval_vsm13.sh lora_only /path/to/best_lora

# 评估 CultureMoE 模型
sh run_eval_vsm13.sh culturemoe /path/to/moe_model
```

### 方法 2：直接使用 Python 脚本

```bash
# 评估 LoRA Only 模型
python eval_vsm13.py \
    --model_path /path/to/best_lora \
    --data_path /root/autodl-fs/vsm13.json \
    --output_dir /path/to/output \
    --model_type lora_only \
    --device cuda

# 评估 CultureMoE 模型
python eval_vsm13.py \
    --model_path /path/to/moe_model \
    --data_path /root/autodl-fs/vsm13.json \
    --output_dir /path/to/output \
    --model_type culturemoe \
    --device cuda
```

---

## 📊 输出结果

### 结果文件

```
output_dir/
├── vsm13_results.json          # 详细结果
```

### 结果格式

```json
{
  "country_dimension_scores": {
    "China": {
      "PDI": 75.0,
      "IDV": 20.0,
      "MAS": 65.0,
      "UAI": 30.0,
      "LTO": 85.0,
      "IVR": 25.0
    },
    ...
  },
  "hofstede_scores": {
    "China": {
      "PDI": 80,
      "IDV": 20,
      "MAS": 66,
      "UAI": 30,
      "LTO": 87,
      "IVR": 24
    },
    ...
  },
  "euclidean_distances": {
    "China": 5.57,
    "India": 12.34,
    ...
  },
  "average_euclidean_distance": 8.92
}
```

### 控制台输出

```
================================================================================
VSM13 Evaluation Results
================================================================================

📊 Dimension Scores by Country:
--------------------------------------------------------------------------------
Country              PDI        IDV        MAS        UAI        LTO        IVR
--------------------------------------------------------------------------------
Brazil               68.75      37.50      48.75      75.00      43.75      58.75
China                75.00      20.00      65.00      30.00      85.00      25.00
France               67.50      70.00      42.50      85.00      62.50      47.50
Germany              35.00      66.25      65.00      65.00      82.50      40.00
India                76.25      47.50      55.00      40.00      50.00      25.00
Japan                53.75      45.00      95.00      91.25      87.50      41.25
Mexico               80.00      30.00      68.75      81.25      23.75      96.25
Russia               92.50      38.75      35.00      95.00      80.00      20.00
South Korea          60.00      17.50      83.75      100.00     100.00     28.75
United States        40.00      90.00      61.25      45.00      25.00      67.50

📊 Hofstede Standard Scores:
--------------------------------------------------------------------------------
Country              PDI        IDV        MAS        UAI        LTO        IVR
--------------------------------------------------------------------------------
Brazil               69         38         49         76         44         59
China                80         20         66         30         87         24
France               68         71         43         86         63         48
Germany              35         67         66         65         83         40
India                77         48         56         40         51         26
Japan                54         46         95         92         88         42
Mexico               81         30         69         82         24         97
Russia               93         39         36         95         81         20
South Korea          60         18         84         100        100        29
United States        40         91         62         46         26         68

📊 Euclidean Distances:
--------------------------------------------------------------------------------
China                5.57
Germany              2.34
United States        3.89
Japan                4.12
India                6.78
France               7.23
Brazil               8.45
Mexico               9.12
Russia               10.34
South Korea         11.56
--------------------------------------------------------------------------------
Average              6.94

✅ Results saved to /path/to/output/vsm13_results.json
```

---

## 🔍 结果解释

### 欧式距离含义

- **距离越小**：模型预测的分数越接近霍夫斯泰德标准分数
- **距离越大**：模型预测的分数与标准分数差异越大

### 维度含义

- **PDI（权力距离）**：社会对不平等的接受程度
- **IDV（个人主义）**：个人与集体的关系
- **MAS（男性气质）**：竞争与合作的平衡
- **UAI（不确定性规避）**：对不确定性的容忍度
- **LTO（长期导向）**：对长期目标的重视
- **IVR（放纵指数）**：对欲望的满足程度

---

## 📈 对比分析

### 比较两个模型

```bash
# 评估 LoRA Only 模型
sh run_eval_vsm13.sh lora_only /path/to/lora_best_lora

# 评估 CultureMoE 模型
sh run_eval_vsm13.sh culturemoe /path/to/moe_best_moe

# 比较两个结果
# 查看 vsm13_results.json 中的 average_euclidean_distance
# 距离更小的模型表现更好
```

### 性能指标

| 指标 | 说明 |
|------|------|
| **Average Euclidean Distance** | 所有国家的平均欧式距离，越小越好 |
| **Per-Country Distance** | 每个国家的欧式距离，用于识别模型在哪些国家表现好/差 |
| **Dimension Scores** | 每个维度的分数，用于分析模型在哪些维度表现好/差 |

---

## 🎯 最佳实践

### 1. 数据准备

确保 VSM13 数据集格式正确：
```json
{
  "instruction": "...",
  "instruction_mask": "...",
  "input": "Please select the most appropriate option by replying with the option number only (1, 2, 3, ...). Your answer must be a single number only. Your answer is:",
  "output": "",
  "label": ""
}
```

### 2. 模型准备

- **LoRA Only**：使用 `best_lora` 目录（包含 adapter_config.json 和权重）
- **CultureMoE**：使用模型目录

### 3. 运行评估

```bash
# 确保有足够的显存（至少 20GB）
nvidia-smi

# 运行评估
sh run_eval_vsm13.sh lora_only /path/to/best_lora
```

### 4. 分析结果

```bash
# 查看详细结果
cat /path/to/output/vsm13_results.json | python -m json.tool

# 提取平均欧式距离
python -c "import json; data = json.load(open('/path/to/output/vsm13_results.json')); print(f\"Average Distance: {data['average_euclidean_distance']:.2f}\")"
```

---

## 🐛 常见问题

### Q1：为什么某些答案提取失败？

**原因**：
- 模型没有生成数字
- 生成的数字不在 1-5 范围内

**解决**：
- 检查模型生成的文本
- 调整提示词引导模型生成数字

### Q2：欧式距离很大是什么原因？

**可能原因**：
- 模型没有充分学习文化特征
- 提示词不够有效
- 数据集不够多样化

**解决**：
- 增加训练数据
- 改进提示词
- 调整模型参数

### Q3：如何比较两个模型的性能？

**方法**：
1. 分别评估两个模型
2. 比较 `average_euclidean_distance`
3. 距离更小的模型表现更好

---

## 🎉 总结

### ✅ 功能

1. **文化提示词**：为每个国家添加文化背景
2. **答案生成**：使用模型生成答案
3. **答案提取**：从生成的文本中提取数字
4. **分数计算**：计算每个维度的分数
5. **距离计算**：计算与标准分数的欧式距离

### 📊 输出

- **vsm13_results.json**：详细的评估结果
- **控制台输出**：可视化的结果展示

### 🚀 使用

```bash
# 评估 LoRA Only 模型
sh run_eval_vsm13.sh lora_only /path/to/best_lora

# 评估 CultureMoE 模型
sh run_eval_vsm13.sh culturemoe /path/to/moe_model
```

---

**现在可以开始评估模型了！** 🚀

