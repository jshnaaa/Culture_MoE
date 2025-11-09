# VSM13 评估输出文件指南

## 📁 输出文件结构

运行 VSM13 评估后，输出目录中会包含以下文件：

```
output_dir/
├── vsm13_results.json                    # 主要结果（汇总）
├── vsm13_detailed_scores.json            # 详细的分数和距离报告
├── vsm13_all_generated.json              # 所有国家的生成数据汇总
├── vsm13_generated_China.json            # 中国的生成数据
├── vsm13_generated_India.json            # 印度的生成数据
├── vsm13_generated_Japan.json            # 日本的生成数据
├── vsm13_generated_Brazil.json           # 巴西的生成数据
├── vsm13_generated_Germany.json          # 德国的生成数据
├── vsm13_generated_France.json           # 法国的生成数据
├── vsm13_generated_United States.json    # 美国的生成数据
├── vsm13_generated_Mexico.json           # 墨西哥的生成数据
├── vsm13_generated_Russia.json           # 俄罗斯的生成数据
└── vsm13_generated_South Korea.json      # 韩国的生成数据
```

---

## 📄 文件详细说明

### 1️⃣ vsm13_results.json（主要结果）

**内容**：汇总的评估结果

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

**用途**：
- ✅ 快速查看所有国家的维度分数
- ✅ 查看霍夫斯泰德标准分数
- ✅ 查看每个国家的欧式距离
- ✅ 查看平均欧式距离

---

### 2️⃣ vsm13_detailed_scores.json（详细的分数和距离报告）

**内容**：每个国家的详细分数和距离

```json
{
  "Brazil": {
    "dimension_scores": {
      "PDI": 68.75,
      "IDV": 37.50,
      "MAS": 48.75,
      "UAI": 75.00,
      "LTO": 43.75,
      "IVR": 58.75
    },
    "hofstede_scores": {
      "PDI": 69,
      "IDV": 38,
      "MAS": 49,
      "UAI": 76,
      "LTO": 44,
      "IVR": 59
    },
    "euclidean_distance": 1.23
  },
  "China": {
    "dimension_scores": {
      "PDI": 75.0,
      "IDV": 20.0,
      "MAS": 65.0,
      "UAI": 30.0,
      "LTO": 85.0,
      "IVR": 25.0
    },
    "hofstede_scores": {
      "PDI": 80,
      "IDV": 20,
      "MAS": 66,
      "UAI": 30,
      "LTO": 87,
      "IVR": 24
    },
    "euclidean_distance": 5.57
  },
  ...
}
```

**用途**：
- ✅ 查看每个国家的详细分数
- ✅ 对比预测分数和标准分数
- ✅ 查看每个国家的欧式距离

---

### 3️⃣ vsm13_all_generated.json（所有国家的生成数据汇总）

**内容**：所有国家的生成数据汇总

```json
{
  "China": [
    {
      "instruction": "Please read the following questions and options...",
      "instruction_mask": "Please read the following questions and options...",
      "input": "Please select the most appropriate option...",
      "output": "1",
      "label": "",
      "generated_text": "1",
      "question_idx": 0
    },
    {
      "instruction": "Please read the following questions and options...",
      "instruction_mask": "Please read the following questions and options...",
      "input": "Please select the most appropriate option...",
      "output": "2",
      "label": "",
      "generated_text": "The answer is 2",
      "question_idx": 1
    },
    ...
  ],
  "India": [
    ...
  ],
  ...
}
```

**用途**：
- ✅ 查看所有国家的生成数据
- ✅ 查看模型生成的完整文本
- ✅ 查看提取的数字答案

---

### 4️⃣ vsm13_generated_[Country].json（单个国家的生成数据）

**内容**：单个国家的生成数据

```json
[
  {
    "instruction": "Please read the following questions and options...",
    "instruction_mask": "Please read the following questions and options...",
    "input": "Please select the most appropriate option...",
    "output": "1",
    "label": "",
    "generated_text": "1",
    "question_idx": 0
  },
  {
    "instruction": "Please read the following questions and options...",
    "instruction_mask": "Please read the following questions and options...",
    "input": "Please select the most appropriate option...",
    "output": "2",
    "label": "",
    "generated_text": "The answer is 2",
    "question_idx": 1
  },
  ...
]
```

**用途**：
- ✅ 查看特定国家的生成数据
- ✅ 分析模型在该国家的表现
- ✅ 调试和改进模型

---

## 🔍 文件字段说明

### 生成数据文件中的字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `instruction` | 原始指令 | "Please read the following questions..." |
| `instruction_mask` | 指令掩码版本 | "Please read the following questions..." |
| `input` | 输入文本 | "Please select the most appropriate option..." |
| `output` | **模型生成的答案** | "1" |
| `label` | 原始标签（通常为空） | "" |
| `generated_text` | **模型生成的完整文本** | "The answer is 1" |
| `question_idx` | 问题索引 | 0 |

---

## 📊 维度说明

| 维度 | 英文 | 说明 |
|------|------|------|
| PDI | Power Distance Index | 权力距离 |
| IDV | Individualism | 个人主义 |
| MAS | Masculinity | 男性气质 |
| UAI | Uncertainty Avoidance Index | 不确定性规避 |
| LTO | Long-Term Orientation | 长期导向 |
| IVR | Indulgence vs. Restraint | 放纵指数 |

---

## 🚀 使用示例

### 1. 查看主要结果

```bash
cat output_dir/vsm13_results.json | python -m json.tool
```

### 2. 查看详细分数

```bash
cat output_dir/vsm13_detailed_scores.json | python -m json.tool
```

### 3. 查看特定国家的生成数据

```bash
cat output_dir/vsm13_generated_China.json | python -m json.tool | head -50
```

### 4. 提取平均欧式距离

```bash
python -c "import json; data = json.load(open('output_dir/vsm13_results.json')); print(f'Average Distance: {data[\"average_euclidean_distance\"]:.2f}')"
```

### 5. 提取特定国家的欧式距离

```bash
python -c "import json; data = json.load(open('output_dir/vsm13_results.json')); print(f'China Distance: {data[\"euclidean_distances\"][\"China\"]:.2f}')"
```

### 6. 提取特定国家的维度分数

```bash
python -c "import json; data = json.load(open('output_dir/vsm13_results.json')); scores = data['country_dimension_scores']['China']; print(f'China PDI: {scores[\"PDI\"]:.2f}')"
```

### 7. 比较预测分数和标准分数

```bash
python -c "
import json
data = json.load(open('output_dir/vsm13_detailed_scores.json'))
for country in ['China', 'United States', 'Japan']:
    report = data[country]
    print(f'{country}:')
    for dim in ['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR']:
        pred = report['dimension_scores'][dim]
        std = report['hofstede_scores'][dim]
        print(f'  {dim}: {pred:.2f} (predicted) vs {std} (standard)')
"
```

---

## 💡 数据分析示例

### 1. 找出表现最好的国家

```bash
python -c "
import json
data = json.load(open('output_dir/vsm13_results.json'))
distances = data['euclidean_distances']
best_country = min(distances, key=distances.get)
print(f'Best: {best_country} ({distances[best_country]:.2f})')
"
```

### 2. 找出表现最差的国家

```bash
python -c "
import json
data = json.load(open('output_dir/vsm13_results.json'))
distances = data['euclidean_distances']
worst_country = max(distances, key=distances.get)
print(f'Worst: {worst_country} ({distances[worst_country]:.2f})')
"
```

### 3. 计算维度的平均分数

```bash
python -c "
import json
import numpy as np
data = json.load(open('output_dir/vsm13_results.json'))
scores = data['country_dimension_scores']
for dim in ['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR']:
    values = [s[dim] for s in scores.values()]
    print(f'{dim}: {np.mean(values):.2f} (avg), {np.std(values):.2f} (std)')
"
```

---

## 🎯 快速查看结果

### 查看所有文件

```bash
ls -lh output_dir/
```

### 查看文件大小

```bash
du -sh output_dir/*
```

### 查看文件数量

```bash
ls output_dir/ | wc -l
```

### 查看 JSON 文件的键

```bash
python -c "import json; data = json.load(open('output_dir/vsm13_results.json')); print(list(data.keys()))"
```

---

## 📈 对比多个模型的结果

### 创建对比脚本

```bash
#!/bin/bash

echo "Model Type | Backbone | Average Distance"
echo "-----------|----------|------------------"

for model_type in base lora_only culturemoe; do
    for backbone in qwen llama; do
        dir=$(ls -d /root/autodl-tmp/CultureMoE/Culture_Alignment/vsm13/${model_type}_${backbone}_* 2>/dev/null | sort -r | head -1)
        if [ -n "$dir" ]; then
            distance=$(python -c "import json; data = json.load(open('$dir/vsm13_results.json')); print(f'{data[\"average_euclidean_distance\"]:.2f}')")
            printf "%-10s | %-8s | %s\n" "$model_type" "$backbone" "$distance"
        fi
    done
done
```

---

## 🎉 总结

### ✅ 输出文件

1. **vsm13_results.json**：主要结果汇总
2. **vsm13_detailed_scores.json**：详细的分数和距离报告
3. **vsm13_all_generated.json**：所有国家的生成数据汇总
4. **vsm13_generated_[Country].json**：单个国家的生成数据（10 个文件）

### 📊 包含的信息

- ✅ 每个国家的维度分数
- ✅ 霍夫斯泰德标准分数
- ✅ 欧式距离
- ✅ 模型生成的完整答案
- ✅ 提取的数字答案

### 🚀 使用

```bash
# 运行评估
sh run_eval_vsm13.sh lora_only qwen

# 查看结果
cat output_dir/vsm13_results.json | python -m json.tool
```

---

**现在可以查看和分析评估结果了！** 🚀

