# 霍夫斯泰德文化维度评估

## 概述

这个工具用于评估 Base LLaMA 模型在霍夫斯泰德文化维度上的一致性。通过让模型回答 VSM2013 的 24 个问题，计算模型的文化维度得分，并与霍夫斯泰德给定的标准得分进行对比。

## 文件说明

- `eval_hofstede_culture.py` - 主评估脚本
- `run_eval_hofstede.sh` - Shell 启动脚本
- `VSM2013_questions_template.json` - VSM2013 问题模板

## 使用方法

### 1. 准备 VSM2013 问题文件

将 `VSM2013_questions_template.json` 复制到服务器：

```bash
# 上传到服务器
scp VSM2013_questions_template.json root@your-server:/root/autodl-fs/VSM2013_questions.json
```

或者直接在服务器上创建该文件。

### 2. 运行评估

```bash
sh run_eval_hofstede.sh
```

### 3. 自定义参数

如果需要修改参数，编辑 `run_eval_hofstede.sh`：

```bash
MODEL_PATH="/path/to/your/model"
DATA_PATH="/path/to/VSM2013_questions.json"
OUTPUT_DIR="/path/to/output"
DEVICE="cuda"  # 或 "cpu"
```

## 评估流程

### 1. 加载模型

从指定路径加载 LLaMA 3.1-8B-Instruct 模型。

### 2. 对每个国家进行评估

评估的国家列表：
- China (Chinese)
- South Korea (South Korean)
- Turkey (Turkish)
- Saudi Arabia (Saudi Arabian)
- Bangladesh (Bangladeshi)
- Germany (German)
- Portugal (Portuguese)
- Spain (Spanish)

### 3. 生成答案

对每个国家，使用以下提示词格式：

```
You are a [Country Adjective] culture chatbot that knows [Country Adjective] culture very well.

[Question]

Please answer with only a number (1-5):
```

例如，对于中国：
```
You are a Chinese culture chatbot that knows Chinese culture very well.

How important is it for you to have a job respected by your family and friends?

Please answer with only a number (1-5):
```

### 4. 计算文化维度得分

根据 VSM2013 公式计算六个维度：

- **PDI** (Power Distance Index): `35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 3`
- **IDV** (Individualism): `35(μQ4 − μQ1) + 35(μQ9 − μQ6) + 3`
- **MAS** (Masculinity): `35(μQ5 − μQ3) + 25(μQ8 − μQ10) + 3`
- **UAI** (Uncertainty Avoidance): `40(μQ18 − μQ15) + 25(μQ21 − μQ24) + 3`
- **LTO** (Long-Term Orientation): `40(μQ13 − μQ14) + 25(μQ19 − μQ22) + 3`
- **IVR** (Indulgence vs Restraint): `35(μQ12 − μQ11) + 40(μQ17 − μQ16) + 3`

### 5. 计算欧氏距离

```
Distance = √∑(d_model − d_hofstede)², ∀d ∈ {PDI, IDV, MAS, UAI, LTO, IVR}
```

## 输出文件

### 1. 每个国家的答案文件

格式：`VSM2013_<Country>_<Timestamp>.json`

示例：`VSM2013_China_20251030_143025.json`

内容：
```json
[
  {
    "id": "Q1",
    "Question": "How important is it for you to have a job respected by your family and friends?",
    "Answer": 4
  },
  ...
]
```

### 2. 总结文件

格式：`hofstede_evaluation_summary_<Timestamp>.json`

示例：`hofstede_evaluation_summary_20251030_143025.json`

内容：
```json
{
  "model_path": "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct",
  "model_name": "Meta-Llama-3.1-8B-Instruct",
  "evaluation_time": "2025-10-30T14:30:25",
  "results": {
    "China": {
      "model_scores": {
        "PDI": 75.5,
        "IDV": 45.2,
        "MAS": 62.8,
        "UAI": 35.6,
        "LTO": 70.3,
        "IVR": 28.9
      },
      "hofstede_scores": {
        "PDI": 80,
        "IDV": 43,
        "MAS": 66,
        "UAI": 30,
        "LTO": 77,
        "IVR": 24
      },
      "euclidean_distance": 12.34
    },
    ...
  }
}
```

## 输出示例

### 控制台输出

```
================================================================================
Loading model: /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
================================================================================
✅ Model loaded successfully
   Model name: Meta-Llama-3.1-8B-Instruct

Loading VSM2013 questions from: /root/autodl-fs/VSM2013_questions.json
✅ Loaded 24 questions

================================================================================
Evaluating country: China (Chinese)
================================================================================

[1/24] Question Q1:
  How important is it for you to have a job respected by your family and friends?...
  Response: 4
  Answer: 4

[2/24] Question Q2:
  How important is it for you to have sufficient time for your personal or family life?...
  Response: 3
  Answer: 3

...

✅ Saved answered questions to: /root/autodl-fs/output/hofstede_evaluation/.../VSM2013_China_20251030_143025.json

================================================================================
Results for China:
================================================================================
Dimension       Model Score     Hofstede Score  Difference
--------------------------------------------------------------------------------
PDI             75.50           80              -4.50
IDV             45.20           43              2.20
MAS             62.80           66              -3.20
UAI             35.60           30              5.60
LTO             70.30           77              -6.70
IVR             28.90           24              4.90
--------------------------------------------------------------------------------
Euclidean Distance: 12.34
================================================================================

...

================================================================================
Overall Statistics:
================================================================================
Country              Euclidean Distance
--------------------------------------------------------------------------------
China                12.34
South Korea          15.67
Turkey               18.92
Saudi Arabia         14.23
Bangladesh           16.45
Germany              11.89
Portugal             13.56
Spain                12.78
--------------------------------------------------------------------------------
Average              14.48
================================================================================
```

## 霍夫斯泰德标准得分

| Country | PDI | IDV | MAS | UAI | LTO | IVR |
|---------|-----|-----|-----|-----|-----|-----|
| China | 80 | 43 | 66 | 30 | 77 | 24 |
| South Korea | 60 | 58 | 39 | 85 | 86 | 29 |
| Turkey | 66 | 46 | 45 | 85 | 35 | 49 |
| Saudi Arabia | 72 | 48 | 43 | 64 | 27 | 14 |
| Bangladesh | 80 | 5 | 55 | 60 | 38 | 20 |
| Germany | 35 | 79 | 66 | 65 | 57 | 40 |
| Portugal | 63 | 59 | 31 | 99 | 42 | 33 |
| Spain | 57 | 67 | 42 | 86 | 47 | 44 |

## 维度说明

- **PDI (Power Distance Index)**: 权力距离指数 - 社会对权力不平等的接受程度
- **IDV (Individualism)**: 个人主义 vs 集体主义
- **MAS (Masculinity)**: 男性化 vs 女性化 - 竞争性 vs 关怀性
- **UAI (Uncertainty Avoidance)**: 不确定性规避 - 对不确定性和模糊性的容忍度
- **LTO (Long-Term Orientation)**: 长期导向 vs 短期导向
- **IVR (Indulgence vs Restraint)**: 放纵 vs 克制 - 对享乐和欲望的态度

## 注意事项

1. **答案提取**：脚本会尝试从模型回答中提取 1-5 的数字。如果提取失败，会使用默认值 3。

2. **计算公式**：所有公式中的常量 C 都设置为 3。

3. **评估时间**：评估 8 个国家 × 24 个问题 = 192 次推理，可能需要较长时间。

4. **GPU 内存**：确保有足够的 GPU 内存加载 LLaMA 3.1-8B 模型。

5. **问题文件格式**：确保 VSM2013 问题文件格式正确，Answer 字段可以为 null。

## 结果解读

- **欧氏距离越小**：表示模型的文化倾向越接近霍夫斯泰德的标准得分
- **欧氏距离越大**：表示模型的文化倾向与标准得分差异较大

典型的距离范围：
- < 15：非常接近
- 15-25：较为接近
- 25-40：有一定差异
- > 40：差异较大

## 故障排除

### 问题：模型加载失败

```bash
# 检查模型路径是否正确
ls -la /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
```

### 问题：CUDA OOM

```bash
# 使用 CPU
DEVICE="cpu"
```

或者在 Python 脚本中使用 8-bit 量化：

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map="auto"
)
```

### 问题：问题文件格式错误

确保 JSON 格式正确：
```bash
python -m json.tool VSM2013_questions.json
```

## 扩展

### 评估其他模型

修改 `run_eval_hofstede.sh` 中的 `MODEL_PATH`：

```bash
MODEL_PATH="/path/to/your/model"
```

### 添加更多国家

在 `eval_hofstede_culture.py` 中添加：

```python
COUNTRY_TO_ADJECTIVE = {
    # ... 现有国家 ...
    "Japan": "Japanese",
    "India": "Indian",
}

HOFSTEDE_SCORES = {
    # ... 现有得分 ...
    "Japan": [54, 46, 95, 92, 88, 42],
    "India": [77, 48, 56, 40, 51, 26],
}
```

## 参考文献

- Hofstede, G. (2011). Dimensionalizing Cultures: The Hofstede Model in Context.
- VSM2013 Manual: Values Survey Module 2013

