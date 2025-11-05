# VSM13 诊断指南

## 🔍 问题现象

答案分布：`1:8, 2:7, 3:7, 4:2`

**说明**：模型**不是**保守地都选 3，而是在做选择。但为什么不同模型的 VSM13 分数还是相同或相似？

## 💡 可能的原因

### 原因 1：VSM13 公式的抵消效应

VSM13 计算公式有**正项**和**负项**：

```python
pdi = Q2 + Q5 + Q8 - Q11 - Q14 - Q17 + 50
     ↑  正项  ↑    ↑   负项   ↑
```

**关键点**：
- 如果正项和负项的答案**分布相似**
- 它们会**相互抵消**
- 最终结果都接近 **50**（基准值）

### 示例

假设两个模型的答案：

```
Model A: [2, 1, 3, 2, 4, 1, 3, 2, 1, 3, 2, 4, 1, 3, 2, 1, 3, 2, 1, 3, 2, 4, 1, 3]
Model B: [3, 2, 2, 3, 3, 2, 2, 3, 2, 2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 2, 2]
```

虽然答案不同，但如果：
- 正项的平均值 ≈ 负项的平均值
- 计算结果都会接近 50

### 计算示例

```python
# Model A
PDI = (1 + 4 + 2) - (3 + 3 + 2) + 50
    = 7 - 8 + 50
    = 49

# Model B
PDI = (2 + 3 + 3) - (2 + 2 + 3) + 50
    = 8 - 7 + 50
    = 51

# 差异很小！
```

## 🔬 诊断方法

### 步骤 1：使用诊断脚本

```bash
# 分析单个模型
python diagnose_vsm13.py --files base:vsm13_test_China_base_llama.json

# 对比两个模型
python diagnose_vsm13.py --files \
    base:vsm13_test_China_base_llama.json \
    lora:vsm13_test_China_lora_llama.json

# 对比三个模型
python diagnose_vsm13.py --files \
    base:vsm13_test_China_base_llama.json \
    lora:vsm13_test_China_lora_llama.json \
    moe:vsm13_test_China_moe_llama.json
```

### 步骤 2：查看输出

#### 2.1 答案分布

```
1. Answer Distribution:
   1:  8 ( 33.3%)
   2:  7 ( 29.2%)
   3:  7 ( 29.2%)
   4:  2 (  8.3%)

   Mean: 2.13
   Std:  0.95
```

**分析**：
- 如果 Mean ≈ 2.5-3.0，说明答案比较均匀
- 如果 Std 很小（< 0.5），说明答案集中

#### 2.2 维度计算详情

```
2. Dimension Calculations:

   PDI (Power Distance):
      Positive terms (Q2, Q5, Q8):    [1, 4, 2] → sum = 7
      Negative terms (Q11, Q14, Q17): [3, 3, 2] → sum = 8
      Difference: 7 - 8 = -1
      Final: -1 + 50 = 49
```

**关键观察**：
- 如果 Difference ≈ 0，说明正负项抵消
- 如果所有维度的 Difference 都很小，最终分数都接近 50

#### 2.3 模型对比

```
VSM13 Scores Comparison
Dimension     base     lora      moe
-----------------------------------------
PDI             49       50       51
IDV             50       49       50
MAS             51       50       49
UAI             50       51       50
LTO             49       50       51
IVR             51       50       49

Euclidean Distance: 2.45
```

**分析**：
- 如果 Euclidean Distance < 5，说明非常相似
- 如果 Euclidean Distance < 10，说明比较相似
- 如果 Euclidean Distance > 20，说明有明显差异

#### 2.4 原始答案对比

```
Comparing raw answers:
   Same answers: 18/24 (75.0%)

   ❌ Models are giving almost identical answers!
      This is the main problem.
```

**结论**：
- 如果 Same answers > 75%，说明模型答案几乎相同
- 如果 Same answers < 50%，说明答案不同，但 VSM13 公式导致抵消

## 🎯 判断标准

### 情况 A：模型答案几乎相同

```
Same answers: 20/24 (83.3%)
Euclidean Distance: 2.45
```

**原因**：模型没有学到区分性特征

**解决方案**：
1. 检查训练数据质量
2. 增加训练轮数
3. 改进模型架构
4. 使用更强的 Prompt

### 情况 B：答案不同，但 VSM13 抵消

```
Same answers: 10/24 (41.7%)
Euclidean Distance: 3.12
```

**原因**：VSM13 公式的抵消效应

**解决方案**：
1. 使用其他评估指标（如直接对比答案）
2. 分析每个问题的答案差异
3. 使用更细粒度的文化维度

### 情况 C：真正的差异

```
Same answers: 5/24 (20.8%)
Euclidean Distance: 25.67
```

**原因**：模型学到了不同的文化特征

**结论**：这是期望的结果！

## 📊 实际案例分析

### 案例 1：你的情况

```
答案分布：1:8, 2:7, 3:7, 4:2
Mean: 2.13
Std: 0.95
```

**分析**：
1. 答案分布比较均匀（不是都选 3）
2. 但需要检查：
   - 不同模型的答案是否相同？
   - 正负项是否抵消？

**诊断命令**：
```bash
python diagnose_vsm13.py --files \
    base:vsm13_test_China_base_llama.json \
    lora:vsm13_test_China_lora_llama.json \
    moe:vsm13_test_China_moe_llama.json
```

**预期输出**：
- 如果 `Same answers > 75%` → 模型问题
- 如果 `Same answers < 50%` → VSM13 公式问题

## 🔧 解决方案

### 方案 1：改进模型（如果是模型问题）

```bash
# 1. 检查训练数据
cat training_data.json | jq '.[] | .output' | sort | uniq -c

# 2. 增加训练轮数
--num_train_epochs 10  # 从 3 增加到 10

# 3. 使用更强的 Prompt
prompt = f"""
You are answering a cultural values questionnaire.
Think carefully about your cultural background and values.

Question: {question}

Rate from 1 to 5:
1 = of utmost importance
2 = very important
3 = of moderate importance
4 = of little importance
5 = of very little or no importance

Your answer (just the number):"""
```

### 方案 2：使用其他评估指标（如果是 VSM13 问题）

```python
# 1. 直接对比答案
def compare_answers(answers1, answers2):
    same = sum(1 for a, b in zip(answers1, answers2) if a == b)
    return same / len(answers1)

# 2. 计算答案分布的 KL 散度
from scipy.stats import entropy

def kl_divergence(dist1, dist2):
    return entropy(dist1, dist2)

# 3. 使用余弦相似度
from sklearn.metrics.pairwise import cosine_similarity

def answer_similarity(answers1, answers2):
    return cosine_similarity([answers1], [answers2])[0][0]
```

### 方案 3：分析每个问题

```python
# 查看哪些问题的答案差异最大
for i in range(24):
    if answers1[i] != answers2[i]:
        print(f"Q{i+1}: Model1={answers1[i]}, Model2={answers2[i]}")
```

## 🎉 总结

### 诊断流程

1. ✅ 运行诊断脚本
2. ✅ 查看答案分布
3. ✅ 检查 Same answers 比例
4. ✅ 分析 Euclidean Distance
5. ✅ 判断是模型问题还是 VSM13 问题

### 判断标准

| Same Answers | Euclidean Distance | 结论 |
|--------------|-------------------|------|
| > 75% | < 5 | 模型答案几乎相同 |
| 50-75% | 5-10 | 答案有差异，但 VSM13 抵消 |
| < 50% | > 10 | 模型学到了不同特征 |

### 立即诊断

```bash
python diagnose_vsm13.py --files \
    base:vsm13_test_China_base_llama.json \
    lora:vsm13_test_China_lora_llama.json \
    moe:vsm13_test_China_moe_llama.json
```

查看输出中的 `Same answers` 和 `Euclidean Distance`，就能知道问题所在！

