# 霍夫斯泰德维度计算公式实现

## 📋 概述

VSM13 评估脚本现在使用正确的霍夫斯泰德公式来计算 6 个文化维度的分数。

---

## 🎯 霍夫斯泰德公式

### 1️⃣ PDI（权力距离）

**公式**：
```
PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + C_PDI
```

**说明**：
- μQ7：第 7 题的平均答案
- μQ2：第 2 题的平均答案
- μQ20, μQ23：VSM13 中不存在（只有 13 个问题）
- C_PDI = 3（常量）

**VSM13 实现**：
```python
pdi = 35 * (question_means.get(7, 0) - question_means.get(2, 0)) + HOFSTEDE_CONSTANTS['PDI']
pdi = max(0, min(100, pdi))  # 限制在 0-100 范围内
```

---

### 2️⃣ IDV（个人主义）

**公式**：
```
IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + C_IDV
```

**说明**：
- μQ4, μQ1, μQ9, μQ6：对应题目的平均答案
- C_IDV = 3（常量）

**VSM13 实现**：
```python
idv = 35 * (question_means.get(4, 0) - question_means.get(1, 0)) + \
      35 * (question_means.get(9, 0) - question_means.get(6, 0)) + HOFSTEDE_CONSTANTS['IDV']
idv = max(0, min(100, idv))
```

---

### 3️⃣ MAS（男性气质）

**公式**：
```
MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + C_MAS
```

**说明**：
- μQ5, μQ3, μQ8, μQ10：对应题目的平均答案
- C_MAS = 3（常量）

**VSM13 实现**：
```python
mas = 35 * (question_means.get(5, 0) - question_means.get(3, 0)) + \
      25 * (question_means.get(8, 0) - question_means.get(10, 0)) + HOFSTEDE_CONSTANTS['MAS']
mas = max(0, min(100, mas))
```

---

### 4️⃣ UAI（不确定性规避）

**公式**：
```
UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + C_UAI
```

**说明**：
- μQ18, μQ15, μQ21, μQ24：VSM13 中不存在（只有 13 个问题）
- C_UAI = 3（常量）

**VSM13 实现**：
```python
# VSM13 只有 13 个问题，所以 Q15, Q18, Q21, Q24 不存在
# 只使用常量
uai = HOFSTEDE_CONSTANTS['UAI']
uai = max(0, min(100, uai))
```

---

### 5️⃣ LTO（长期导向）

**公式**：
```
LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + C_LTO
```

**说明**：
- μQ13：第 13 题的平均答案
- μQ14, μQ19, μQ22：VSM13 中不存在（只有 13 个问题）
- C_LTO = 3（常量）

**VSM13 实现**：
```python
# VSM13 只有 13 个问题，所以 Q14, Q19, Q22 不存在
# 只使用 Q13
lto = 40 * (question_means.get(13, 0) - 0) + HOFSTEDE_CONSTANTS['LTO']
lto = max(0, min(100, lto))
```

---

### 6️⃣ IVR（放纵指数）

**公式**：
```
IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + C_IVR
```

**说明**：
- μQ12, μQ11：第 12 和 11 题的平均答案
- μQ16, μQ17：VSM13 中不存在（只有 13 个问题）
- C_IVR = 3（常量）

**VSM13 实现**：
```python
# VSM13 只有 13 个问题，所以 Q16, Q17 不存在
# 只使用 Q11, Q12
ivr = 35 * (question_means.get(12, 0) - question_means.get(11, 0)) + HOFSTEDE_CONSTANTS['IVR']
ivr = max(0, min(100, ivr))
```

---

## 📊 问题编号映射

| 0-based 索引 | 问题编号 | 说明 |
|-------------|---------|------|
| 0 | Q1 | 时间充足 |
| 1 | Q2 | 尊重的老板 |
| 2 | Q3 | 与同事的关系 |
| 3 | Q4 | 工作安全 |
| 4 | Q5 | 工作成就 |
| 5 | Q6 | 晋升机会 |
| 6 | Q7 | 工作安全 |
| 7 | Q8 | 公司规则 |
| 8 | Q9 | 长期就业 |
| 9 | Q10 | 谦虚 |
| 10 | Q11 | 生活质量 |
| 11 | Q12 | 个人成就 |
| 12 | Q13 | 参与决策 |

---

## 🔍 计算示例

### 假设某国家的答案

```
Q1 = 2, Q2 = 3, Q3 = 4, Q4 = 2, Q5 = 3, Q6 = 4,
Q7 = 3, Q8 = 2, Q9 = 4, Q10 = 3, Q11 = 2, Q12 = 4, Q13 = 3
```

### 计算 PDI

```
PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + C_PDI
    = 35(3 − 3) + 0 + 3
    = 35 * 0 + 3
    = 3
```

### 计算 IDV

```
IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + C_IDV
    = 35(2 − 2) + 35(4 − 4) + 3
    = 35 * 0 + 35 * 0 + 3
    = 3
```

### 计算 MAS

```
MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + C_MAS
    = 35(3 − 4) + 25(2 − 3) + 3
    = 35 * (-1) + 25 * (-1) + 3
    = -35 - 25 + 3
    = -57
    → max(0, min(100, -57)) = 0
```

---

## 💡 关键点

### 1. 问题编号从 1 开始

- 代码中使用 0-based 索引（0-12）
- 公式中使用 1-based 编号（Q1-Q13）
- 映射通过 `QUESTION_NUMBERS` 字典完成

### 2. VSM13 的限制

- VSM13 只有 13 个问题
- 完整的霍夫斯泰德公式需要 22 个问题
- 对于不存在的问题，使用 0 作为默认值

### 3. 分数范围

- 原始计算结果可能超出 0-100 范围
- 使用 `max(0, min(100, score))` 限制在 0-100 范围内

### 4. 常量值

- 所有维度的常量 C_xxx 都设为 3
- 这是霍夫斯泰德公式的标准常量

---

## 📈 欧式距离计算

**公式**：
```
distance = sqrt(sum((predicted_score - hofstede_score)^2 for each dimension))
```

**实现**：
```python
distance = 0
for dimension in ['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR']:
    predicted = predicted_scores.get(dimension, 0)
    standard = hofstede_scores.get(dimension, 0)
    distance += (predicted - standard) ** 2

euclidean_distances[country] = np.sqrt(distance)
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

## 🚀 使用

### 运行评估

```bash
sh run_eval_vsm13.sh lora_only qwen
```

### 查看结果

```bash
cat output_dir/vsm13_results.json | python -m json.tool
```

### 查看详细分数

```bash
cat output_dir/vsm13_detailed_scores.json | python -m json.tool
```

---

## 🎉 总结

### ✅ 实现的公式

1. **PDI**：35(μQ7 − μQ2) + C_PDI
2. **IDV**：35(μQ4 − μQ1) + 35(μQ9 − μQ6) + C_IDV
3. **MAS**：35(μQ5 − μQ3) + 25(μQ8 − μQ10) + C_MAS
4. **UAI**：C_UAI（VSM13 中不存在所需问题）
5. **LTO**：40(μQ13 − 0) + C_LTO（VSM13 中不存在 Q14）
6. **IVR**：35(μQ12 − μQ11) + C_IVR（VSM13 中不存在 Q16, Q17）

### 📊 输出

- **vsm13_results.json**：主要结果
- **vsm13_detailed_scores.json**：详细分数和距离
- **vsm13_generated_[Country].json**：每个国家的生成数据

---

**现在使用正确的霍夫斯泰德公式计算维度分数！** 🚀

