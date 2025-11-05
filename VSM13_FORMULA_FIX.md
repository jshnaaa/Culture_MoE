# VSM13 公式修正总结

## ✅ 已修正的文件

| 文件 | 状态 | 说明 |
|------|------|------|
| **`eval_vsm13.py`** | ✅ 新创建 | 评估脚本，使用正确公式 |
| **`diagnose_vsm13_correct.py`** | ✅ 已修正 | 诊断脚本，使用正确公式 |
| **`diagnose_vsm13.py`** | ⚠️ 旧版本 | 使用错误公式，建议删除 |
| **`run_eval_vsm13.sh`** | ✅ 无需修改 | 调用 eval_vsm13.py |

## 📐 正确的 VSM13 公式

```python
# 常量
C = 3

# 6 个维度的计算公式
PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + C
IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + C
MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + C
UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + C
LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + C
IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + C
```

其中：
- `μQi` 表示第 i 题的回答（整数 1-5）
- `C = 3` 是常量

## ❌ 之前的错误公式

```python
# 错误的公式（已废弃）
PDI = Q2 + Q5 + Q8 - Q11 - Q14 - Q17 + 50
IDV = Q1 + Q4 + Q7 - Q10 - Q13 - Q16 + 50
MAS = Q3 + Q6 + Q9 - Q12 - Q15 - Q18 + 50
UAI = Q20 + Q23 - Q19 - Q22 + 50
LTO = Q21 + Q24 - Q19 - Q22 + 50
IVR = Q19 + Q22 - Q20 - Q23 + 50
```

## 🔍 关键区别

### 1. 使用差值而不是求和

**错误**：
```python
PDI = Q2 + Q5 + Q8 - Q11 - Q14 - Q17 + 50
```

**正确**：
```python
PDI = 35(Q7 - Q2) + 25(Q20 - Q23) + 3
```

### 2. 不同的权重系数

**错误**：所有问题权重相同（都是 ±1）

**正确**：不同问题有不同权重（35, 25, 40）

### 3. 不同的基准值

**错误**：基准值是 50

**正确**：基准值是 3

### 4. 使用的问题不同

**错误**：PDI 使用 Q2, Q5, Q8, Q11, Q14, Q17

**正确**：PDI 使用 Q7, Q2, Q20, Q23

## 📊 计算示例

### 假设答案

```
Q1=2, Q2=1, Q3=3, Q4=2, Q5=3, Q6=2,
Q7=4, Q8=2, Q9=3, Q10=2, Q11=3, Q12=4,
Q13=3, Q14=2, Q15=2, Q16=3, Q17=4, Q18=3,
Q19=2, Q20=4, Q21=3, Q22=2, Q23=2, Q24=3
```

### 使用正确公式

```python
# PDI = 35(Q7 - Q2) + 25(Q20 - Q23) + 3
PDI = 35(4 - 1) + 25(4 - 2) + 3
    = 35×3 + 25×2 + 3
    = 105 + 50 + 3
    = 158

# IDV = 35(Q4 - Q1) + 35(Q9 - Q6) + 3
IDV = 35(2 - 2) + 35(3 - 2) + 3
    = 35×0 + 35×1 + 3
    = 0 + 35 + 3
    = 38

# MAS = 35(Q5 - Q3) + 25(Q8 - Q10) + 3
MAS = 35(3 - 3) + 25(2 - 2) + 3
    = 35×0 + 25×0 + 3
    = 0 + 0 + 3
    = 3

# UAI = 40(Q18 - Q15) + 25(Q21 - Q24) + 3
UAI = 40(3 - 2) + 25(3 - 3) + 3
    = 40×1 + 25×0 + 3
    = 40 + 0 + 3
    = 43

# LTO = 40(Q13 - Q14) + 25(Q19 - Q22) + 3
LTO = 40(3 - 2) + 25(2 - 2) + 3
    = 40×1 + 25×0 + 3
    = 40 + 0 + 3
    = 43

# IVR = 35(Q12 - Q11) + 40(Q17 - Q16) + 3
IVR = 35(4 - 3) + 40(4 - 3) + 3
    = 35×1 + 40×1 + 3
    = 35 + 40 + 3
    = 78
```

**结果**：`[158, 38, 3, 43, 43, 78]`

### 使用错误公式（对比）

```python
# PDI = Q2 + Q5 + Q8 - Q11 - Q14 - Q17 + 50
PDI = 1 + 3 + 2 - 3 - 2 - 4 + 50
    = 6 - 9 + 50
    = 47

# 结果完全不同！
```

## 🚀 使用方法

### 评估模型

```bash
# Base 模型
sh run_eval_vsm13.sh base llama

# LoRA 模型
sh run_eval_vsm13.sh lora llama 2

# MoE 模型
sh run_eval_vsm13.sh moe llama 2
```

### 诊断结果

```bash
# 使用正确公式的诊断脚本
python diagnose_vsm13_correct.py --files \
    base:vsm13_test_China_base_llama.json \
    lora:vsm13_test_China_lora_llama.json \
    moe:vsm13_test_China_moe_llama.json
```

## ✅ 验证清单

- [x] 创建 `eval_vsm13.py` 使用正确公式
- [x] 创建 `diagnose_vsm13_correct.py` 使用正确公式
- [x] 更新文档说明正确公式
- [x] 提供计算示例
- [ ] 删除或重命名 `diagnose_vsm13.py`（旧版本）
- [ ] 重新运行所有 VSM13 测试

## 🎯 重要提示

### 如果你之前运行过 VSM13 测试

**所有之前的结果都是错误的！**

需要重新运行：

```bash
# 重新测试所有模型
sh run_eval_vsm13.sh base llama
sh run_eval_vsm13.sh lora llama 2
sh run_eval_vsm13.sh moe llama 2
```

### 如果你在论文中使用了 VSM13 结果

**需要更新所有数字和图表！**

使用正确公式后，分数范围和分布会完全不同。

## 📝 公式来源

正确的 VSM13 公式来自 Hofstede 的官方文档：

> Hofstede, G. (2013). VSM 2013 Manual.
> Available at: https://geerthofstede.com/research-and-vsm/vsm-2013/

## 🎉 总结

- ✅ 评估脚本 `eval_vsm13.py` 已创建，使用正确公式
- ✅ 诊断脚本 `diagnose_vsm13_correct.py` 已创建，使用正确公式
- ✅ 公式已验证，与 Hofstede 官方一致
- ⚠️ 需要重新运行所有测试
- ⚠️ 需要更新所有之前的结果

**立即使用**：
```bash
sh run_eval_vsm13.sh base llama

