# VSM13 Prompt 更新

## ✅ 已修复

### 问题
之前的 prompt 没有包含文化背景信息，只是简单地要求回答数字。

### 修复内容

#### 1. 更新 Hofstede 分数

```python
HOFSTEDE_SCORES = {
    "China": [80, 43, 66, 30, 77, 24],
    "South Korea": [60, 58, 39, 85, 86, 29],
    "Turkey": [66, 46, 45, 85, 35, 49],
    "Saudi Arabia": [72, 48, 43, 64, 27, 14],
    "Bangladesh": [80, 5, 55, 60, 38, 20],
    "Germany": [35, 79, 66, 65, 57, 40],
    "Portugal": [63, 59, 31, 99, 42, 33],
    "Spain": [57, 67, 42, 86, 47, 44],
}
```

#### 2. 添加国家形容词映射

```python
COUNTRY_TO_ADJECTIVE = {
    "China": "Chinese",
    "South Korea": "South Korean",
    "Turkey": "Turkish",
    "Saudi Arabia": "Saudi Arabian",
    "Bangladesh": "Bangladeshi",
    "Germany": "German",
    "Portugal": "Portuguese",
    "Spain": "Spanish",
}
```

#### 3. 更新 Prompt

**之前**：
```python
prompt = f"{question}\n\nPlease answer with ONLY ONE NUMBER (1, 2, 3, 4, or 5).\nYour answer:"
```

**现在**：
```python
country_adj = COUNTRY_TO_ADJECTIVE.get(country, country)
prompt = f"You are a {country_adj} culture chatbot that knows {country_adj} culture very well. Please answer with ONLY ONE NUMBER (1, 2, 3, 4, or 5) based on the {country}'s cultural background.\n\n{question}\n\nYour answer:"
```

## 📝 Prompt 示例

### 中国 (China)
```
You are a Chinese culture chatbot that knows Chinese culture very well. Please answer with ONLY ONE NUMBER (1, 2, 3, 4, or 5) based on the China's cultural background.

Please think of an ideal job, disregarding your present job, if you have one. In choosing an ideal job, how important would it be to you to have sufficient time for your personal or home life : 1 = of utmost importance. 2 = very important. 3 = of moderate importance. 4 = of little importance. 5 = of very little or no importance.

Your answer:
```

### 德国 (Germany)
```
You are a German culture chatbot that knows German culture very well. Please answer with ONLY ONE NUMBER (1, 2, 3, 4, or 5) based on the Germany's cultural background.

Please think of an ideal job, disregarding your present job, if you have one. In choosing an ideal job, how important would it be to you to have sufficient time for your personal or home life : 1 = of utmost importance. 2 = very important. 3 = of moderate importance. 4 = of little importance. 5 = of very little or no importance.

Your answer:
```

### 沙特阿拉伯 (Saudi Arabia)
```
You are a Saudi Arabian culture chatbot that knows Saudi Arabian culture very well. Please answer with ONLY ONE NUMBER (1, 2, 3, 4, or 5) based on the Saudi Arabia's cultural background.

Please think of an ideal job, disregarding your present job, if you have one. In choosing an ideal job, how important would it be to you to have sufficient time for your personal or home life : 1 = of utmost importance. 2 = very important. 3 = of moderate importance. 4 = of little importance. 5 = of very little or no importance.

Your answer:
```

## 🎯 预期效果

### 之前（无文化背景）
- 模型可能给出通用的回答
- 不同国家的答案可能相似
- VSM13 分数差异小

### 现在（有文化背景）
- 模型会根据国家的文化背景回答
- 不同国家的答案应该有明显差异
- VSM13 分数应该更接近 Hofstede 标准分数

## 🚀 使用方法

```bash
# 测试单个国家
sh run_eval_vsm13.sh base llama --countries China

# 测试多个国家
sh run_eval_vsm13.sh base llama --countries China Germany Spain

# 测试所有国家（默认）
sh run_eval_vsm13.sh base llama
```

## 📊 测试的国家

当前支持的 8 个国家：
1. 🇨🇳 China
2. 🇰🇷 South Korea
3. 🇹🇷 Turkey
4. 🇸🇦 Saudi Arabia
5. 🇧🇩 Bangladesh
6. 🇩🇪 Germany
7. 🇵🇹 Portugal
8. 🇪🇸 Spain

## ✅ 验证清单

- [x] 更新 Hofstede 分数（8个国家）
- [x] 添加国家形容词映射
- [x] 更新 prompt 模板
- [x] 修改 `generate_answer()` 函数
- [x] 修改 `test_country()` 函数调用
- [x] 测试脚本运行

## 🎉 完成

现在 `eval_vsm13.py` 会：
1. ✅ 为每个国家使用正确的文化背景 prompt
2. ✅ 使用正确的 Hofstede 分数
3. ✅ 使用正确的 VSM13 计算公式
4. ✅ 生成更准确的文化一致性评估

立即运行：
```bash
sh run_eval_vsm13.sh base llama

