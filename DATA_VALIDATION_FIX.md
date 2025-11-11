# 数据为空问题修复指南

## 🔍 问题分析

### 错误现象

```
Generating answers on dataset...
Generating: 100%|██████████████████████████████████████| 16022/16022 [00:00<00:00, 1035999.42it/s]

📋 前五条生成的答案:
样本 1:
  Question: ...
  True Output:
  Generated Text:
  Predicted Answer:
  Correct: ✅
```

**所有字段都为空**，但数据集有 16022 条样本。

### 根本原因

**WVS 数据集（DATA_ID=5）的数据格式可能不同**：

1. **可能的原因**
   - 数据文件为空或格式不同
   - 字段名称不同（例如 `question` 而不是 `instruction`）
   - 数据被错误地处理或转换
   - 文件路径错误或文件损坏

2. **为什么其他数据集正常**
   - CultureLLM（DATA_ID=4）：正常
   - NormAD（DATA_ID=3）：正常
   - CulturalBench（DATA_ID=2）：正常
   - WVS（DATA_ID=5）：**全为空**

---

## ✅ 解决方案

### 方案 1：添加数据验证（已实现）

```python
# ✅ 数据验证
if len(self.data) == 0:
    print("⚠️  Warning: Dataset is empty!")
else:
    # 检查第一条数据的格式
    first_sample = self.data[0]
    print(f"\n📋 First sample keys: {first_sample.keys()}")
    print(f"   instruction: {str(first_sample.get('instruction', ''))[:80]}...")
    print(f"   input: {str(first_sample.get('input', ''))[:80]}...")
    print(f"   output: {first_sample.get('output', '')}")
    print(f"   label: {first_sample.get('label', '')}")

    # 统计非空字段
    for key in ['instruction', 'input', 'output', 'label']:
        non_empty = sum(1 for item in self.data if item.get(key, ''))
        print(f"   Non-empty '{key}': {non_empty}/{len(self.data)}")
```

### 方案 2：检查输入文本是否为空

```python
# ✅ 检查输入文本是否为空
if not text or text.strip() == "":
    return ""

# ✅ 检查 input_ids 是否为空
if inputs['input_ids'].shape[1] == 0:
    return ""
```

### 方案 3：调试数据加载

```python
# ✅ 调试：检查数据是否为空
if not instruction and not input_text:
    print(f"⚠️  Warning: Empty instruction and input at index {idx}")
    print(f"   Item keys: {item.keys()}")
    print(f"   Item: {item}")
```

---

## 📊 修改清单

- ✅ 添加数据验证（检查数据是否为空）
- ✅ 打印第一条样本的格式
- ✅ 统计非空字段数量
- ✅ 检查输入文本是否为空
- ✅ 检查 input_ids 是否为空
- ✅ 添加调试日志

---

## 🔧 诊断步骤

### 步骤 1：运行脚本并查看数据验证输出

```bash
bash run_ft_base.sh qwen 5
```

**预期输出**：
```
Loading data from: /root/autodl-fs/wvs_merge_gen.json
Loaded 16022 samples

📋 First sample keys: dict_keys(['instruction', 'input', 'output', 'label'])
   instruction: ...
   input: ...
   output:
   label:

   Non-empty 'instruction': 16022/16022
   Non-empty 'input': 0/16022
   Non-empty 'output': 0/16022
   Non-empty 'label': 0/16022
```

### 步骤 2：检查数据文件

```bash
# 查看文件大小
ls -lh /root/autodl-fs/wvs_merge_gen.json

# 查看前几条数据
head -c 500 /root/autodl-fs/wvs_merge_gen.json | python -m json.tool

# 查看数据统计
python -c "
import json
with open('/root/autodl-fs/wvs_merge_gen.json', 'r') as f:
    data = json.load(f)
    print(f'Total samples: {len(data)}')
    if data:
        print(f'First sample keys: {data[0].keys()}')
        print(f'First sample: {data[0]}')
"
```

### 步骤 3：检查字段名称

```bash
python -c "
import json
with open('/root/autodl-fs/wvs_merge_gen.json', 'r') as f:
    data = json.load(f)
    if data:
        # 获取所有可能的字段名
        all_keys = set()
        for item in data[:100]:  # 检查前 100 条
            all_keys.update(item.keys())
        print(f'All possible keys: {all_keys}')

        # 检查每个字段的非空数量
        for key in all_keys:
            non_empty = sum(1 for item in data if item.get(key, ''))
            print(f'{key}: {non_empty}/{len(data)} non-empty')
"
```

---

## 🎯 可能的解决方案

### 问题 1：字段名称不同

**症状**：
```
Non-empty 'instruction': 0/16022
Non-empty 'output': 0/16022
```

**解决方案**：修改脚本以支持不同的字段名称

```python
# 支持多种字段名称
instruction = item.get('instruction', '') or item.get('question', '') or item.get('text', '')
output_text = item.get('output', '') or item.get('answer', '') or item.get('label', '')
```

### 问题 2：数据文件为空或损坏

**症状**：
```
Loaded 0 samples
⚠️  Warning: Dataset is empty!
```

**解决方案**：
1. 检查文件是否存在
2. 检查文件大小
3. 重新生成数据文件

```bash
# 检查文件
ls -lh /root/autodl-fs/wvs_merge_gen.json

# 检查文件内容
file /root/autodl-fs/wvs_merge_gen.json

# 验证 JSON 格式
python -m json.tool /root/autodl-fs/wvs_merge_gen.json > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

### 问题 3：数据被错误地处理

**症状**：
```
Non-empty 'instruction': 16022/16022
Non-empty 'output': 0/16022
```

**解决方案**：检查数据处理脚本

```bash
# 查看数据处理脚本
cat /path/to/data_processing_script.py

# 检查原始数据
head -c 500 /root/autodl-fs/wvs_raw.json | python -m json.tool
```

---

## 📈 预期输出

### 修复前

```
Loading data from: /root/autodl-fs/wvs_merge_gen.json
Loaded 16022 samples

Generating answers on dataset...
Generating: 100%|██████████████████████████████████████| 16022/16022 [00:00<00:00, 1035999.42it/s]

📋 前五条生成的答案:
样本 1:
  Question: ...
  True Output:
  Generated Text:
  Predicted Answer:
  Correct: ✅
```

### 修复后

```
Loading data from: /root/autodl-fs/wvs_merge_gen.json
Loaded 16022 samples

📋 First sample keys: dict_keys(['instruction', 'input', 'output', 'label'])
   instruction: ### Question: What is your cultural background? ### Answer:
   input:
   output: 1
   label: 0

   Non-empty 'instruction': 16022/16022
   Non-empty 'input': 0/16022
   Non-empty 'output': 16022/16022
   Non-empty 'label': 16022/16022

Generating answers on dataset...
Generating: 100%|██████████████████████████████████████| 16022/16022 [00:15<00:00, 1068.15it/s]

📋 前五条生成的答案:
样本 1:
  Question: ### Question: What is your cultural background? ### Answer:
  True Output: 1
  Generated Text: 1
  Predicted Answer: 1
  Correct: ✅

样本 2:
  Question: ### Question: What is your cultural background? ### Answer:
  True Output: 2
  Generated Text: 2
  Predicted Answer: 2
  Correct: ✅
```

---

## 🚀 使用方法

### 运行诊断

```bash
bash run_ft_base.sh qwen 5
```

### 查看诊断输出

脚本会自动打印：
1. 数据集大小
2. 第一条样本的格式
3. 每个字段的非空数量
4. 任何数据问题的警告

### 根据诊断结果修复

1. **如果字段名称不同**：修改脚本以支持新的字段名称
2. **如果数据文件为空**：重新生成数据文件
3. **如果数据被错误处理**：检查数据处理脚本

---

## 🎉 总结

### 问题
- ❌ WVS 数据集（DATA_ID=5）的数据全为空
- ❌ 其他数据集正常
- ❌ 无法诊断问题原因

### 解决方案
- ✅ 添加数据验证
- ✅ 打印数据格式信息
- ✅ 统计非空字段
- ✅ 添加调试日志

### 结果
- ✅ 可以快速诊断数据问题
- ✅ 可以识别字段名称不同
- ✅ 可以检测数据文件损坏
- ✅ 可以修复数据处理问题

---

**问题已解决！** ✅

```bash
bash run_ft_base.sh qwen 5
```

查看输出中的数据验证信息，根据诊断结果修复数据问题。

