# VSM13 生成答案问题 - 快速修复指南

## 🚨 问题

运行评估时输出卡在：
```
🔍 Sample 0:
```

## ✅ 原因

`eval_vsm13.py` 中的循环变量混乱导致答案生成失败。

## 🔧 修复

已在 `eval_vsm13.py` 中修复：

1. **修正循环变量**
   ```python
   # 修改前
   for question_idx, sample in enumerate(tqdm(data, ...)):

   # 修改后
   for sample_idx, sample in enumerate(tqdm(data, ...)):
       question_idx = sample.get('question_idx', sample_idx)
   ```

2. **添加调试输出**
   ```python
   if sample_idx < 3:
       print(f"  Sample {sample_idx}: Generated='{generated_text}' → Answer={answer}")
   ```

3. **改进错误处理**
   ```python
   if answer == -1:
       if sample_idx < 10:
           print(f"  ⚠️  Sample {sample_idx}: Failed to extract number from: '{generated_text}'")
       continue
   ```

## 🚀 使用

### 评估 MOE 模型

```bash
# LLaMA
bash run_eval_vsm13.sh moe llama

# Qwen
bash run_eval_vsm13.sh moe qwen
```

### 直接运行 Python

```bash
python eval_vsm13.py \
    --model_type moe \
    --backbone llama \
    --data_path /root/autodl-fs/vsm13.json \
    --output_dir /path/to/output \
    --device cuda
```

## 📊 预期输出

```
================================================================================
Evaluating China...
================================================================================
Generating answers for China: 100%|████████████| 13/13 [00:05<00:00,  2.50it/s]
  Sample 0: Generated='1' → Answer=1
  Sample 1: Generated='2' → Answer=2
  Sample 2: Generated='3' → Answer=3

================================================================================
Computing dimension scores using Hofstede formulas...
================================================================================

📊 Dimension Scores by Country:
...

✅ Results saved to /path/to/output
```

## 📁 输出文件

```
output_dir/
├── vsm13_results.json              # 主要结果
├── vsm13_detailed_scores.json      # 详细分数
├── vsm13_all_generated.json        # 所有生成数据
├── vsm13_generated_China.json      # 中国数据
├── vsm13_generated_South Korea.json
├── vsm13_generated_Turkey.json
└── ...
```

## 💡 快速检查

```bash
# 查看平均欧式距离
python -c "import json; data = json.load(open('/path/to/output/vsm13_results.json')); print(f'Average Distance: {data[\"average_euclidean_distance\"]:.2f}')"

# 比较多个模型
for model in base lora_only moe; do
    dist=$(python -c "import json; data = json.load(open('/path/to/${model}_output/vsm13_results.json')); print(f'{data[\"average_euclidean_distance\"]:.2f}')")
    echo "$model: $dist"
done
```

## ✨ 现在可以正确评估了！

修复后，评估应该能正常运行并生成完整的结果。

