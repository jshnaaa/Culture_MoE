# 保存每个 Epoch 的验证集评估结果

## 🎯 功能说明

现在 `run_ft_lora_only_gen.sh` 和 `run_ft_culturemoe_gen.sh` 会在输出目录里保存**每一次在验证集上的评估结果**。

---

## 📁 输出文件

### 1. 最新的生成答案

**文件名**: `generated_answers.json`

**说明**: 保存最后一次评估的生成答案（会被覆盖）

**用途**: 快速查看最新的评估结果

### 2. 每个 Epoch 的生成答案

**文件名**: `generated_answers_epoch_{epoch}.json`

**说明**: 保存每个 epoch 的生成答案（不会被覆盖）

**用途**: 追踪模型在不同 epoch 的表现

**示例**:
- `generated_answers_epoch_1.json` - 第 1 个 epoch 的评估结果
- `generated_answers_epoch_2.json` - 第 2 个 epoch 的评估结果
- `generated_answers_epoch_3.json` - 第 3 个 epoch 的评估结果
- ...

### 3. Epoch 评估结果汇总

**文件名**: `epoch_eval_results.json`

**说明**: 汇总所有 epoch 的评估指标

**内容**:
```json
[
  {
    "epoch": 1,
    "train_loss": 0.5234,
    "eval_loss": 0.4521,
    "eval_accuracy": 0.3545,
    "correct": 39,
    "total": 110
  },
  {
    "epoch": 2,
    "train_loss": 0.4987,
    "eval_loss": 0.4234,
    "eval_accuracy": 0.3636,
    "correct": 40,
    "total": 110
  },
  ...
]
```

---

## 📊 文件结构

### 训练完成后的输出目录

```
output_dir/
├── best_lora/                          # 最佳模型权重
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── generated_answers.json              # 最新的生成答案
├── generated_answers_epoch_1.json      # 第 1 个 epoch 的生成答案
├── generated_answers_epoch_2.json      # 第 2 个 epoch 的生成答案
├── generated_answers_epoch_3.json      # 第 3 个 epoch 的生成答案
├── ...
├── generated_answers_epoch_12.json     # 第 12 个 epoch 的生成答案
├── epoch_eval_results.json             # Epoch 评估结果汇总
└── config.json                         # 训练配置
```

---

## 🔍 查看评估结果

### 1. 查看最新的生成答案

```bash
cat output_dir/generated_answers.json | python -m json.tool | head -50
```

### 2. 查看特定 epoch 的生成答案

```bash
# 查看第 3 个 epoch 的生成答案
cat output_dir/generated_answers_epoch_3.json | python -m json.tool | head -50
```

### 3. 查看所有 epoch 的评估指标

```bash
cat output_dir/epoch_eval_results.json | python -m json.tool
```

### 4. 查看准确率变化趋势

```python
import json
import matplotlib.pyplot as plt

# 读取评估结果
with open('output_dir/epoch_eval_results.json', 'r') as f:
    results = json.load(f)

# 提取 epoch 和准确率
epochs = [r['epoch'] for r in results]
accuracies = [r['eval_accuracy'] for r in results]

# 绘制曲线
plt.plot(epochs, accuracies, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.grid(True)
plt.savefig('accuracy_curve.png')
plt.show()
```

---

## 📈 生成答案文件格式

### 文件内容

```json
[
  {
    "instruction": "### Question: ... ### Answer: ",
    "input": "",
    "true_output": "2",
    "label": 0,
    "generated_text": "2",
    "predicted_answer": "2",
    "correct": true
  },
  {
    "instruction": "### Question: ... ### Answer: ",
    "input": "",
    "true_output": "1",
    "label": 0,
    "generated_text": "3",
    "predicted_answer": "3",
    "correct": false
  },
  ...
]
```

### 字段说明

- `instruction`: 输入的指令
- `input`: 输入的文本（通常为空）
- `true_output`: 真实答案
- `label`: 文化标签
- `generated_text`: 模型生成的文本
- `predicted_answer`: 从生成文本中提取的答案
- `correct`: 是否正确

---

## 🎯 使用场景

### 1. 追踪模型性能

查看模型在不同 epoch 的表现，找出最佳 epoch：

```bash
# 查看所有 epoch 的准确率
cat output_dir/epoch_eval_results.json | python -m json.tool | grep "eval_accuracy"
```

### 2. 分析错误样本

查看特定 epoch 的错误样本：

```python
import json

# 读取第 3 个 epoch 的生成答案
with open('output_dir/generated_answers_epoch_3.json', 'r') as f:
    answers = json.load(f)

# 找出错误样本
errors = [a for a in answers if not a['correct']]

print(f"错误样本数: {len(errors)}")
for i, error in enumerate(errors[:5]):
    print(f"\n错误样本 {i+1}:")
    print(f"  Question: {error['instruction'][:80]}...")
    print(f"  True Output: {error['true_output']}")
    print(f"  Predicted Answer: {error['predicted_answer']}")
```

### 3. 对比不同 epoch

对比不同 epoch 的表现：

```python
import json

# 读取两个 epoch 的生成答案
with open('output_dir/generated_answers_epoch_1.json', 'r') as f:
    epoch1 = json.load(f)

with open('output_dir/generated_answers_epoch_12.json', 'r') as f:
    epoch12 = json.load(f)

# 计算准确率
acc1 = sum(1 for a in epoch1 if a['correct']) / len(epoch1)
acc12 = sum(1 for a in epoch12 if a['correct']) / len(epoch12)

print(f"Epoch 1 Accuracy: {acc1:.4f}")
print(f"Epoch 12 Accuracy: {acc12:.4f}")
print(f"Improvement: {(acc12 - acc1):.4f}")
```

---

## 🔧 修改清单

### ft_lora_only_gen.py

- ✅ 添加 `epoch` 参数到 `generate_and_evaluate_answers` 函数
- ✅ 保存每个 epoch 的生成答案到 `generated_answers_epoch_{epoch}.json`
- ✅ 保留最新的生成答案到 `generated_answers.json`

### ft_culturemoe_from_base_gen.py

- ✅ 添加 `epoch` 参数到 `generate_and_evaluate_answers` 函数
- ✅ 保存每个 epoch 的生成答案到 `generated_answers_epoch_{epoch}.json`
- ✅ 保留最新的生成答案到 `generated_answers.json`

---

## 📝 示例输出

### 训练过程

```
Epoch 1/12
Training: 100%|████████████████████████████████████████| 124/124 [00:45<00:00,  2.75it/s]
  Train Loss: 0.5234
Evaluating: 100%|████████████████████████████████████████| 14/14 [00:05<00:00,  2.80it/s]
  Eval Loss: 0.4521
Generating: 100%|████████████████████████████████████████| 110/110 [00:15<00:00,  7.33it/s]
  Eval Accuracy: 0.3545
  ✅ Best model saved (loss: 0.4521)

Epoch 2/12
Training: 100%|████████████████████████████████████████| 124/124 [00:45<00:00,  2.75it/s]
  Train Loss: 0.4987
Evaluating: 100%|████████████████████████████████████████| 14/14 [00:05<00:00,  2.80it/s]
  Eval Loss: 0.4234
Generating: 100%|████████████████████████████████████████| 110/110 [00:15<00:00,  7.33it/s]
  Eval Accuracy: 0.3636
  ✅ Best model saved (loss: 0.4234)

...

✅ Training completed!
```

### 输出文件

```
Files generated:
  - best_lora/ (Best LoRA weights)
  - epoch_eval_results.json (Epoch-by-epoch results)
  - generated_answers.json (Latest generated answers)
  - generated_answers_epoch_1.json (Epoch 1 answers)
  - generated_answers_epoch_2.json (Epoch 2 answers)
  - ...
  - generated_answers_epoch_12.json (Epoch 12 answers)
  - config.json (Training configuration)
```

---

## 🎉 总结

### 功能

- ✅ 保存每个 epoch 的生成答案
- ✅ 保留最新的生成答案
- ✅ 汇总所有 epoch 的评估指标

### 优势

- ✅ 追踪模型性能变化
- ✅ 分析错误样本
- ✅ 对比不同 epoch
- ✅ 找出最佳 epoch

### 使用

```bash
# 运行训练
sh run_ft_lora_only_gen.sh llama 4

# 查看评估结果
cat output_dir/epoch_eval_results.json | python -m json.tool

# 查看特定 epoch 的生成答案
cat output_dir/generated_answers_epoch_3.json | python -m json.tool | head -50
```

---

**功能已添加！** ✅

