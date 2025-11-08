# CultureMoE 训练输出文件快速参考

## ✅ 是否保存这些文件？

| 文件 | 保存？ | 说明 |
|------|-------|------|
| **generated_answers.json** | ✅ 是 | 每轮生成的答案（最后一轮） |
| **每轮验证集评估结果** | ✅ 是 | 保存在 `epoch_eval_results.json` |
| **最佳模型 MoE 权重** | ✅ 是 | 保存在 `best_moe/moe_state_dict.pt` |
| **完整模型** | ❌ 否 | 只保存 MoE 部分，不保存完整模型 |

## 📁 输出文件位置

```
output_dir/
├── best_moe/
│   ├── moe_config.json              # MoE 配置
│   └── moe_state_dict.pt            # ✅ MoE 权重（仅 MoE 部分）
├── generated_answers.json            # ✅ 最后一轮生成的答案
├── eval_metrics.json                 # 最后一轮的评估指标
├── epoch_eval_results.json           # ✅ 所有轮次的评估结果
├── final_eval_results.json           # 最终评估结果
└── config.json                       # 训练配置
```

## 📊 关键文件说明

### 1. **generated_answers.json**

```json
[
  {"predicted": "1", "true": "1"},
  {"predicted": "2", "true": "2"},
  ...
]
```

- ✅ 每轮都生成
- 📝 只保留最后一轮
- 📊 用于验证生成质量

### 2. **epoch_eval_results.json**

```json
[
  {
    "epoch": 1,
    "train_loss": 1.2345,
    "train_accuracy": 0.7234,
    "eval_loss": 1.1234,
    "eval_accuracy": 0.7456,
    "eval_precision": 0.7345,
    "eval_recall": 0.7234,
    "eval_f1": 0.7289
  },
  ...
]
```

- ✅ 每轮都追加
- 📈 包含所有轮次
- 📊 用于分析训练过程

### 3. **best_moe/moe_state_dict.pt**

- ✅ 最佳模型时保存
- 📦 仅包含 MoE 部分（不包含 LLaMA）
- 💾 大小：100-500MB
- 🎯 用于推理和评估

## 🔄 训练流程

```
每个 Epoch
├── 训练
├── 验证
├── 生成式评估
│   ├── 生成答案 → generated_answers.json（覆盖）
│   ├── 计算指标 → eval_metrics.json（覆盖）
│   └── 返回准确率
├── 检查最佳模型
│   └── 如果是最佳 → best_moe/moe_state_dict.pt（保存）
└── 保存轮次结果 → epoch_eval_results.json（追加）

训练结束
├── 加载最佳模型
├── 最终评估
├── 保存 final_eval_results.json
└── 保存 config.json
```

## 💡 使用示例

### 查看训练过程

```bash
# 查看所有轮次的结果
cat output_dir/epoch_eval_results.json | python -m json.tool

# 查看最后一轮的生成答案
cat output_dir/generated_answers.json | python -m json.tool | head -20

# 查看最终结果
cat output_dir/final_eval_results.json | python -m json.tool
```

### 分析训练曲线

```python
import json
import matplotlib.pyplot as plt

with open('output_dir/epoch_eval_results.json', 'r') as f:
    results = json.load(f)

epochs = [r['epoch'] for r in results]
eval_acc = [r['eval_accuracy'] for r in results]

plt.plot(epochs, eval_acc, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Eval Accuracy')
plt.title('Training Progress')
plt.show()
```

### 加载最佳模型

```python
import torch

# 加载 MoE 权重
moe_weights = torch.load('output_dir/best_moe/moe_state_dict.pt')

# 加载到模型
model.load_state_dict(moe_weights, strict=False)
```

### 在测试集上评估

```bash
python eval_culturemoe_from_components_gen.py \
    --base_model_path /path/to/base_model \
    --lora_weights_path /path/to/lora_weights \
    --moe_weights_path output_dir/best_moe \
    --test_file /path/to/test.json \
    --output_dir /path/to/eval_output
```

## 📊 文件大小参考

| 文件 | 大小 |
|------|------|
| generated_answers.json | 1-10MB |
| epoch_eval_results.json | 10-100KB |
| best_moe/moe_state_dict.pt | 100-500MB |
| 其他文件 | <1KB |

## ⚠️ 重要提示

1. **MoE 权重只包含 MoE 部分**
   - 不包含 LLaMA 模型
   - 需要配合 Base 模型 + LoRA 权重使用

2. **generated_answers.json 只保留最后一轮**
   - 每轮都会覆盖
   - 如果需要保存所有轮次，需要手动备份

3. **epoch_eval_results.json 是累积的**
   - 每轮都会追加
   - 包含所有轮次的结果

## 🎯 总结

✅ **是的，所有功能都已实现！**

- ✅ 每轮生成 `generated_answers.json`
- ✅ 每轮保存验证集评估结果到 `epoch_eval_results.json`
- ✅ 保存最佳模型的 MoE 权重到 `best_moe/moe_state_dict.pt`
- ✅ 只保存 MoE 部分，不保存完整模型

