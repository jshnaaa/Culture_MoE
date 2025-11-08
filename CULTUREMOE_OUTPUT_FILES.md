# CultureMoE 训练输出文件说明

## 📋 概述

运行 `bash run_train_culturemoe_from_base_gen.sh` 后，脚本会在每一轮训练后保存以下文件：

✅ **每轮生成的答案**：`generated_answers.json`
✅ **每轮验证集评估结果**：`epoch_eval_results.json`
✅ **最佳模型的 MoE 权重**：`best_moe/moe_state_dict.pt`（仅 MoE 部分，不是完整模型）
✅ **最终评估结果**：`final_eval_results.json`
✅ **训练配置**：`config.json`

## 📁 输出目录结构

```
output_dir/
├── best_moe/                          # 最佳模型的 MoE 权重
│   ├── moe_config.json               # MoE 配置
│   └── moe_state_dict.pt             # MoE 权重（仅 MoE 部分）
├── generated_answers.json             # 最后一轮生成的答案
├── eval_metrics.json                  # 最后一轮的评估指标
├── epoch_eval_results.json            # 所有轮次的评估结果
├── final_eval_results.json            # 最终评估结果
└── config.json                        # 训练配置
```

## 📊 文件详细说明

### 1. **generated_answers.json**（每轮生成）

**内容**：最后一轮训练后在验证集上生成的答案

```json
[
  {
    "predicted": "1",
    "true": "1"
  },
  {
    "predicted": "2",
    "true": "2"
  },
  ...
]
```

**说明**：
- 保存最后一轮的生成答案
- 用于验证模型的生成质量
- 每轮都会覆盖（只保留最后一轮）

### 2. **eval_metrics.json**（每轮生成）

**内容**：最后一轮的评估指标

```json
{
  "accuracy": 0.8234,
  "precision": 0.8123,
  "recall": 0.8045,
  "f1": 0.8084
}
```

**说明**：
- 基于 `generated_answers.json` 计算
- 每轮都会覆盖（只保留最后一轮）

### 3. **epoch_eval_results.json**（累积保存）

**内容**：所有轮次的评估结果

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
  {
    "epoch": 2,
    "train_loss": 0.9876,
    "train_accuracy": 0.8123,
    "eval_loss": 0.9234,
    "eval_accuracy": 0.8234,
    "eval_precision": 0.8145,
    "eval_recall": 0.8056,
    "eval_f1": 0.8100
  },
  ...
]
```

**说明**：
- 累积保存所有轮次的结果
- 用于分析训练过程
- 每轮都会追加新的结果

### 4. **best_moe/moe_state_dict.pt**（最佳模型）

**内容**：最佳模型的 MoE 权重（仅 MoE 部分）

**特点**：
- ✅ **只包含 MoE 部分**，不包含 LLaMA 模型
- ✅ **大小较小**（通常 100-500MB）
- ✅ **可以单独加载和使用**
- ❌ **不能单独运行**（需要配合 Base 模型 + LoRA 权重）

**包含的参数**：
```python
# 包含的参数（不包含 llama_model.* 的参数）
- shared.*              # Shared 层
- router.*              # Router 层
- experts_layer.*       # Experts 层
- classifier.*          # 分类头
```

**不包含的参数**：
```python
# 不包含的参数
- llama_model.*         # LLaMA 模型（太大）
```

### 5. **best_moe/moe_config.json**

**内容**：MoE 配置

```json
{
  "num_experts": 6,
  "shared_hidden_dim": 2048,
  "router_hidden_dim": 1024,
  "experts_hidden_dim": 2048,
  "moe_lora_rank": 16,
  "num_classes": 10,
  "classification_hidden_dim": 512,
  "dropout": 0.1,
  "num_heads": 8
}
```

### 6. **final_eval_results.json**

**内容**：最终评估结果

```json
{
  "best_epoch": 5,
  "best_accuracy": 0.8456,
  "final_eval_loss": 0.8234,
  "final_eval_accuracy": 0.8456,
  "final_eval_precision": 0.8345,
  "final_eval_recall": 0.8267,
  "final_eval_f1": 0.8306,
  "num_val_samples": 1234,
  "evaluation_time": "2025-11-08 15:30:45"
}
```

### 7. **config.json**

**内容**：训练配置

```json
{
  "base_model": "/path/to/base_model",
  "lora_weights": "/path/to/lora_weights",
  "num_classes": 10,
  "use_culture_loss": true,
  "culture_loss_lambda": 0.5,
  "num_epochs": 10,
  "best_epoch": 5,
  "best_accuracy": 0.8456,
  "final_eval_accuracy": 0.8456,
  "training_time": "2025-11-08 15:30:45"
}
```

## 🔄 训练流程中的保存时机

### 每个 Epoch 的流程

```
Epoch 1
├── 训练（train_epoch）
│   └── 保存 train_loss, train_accuracy
├── 验证（evaluate）
│   └── 保存 eval_loss, eval_accuracy
├── 生成式评估（generate_and_evaluate）
│   ├── 生成答案
│   ├── 保存 generated_answers.json（覆盖）
│   ├── 保存 eval_metrics.json（覆盖）
│   └── 返回 accuracy
├── 检查是否是最佳模型
│   └── 如果是，保存 best_moe/moe_state_dict.pt
└── 保存 epoch_eval_results.json（追加）

Epoch 2
├── ...（重复）
└── 保存 epoch_eval_results.json（追加）

...

Epoch 10
├── ...（重复）
└── 保存 epoch_eval_results.json（追加）

训练结束
├── 加载最佳模型
├── 最终评估
├── 保存 final_eval_results.json
└── 保存 config.json
```

## 💾 文件大小参考

| 文件 | 大小 | 说明 |
|------|------|------|
| **generated_answers.json** | 1-10MB | 取决于验证集大小 |
| **eval_metrics.json** | <1KB | 固定大小 |
| **epoch_eval_results.json** | 10-100KB | 取决于 epoch 数 |
| **best_moe/moe_state_dict.pt** | 100-500MB | 仅 MoE 部分 |
| **best_moe/moe_config.json** | <1KB | 固定大小 |
| **final_eval_results.json** | <1KB | 固定大小 |
| **config.json** | <1KB | 固定大小 |

## 🎯 如何使用这些文件

### 1. **分析训练过程**

```python
import json

# 加载所有轮次的结果
with open('epoch_eval_results.json', 'r') as f:
    results = json.load(f)

# 找到最佳轮次
best_epoch = max(results, key=lambda x: x['eval_accuracy'])
print(f"Best epoch: {best_epoch['epoch']}")
print(f"Best accuracy: {best_epoch['eval_accuracy']:.4f}")

# 绘制训练曲线
import matplotlib.pyplot as plt

epochs = [r['epoch'] for r in results]
train_loss = [r['train_loss'] for r in results]
eval_acc = [r['eval_accuracy'] for r in results]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, eval_acc)
plt.xlabel('Epoch')
plt.ylabel('Eval Accuracy')
plt.title('Evaluation Accuracy')
plt.tight_layout()
plt.show()
```

### 2. **加载最佳模型进行推理**

```python
import torch
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel

# 加载 Base 模型 + LoRA 权重
base_model = ...  # 加载 base 模型
lora_model = ...  # 加载 LoRA 权重

# 创建 CultureMoE 模型
moe_model = LlamaSharedRouterExpertsModel(lora_model, config, moe_args)

# 加载最佳 MoE 权重
best_moe_weights = torch.load('best_moe/moe_state_dict.pt')
moe_model.load_state_dict(best_moe_weights, strict=False)

# 进行推理
outputs = moe_model(input_ids, attention_mask)
```

### 3. **评估最佳模型**

```bash
# 使用最佳 MoE 权重在测试集上评估
python eval_culturemoe_from_components_gen.py \
    --base_model_path /path/to/base_model \
    --lora_weights_path /path/to/lora_weights \
    --moe_weights_path output_dir/best_moe \
    --test_file /path/to/test.json \
    --output_dir /path/to/eval_output
```

## ⚠️ 重要说明

### 1. **MoE 权重只包含 MoE 部分**

```python
# ✅ 正确：只保存 MoE 部分
moe_state_dict = {}
for name, param in model.named_parameters():
    if not name.startswith('llama_model.'):
        moe_state_dict[name] = param.cpu()

# ❌ 错误：保存完整模型
torch.save(model.state_dict(), 'full_model.pt')  # 太大！
```

### 2. **generated_answers.json 只保留最后一轮**

- 每轮都会覆盖
- 如果需要保存所有轮次，需要手动备份

### 3. **epoch_eval_results.json 是累积的**

- 每轮都会追加
- 包含所有轮次的结果
- 用于分析训练过程

## 📝 总结

| 文件 | 更新频率 | 用途 |
|------|---------|------|
| **generated_answers.json** | 每轮覆盖 | 查看最后一轮的生成答案 |
| **eval_metrics.json** | 每轮覆盖 | 查看最后一轮的评估指标 |
| **epoch_eval_results.json** | 每轮追加 | 分析整个训练过程 |
| **best_moe/moe_state_dict.pt** | 最佳时保存 | 保存最佳模型的 MoE 权重 |
| **final_eval_results.json** | 训练结束保存 | 最终评估结果 |
| **config.json** | 训练结束保存 | 训练配置 |

✅ **所有功能都已实现！** 🎉

