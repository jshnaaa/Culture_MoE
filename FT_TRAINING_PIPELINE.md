# 标准语言建模损失微调 - 训练流程图

## 🏗️ 完整的训练流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         开始训练                                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    第一步：训练 LoRA Only 模型                        │
│                                                                      │
│  sh run_ft_lora_only_from_components.sh llama 4                    │
│                                                                      │
│  时间：1-2 小时                                                      │
│  GPU 内存：~20GB                                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    加载 Base 模型 (LLaMA 3.1-8B)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    添加 LoRA 适配器                                   │
│                    (r=64, alpha=16, dropout=0.1)                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    使用标准语言建模损失微调                           │
│                                                                      │
│  Loss = -Σ log P(token_t | token_<t)                               │
│                                                                      │
│  6 个 Epoch，每个 Epoch：                                            │
│  - 训练：计算生成损失，更新 LoRA 权重                                │
│  - 验证：评估损失和准确率                                            │
│  - 生成：在验证集上生成答案                                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    保存最好的 LoRA 权重                               │
│                                                                      │
│  /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/                 │
│  ft_lora_only_cultureLLM_llama_*/best_lora/                        │
│                                                                      │
│  输出文件：                                                          │
│  - best_lora/                    (LoRA 权重)                        │
│  - epoch_eval_results.json       (每个 epoch 的结果)                │
│  - generated_answers.json        (生成的答案)                       │
│  - config.json                   (配置)                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    第二步：训练 CultureMoE 模型                       │
│                                                                      │
│  sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5       │
│                                                                      │
│  时间：3-4 小时                                                      │
│  GPU 内存：~30GB                                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    加载 Base 模型 (LLaMA 3.1-8B)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    加载第一步保存的 LoRA 权重                         │
│                    (is_trainable=False，冻结)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    合并 LoRA 权重到 Base 模型                        │
│                    (merge_and_unload)                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    添加 MOE 层                                        │
│                                                                      │
│  ├── Shared Experts (共享专家)                                      │
│  ├── Router (路由器)                                                │
│  └── Task-specific Experts (任务特定专家)                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    使用标准语言建模损失 + 文化专注性损失微调          │
│                                                                      │
│  Total Loss = Generation Loss + λ * Culture Loss                   │
│                                                                      │
│  30 个 Epoch，每个 Epoch：                                           │
│  - 训练：计算总损失，更新 MOE 权重（LoRA 冻结）                     │
│  - 验证：评估损失和准确率                                            │
│  - 生成：在验证集上生成答案                                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    保存最好的 MOE 权重                                │
│                                                                      │
│  /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/                 │
│  ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/ │
│                                                                      │
│  输出文件：                                                          │
│  - best_moe/                     (MOE 权重)                         │
│  - epoch_eval_results.json       (每个 epoch 的结果)                │
│  - generated_answers.json        (生成的答案)                       │
│  - config.json                   (配置)                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    消融实验（可选）                                   │
│                                                                      │
│  for weight in 0.1 0.3 0.5 0.7 0.9; do                             │
│    sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 $weight │
│  done                                                               │
│                                                                      │
│  比较不同 culture_loss_lambda 的性能                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         训练完成                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 模型架构演变

### 第一步：LoRA Only 模型

```
┌──────────────────────────────────────────┐
│         Base Model (LLaMA 3.1-8B)        │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Transformer Layers                │  │
│  │  (冻结，不更新)                     │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│         LoRA Adapter (微调)               │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  q_proj: r=64                      │  │
│  │  v_proj: r=64                      │  │
│  │  (更新)                             │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│         Output (生成式)                   │
│                                          │
│  Loss = -Σ log P(token_t | token_<t)    │
└──────────────────────────────────────────┘
```

### 第二步：CultureMoE 模型

```
┌──────────────────────────────────────────┐
│    Base Model + LoRA (合并，冻结)        │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Transformer Layers                │  │
│  │  + LoRA Weights (冻结)              │  │
│  │  (不更新)                           │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│         MOE 层 (新增，微调)               │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Shared Experts                    │  │
│  │  (共享知识，更新)                   │  │
│  └────────────────────────────────────┘  │
│                    ↓                      │
│  ┌────────────────────────────────────┐  │
│  │  Router                            │  │
│  │  (路由决策，更新)                   │  │
│  └────────────────────────────────────┘  │
│                    ↓                      │
│  ┌────────────────────────────────────┐  │
│  │  Task-specific Experts             │  │
│  │  (文化特定知识，更新)               │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│    Output (生成式 + 文化对齐)             │
│                                          │
│  Total Loss = Gen Loss + λ * Culture Loss│
└──────────────────────────────────────────┘
```

---

## 🔄 数据流向

### LoRA Only 训练

```
输入数据
  │
  ├─ text: "### Question: ... ### Answer: 2"
  │
  ▼
Tokenize
  │
  ├─ input_ids: [101, 2054, ...]
  ├─ attention_mask: [1, 1, 1, ...]
  │
  ▼
Base Model + LoRA
  │
  ├─ 前向传播
  ├─ 计算生成损失
  │
  ▼
反向传播
  │
  ├─ 更新 LoRA 权重
  │
  ▼
保存最好的 LoRA 权重
```

### CultureMoE 训练

```
输入数据
  │
  ├─ text: "### Question: ... ### Answer: 2"
  ├─ text_mask: "### Question: ... ### Answer: 2"
  ├─ label: "0"
  │
  ▼
Tokenize
  │
  ├─ input_ids: [101, 2054, ...]
  ├─ attention_mask: [1, 1, 1, ...]
  ├─ input_ids_mask: [101, 2054, ...]
  ├─ attention_mask_mask: [1, 1, 1, ...]
  │
  ▼
Base Model + LoRA (冻结)
  │
  ├─ 提取隐藏状态
  │
  ▼
MOE 层
  │
  ├─ Shared Experts 处理 text_mask
  ├─ Router 决策路由
  ├─ Task-specific Experts 处理 text
  │
  ▼
融合输出
  │
  ├─ 计算生成损失
  ├─ 计算文化损失
  ├─ 总损失 = 生成损失 + λ * 文化损失
  │
  ▼
反向传播
  │
  ├─ 更新 MOE 权重（LoRA 冻结）
  │
  ▼
保存最好的 MOE 权重
```

---

## 📈 训练曲线预期

### LoRA Only 模型

```
Loss
  │
  │  ╱╲
  │ ╱  ╲
  │╱    ╲___
  │         ╲___
  │             ╲___
  │                 ╲___
  │                     ╲___
  └─────────────────────────────► Epoch
  0   1   2   3   4   5   6

预期：
- 损失逐步下降
- 准确率逐步上升
- 6 个 epoch 后收敛
```

### CultureMoE 模型

```
Loss
  │
  │  ╱╲
  │ ╱  ╲
  │╱    ╲___
  │         ╲___
  │             ╲___
  │                 ╲___
  │                     ╲___
  │                         ╲___
  │                             ╲___
  │                                 ╲___
  │                                     ╲___
  │                                         ╲___
  └─────────────────────────────────────────────► Epoch
  0   5   10  15  20  25  30

预期：
- 损失逐步下降（比 LoRA Only 更平缓）
- 准确率逐步上升
- 30 个 epoch 后收敛
- 文化损失逐步减小
```

---

## 🎯 关键检查点

### 第一步完成后

```bash
# 检查 LoRA 权重是否保存
ls -la /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/best_lora/

# 应该看到：
# adapter_config.json
# adapter_model.bin
# ...

# 检查训练结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool

# 应该看到 6 个 epoch 的结果
```

### 第二步完成后

```bash
# 检查 MOE 权重是否保存
ls -la /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/best_moe/

# 应该看到：
# pytorch_model.bin

# 检查训练结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/epoch_eval_results.json | python -m json.tool

# 应该看到 30 个 epoch 的结果
```

---

## 🚀 快速命令参考

### 第一步：LoRA Only

```bash
# 开始训练
sh run_ft_lora_only_from_components.sh llama 4

# 监控进度
tail -f /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/training.log

# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool
```

### 第二步：CultureMoE

```bash
# 开始训练
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5

# 监控进度
tail -f /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/training.log

# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/epoch_eval_results.json | python -m json.tool
```

### 消融实验

```bash
# 测试不同的文化损失权重
for weight in 0.1 0.3 0.5 0.7 0.9; do
    sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 $weight
done

# 比较结果
python -c "
import json
import os

weights = [0.1, 0.3, 0.5, 0.7, 0.9]
for weight in weights:
    # 找到对应的输出目录
    # 加载结果并比较
    pass
"
```

---

## 🎉 总结

### 两步训练流程

1. **第一步：LoRA Only**
   - 加载 Base 模型
   - 添加 LoRA 适配器
   - 使用标准语言建模损失微调
   - 保存 LoRA 权重

2. **第二步：CultureMoE**
   - 加载 Base 模型
   - 加载第一步的 LoRA 权重（冻结）
   - 合并 LoRA 权重
   - 添加 MOE 层
   - 使用标准语言建模损失 + 文化专注性损失微调
   - 保存 MOE 权重

### 关键点

✅ MOE 是生成式的
✅ 只训练 MOE 部分（LoRA 冻结）
✅ 支持消融实验
✅ 每个 epoch 生成答案并评估

---

**现在你完全理解了训练流程！** 🚀

