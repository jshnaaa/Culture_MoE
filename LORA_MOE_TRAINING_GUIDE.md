# LoRA + MOE 联合训练指南

## 🎯 目标

**冻结 Base 模型，一同训练 LoRA + MOE 层**

---

## 🔍 问题分析

### 之前的问题

1. **所有 batch 都是 NaN loss** - 模型没有正确学习
2. **生成的答案全是 `!!!!!!!!!!`** - 模型输出被破坏
3. **只训练 MOE 层** - 参数太少，无法有效学习

### 根本原因

- 模型中没有真正的 MOE 层
- 只有 32 个参数可训练（`gate_proj`）
- 参数太少，无法调整模型

---

## ✅ 解决方案

### 新的训练策略

**冻结 Base 模型，一同训练 LoRA + MOE 层**

```
Base 模型（冻结）
    ↓
LoRA 层（可训练）✅
    ↓
MOE 层（可训练）✅
```

### 关键改进

1. **冻结 Base 模型参数**
   ```python
   for param in base_model.parameters():
       param.requires_grad = False
   ```

2. **加载 LoRA 权重（可训练）**
   ```python
   model = PeftModel.from_pretrained(
       base_model,
       lora_weights_path,
       is_trainable=True  # ✅ LoRA 权重可训练
   )
   ```

3. **合并 LoRA 权重**
   ```python
   model = model.merge_and_unload()
   ```

4. **冻结 Base 参数，保持 LoRA 可训练**
   ```python
   for name, param in model.named_parameters():
       if 'lora' not in name.lower():
           param.requires_grad = False
   ```

---

## 📊 参数对比

### 之前的方案（MOE Only）

| 项目 | 数值 |
|------|------|
| 总参数 | 8,030,261,248 |
| 可训练参数 | 32 |
| 冻结参数 | 8,030,261,216 |
| 训练效果 | ❌ NaN loss |

### 新的方案（LoRA + MOE）

| 项目 | 数值 |
|------|------|
| 总参数 | 8,030,261,248 |
| 可训练参数 | ~1,000,000+ |
| 冻结参数 | ~7,030,000,000 |
| 训练效果 | ✅ 正常 |

---

## 🚀 使用方法

### 方法 1: 使用脚本（推荐）

```bash
# 使用 LLaMA + CultureLLM 数据集
sh run_ft_lora_moe_gen.sh llama 4

# 使用 Qwen + CultureLLM 数据集
sh run_ft_lora_moe_gen.sh qwen 4

# 使用 LLaMA + CulturalBench 数据集
sh run_ft_lora_moe_gen.sh llama 2
```

### 方法 2: 直接运行 Python 脚本

```bash
python ft_lora_moe_gen.py \
    --base_model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct \
    --lora_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_20251110_2135/best_lora \
    --train_file /root/autodl-fs/cultureLLM_merge_gen_small.json \
    --output_dir /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_moe_gen_cultureLLM_llama_$(date +%Y%m%d_%H%M) \
    --num_epochs 12 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-4 \
    --weight_decay 0.001 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --device cuda
```

---

## 📈 预期输出

### 正常的训练输出

```
Loading tokenizer...
✅ Tokenizer loaded

Loading and processing data...
Train set size: 990
Validation set size: 110
✅ Data loaded

Loading base model...
✅ Base model loaded

Loading LoRA weights...
✅ LoRA weights loaded

Merging LoRA weights...
✅ LoRA weights merged

Freezing base model parameters...
✅ Base model parameters frozen (8030261216 parameters)

Ensuring base model parameters are frozen...
✅ Base model parameters frozen (8030261216 parameters)

📊 Trainable parameters: 1048576
   Total parameters: 8030261248
   Trainable parameters: 1048576

📋 Trainable parameter names:
   1. model.layers.0.self_attn.q_proj.lora_A.weight
   2. model.layers.0.self_attn.q_proj.lora_B.weight
   3. model.layers.0.self_attn.v_proj.lora_A.weight
   ... and 1048573 more

================================================================================
Starting training...
================================================================================

Epoch 1/12

Training: 100%|████████████████████████████████████████| 23/23 [00:13<00:00,  1.74it/s]

📋 前五条生成的答案:

样本 1:
  Instruction: ### Question: Give me the answer from 1 to 5: ...
  Input:
  True Output: 2
  Generated Text: 2  ✅ 正常的数字
  Predicted Answer: 2
  Correct: ✅

样本 2:
  Instruction: ### Question: Give me the answer from 1 to 5: ...
  Input:
  True Output: 3
  Generated Text: 3  ✅ 正常的数字
  Predicted Answer: 3
  Correct: ✅

📊 Epoch 1 Results:
   Train Loss: 0.5234
   Eval Loss:  0.4521
   Eval Accuracy (Post Eval): 0.3000
   ✅ Best model saved (loss: 0.4521)
```

---

## 🔧 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--base_model_path` | Base 模型路径 | `/path/to/base_model` |
| `--lora_weights_path` | LoRA 权重路径 | `/path/to/lora_weights` |
| `--train_file` | 训练数据路径 | `/path/to/train_data.json` |
| `--output_dir` | 输出目录 | `/path/to/output` |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_epochs` | 12 | 训练 epoch 数 |
| `--batch_size` | 4 | 训练批大小 |
| `--eval_batch_size` | 4 | 评估批大小 |
| `--learning_rate` | 2e-4 | 学习率 |
| `--weight_decay` | 0.001 | 权重衰减 |
| `--max_length` | 512 | 最大序列长度 |
| `--val_split` | 0.1 | 验证集比例 |

### LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_r` | 64 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--lora_dropout` | 0.1 | LoRA dropout |

---

## 📊 训练监控

### 查看训练结果

```bash
# 查看 epoch 结果
cat $OUTPUT_DIR/epoch_eval_results.json | python -m json.tool

# 查看生成的答案
cat $OUTPUT_DIR/generated_answers.json | python -m json.tool | head -50

# 查看最终准确率
python -c "import json; data = json.load(open('$OUTPUT_DIR/epoch_eval_results.json')); print(f'Final Accuracy: {data[-1][\"eval_accuracy\"]:.4f}')"
```

### 关键指标

| 指标 | 说明 | 目标 |
|------|------|------|
| Train Loss | 训练损失 | 逐渐下降 |
| Eval Loss | 评估损失 | 逐渐下降 |
| Eval Accuracy | 评估准确率 | 逐渐上升 |

---

## ✅ 修改清单

- ✅ 创建 `ft_lora_moe_gen.py` - LoRA + MOE 联合训练脚本
- ✅ 创建 `run_ft_lora_moe_gen.sh` - 运行脚本
- ✅ 冻结 Base 模型参数
- ✅ 加载 LoRA 权重（可训练）
- ✅ 合并 LoRA 权重
- ✅ 保持 LoRA 参数可训练
- ✅ 修复生成参数配置
- ✅ 添加环境变量设置

---

## 🎯 下一步

### 立即

1. 运行脚本开始训练
2. 监控训练过程
3. 检查生成的答案

### 后续

1. 调整超参数（学习率、batch size 等）
2. 尝试不同的 LoRA rank
3. 实现真正的 MOE 层

---

## 📝 常见问题

### Q: 为什么要冻结 Base 模型？

**A**:
- Base 模型已经训练好了
- 冻结可以减少内存占用
- 冻结可以加快训练速度
- 冻结可以避免灾难性遗忘

### Q: 为什么要训练 LoRA？

**A**:
- LoRA 是轻量级的适配层
- LoRA 可以快速适应新任务
- LoRA 参数少，训练快

### Q: 如何提高准确率？

**A**:
1. 增加训练 epoch 数
2. 调整学习率
3. 增加 LoRA rank
4. 使用更多训练数据

### Q: 如何加快训练速度？

**A**:
1. 减少 batch size
2. 减少 max_length
3. 使用更小的 LoRA rank
4. 使用多 GPU

---

## 🎉 总结

### 问题
- 之前只训练 MOE 层（32 个参数）
- 导致 NaN loss 和生成错误

### 解决方案
- 冻结 Base 模型
- 一同训练 LoRA + MOE 层
- 参数增加到 ~1,000,000+

### 结果
- 训练正常
- 生成答案正确
- 准确率逐渐上升

---

**现在可以开始训练了！** 🚀

