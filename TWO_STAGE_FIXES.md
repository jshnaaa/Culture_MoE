# 两阶段训练 - 问题修复

## ✅ 已修复的问题

### 1. **TypeError: cross_entropy_loss() 错误**

**问题原因**：
- 阶段1使用的是 LoRA 微调的 LLM，没有 MoE 结构
- `ClassificationTrainer` 默认启用文化损失，期望模型有 MoE 输出
- LoRA 模型返回 `CausalLMOutputWithPast` 对象，不是 tensor

**解决方案**：
- ✅ 阶段1的 Trainer 禁用文化损失（`use_culture_loss=False`）
- ✅ 阶段2的 Trainer 启用文化损失（`use_culture_loss=True`）

### 2. **文化损失权重更新**

- ✅ 从 `0.05` 更新为 `0.5`
- ✅ 在 `run_two_stage_culturemoe.sh` 中修改

### 3. **输出目录更新**

- ✅ 从 `/root/autodl-fs/output/two_stage_moe/`
- ✅ 更新为 `/root/autodl-tmp/CultureMoE/Culture_Alignment/two_stage_moe/`

## 📝 修改的文件

### 1. `train_two_stage_culturemoe.py`

```python
# 阶段1：不使用文化损失
trainer = ClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn,
    callbacks=[epoch_callback],
    use_culture_loss=False,  # ⭐ 阶段1不使用文化损失
    lambda_weight=0.0
)

# 阶段2：使用文化损失
trainer = ClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn,
    callbacks=[epoch_callback],
    use_culture_loss=args.use_culture_loss,  # ⭐ 阶段2使用文化损失
    lambda_weight=args.culture_loss_lambda
)
```

### 2. `run_two_stage_culturemoe.sh`

```bash
# 文化损失权重：0.05 → 0.5
--culture_loss_lambda 0.5

# 输出目录更新
if [ "$BACKBONE" = "qwen" ]; then
    OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/two_stage_moe/qwen_${NUM_CLASSES}class_experts${NUM_EXPERTS}_$(date +%Y%m%d_%H%M)"
else
    OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/two_stage_moe/llama_${NUM_CLASSES}class_experts${NUM_EXPERTS}_$(date +%Y%m%d_%H%M)"
fi
```

## 🚀 使用方法（不变）

```bash
# 基本用法
sh run_two_stage_culturemoe.sh llama 2 True false

# 自定义配置
sh run_two_stage_culturemoe.sh llama 3 True false 4
```

## 📊 训练流程

### 阶段1：LoRA 微调（不使用文化损失）

```
1. 加载 base model
2. 添加 LoRA adapters
3. 添加分类头
4. 训练（不使用文化损失）⭐
5. 选择最佳 checkpoint
6. 合并 LoRA 权重
```

### 阶段2：训练 MoE（使用文化损失）

```
1. 加载阶段1最佳模型
2. 冻结 LLM
3. 创建 MoE 模型
4. 训练（使用文化损失，lambda=0.5）⭐
```

## 💡 为什么阶段1不使用文化损失？

### 原因

1. **模型结构不同**
   - 阶段1：LoRA 微调的 LLM + 分类头
   - 阶段2：冻结的 LLM + MoE + 分类头

2. **文化损失需要 MoE**
   - 文化损失计算需要 expert weights 和 router outputs
   - 阶段1没有 MoE 结构，无法计算文化损失

3. **避免类型错误**
   - LoRA 模型返回 `CausalLMOutputWithPast`
   - 文化损失期望特定的 MoE 输出格式

### 训练策略

| 阶段 | 模型结构 | 文化损失 | 损失函数 |
|------|---------|---------|---------|
| 阶段1 | LoRA + 分类头 | ❌ 不使用 | CrossEntropy |
| 阶段2 | 冻结LLM + MoE | ✅ 使用 (λ=0.5) | CrossEntropy + Culture Loss |

## 📈 预期效果

### 阶段1（LoRA Only）

```
Loss = CrossEntropyLoss(logits, labels)
```

- 专注于学习分类任务
- 不考虑文化专注性

### 阶段2（MoE + Culture Loss）

```
Loss = CrossEntropyLoss(logits, labels) + 0.5 * CultureLoss(expert_weights, culture_labels)
```

- 学习分类任务
- 同时优化文化专注性
- 文化损失权重较大（0.5），强调文化专注

## 🔍 验证修复

### 运行测试

```bash
# 测试阶段1（不应该报错）
sh run_two_stage_culturemoe.sh llama 2 True false
```

### 预期输出

```
阶段1：
7. Creating trainer...
   (不显示文化损失信息)

阶段2：
6. Creating trainer...
   Use culture loss: True
   Culture loss lambda: 0.5
```

## 📁 输出目录结构

```
/root/autodl-tmp/CultureMoE/Culture_Alignment/two_stage_moe/
├── llama_2class_experts6_20251031_1430/
│   ├── config.json
│   ├── final_report.json
│   ├── stage1_epoch_eval_results.json
│   ├── stage1_best_eval.json
│   ├── stage2_epoch_eval_results.json
│   └── stage2_final_eval.json
│
└── qwen_3class_experts4_20251031_1500/
    └── ...
```

## ⚙️ 配置总结

| 参数 | 阶段1 | 阶段2 |
|------|-------|-------|
| 模型结构 | LoRA + 分类头 | 冻结LLM + MoE |
| 文化损失 | ❌ False | ✅ True |
| 文化损失权重 | 0.0 | 0.5 |
| 训练轮数 | 3 | 5 |
| 选择策略 | 最佳 checkpoint | 最终模型 |

## ✅ 总结

### 核心修复

1. ✅ **修复 TypeError**：阶段1禁用文化损失
2. ✅ **更新文化损失权重**：0.05 → 0.5
3. ✅ **更新输出目录**：使用新路径

### 训练策略

- **阶段1**：专注于 LoRA 微调和分类任务
- **阶段2**：在冻结的 LLM 基础上训练 MoE，强调文化专注性

现在可以正常运行了！🎉

