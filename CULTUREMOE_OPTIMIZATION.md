# CultureMoE 模型参数优化

## 问题分析

### 原始问题
1. **CultureMoE 准确率不如 LoRA Only** - 这是不正常的
2. **学习率保持常数** - 使用 `constant_with_warmup`，训练后期无法精细调整
3. **训练轮数过多** - 20 epoch 过拟合，10 epoch 也过拟合

### 根本原因
1. **学习率调度不合理**：常数学习率导致训练后期无法收敛到更好的局部最优
2. **训练轮数过多**：MoE 模型参数更多，更容易过拟合
3. **学习率过低**：5e-6 对于 MoE 训练可能太保守
4. **LoRA rank 过小**：rank=8 可能限制了模型表达能力
5. **文化损失权重过大**：0.1 可能干扰主任务学习

## 优化方案

### 1. 学习率调度器：constant_with_warmup → cosine

**修改文件**：`src/llamafactory/train/classification/workflow.py`

**修改前**：
```python
lr_scheduler_type="constant_with_warmup"
```

**修改后**：
```python
lr_scheduler_type="cosine"  # ✅ 改为 cosine 学习率衰减
```

**效果**：
- Warmup 阶段（前 10%）：学习率从 0 逐渐增加到 1e-5
- 训练阶段（10%-100%）：学习率按余弦曲线从 1e-5 衰减到接近 0
- 训练后期可以精细调整参数，避免震荡

**学习率变化示例**（5 epochs）：
```
Epoch 0.5:  lr = 5.0e-6  (warmup)
Epoch 1.0:  lr = 1.0e-5  (peak)
Epoch 2.0:  lr = 8.5e-6
Epoch 3.0:  lr = 5.0e-6
Epoch 4.0:  lr = 2.0e-6
Epoch 5.0:  lr = 0.5e-6  (end)
```

### 2. 训练轮数：20/10 → 5

**修改文件**：`run_train_ddp_lora_dual.sh`

**修改前**：
```bash
--num_train_epochs 2  # 或 10, 20
```

**修改后**：
```bash
--num_train_epochs 5  # ✅ 减少到 5 epoch
```

**理由**：
- MoE 模型参数更多，更容易过拟合
- 5 epoch 足够学习主要模式，避免过拟合
- 配合 cosine 学习率衰减，5 epoch 可以充分收敛

### 3. 学习率：5e-6 → 1e-5

**修改前**：
```bash
--learning_rate 5e-6
```

**修改后**：
```bash
--learning_rate 1e-5  # ✅ 提高学习率
```

**理由**：
- 5e-6 对于 MoE 训练太保守
- 1e-5 是 LoRA 微调的常用学习率
- 配合 cosine 衰减，可以快速收敛

### 4. LoRA Rank：8 → 16

**修改前**：
```bash
--llama_lora_rank 8
--llama_lora_alpha 16
```

**修改后**：
```bash
--llama_lora_rank 16  # ✅ 增加 rank
--llama_lora_alpha 32  # ✅ alpha = 2 × rank
```

**理由**：
- Rank=8 可能限制了模型表达能力
- Rank=16 提供更多可训练参数
- Alpha=32 保持 alpha/rank=2 的比例

### 5. 文化损失权重：0.1 → 0.05

**修改前**：
```bash
--culture_loss_lambda 0.1
```

**修改后**：
```bash
--culture_loss_lambda 0.05  # ✅ 降低文化损失权重
```

**理由**：
- 0.1 可能过大，干扰主任务学习
- 0.05 更平衡，既保留文化信息，又不影响分类性能

### 6. 评估频率：500 → 100

**修改前**：
```bash
--eval_steps 500
```

**修改后**：
```bash
--eval_steps 100  # ✅ 更频繁评估
```

**理由**：
- 更频繁评估可以及时发现过拟合
- 可以选择最佳 checkpoint

### 7. 添加 Weight Decay 和 Warmup Ratio

**新增**：
```bash
--weight_decay 0.01    # ✅ 添加权重衰减，防止过拟合
--warmup_ratio 0.1     # ✅ 10% 步数用于 warmup
```

**理由**：
- Weight decay 有助于正则化，防止过拟合
- Warmup ratio 让学习率平滑上升

### 8. LoRA Dropout：0.05 → 0.1

**修改前**：
```bash
--llama_lora_dropout 0.05
```

**修改后**：
```bash
--llama_lora_dropout 0.1  # ✅ 增加 dropout
```

**理由**：
- 更高的 dropout 有助于防止过拟合
- 0.1 是常用值

## 完整参数对比

| 参数 | 修改前 | 修改后 | 说明 |
|------|--------|--------|------|
| `lr_scheduler_type` | `constant_with_warmup` | `cosine` | 学习率衰减 |
| `num_train_epochs` | 2/10/20 | **5** | 减少训练轮数 |
| `learning_rate` | 5e-6 | **1e-5** | 提高学习率 |
| `llama_lora_rank` | 8 | **16** | 增加 LoRA rank |
| `llama_lora_alpha` | 16 | **32** | 对应调整 alpha |
| `llama_lora_dropout` | 0.05 | **0.1** | 增加 dropout |
| `culture_loss_lambda` | 0.1 | **0.05** | 降低文化损失权重 |
| `eval_steps` | 500 | **100** | 更频繁评估 |
| `weight_decay` | - | **0.01** | 新增权重衰减 |
| `warmup_ratio` | - | **0.1** | 新增 warmup |

## 预期效果

### 训练过程

```
Epoch 1: lr=1e-5, loss=1.5, acc=0.65  (快速学习)
Epoch 2: lr=8.5e-6, loss=1.2, acc=0.72  (继续提升)
Epoch 3: lr=5e-6, loss=1.0, acc=0.76  (稳定收敛)
Epoch 4: lr=2e-6, loss=0.95, acc=0.78  (精细调整)
Epoch 5: lr=0.5e-6, loss=0.92, acc=0.79  (最终收敛)
```

### 性能对比

| 模型 | 修改前 | 修改后（预期） |
|------|--------|---------------|
| LoRA Only | 0.75 | 0.75 |
| CultureMoE | 0.70 ❌ | **0.78** ✅ |

**预期**：CultureMoE 应该**超过** LoRA Only

## 使用方法

### 运行优化后的训练

```bash
# 使用文化损失
sh run_train_ddp_lora_dual.sh llama 3 True false

# 不使用文化损失（baseline）
sh run_train_ddp_lora_dual.sh llama 3 False false
```

### 监控训练过程

```bash
# 启动 TensorBoard
tensorboard --logdir output/CultureMoE/

# 查看学习率曲线
# 应该看到余弦衰减曲线
```

### 对比实验

```bash
# 实验 1：优化后的 CultureMoE（有文化损失）
sh run_train_ddp_lora_dual.sh llama 3 True false

# 实验 2：优化后的 CultureMoE（无文化损失）
sh run_train_ddp_lora_dual.sh llama 3 False false

# 实验 3：LoRA Only（baseline）
sh run_train_lora_only.sh llama 3 false

# 对比三个模型的性能
```

## 调试建议

### 如果仍然过拟合

1. **进一步减少 epoch**：
   ```bash
   --num_train_epochs 3
   ```

2. **增加 dropout**：
   ```bash
   --llama_lora_dropout 0.15
   ```

3. **增加 weight decay**：
   ```bash
   --weight_decay 0.05
   ```

### 如果欠拟合

1. **增加 epoch**：
   ```bash
   --num_train_epochs 7
   ```

2. **提高学习率**：
   ```bash
   --learning_rate 2e-5
   ```

3. **增加 LoRA rank**：
   ```bash
   --llama_lora_rank 32
   --llama_lora_alpha 64
   ```

### 如果文化损失效果不明显

1. **调整文化损失权重**：
   ```bash
   --culture_loss_lambda 0.1  # 增加到 0.1
   ```

2. **或者尝试不同的值**：
   ```bash
   --culture_loss_lambda 0.01  # 降低到 0.01
   --culture_loss_lambda 0.02
   --culture_loss_lambda 0.05
   ```

## 理论依据

### 为什么 Cosine 学习率更好？

1. **训练初期**：高学习率快速探索参数空间
2. **训练中期**：逐渐降低学习率，稳定收敛
3. **训练后期**：极低学习率精细调整，避免震荡

### 为什么 5 Epoch 合适？

1. **MoE 模型参数多**：更容易过拟合
2. **LoRA 微调快**：不需要太多 epoch
3. **配合学习率衰减**：5 epoch 足够完成一个完整的学习周期

### 为什么提高学习率？

1. **5e-6 太保守**：收敛太慢
2. **1e-5 是标准值**：LoRA 微调的常用学习率
3. **配合 cosine 衰减**：可以快速收敛后精细调整

## 总结

### 关键改进

1. ✅ **Cosine 学习率衰减** - 最重要的改进
2. ✅ **减少到 5 epoch** - 防止过拟合
3. ✅ **提高学习率到 1e-5** - 加快收敛
4. ✅ **增加 LoRA rank 到 16** - 提升表达能力
5. ✅ **降低文化损失权重到 0.05** - 平衡主任务

### 预期结果

- CultureMoE 性能应该**超过** LoRA Only
- 训练 5 epoch 不会过拟合
- 学习率会平滑衰减
- 文化损失有助于提升性能

现在可以重新训练并对比结果了！🚀

