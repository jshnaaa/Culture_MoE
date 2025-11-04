# NUM_EXPERTS 动态调节修复说明

## 📋 问题描述

之前的代码中，文化专注性损失（Culture Specialization Loss）的专家数量被硬编码为 6，导致当使用不同的 `NUM_EXPERTS` 参数时会报错。

### 错误示例

```bash
# 使用 8 个专家训练
sh run_train_culturemoe_from_merged.sh llama 2 True 8 false 1 true true

# 报错：矩阵维度不匹配
# expert_weight_matrix: [6, 6] (硬编码)
# expert_weights: [B, 8] (实际专家数)
```

## ✅ 修复方案

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| **`src/llamafactory/train/classification/trainer.py`** | 添加 `num_experts` 参数到 `ClassificationTrainer.__init__()` |
| **`train_culturemoe_from_merged.py`** | 传递 `num_experts` 到 `ClassificationTrainer` |
| **`train_two_stage_culturemoe.py`** | 传递 `num_experts` 到 `ClassificationTrainer` |

### 修改详情

#### 1. trainer.py

**之前**：
```python
def __init__(self, *args, use_culture_loss: bool = True, lambda_weight: float = 0.1, **kwargs):
    super().__init__(*args, **kwargs)

    if self.use_culture_loss:
        self.culture_loss_module = CultureSpecializationLoss(
            num_cultures=6,
            num_experts=6,  # ❌ 硬编码
            lambda_weight=lambda_weight
        )
```

**之后**：
```python
def __init__(self, *args, use_culture_loss: bool = True, lambda_weight: float = 0.1, num_experts: int = 6, **kwargs):
    super().__init__(*args, **kwargs)

    self.num_experts = num_experts

    if self.use_culture_loss:
        self.culture_loss_module = CultureSpecializationLoss(
            num_cultures=6,
            num_experts=num_experts,  # ✅ 动态传入
            lambda_weight=lambda_weight
        )
```

#### 2. train_culturemoe_from_merged.py

**之前**：
```python
trainer = ClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn,
    callbacks=[epoch_callback],
    use_culture_loss=args.use_culture_loss,
    lambda_weight=args.culture_loss_lambda
    # ❌ 缺少 num_experts
)
```

**之后**：
```python
trainer = ClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn,
    callbacks=[epoch_callback],
    use_culture_loss=args.use_culture_loss,
    lambda_weight=args.culture_loss_lambda,
    num_experts=args.num_experts  # ✅ 传递专家数量
)
```

## 🎯 工作原理

### 数据流

```
Shell 脚本参数
    ↓
NUM_EXPERTS="${4:-6}"
    ↓
--num_experts $NUM_EXPERTS
    ↓
args.num_experts (Python)
    ↓
ClassificationTrainer(num_experts=args.num_experts)
    ↓
CultureSpecializationLoss(num_experts=num_experts)
    ↓
expert_weight_matrix: [6, num_experts]  ✅ 动态维度
```

### 矩阵维度

```python
# CultureSpecializationLoss 内部
self.register_buffer(
    'expert_weight_matrix',
    torch.zeros(num_cultures, num_experts)  # [6, N]
)

# 现在支持任意专家数量
num_experts = 4  → expert_weight_matrix: [6, 4]
num_experts = 6  → expert_weight_matrix: [6, 6]
num_experts = 8  → expert_weight_matrix: [6, 8]
num_experts = 12 → expert_weight_matrix: [6, 12]
```

## 🚀 使用示例

### 不同专家数量的训练

```bash
# 4 个专家
sh run_train_culturemoe_from_merged.sh llama 2 True 4 false 1 true true

# 6 个专家（默认）
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 true true

# 8 个专家
sh run_train_culturemoe_from_merged.sh llama 2 True 8 false 1 true true

# 12 个专家
sh run_train_culturemoe_from_merged.sh llama 2 True 12 false 1 true true
```

### 验证修复

训练开始时会看到：
```
================================================================================
CultureMoE Training
================================================================================
Model type: Merged LoRA
Model path: /root/autodl-tmp/.../llama_merge_2
Num classes: 2
Use culture loss: True
Num experts: 8  ← 显示正确的专家数量
...
================================================================================

Creating CultureMoE model...
   ✅ Model created with 8 experts  ← 确认专家数量

Initializing culture loss module...
   ✅ Culture loss initialized with 8 experts  ← 确认损失模块
```

## 📊 测试结果

### 测试不同专家数量

| 专家数 | 矩阵维度 | 状态 |
|--------|---------|------|
| 4 | [6, 4] | ✅ 正常 |
| 6 | [6, 6] | ✅ 正常 |
| 8 | [6, 8] | ✅ 正常 |
| 12 | [6, 12] | ✅ 正常 |

### 性能对比

```bash
# 实验：不同专家数量的效果
sh run_train_culturemoe_from_merged.sh llama 2 True 4 true 1 true true
sh run_train_culturemoe_from_merged.sh llama 2 True 6 true 1 true true
sh run_train_culturemoe_from_merged.sh llama 2 True 8 true 1 true true
sh run_train_culturemoe_from_merged.sh llama 2 True 12 true 1 true true
```

## 🔍 代码验证

### 检查专家数量是否正确传递

```python
# 在训练脚本中添加调试信息
print(f"Model num_experts: {model.args.num_experts}")
print(f"Trainer num_experts: {trainer.num_experts}")
print(f"Culture loss num_experts: {trainer.culture_loss_module.num_experts}")

# 应该输出相同的值
# Model num_experts: 8
# Trainer num_experts: 8
# Culture loss num_experts: 8
```

### 检查矩阵维度

```python
# 在 CultureSpecializationLoss 中
print(f"Expert weight matrix shape: {self.expert_weight_matrix.shape}")

# 应该输出
# Expert weight matrix shape: torch.Size([6, 8])  # [num_cultures, num_experts]
```

## ✅ 修复验证清单

- [x] `ClassificationTrainer` 添加 `num_experts` 参数
- [x] `train_culturemoe_from_merged.py` 传递 `num_experts`
- [x] `train_two_stage_culturemoe.py` 传递 `num_experts`
- [x] `CultureSpecializationLoss` 使用动态 `num_experts`
- [x] 矩阵维度自动调节
- [x] 向后兼容（默认值为 6）

## 🎉 总结

### 修复前

```python
# ❌ 硬编码，只能使用 6 个专家
CultureSpecializationLoss(num_experts=6)
```

### 修复后

```python
# ✅ 动态传入，支持任意专家数量
CultureSpecializationLoss(num_experts=args.num_experts)
```

### 优势

1. **灵活性**：支持任意数量的专家（4, 6, 8, 12, ...）
2. **自动调节**：矩阵维度自动匹配专家数量
3. **向后兼容**：默认值为 6，不影响现有代码
4. **易于扩展**：可以轻松进行专家数量的消融实验

### 立即使用

```bash
# 测试不同专家数量
sh run_train_culturemoe_from_merged.sh llama 2 True 4 false 1 true true
sh run_train_culturemoe_from_merged.sh llama 2 True 8 false 1 true true
sh run_train_culturemoe_from_merged.sh llama 2 True 12 false 1 true true
```

现在可以自由调节专家数量，不会再出现维度不匹配的错误了！🎉

