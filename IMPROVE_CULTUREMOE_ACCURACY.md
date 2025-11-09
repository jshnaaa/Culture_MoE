# 提高 CultureMoE 准确率的方法

## 🎯 目标

提高 `train_culturemoe_from_base_gen.py` 训练的 CultureMoE 模型的准确率。

---

## 📊 当前配置分析

### 当前参数（run_train_culturemoe_from_base_gen.sh）

```bash
--num_epochs 20
--num_experts 6
--shared_hidden_dim 2048
--router_hidden_dim 1024
--experts_hidden_dim 2048
--moe_lora_rank 16
--classification_hidden_dim 512
--dropout 0.1
--num_heads 8

--batch_size 4
--eval_batch_size 4
--learning_rate 5e-6
--weight_decay 0.01
--max_length 512
--val_split 0.1
--num_workers 2
```

---

## 🚀 提高准确率的 10 个方法

### 方法 1：增加训练 Epoch 数 ⭐⭐⭐⭐⭐

**当前**：20 epochs
**建议**：30-50 epochs

```bash
# 修改 run_train_culturemoe_from_base_gen.sh
--num_epochs 30  # 或 40, 50
```

**原因**：
- ✅ MoE 模型收敛慢，需要更多 epoch
- ✅ 从日志看，20 epoch 可能还没有完全收敛
- ✅ 增加 epoch 是最简单有效的方法

**预期提升**：+2-5% 准确率

---

### 方法 2：调整学习率 ⭐⭐⭐⭐⭐

**当前**：5e-6
**建议**：尝试不同的学习率

```bash
# 方案 A：提高学习率（如果训练稳定）
--learning_rate 1e-5  # 提高 2 倍

# 方案 B：使用学习率调度器
# 在 train_culturemoe_from_base_gen.py 中添加
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(len(train_loader) * 0.1),  # 10% warmup
    num_training_steps=len(train_loader) * args.num_epochs
)

# 在训练循环中
optimizer.step()
scheduler.step()  # ✅ 添加这一行
```

**原因**：
- ✅ 5e-6 可能太低，导致收敛慢
- ✅ 学习率调度器可以帮助更好地收敛
- ✅ Warmup 可以避免初期梯度不稳定

**预期提升**：+3-8% 准确率

---

### 方法 3：增加 MoE 专家数量 ⭐⭐⭐⭐

**当前**：6 个专家
**建议**：8-12 个专家

```bash
# 修改 run_train_culturemoe_from_base_gen.sh
--num_experts 8  # 或 10, 12
```

**原因**：
- ✅ 更多专家可以学习更细粒度的文化特征
- ✅ 6 个专家可能不够表达复杂的文化差异
- ✅ 但要注意显存占用

**预期提升**：+1-3% 准确率

**注意**：显存占用会增加

---

### 方法 4：增加 MoE 层的维度 ⭐⭐⭐⭐

**当前**：
- shared_hidden_dim: 2048
- router_hidden_dim: 1024
- experts_hidden_dim: 2048

**建议**：增加维度

```bash
# 修改 run_train_culturemoe_from_base_gen.sh
--shared_hidden_dim 4096      # 2048 → 4096
--router_hidden_dim 2048      # 1024 → 2048
--experts_hidden_dim 4096     # 2048 → 4096
```

**原因**：
- ✅ 更大的维度可以学习更复杂的特征
- ✅ LLaMA 的 hidden_size 是 4096，MoE 层应该匹配
- ✅ 当前维度可能太小

**预期提升**：+2-5% 准确率

**注意**：显存占用会显著增加

---

### 方法 5：增加 LoRA Rank ⭐⭐⭐

**当前**：moe_lora_rank: 16
**建议**：32 或 64

```bash
# 修改 run_train_culturemoe_from_base_gen.sh
--moe_lora_rank 32  # 或 64
```

**原因**：
- ✅ 更高的 rank 可以学习更复杂的变换
- ✅ 16 可能太小，限制了表达能力
- ✅ 32-64 是常用的值

**预期提升**：+1-3% 准确率

---

### 方法 6：减小 Dropout ⭐⭐⭐

**当前**：dropout: 0.1
**建议**：0.05 或 0.03

```bash
# 修改 run_train_culturemoe_from_base_gen.sh
--dropout 0.05  # 或 0.03
```

**原因**：
- ✅ 0.1 可能太高，导致欠拟合
- ✅ MoE 层本身就有正则化效果
- ✅ 减小 dropout 可以提高拟合能力

**预期提升**：+1-2% 准确率

---

### 方法 7：增加 Batch Size ⭐⭐⭐⭐

**当前**：batch_size: 4
**建议**：8 或 16（如果显存允许）

```bash
# 修改 run_train_culturemoe_from_base_gen.sh
--batch_size 8  # 或 16

# 如果显存不足，使用梯度累积
--batch_size 4
--gradient_accumulation_steps 2  # 等效 batch_size = 8
```

**原因**：
- ✅ 更大的 batch size 可以提供更稳定的梯度
- ✅ 4 可能太小，导致梯度噪声大
- ✅ 8-16 是常用的值

**预期提升**：+1-3% 准确率

---

### 方法 8：调整 Culture Loss 权重 ⭐⭐⭐⭐

**当前**：culture_loss_lambda: 0.5
**建议**：尝试不同的权重

```bash
# 修改 run_train_culturemoe_from_base_gen.sh

# 方案 A：增加 culture loss 权重
--culture_loss_lambda 1.0  # 或 1.5, 2.0

# 方案 B：减小 culture loss 权重
--culture_loss_lambda 0.3  # 或 0.2, 0.1

# 方案 C：动态调整（需要修改代码）
# 在训练过程中逐渐增加 culture_loss_lambda
```

**原因**：
- ✅ Culture loss 的权重直接影响专家的分配
- ✅ 0.5 可能不是最优值
- ✅ 需要实验找到最佳权重

**预期提升**：+2-5% 准确率

---

### 方法 9：使用更好的数据增强 ⭐⭐⭐

**当前**：只使用 instruction_mask
**建议**：添加更多数据增强

```python
# 在 load_and_process_generative_data 中添加

# 方案 A：随机替换同义词
def augment_text(text):
    # 随机替换一些词
    return text

# 方案 B：回译（Back Translation）
def back_translate(text):
    # 翻译成其他语言再翻译回来
    return text

# 方案 C：随机删除/插入词
def random_edit(text):
    # 随机删除或插入一些词
    return text
```

**原因**：
- ✅ 数据增强可以提高模型的泛化能力
- ✅ 当前只使用了 instruction_mask，可能不够
- ✅ 更多的数据变化可以帮助模型学习更鲁棒的特征

**预期提升**：+2-4% 准确率

---

### 方法 10：使用 Focal Loss ⭐⭐⭐

**当前**：使用标准的 CrossEntropyLoss
**建议**：使用 Focal Loss 处理类别不平衡

```python
# 在 train_culturemoe_from_base_gen.py 中修改

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 在 CultureMoE 的 forward 中使用
loss_fct = FocalLoss(alpha=1, gamma=2)
generation_loss = loss_fct(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1)
)
```

**原因**：
- ✅ Focal Loss 可以处理类别不平衡
- ✅ 对难分类的样本给予更多关注
- ✅ 特别适合文化分类任务

**预期提升**：+1-3% 准确率

---

## 📊 参数调整优先级

### 🔥 高优先级（立即尝试）

| 方法 | 难度 | 预期提升 | 显存影响 |
|------|------|---------|---------|
| **1. 增加 Epoch** | ⭐ | +2-5% | 无 |
| **2. 调整学习率** | ⭐⭐ | +3-8% | 无 |
| **4. 增加 MoE 维度** | ⭐ | +2-5% | 高 |
| **7. 增加 Batch Size** | ⭐ | +1-3% | 中 |
| **8. 调整 Culture Loss** | ⭐ | +2-5% | 无 |

### ⚠️ 中优先级（稳定后尝试）

| 方法 | 难度 | 预期提升 | 显存影响 |
|------|------|---------|---------|
| **3. 增加专家数量** | ⭐ | +1-3% | 中 |
| **5. 增加 LoRA Rank** | ⭐ | +1-3% | 低 |
| **6. 减小 Dropout** | ⭐ | +1-2% | 无 |

### 💡 低优先级（高级优化）

| 方法 | 难度 | 预期提升 | 显存影响 |
|------|------|---------|---------|
| **9. 数据增强** | ⭐⭐⭐ | +2-4% | 无 |
| **10. Focal Loss** | ⭐⭐ | +1-3% | 无 |

---

## 🎯 推荐的调整方案

### 方案 A：保守优化（显存有限）

```bash
# 修改 run_train_culturemoe_from_base_gen.sh

--num_epochs 30                    # ✅ 增加 epoch
--learning_rate 1e-5               # ✅ 提高学习率
--culture_loss_lambda 1.0          # ✅ 增加 culture loss 权重
--dropout 0.05                     # ✅ 减小 dropout
--batch_size 4                     # 保持不变
--gradient_accumulation_steps 2    # ✅ 添加梯度累积
```

**预期提升**：+5-10% 准确率
**显存影响**：低

### 方案 B：激进优化（显存充足）

```bash
# 修改 run_train_culturemoe_from_base_gen.sh

--num_epochs 40                    # ✅ 大幅增加 epoch
--num_experts 8                    # ✅ 增加专家数量
--shared_hidden_dim 4096           # ✅ 增加维度
--router_hidden_dim 2048           # ✅ 增加维度
--experts_hidden_dim 4096          # ✅ 增加维度
--moe_lora_rank 32                 # ✅ 增加 LoRA rank
--dropout 0.05                     # ✅ 减小 dropout

--batch_size 8                     # ✅ 增加 batch size
--learning_rate 1e-5               # ✅ 提高学习率
--culture_loss_lambda 1.0          # ✅ 增加 culture loss 权重
```

**预期提升**：+10-15% 准确率
**显存影响**：高

### 方案 C：平衡优化（推荐）

```bash
# 修改 run_train_culturemoe_from_base_gen.sh

--num_epochs 30                    # ✅ 增加 epoch
--num_experts 8                    # ✅ 增加专家数量
--shared_hidden_dim 3072           # ✅ 适度增加维度
--router_hidden_dim 1536           # ✅ 适度增加维度
--experts_hidden_dim 3072          # ✅ 适度增加维度
--moe_lora_rank 24                 # ✅ 适度增加 LoRA rank
--dropout 0.05                     # ✅ 减小 dropout

--batch_size 6                     # ✅ 适度增加 batch size
--learning_rate 8e-6               # ✅ 适度提高学习率
--culture_loss_lambda 0.8          # ✅ 适度增加 culture loss 权重
```

**预期提升**：+8-12% 准确率
**显存影响**：中

---

## 🔧 代码修改示例

### 修改 1：添加学习率调度器

```python
# 在 train_culturemoe_from_base_gen.py 的 main() 函数中

# 创建优化器（原有代码）
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

# ✅ 添加学习率调度器
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(len(train_loader) * 0.1),  # 10% warmup
    num_training_steps=len(train_loader) * args.num_epochs
)

# 在训练循环中（train_epoch 函数）
optimizer.step()
scheduler.step()  # ✅ 添加这一行
```

### 修改 2：添加梯度累积

```python
# 在 argparse 中添加
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="梯度累积步数")

# 在 train_epoch 函数中修改
accumulation_steps = 0

for batch in progress_bar:
    # 前向传播
    outputs = model(...)
    loss = outputs['loss']

    # ✅ 梯度累积
    loss = loss / args.gradient_accumulation_steps
    loss.backward()

    accumulation_steps += 1

    if accumulation_steps % args.gradient_accumulation_steps == 0:
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
```

### 修改 3：添加 Focal Loss

```python
# 在 train_culturemoe_from_base_gen.py 中添加

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [N, C]
        # targets: [N]

        # 过滤 ignore_index
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        if inputs.size(0) == 0:
            return torch.tensor(0.0, device=inputs.device)

        # 计算 CE loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算 pt
        pt = torch.exp(-ce_loss)

        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

# 在 CultureMoE 的 forward 中使用
loss_fct = FocalLoss(alpha=1, gamma=2, ignore_index=-100)
generation_loss = loss_fct(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1)
)
```

---

## 📈 实验建议

### 第 1 步：基线测试

```bash
# 使用当前配置训练
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 1 true

# 记录准确率
```

### 第 2 步：单变量测试

```bash
# 测试 1：只增加 epoch
--num_epochs 30

# 测试 2：只提高学习率
--learning_rate 1e-5

# 测试 3：只增加 batch size
--batch_size 8

# 每次只改一个参数，观察效果
```

### 第 3 步：组合优化

```bash
# 使用方案 C（平衡优化）
sh run_train_culturemoe_from_base_gen.sh llama 4 True 8 1 true \
    --num_epochs 30 \
    --shared_hidden_dim 3072 \
    --router_hidden_dim 1536 \
    --experts_hidden_dim 3072 \
    --moe_lora_rank 24 \
    --dropout 0.05 \
    --batch_size 6 \
    --learning_rate 8e-6 \
    --culture_loss_lambda 0.8
```

---

## 🎉 总结

### 最有效的 5 个方法

1. **增加 Epoch**（+2-5%）- 最简单
2. **调整学习率**（+3-8%）- 最有效
3. **增加 MoE 维度**（+2-5%）- 需要显存
4. **调整 Culture Loss**（+2-5%）- 需要实验
5. **增加 Batch Size**（+1-3%）- 稳定梯度

### 推荐的优化顺序

1. ✅ 增加 Epoch 到 30
2. ✅ 提高学习率到 1e-5
3. ✅ 增加 Batch Size 到 8（或使用梯度累积）
4. ✅ 调整 Culture Loss Lambda 到 1.0
5. ✅ 减小 Dropout 到 0.05
6. ✅ 增加专家数量到 8
7. ✅ 增加 MoE 维度

### 预期总提升

- **保守方案**：+5-10% 准确率
- **平衡方案**：+8-12% 准确率
- **激进方案**：+10-15% 准确率

---

**现在可以开始优化 CultureMoE 模型了！** 🚀

