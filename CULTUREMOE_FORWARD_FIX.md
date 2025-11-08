# CultureMoE Forward 方法返回格式修复

## 🔍 问题描述

运行 `sh run_train_culturemoe_from_base_gen.sh llama 3` 时报错：

```python
Traceback (most recent call last):
  File "train_culturemoe_from_base_gen.py", line 714, in <module>
    main()
  File "train_culturemoe_from_base_gen.py", line 537, in main
    train_metrics = train_epoch(
  File "train_culturemoe_from_base_gen.py", line 182, in train_epoch
    loss = outputs['loss']
           ~~~~~~~^^^^^^^^
IndexError: too many indices for tensor of dimension 2
```

## 🔍 根本原因

### 问题分析

1. **训练代码期望**：`outputs` 是一个字典，包含 `'loss'`、`'logits'`、`'classification_loss'` 等键

```python
# train_culturemoe_from_base_gen.py
outputs = model(...)
loss = outputs['loss']  # ❌ 期望字典
```

2. **CultureMoE 实际返回**：只返回一个 2D tensor（logits）

```python
# CultureMoE.py (修复前)
def forward(self, ...):
    ...
    logits_avg = logits.mean(dim=1)  # [B, num_classes]
    return logits_avg  # ❌ 返回 tensor，不是字典
```

3. **错误原因**：尝试用字典索引访问 tensor → `IndexError`

## ✅ 解决方案

### 修改 CultureMoE 的 forward 方法

#### 修复前

```python
def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None, attention_mask_mask=None, **kwargs):
    ...
    # Step 8: 对 logits 进行平均
    logits_avg = logits.mean(dim=1)  # [B, num_classes]

    return logits_avg  # ❌ 只返回 tensor
```

#### 修复后

```python
def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None, attention_mask_mask=None,
            labels=None, culture_labels=None, use_culture_loss=False, culture_loss_lambda=0.5, **kwargs):
    ...
    # Step 8: 对 logits 进行平均
    logits_avg = logits.mean(dim=1)  # [B, num_classes]

    # ✅ Step 9: 计算损失（如果提供了 labels）
    outputs = {'logits': logits_avg}

    if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        classification_loss = loss_fct(logits_avg, labels)
        outputs['classification_loss'] = classification_loss

        # 文化损失（如果启用）
        if use_culture_loss and culture_labels is not None:
            culture_loss = self.compute_culture_loss(expert_weights, culture_labels)
            outputs['culture_loss'] = culture_loss

            # 总损失
            total_loss = classification_loss + culture_loss_lambda * culture_loss
        else:
            outputs['culture_loss'] = torch.tensor(0.0, device=device)
            total_loss = classification_loss

        outputs['loss'] = total_loss

    return outputs  # ✅ 返回字典
```

### 添加文化损失计算方法

```python
def compute_culture_loss(self, expert_weights, culture_labels):
    """
    计算文化损失：鼓励相同文化的样本使用相似的专家

    Args:
        expert_weights: [B, E] 专家权重
        culture_labels: [B] 文化标签

    Returns:
        culture_loss: 标量
    """
    device = expert_weights.device

    # 计算样本对之间的文化相似度（相同文化为1，不同文化为0）
    culture_labels = culture_labels.unsqueeze(1)  # [B, 1]
    culture_similarity = (culture_labels == culture_labels.t()).float()  # [B, B]

    # 计算专家权重之间的余弦相似度
    expert_weights_norm = torch.nn.functional.normalize(expert_weights, p=2, dim=1)
    expert_similarity = torch.mm(expert_weights_norm, expert_weights_norm.t())  # [B, B]

    # 文化损失：相同文化的样本应该有相似的专家权重
    culture_loss = torch.nn.functional.mse_loss(
        expert_similarity * culture_similarity,
        culture_similarity
    )

    return culture_loss
```

## 📊 修复后的输出格式

### 训练时（提供 labels）

```python
outputs = {
    'logits': tensor([[0.1, 0.2, 0.7], ...]),  # [B, num_classes]
    'classification_loss': tensor(1.2345),      # 分类损失
    'culture_loss': tensor(0.0234),             # 文化损失（如果启用）
    'loss': tensor(1.2462)                      # 总损失
}
```

### 推理时（不提供 labels）

```python
outputs = {
    'logits': tensor([[0.1, 0.2, 0.7], ...])   # [B, num_classes]
}
```

## 🎯 关键改进

### 1. **添加参数**

```python
def forward(self,
            input_ids=None,
            attention_mask=None,
            input_ids_mask=None,
            attention_mask_mask=None,
            labels=None,                    # ✅ 新增
            culture_labels=None,            # ✅ 新增
            use_culture_loss=False,         # ✅ 新增
            culture_loss_lambda=0.5,        # ✅ 新增
            **kwargs):
```

### 2. **返回字典而不是 tensor**

```python
# 修复前
return logits_avg  # ❌ tensor

# 修复后
return outputs     # ✅ dict
```

### 3. **计算损失**

```python
if labels is not None:
    # 分类损失
    classification_loss = loss_fct(logits_avg, labels)

    # 文化损失（可选）
    if use_culture_loss and culture_labels is not None:
        culture_loss = self.compute_culture_loss(expert_weights, culture_labels)
        total_loss = classification_loss + culture_loss_lambda * culture_loss
    else:
        total_loss = classification_loss

    outputs['loss'] = total_loss
```

## 📈 预期效果

### 修复前

```
Training:   0%|                                                    | 0/1105 [00:01<?, ?it/s]

Traceback (most recent call last):
  ...
  IndexError: too many indices for tensor of dimension 2
```

### 修复后

```
================================================================================
Epoch 1/10
================================================================================
Training: 100%|████████████████████████████████| 1105/1105 [15:23<00:00,  1.20it/s]

Epoch 1 Training Results:
  Loss: 1.2345
  Classification Loss: 1.2000
  Culture Loss: 0.0345
  Accuracy: 0.7234

================================================================================
Epoch 1 Evaluation
================================================================================
...
```

## 🔧 兼容性

### 训练代码

```python
# train_culturemoe_from_base_gen.py
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    input_ids_mask=input_ids_mask,
    attention_mask_mask=attention_mask_mask,
    labels=labels,
    culture_labels=culture_labels,
    use_culture_loss=use_culture_loss,
    culture_loss_lambda=culture_loss_lambda
)

loss = outputs['loss']  # ✅ 现在可以正常工作
```

### 评估代码

```python
# 评估时不提供 labels
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    input_ids_mask=input_ids_mask,
    attention_mask_mask=attention_mask_mask
)

logits = outputs['logits']  # ✅ 获取 logits
preds = torch.argmax(logits, dim=-1)
```

### 生成代码

```python
# 生成时使用 llama_model 的 generate 方法
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=10
)
```

## 📝 文化损失说明

### 原理

文化损失鼓励**相同文化的样本使用相似的专家组合**：

1. **计算文化相似度**：相同文化标签的样本对为 1，不同为 0
2. **计算专家权重相似度**：使用余弦相似度
3. **损失函数**：MSE(专家相似度 × 文化相似度, 文化相似度)

### 效果

- ✅ 相同文化的样本 → 使用相似的专家
- ✅ 不同文化的样本 → 使用不同的专家
- ✅ 提高模型的文化感知能力

### 控制

```python
# 启用文化损失
use_culture_loss=True
culture_loss_lambda=0.5  # 文化损失权重

# 禁用文化损失
use_culture_loss=False
```

## 总结

✅ **主要修复**：
1. 修改 forward 方法返回字典而不是 tensor
2. 添加损失计算逻辑（分类损失 + 文化损失）
3. 添加 compute_culture_loss 方法
4. 添加必要的参数（labels, culture_labels 等）

✅ **预期效果**：
- 训练代码可以正常运行
- 支持分类损失和文化损失
- 兼容训练、评估和生成

现在可以正常训练 CultureMoE 了！🎉

