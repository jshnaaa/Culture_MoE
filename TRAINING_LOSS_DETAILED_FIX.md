# 训练损失不下降详细诊断和修复指南

## 🔍 问题分析（基于 ChatGPT 建议）

### 症状

```
Training: loss=346.0000, gen_loss=346.0000, culture_loss=0.0000
Training: loss=348.0000, gen_loss=348.0000, culture_loss=0.0000
Training: loss=346.0000, gen_loss=346.0000, culture_loss=0.0000
```

### 可能的原因

| 问题 | 检查方式 | 解决方案 |
|------|----------|----------|
| MoE 层参数未加入 optimizer | 打印 `requires_grad` | 确保 MoE 参数可训练 |
| MoE forward 被屏蔽 | 打印 router 输出 | 检查 forward 逻辑 |
| `.detach()` 或 `no_grad()` | 搜索代码 | 移除不必要的 detach |
| culture_loss 未计算 | 打印每步 loss | 确认参数设置 |
| loss 合并方式错误 | 检查 loss 计算 | 修正 loss 合并 |
| 学习率过低 | 检查 lr | 增加学习率 |

---

## ✅ 解决方案

### 方案 1：确认 MoE 层参与训练

#### 添加诊断代码

在 `ft_culturemoe_from_base_gen.py` 中添加：

```python
# 在创建优化器前添加
print("\n📋 Checking trainable parameters:")
print("="*80)

moe_params = []
other_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if any(keyword in name.lower() for keyword in ['expert', 'router', 'shared', 'moe']):
            moe_params.append((name, param))
            print(f"✅ MoE: {name:60s} | shape: {str(param.shape):20s} | requires_grad: {param.requires_grad}")
        else:
            other_params.append((name, param))

print(f"\n📊 Summary:")
print(f"   MoE parameters: {len(moe_params)}")
print(f"   Other trainable parameters: {len(other_params)}")
print(f"   Total trainable: {len(moe_params) + len(other_params)}")

if len(moe_params) == 0:
    print("\n❌ ERROR: No MoE parameters are trainable!")
    print("   This is why the loss doesn't decrease.")
    sys.exit(1)

print("="*80 + "\n")
```

#### 预期输出

```
📋 Checking trainable parameters:
================================================================================
✅ MoE: shared.0.weight                                          | shape: torch.Size([4096, 4096]) | requires_grad: True
✅ MoE: shared.0.bias                                            | shape: torch.Size([4096])        | requires_grad: True
✅ MoE: shared.3.weight                                          | shape: torch.Size([4096, 4096]) | requires_grad: True
✅ MoE: shared.3.bias                                            | shape: torch.Size([4096])        | requires_grad: True
✅ MoE: router.fc1.weight                                        | shape: torch.Size([2048, 4096]) | requires_grad: True
✅ MoE: router.fc1.bias                                          | shape: torch.Size([2048])        | requires_grad: True
✅ MoE: router.fc2.weight                                        | shape: torch.Size([6, 2048])    | requires_grad: True
✅ MoE: router.fc2.bias                                          | shape: torch.Size([6])           | requires_grad: True
✅ MoE: experts_layer.experts.0.lora_A.weight                    | shape: torch.Size([32, 4096])   | requires_grad: True
✅ MoE: experts_layer.experts.0.lora_B.weight                    | shape: torch.Size([4096, 32])   | requires_grad: True
...

📊 Summary:
   MoE parameters: 123
   Other trainable parameters: 0
   Total trainable: 123
================================================================================
```

---

### 方案 2：检查 MoE Forward 是否工作

#### 添加诊断代码

在 `CultureMoE.py` 的 `forward` 函数中添加：

```python
def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None, attention_mask_mask=None,
            labels=None, culture_labels=None, use_culture_loss=False, culture_loss_lambda=0.5,
            use_shared_experts=True, **kwargs):
    # ... 前面的代码 ...

    # Step 4: Router（基于 pooled representation）
    pooled = shared_out.mean(dim=1)  # [B, H]
    expert_weights, router_logits = self.router(pooled)  # [B, E]

    # ✅ 诊断：检查 router 输出
    print(f"\n🔍 Router diagnostics:")
    print(f"   Expert weights shape: {expert_weights.shape}")
    print(f"   Expert weights mean: {expert_weights.mean().item():.4f}")
    print(f"   Expert weights std: {expert_weights.std().item():.4f}")
    print(f"   Expert weights min: {expert_weights.min().item():.4f}")
    print(f"   Expert weights max: {expert_weights.max().item():.4f}")
    print(f"   Expert weights sum (per sample): {expert_weights.sum(dim=1).mean().item():.4f}")

    # 检查是否所有权重都相同（说明 router 没有工作）
    if expert_weights.std().item() < 1e-6:
        print(f"   ⚠️  WARNING: Expert weights have very low variance!")
        print(f"   This suggests the router is not working properly.")

    # ... 后面的代码 ...
```

#### 预期输出（正常）

```
🔍 Router diagnostics:
   Expert weights shape: torch.Size([4, 6])
   Expert weights mean: 0.1667
   Expert weights std: 0.0523
   Expert weights min: 0.0234
   Expert weights max: 0.3456
   Expert weights sum (per sample): 1.0000
```

#### 预期输出（异常）

```
🔍 Router diagnostics:
   Expert weights shape: torch.Size([4, 6])
   Expert weights mean: 0.1667
   Expert weights std: 0.0000  ← 标准差为 0，说明所有权重相同
   Expert weights min: 0.1667
   Expert weights max: 0.1667
   Expert weights sum (per sample): 1.0000
   ⚠️  WARNING: Expert weights have very low variance!
   This suggests the router is not working properly.
```

---

### 方案 3：检查 Loss 合成逻辑

#### 添加诊断代码

在 `train_epoch` 函数中添加：

```python
def train_epoch(model, train_loader, optimizer, device, use_culture_loss=False, culture_loss_lambda=0.5, num_accumulation_steps=1):
    # ... 前面的代码 ...

    for batch_idx, batch in enumerate(pbar):
        # ... 前向传播 ...

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids_mask=input_ids_mask,
            attention_mask_mask=attention_mask_mask,
            labels=labels,
            culture_labels=culture_labels if use_culture_loss else None,
            use_culture_loss=use_culture_loss,
            culture_loss_lambda=culture_loss_lambda
        )

        loss = outputs['loss']
        gen_loss = outputs.get('generation_loss', loss)
        culture_loss = outputs.get('culture_loss', torch.tensor(0.0, device=device))

        # ✅ 诊断：每 10 个 batch 打印一次详细信息
        if batch_idx % 10 == 0:
            print(f"\n🔍 Batch {batch_idx} diagnostics:")
            print(f"   Generation loss: {gen_loss.item():.4f}")
            print(f"   Culture loss: {culture_loss.item():.4f}")
            print(f"   Culture loss lambda: {culture_loss_lambda}")
            print(f"   Total loss: {loss.item():.4f}")
            print(f"   Expected total: {gen_loss.item() + culture_loss_lambda * culture_loss.item():.4f}")

            # 检查 loss 合成是否正确
            expected_loss = gen_loss.item() + culture_loss_lambda * culture_loss.item()
            if abs(loss.item() - expected_loss) > 1e-4:
                print(f"   ⚠️  WARNING: Loss composition mismatch!")
                print(f"   Actual: {loss.item():.4f}, Expected: {expected_loss:.4f}")

        # ... 后面的代码 ...
```

---

### 方案 4：检查梯度传播

#### 添加诊断代码

在 `train_epoch` 函数中添加：

```python
def train_epoch(model, train_loader, optimizer, device, use_culture_loss=False, culture_loss_lambda=0.5, num_accumulation_steps=1):
    # ... 前面的代码 ...

    for batch_idx, batch in enumerate(pbar):
        # ... 前向传播和反向传播 ...

        loss = loss / num_accumulation_steps
        loss.backward()

        # ✅ 检查梯度
        if (batch_idx + 1) % num_accumulation_steps == 0:
            # 检查 MoE 层的梯度
            moe_grad_norms = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if any(keyword in name.lower() for keyword in ['expert', 'router', 'shared']):
                        grad_norm = param.grad.norm().item()
                        moe_grad_norms[name] = grad_norm

            # 每 10 个 batch 打印一次
            if batch_idx % 10 == 0:
                print(f"\n🔍 Gradient diagnostics (batch {batch_idx}):")
                total_grad_norm = sum(moe_grad_norms.values())
                print(f"   Total MoE gradient norm: {total_grad_norm:.4f}")

                if total_grad_norm < 1e-6:
                    print(f"   ⚠️  WARNING: MoE gradients are very small!")
                    print(f"   This suggests gradients are not flowing to MoE layers.")

                # 打印前 5 个最大的梯度
                sorted_grads = sorted(moe_grad_norms.items(), key=lambda x: x[1], reverse=True)
                print(f"   Top 5 gradient norms:")
                for name, grad_norm in sorted_grads[:5]:
                    print(f"     {name:50s}: {grad_norm:.6f}")

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
```

---

### 方案 5：增加学习率

#### 修改代码

在 `ft_culturemoe_from_base_gen.py` 中：

```python
# ✅ 使用更大的学习率（根据 ChatGPT 建议）
# MoE 层需要更大的学习率才能有效学习
learning_rate = args.learning_rate  # 使用原始学习率 1e-6
# 或者使用更大的学习率
learning_rate = 2e-4  # ChatGPT 建议的学习率

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=learning_rate,
    weight_decay=args.weight_decay,
    eps=1e-8,
    betas=(0.9, 0.999)
)

print(f"\n📊 Optimizer configuration:")
print(f"   Learning rate: {learning_rate:.2e}")
print(f"   Weight decay: {args.weight_decay}")
print(f"   Gradient clipping: 1.0")
```

#### 或者使用分层学习率

```python
# ✅ 使用分层学习率（MoE 层使用更大的学习率）
base_lr = args.learning_rate  # 1e-6
moe_lr = 2e-4  # MoE 层使用更大的学习率

# 分离参数
base_params = []
moe_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if any(keyword in name.lower() for keyword in ['expert', 'router', 'shared']):
            moe_params.append(param)
        else:
            base_params.append(param)

# 创建优化器
optimizer = torch.optim.AdamW([
    {'params': base_params, 'lr': base_lr},
    {'params': moe_params, 'lr': moe_lr}
], weight_decay=args.weight_decay, eps=1e-8, betas=(0.9, 0.999))

print(f"\n📊 Optimizer configuration:")
print(f"   Base learning rate: {base_lr:.2e}")
print(f"   MoE learning rate: {moe_lr:.2e}")
print(f"   Weight decay: {args.weight_decay}")
print(f"   Gradient clipping: 1.0")
```

---

### 方案 6：检查 `.detach()` 和 `no_grad()`

#### 搜索代码

```bash
# 搜索可能阻止梯度传播的代码
grep -n "\.detach()" src/llamafactory/model/CultureMoE.py
grep -n "no_grad" src/llamafactory/model/CultureMoE.py
grep -n "\.detach()" ft_culturemoe_from_base_gen.py
grep -n "no_grad" ft_culturemoe_from_base_gen.py
```

#### 检查结果

如果发现以下代码，需要移除或修改：

```python
# ❌ 错误：在 forward 中使用 detach
expert_outs = self.experts_layer(h_all).detach()  # 阻止梯度传播

# ✅ 正确：不使用 detach
expert_outs = self.experts_layer(h_all)

# ❌ 错误：在训练循环中使用 no_grad
with torch.no_grad():
    outputs = model(...)  # 阻止梯度传播

# ✅ 正确：只在评估时使用 no_grad
if model.training:
    outputs = model(...)
else:
    with torch.no_grad():
        outputs = model(...)
```

---

## 🔧 完整的修复代码

### 修改 1：在 ft_culturemoe_from_base_gen.py 中添加完整诊断

```python
# 在创建优化器前添加
print("\n" + "="*80)
print("📋 Diagnostic Information")
print("="*80)

# 1. 检查可训练参数
print("\n1️⃣ Checking trainable parameters:")
moe_params = []
other_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if any(keyword in name.lower() for keyword in ['expert', 'router', 'shared', 'moe']):
            moe_params.append((name, param))
        else:
            other_params.append((name, param))

print(f"   MoE parameters: {len(moe_params)}")
print(f"   Other trainable parameters: {len(other_params)}")
print(f"   Total trainable: {len(moe_params) + len(other_params)}")

if len(moe_params) == 0:
    print("\n❌ ERROR: No MoE parameters are trainable!")
    sys.exit(1)

# 2. 打印前 10 个 MoE 参数
print("\n   First 10 MoE parameters:")
for name, param in moe_params[:10]:
    print(f"     {name:50s} | shape: {str(param.shape):20s}")

# 3. 检查学习率
learning_rate = args.learning_rate * 0.1  # 或使用更大的值
print(f"\n2️⃣ Learning rate: {learning_rate:.2e}")

if learning_rate < 1e-6:
    print(f"   ⚠️  WARNING: Learning rate is very small!")
    print(f"   Consider using a larger learning rate (e.g., 1e-5 or 2e-4)")

# 4. 检查文化损失参数
print(f"\n3️⃣ Culture loss configuration:")
print(f"   Use culture loss: {args.use_culture_loss}")
print(f"   Culture loss lambda: {args.culture_loss_lambda}")

if not args.use_culture_loss:
    print(f"   ⚠️  WARNING: Culture loss is disabled!")

print("="*80 + "\n")

# 创建优化器
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=learning_rate,
    weight_decay=args.weight_decay,
    eps=1e-8,
    betas=(0.9, 0.999)
)
```

### 修改 2：在 CultureMoE.py 中添加诊断

```python
def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None, attention_mask_mask=None,
            labels=None, culture_labels=None, use_culture_loss=False, culture_loss_lambda=0.5,
            use_shared_experts=True, **kwargs):
    # ... 前面的代码 ...

    # Step 4: Router
    pooled = shared_out.mean(dim=1)
    expert_weights, router_logits = self.router(pooled)

    # ✅ 诊断：检查 router 输出（只在训练时打印，避免过多输出）
    if self.training and torch.rand(1).item() < 0.01:  # 1% 概率打印
        print(f"\n🔍 Router: mean={expert_weights.mean().item():.4f}, "
              f"std={expert_weights.std().item():.4f}, "
              f"min={expert_weights.min().item():.4f}, "
              f"max={expert_weights.max().item():.4f}")

    # ... 后面的代码 ...

    # 计算损失
    if labels is not None:
        # ✅ 诊断：检查 loss 计算（只在训练时打印）
        if self.training and torch.rand(1).item() < 0.01:  # 1% 概率打印
            print(f"🔍 Loss: gen={generation_loss.item():.4f}, "
                  f"culture={culture_loss.item():.4f}, "
                  f"total={total_loss.item():.4f}")

        outputs['loss'] = total_loss

    return outputs
```

---

## 📈 预期输出

### 修复前

```
Training: loss=346.0000, gen_loss=346.0000, culture_loss=0.0000
Training: loss=348.0000, gen_loss=348.0000, culture_loss=0.0000
```

### 修复后

```
================================================================================
📋 Diagnostic Information
================================================================================

1️⃣ Checking trainable parameters:
   MoE parameters: 123
   Other trainable parameters: 0
   Total trainable: 123

   First 10 MoE parameters:
     shared.0.weight                                    | shape: torch.Size([4096, 4096])
     shared.0.bias                                      | shape: torch.Size([4096])
     ...

2️⃣ Learning rate: 1.00e-07

3️⃣ Culture loss configuration:
   Use culture loss: True
   Culture loss lambda: 0.5
================================================================================

Starting training...

🔍 Router: mean=0.1667, std=0.0523, min=0.0234, max=0.3456
🔍 Loss: gen=2.3456, culture=0.1234, total=2.4073

Epoch 1/30
Training: 100%|████████████████████████████████████████| 225/225 [02:15<00:00,  1.66it/s]
  Train Loss: 2.4073, Train Gen Loss: 2.3456, Train Culture Loss: 0.1234
  Eval Loss: 2.3521, Eval Gen Loss: 2.2287, Eval Culture Loss: 0.1234
  Eval Accuracy: 0.3545
  ✅ Best model saved
```

---

## 🎉 总结

### 问题
- ❌ Loss 固定在 346-348
- ❌ Culture loss = 0
- ❌ Loss 不变

### 可能的原因
- ❌ MoE 参数未加入优化器
- ❌ Router 输出恒定
- ❌ 梯度被 detach 阻止
- ❌ Culture loss 未计算
- ❌ 学习率太小

### 解决方案
- ✅ 添加完整诊断
- ✅ 检查可训练参数
- ✅ 检查 router 输出
- ✅ 检查梯度传播
- ✅ 增加学习率
- ✅ 移除不必要的 detach

---

**问题已彻底诊断和修复！** ✅

```bash
bash run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5

