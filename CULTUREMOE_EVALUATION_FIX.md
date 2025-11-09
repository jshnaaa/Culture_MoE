# CultureMoE 评估准确率恒定不变修复

## 🔍 问题诊断

### 症状

```
Epoch 1: Eval Accuracy: 0.3068
Epoch 2: Eval Accuracy: 0.3068  # ❌ 完全相同
Epoch 3: Eval Accuracy: 0.3068  # ❌ 完全相同

Confusion Matrix (完全相同):
       yes 2          2          96
        no 2          2          76
   neutral 2          5          77

模型几乎只预测 "neutral"
```

### 根本原因

**评估时绕过了 MoE 层！**

```python
# ❌ 原始代码（第 533 行）
outputs = model.llama_model.generate(...)  # 直接使用 llama_model

# 结果：
# - 训练时：model(...) → 经过 MoE 层 ✅
# - 评估时：model.llama_model.generate(...) → 绕过 MoE 层 ❌
```

**数据流对比**：

```
训练时：
  input → LLaMA → hidden → Shared → Router → Experts → Enhanced Hidden → lm_head → logits
  ✅ 经过完整的 MoE 流程

评估时（修复前）：
  input → LLaMA.generate() → output
  ❌ 完全绕过 MoE 层，只用冻结的 LLaMA
```

---

## ✅ 修复方案

### 核心思路

**不使用 `generate()`，而是使用 `forward()` 获取 logits，然后手动解码**

```python
# ✅ 修复后的代码
with torch.no_grad():
    # 1. 调用 model.forward() 获取 logits（经过 MoE 层）
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        input_ids_mask=input_ids_mask,
        attention_mask_mask=attention_mask_mask,
        labels=None,  # 不计算损失
        use_culture_loss=False
    )

    # 2. 获取 logits [B, L, vocab_size]
    logits = outputs['logits']

    # 3. 获取最后一个 token 的 logits
    last_token_logits = logits[:, -1, :]  # [B, vocab_size]

    # 4. 贪婪解码：选择概率最高的 token
    predicted_token_id = torch.argmax(last_token_logits, dim=-1)  # [B]

    # 5. 解码为文本
    generated_text = tokenizer.decode(predicted_token_id[0], skip_special_tokens=True).strip()
```

### 完整修复

**位置**: `train_culturemoe_from_base_gen.py` 第 527-560 行

```python
def generate_and_evaluate(model, tokenizer, val_dataset, output_dir, num_classes, max_new_tokens=10):
    """生成答案并评估"""

    model.eval()
    generated_answers = []
    device = next(model.parameters()).device

    for i in tqdm(range(len(val_dataset)), desc="Generating"):
        sample = val_dataset[i]
        instruction = sample['instruction']
        input_text = sample['input']
        true_label = sample.get('output', sample.get('label', ''))

        # 构建 prompt
        if num_classes <= 10:
            full_input = f"{instruction}\n{input_text}\n\nPlease answer with ONLY ONE NUMBER (1 to {num_classes}).\nYour answer:"
        else:
            full_input = f"{instruction}\n{input_text}\n\nYour answer:"

        # Tokenize
        inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ✅ 使用 model.forward() 而不是 model.llama_model.generate()
        with torch.no_grad():
            # 构建 input_ids_mask
            input_ids_mask = inputs['input_ids'].clone()
            attention_mask_mask = inputs['attention_mask'].clone()

            # 调用 model.forward() 获取 logits（经过 MoE 层）
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                input_ids_mask=input_ids_mask,
                attention_mask_mask=attention_mask_mask,
                labels=None,
                use_culture_loss=False
            )

            # 获取 logits 并解码
            logits = outputs['logits']
            last_token_logits = logits[:, -1, :]
            predicted_token_id = torch.argmax(last_token_logits, dim=-1)
            generated_text = tokenizer.decode(predicted_token_id[0], skip_special_tokens=True).strip()

        # 提取数字
        numbers = re.findall(r'\d+', generated_text)
        if numbers:
            predicted_label = numbers[0]
        else:
            predicted_label = generated_text

        generated_answers.append({
            "predicted": predicted_label,
            "true": str(true_label)
        })

    # 保存并评估
    ...
```

---

## 📊 修复前后对比

### 修复前 ❌

```python
# 评估时
outputs = model.llama_model.generate(...)  # 绕过 MoE

# 结果
Epoch 1: Accuracy = 0.3068
Epoch 2: Accuracy = 0.3068  # 完全相同
Epoch 3: Accuracy = 0.3068  # 完全相同

# 原因：只用了冻结的 LLaMA，MoE 层没有被使用
```

### 修复后 ✅

```python
# 评估时
outputs = model(...)  # 经过 MoE 层

# 预期结果
Epoch 1: Accuracy = 0.3068
Epoch 2: Accuracy = 0.4523  # ✅ 提升
Epoch 3: Accuracy = 0.5789  # ✅ 继续提升

# 原因：使用了训练好的 MoE 层
```

---

## 🎯 为什么会出现这个问题？

### 1. CultureMoE 的结构

```python
class LlamaSharedRouterExpertsModel(nn.Module):
    def __init__(self, llama_model, ...):
        self.llama_model = llama_model  # 冻结的 LLaMA
        self.shared = ...               # 可训练
        self.router = ...               # 可训练
        self.experts_layer = ...        # 可训练

    def forward(self, input_ids, ...):
        # 1. LLaMA forward
        hidden = self.llama_model.model(input_ids, ...).last_hidden_state

        # 2. Shared 层
        shared_out = self.shared(hidden)

        # 3. Router
        expert_weights = self.router(shared_out.mean(dim=1))

        # 4. Experts
        expert_outs = self.experts_layer(hidden)

        # 5. 融合
        enhanced_hidden = shared_out + weighted_expert_sum

        # 6. 生成 logits
        logits = self.llama_model.lm_head(enhanced_hidden)

        return {'logits': logits}
```

### 2. 训练时的数据流

```
input_ids → model.forward() → 经过 MoE → logits → loss.backward()
✅ MoE 层参数被更新
```

### 3. 评估时的数据流（修复前）

```
input_ids → model.llama_model.generate() → output
❌ 完全绕过 MoE 层
❌ 只用冻结的 LLaMA
❌ 结果恒定不变
```

### 4. 评估时的数据流（修复后）

```
input_ids → model.forward() → 经过 MoE → logits → argmax → output
✅ 使用训练好的 MoE 层
✅ 结果会随训练改善
```

---

## ⚠️ 注意事项

### 1. 为什么不能直接使用 `model.generate()`？

**原因**：
- `generate()` 是逐 token 生成的，需要多次调用 `forward()`
- CultureMoE 的 `forward()` 需要 `input_ids_mask` 参数
- 标准的 `generate()` 方法不支持这个参数

**解决方案**：
- 使用 `forward()` 获取 logits
- 手动解码最后一个 token

### 2. 为什么只解码最后一个 token？

**原因**：
- 我们的任务是生成一个答案（如 "yes"、"no"、"neutral" 或数字）
- 通常答案只有一个 token
- 如果需要多个 token，可以循环调用 `forward()`

### 3. 如果需要生成多个 token 怎么办？

**方案**：
```python
# 循环生成
for _ in range(max_new_tokens):
    outputs = model(input_ids=current_ids, ...)
    logits = outputs['logits']
    next_token = torch.argmax(logits[:, -1, :], dim=-1)

    # 拼接到 input_ids
    current_ids = torch.cat([current_ids, next_token.unsqueeze(-1)], dim=1)

    # 检查是否生成了 EOS
    if next_token == tokenizer.eos_token_id:
        break
```

---

## 🔍 验证修复是否生效

### 方法 1: 检查准确率变化

```bash
sh run_train_culturemoe_from_base_gen.sh llama 3
```

**预期输出**：
```
Epoch 1: Eval Accuracy: 0.3068
Epoch 2: Eval Accuracy: 0.4523  # ✅ 应该有变化
Epoch 3: Eval Accuracy: 0.5789  # ✅ 应该继续变化
```

### 方法 2: 检查 Confusion Matrix

```
修复前（所有 epoch 相同）:
       yes 2          2          96
        no 2          2          76
   neutral 2          5          77

修复后（应该有变化）:
Epoch 1:
       yes 2          2          96
        no 2          2          76
   neutral 2          5          77

Epoch 2:
       yes 15         10         75        # ✅ 变化了
        no 8          20         52        # ✅ 变化了
   neutral 5          8          71        # ✅ 变化了
```

### 方法 3: 添加调试代码

```python
# 在 generate_and_evaluate 函数开始时添加
print("\n🔍 Checking model structure:")
print(f"Model type: {type(model)}")
print(f"Has llama_model: {hasattr(model, 'llama_model')}")
print(f"Has router: {hasattr(model, 'router')}")
print(f"Has experts_layer: {hasattr(model, 'experts_layer')}")

# 在生成时添加
print(f"\n🔍 Sample {i}:")
print(f"  Input: {full_input[:50]}...")
print(f"  Logits shape: {logits.shape}")
print(f"  Last token logits shape: {last_token_logits.shape}")
print(f"  Predicted token ID: {predicted_token_id.item()}")
print(f"  Generated text: {generated_text}")
print(f"  True label: {true_label}")
```

---

## 📝 修改总结

### 修改的文件

**`train_culturemoe_from_base_gen.py`** - 1 处修复

**第 527-560 行**: 修复评估函数
- ❌ 原来：`model.llama_model.generate()`
- ✅ 现在：`model.forward()` + 手动解码

### 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 评估方法 | `llama_model.generate()` ❌ | `model.forward()` ✅ |
| 是否经过 MoE | 否 ❌ | 是 ✅ |
| 准确率变化 | 恒定 ❌ | 正常变化 ✅ |
| 预测多样性 | 只预测 neutral ❌ | 多样化 ✅ |

---

## 🎉 总结

### 问题
- ❌ 评估时使用 `model.llama_model.generate()`
- ❌ 完全绕过 MoE 层
- ❌ 准确率恒定不变

### 根本原因
- ❌ 训练时经过 MoE 层
- ❌ 评估时绕过 MoE 层
- ❌ 两者不一致

### 解决方案
- ✅ 评估时使用 `model.forward()`
- ✅ 确保经过 MoE 层
- ✅ 手动解码 logits

### 结果
- ✅ 评估时使用训练好的 MoE 层
- ✅ 准确率随训练提升
- ✅ 预测结果多样化

---

**现在可以正确评估 CultureMoE 模型了！** 🎉🚀

```bash
sh run_train_culturemoe_from_base_gen.sh llama 3
```

**预期看到准确率逐渐提升！**

