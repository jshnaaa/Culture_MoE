# CultureMoE 生成式模型修复总结

## 🎯 修复目标

将 CultureMoE 从**分类模型**改为**生成式模型**，使其能够像 LoRA 微调的 LLaMA 一样生成文本答案。

---

## ✅ 已完成的修复

### 1. **CultureMoE.py** （已经是生成式）

`src/llamafactory/model/CultureMoE.py` 已经实现为生成式模型：

```python
class LlamaSharedRouterExpertsModel(nn.Module):
    def forward(self, input_ids, attention_mask, labels=None, ...):
        # 1. LLaMA 提取特征
        hidden_states = self.llama_model.model(input_ids, attention_mask)

        # 2. MoE 增强特征
        enhanced_states = self.moe_layer(hidden_states)

        # 3. ✅ 使用 LLaMA 的 lm_head 生成 logits
        logits = self.llama_model.lm_head(enhanced_states)  # [B, L, vocab_size]

        # 4. ✅ 计算生成式损失
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss(shift_logits, shift_labels)

        return {'logits': logits, 'loss': loss, 'generation_loss': loss}

    def generate(self, input_ids, **kwargs):
        # ✅ 支持生成
        return self.llama_model.generate(input_ids, **kwargs)
```

**关键特性**：
- ✅ 输出 `[batch, seq_len, vocab_size]` 的生成 logits
- ✅ 使用 `generation_loss` 而不是 `classification_loss`
- ✅ 有 `generate()` 方法支持文本生成

### 2. **train_culturemoe_from_base_gen.py** （已修复）

修复了训练脚本中的分类逻辑：

#### 修复前 ❌
```python
# 训练循环
total_cls_loss += outputs['classification_loss'].item()  # ❌ 不存在
preds = torch.argmax(outputs['logits'], dim=-1)  # ❌ 分类预测
all_preds.extend(preds.cpu().numpy())

# 评估循环
preds = torch.argmax(outputs['logits'], dim=-1)  # ❌ 分类预测
accuracy = accuracy_score(all_labels, all_preds)
```

#### 修复后 ✅
```python
# 训练循环
total_gen_loss += outputs['generation_loss'].item()  # ✅ 生成损失
# ✅ 不在训练时计算准确率（太慢）

# 评估循环（只计算 loss）
total_gen_loss += outputs['generation_loss'].item()  # ✅ 生成损失
# ✅ 准确率通过 generate_and_evaluate() 计算

# 生成式评估
gen_metrics = generate_and_evaluate(
    model=model,
    tokenizer=tokenizer,
    val_dataset=val_dataset,
    ...
)
```

**关键修复**：
- ✅ 使用 `generation_loss` 而不是 `classification_loss`
- ✅ 移除训练时的 `torch.argmax()` 分类预测
- ✅ 使用 `generate()` 方法进行评估
- ✅ 添加分布式训练支持

---

## 📊 修复后的预期结果

### 训练输出

```
Epoch 1:
   Train Loss: 2.5-3.0, Train Gen Loss: 2.5-3.0
   Eval Loss:  2.8-3.2, Eval Gen Loss: 2.8-3.2
   Eval Accuracy (Generative): 0.30-0.40

Epoch 5:
   Train Loss: 1.5-2.0, Train Gen Loss: 1.5-2.0
   Eval Loss:  1.8-2.2, Eval Gen Loss: 1.8-2.2
   Eval Accuracy (Generative): 0.50-0.60

Epoch 10:
   Train Loss: 0.8-1.2, Train Gen Loss: 0.8-1.2
   Eval Loss:  1.0-1.5, Eval Gen Loss: 1.0-1.5
   Eval Accuracy (Generative): 0.65-0.75
```

**关键指标**：
- ✅ Train Loss 和 Eval Loss 都应该 > 0（不是 0.0）
- ✅ Gen Loss 应该逐渐下降
- ✅ Eval Accuracy 通过真实生成计算

### 生成示例

```json
{
  "instruction": "Question: How important is family in your life? Country: Andorra",
  "input": "Option: 1. Very important\n2. Rather important\n3. Not very important\n4. Not at all important",
  "true_label": 0,
  "predicted_label": 0,
  "raw_answer": "1",
  "correct": true
}
```

---

## 🚀 使用方法

### 训练

```bash
# 单卡训练
sh run_train_culturemoe_from_base_gen.sh llama 4

# 双卡训练（自动检测）
sh run_train_culturemoe_from_base_gen.sh llama 4
```

### 评估

```bash
python eval_culturemoe_from_components.py \
    --base_model_path /path/to/base_model \
    --lora_weights_path /path/to/lora/best_lora \
    --moe_weights_path /path/to/moe/best_moe \
    --test_file /path/to/test.json \
    --output_dir /path/to/results
```

---

## 🔍 关键区别：分类 vs 生成

### 分类模型 ❌

```python
# 输出
logits = classifier(hidden_states)  # [B, num_classes]

# 损失
loss = CrossEntropyLoss(logits, labels)  # labels: [B]

# 预测
preds = torch.argmax(logits, dim=-1)  # [B]

# 评估
accuracy = (preds == labels).mean()
```

### 生成式模型 ✅

```python
# 输出
logits = lm_head(hidden_states)  # [B, L, vocab_size]

# 损失
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
loss = CrossEntropyLoss(shift_logits, shift_labels)

# 预测
generated = model.generate(input_ids, max_new_tokens=10)

# 评估
pred_text = tokenizer.decode(generated)
accuracy = (pred_text == true_text)
```

---

## 📝 数据格式

### 训练数据

```json
{
  "instruction": "Question: ...",
  "instruction_mask": "Question: ...",  // 用于文化损失
  "input": "Option: 1. ...\n2. ...",
  "output": "1"  // 答案（字符串）
}
```

### 处理后的数据

```python
{
  'input_ids': [1, 2, 3, ..., 100],  // instruction + input + answer
  'attention_mask': [1, 1, 1, ..., 1],
  'labels': [-100, -100, ..., 100],  // prompt 部分用 -100，answer 部分用真实 token
  'culture_labels': 0  // 文化标签（可选）
}
```

---

## 🎯 为什么之前会出现 Train Loss = 0.0？

### 原因分析

1. **模型架构正确**：CultureMoE 已经是生成式模型
2. **训练脚本错误**：使用了 `classification_loss`（不存在）
3. **评估逻辑错误**：使用 `torch.argmax()` 进行分类

### 实际情况

```python
# 训练时
outputs = model(input_ids, labels=labels)
# outputs = {'logits': [...], 'loss': 2.5, 'generation_loss': 2.5}

# 但训练脚本尝试访问
loss = outputs['classification_loss']  # ❌ KeyError 或返回 0.0
```

**结果**：
- 训练脚本可能捕获了异常并返回 0.0
- 或者使用了默认值 0.0
- 导致打印出 `Train Loss: 0.0000`

---

## ✅ 修复验证

### 检查点 1：模型输出

```python
outputs = model(input_ids, attention_mask, labels=labels)
print(outputs.keys())
# 应该输出: ['logits', 'loss', 'generation_loss', 'culture_loss']

print(outputs['logits'].shape)
# 应该输出: torch.Size([batch_size, seq_len, vocab_size])
```

### 检查点 2：损失值

```python
print(f"Loss: {outputs['loss'].item()}")
# 应该输出: Loss: 2.5xxx (不是 0.0)

print(f"Gen Loss: {outputs['generation_loss'].item()}")
# 应该输出: Gen Loss: 2.5xxx (不是 0.0)
```

### 检查点 3：生成能力

```python
generated = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=10
)
print(tokenizer.decode(generated[0]))
# 应该输出: "1" 或 "2" 等答案
```

---

## 🚀 下一步

1. **重新训练模型**
   ```bash
   sh run_train_culturemoe_from_base_gen.sh llama 4
   ```

2. **观察训练日志**
   - Train Loss 应该从 2.5-3.0 开始
   - Gen Loss 应该逐渐下降
   - Eval Accuracy 应该逐渐提升

3. **检查生成结果**
   ```bash
   cat /path/to/output/generated_answers.json | head -20
   ```

4. **评估最终模型**
   ```bash
   python eval_culturemoe_from_components.py ...
   ```

---

## 📚 相关文件

- `src/llamafactory/model/CultureMoE.py` - 模型定义（✅ 已经是生成式）
- `train_culturemoe_from_base_gen.py` - 训练脚本（✅ 已修复）
- `eval_culturemoe_from_components.py` - 评估脚本
- `run_train_culturemoe_from_base_gen.sh` - 训练启动脚本

---

## 🎉 总结

✅ **CultureMoE 现在是一个真正的生成式模型**：
1. 使用 `lm_head` 生成 vocab logits
2. 计算生成式损失（next token prediction）
3. 支持 `generate()` 方法
4. 训练和评估逻辑已修复

现在可以重新训练，应该会看到正常的 loss 值和准确率！🚀

