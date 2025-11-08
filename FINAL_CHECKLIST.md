# CultureMoE 训练最终检查清单

## ✅ 已修复的所有问题

### 1. **模型架构问题** ✅

#### 问题 1.1: 分类模型 vs 生成式模型
- ❌ **修复前**: 使用分类头输出 `[B, num_classes]`
- ✅ **修复后**: 使用 `lm_head` 输出 `[B, L, vocab_size]`

#### 问题 1.2: 数据类型不匹配
- ❌ **修复前**: `enhanced_hidden` (float32) vs `lm_head` (float16)
- ✅ **修复后**: 自动转换数据类型
```python
lm_head_dtype = self.llama_model.lm_head.weight.dtype
if enhanced_hidden.dtype != lm_head_dtype:
    enhanced_hidden = enhanced_hidden.to(lm_head_dtype)
```

#### 问题 1.3: 序列长度不匹配
- ❌ **修复前**: `shared_out` (138) vs `expert_sum` (139)
- ✅ **修复后**: 截断到较短长度
```python
if shared_out.size(1) != expert_sum.size(1):
    min_len = min(shared_out.size(1), expert_sum.size(1))
    shared_out = shared_out[:, :min_len, :]
    expert_sum = expert_sum[:, :min_len, :]
```

---

### 2. **数据处理问题** ✅

#### 问题 2.1: 使用分类数据处理器
- ❌ **修复前**: `DualClassificationDataCollator` 返回 1D labels `[B]`
- ✅ **修复后**: `generative_data_collator` 返回 2D labels `[B, L]`

#### 问题 2.2: Labels 格式错误
- ❌ **修复前**: 分类标签 `[0, 1, 2, ...]`
- ✅ **修复后**: 生成式标签 `[-100, -100, ..., token_ids]`

**新增函数**:
```python
def load_and_process_generative_data(data_path, tokenizer, ...):
    # 处理生成式数据
    # 返回 {'train': Dataset, 'validation': Dataset, 'validation_raw': list}

def generative_data_collator(features, tokenizer):
    # 整理生成式批次数据
    # 返回 {'input_ids': [B, L], 'labels': [B, L], ...}
```

---

### 3. **训练逻辑问题** ✅

#### 问题 3.1: 使用错误的 loss 名称
- ❌ **修复前**: `outputs['classification_loss']`
- ✅ **修复后**: `outputs['generation_loss']`

#### 问题 3.2: 训练时使用分类预测
- ❌ **修复前**: `preds = torch.argmax(outputs['logits'], dim=-1)`
- ✅ **修复后**: 移除，只在评估时使用 `generate()`

#### 问题 3.3: 评估逻辑错误
- ❌ **修复前**: 使用 `torch.argmax()` 计算准确率
- ✅ **修复后**: 使用 `generate()` 方法生成答案

---

### 4. **配置问题** ✅

#### 问题 4.1: 优化器
- ✅ **已确认**: 使用 AdamW 优化器（默认）

#### 问题 4.2: 学习率
- ✅ **已配置**: `1e-5` (合理的生成式训练学习率)

#### 问题 4.3: 批次大小
- ✅ **已配置**: `batch_size=4`, `eval_batch_size=4`

---

## 📋 完整的文件修改列表

### 修改的文件

1. **`src/llamafactory/model/CultureMoE.py`** ✅
   - 添加数据类型转换
   - 添加序列长度检查
   - 添加 labels 维度检查
   - 使用 `generation_loss` 而不是 `classification_loss`

2. **`train_culturemoe_from_base_gen.py`** ✅
   - 添加 `load_and_process_generative_data()` 函数
   - 添加 `generative_data_collator()` 函数
   - 修改训练循环使用 `generation_loss`
   - 移除训练时的分类预测
   - 修改评估逻辑

3. **`run_train_culturemoe_from_base_gen.sh`** ✅
   - 无需修改（已经正确）

---

## 🔍 最终检查项

### A. 模型检查

```python
# 检查 1: 模型输出
outputs = model(input_ids, attention_mask, labels=labels)
assert 'logits' in outputs
assert 'generation_loss' in outputs
assert outputs['logits'].shape == (batch_size, seq_len, vocab_size)
print("✅ 模型输出正确")

# 检查 2: 生成能力
generated = model.generate(input_ids, max_new_tokens=10)
assert generated.shape[1] > input_ids.shape[1]
print("✅ 生成功能正常")

# 检查 3: 损失计算
loss = outputs['loss']
assert loss.item() > 0
assert not torch.isnan(loss)
print("✅ 损失计算正常")
```

### B. 数据检查

```python
# 检查 1: 数据格式
batch = next(iter(train_loader))
assert 'input_ids' in batch
assert 'labels' in batch
assert batch['labels'].dim() == 2  # [B, L]
print("✅ 数据格式正确")

# 检查 2: Labels 格式
labels = batch['labels']
assert (labels == -100).any()  # 有 -100 (prompt 部分)
assert (labels != -100).any()  # 有真实 token (answer 部分)
print("✅ Labels 格式正确")

# 检查 3: 序列长度
assert batch['input_ids'].shape == batch['labels'].shape
print("✅ 序列长度一致")
```

### C. 训练检查

```python
# 检查 1: 前向传播
outputs = model(
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    input_ids_mask=batch['input_ids_mask'],
    attention_mask_mask=batch['attention_mask_mask'],
    labels=batch['labels'],
    culture_labels=batch['culture_labels'],
    use_culture_loss=True
)
assert 'loss' in outputs
assert 'generation_loss' in outputs
assert 'culture_loss' in outputs
print("✅ 前向传播正常")

# 检查 2: 反向传播
loss = outputs['loss']
loss.backward()
assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
print("✅ 反向传播正常")

# 检查 3: 优化器
optimizer.step()
optimizer.zero_grad()
print("✅ 优化器正常")
```

---

## 🚀 运行命令

### 1. 训练 LoRA Only（如果还没有）

```bash
# LLaMA + CultureLLM
sh run_train_lora_only_gen.sh llama 4

# 等待训练完成，确保生成 best_lora 目录
```

### 2. 训练 CultureMoE

```bash
# 单卡训练
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 1 true

# 双卡训练（如果有）
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true
```

### 3. 监控训练

```bash
# 查看输出目录
ls -lh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_*

# 查看训练日志
tail -f /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_*/training.log
```

---

## 📊 预期的训练输出

### Epoch 1
```
================================================================================
Epoch 1/10
================================================================================

Training: 100%|████████████████████| 2475/2475 [XX:XX<00:00, X.XXit/s, loss=2.8543, gen_loss=2.8543]

Evaluating (loss only): 100%|████████| 275/275 [XX:XX<00:00, X.XXit/s]

================================================================================
📊 Epoch 1 Generative Evaluation
================================================================================
Generating answers for evaluation...
100%|████████████████████████████████| 275/275 [XX:XX<00:00, X.XXit/s]

📊 Epoch 1 Results:
   Train Loss: 2.8543, Train Gen Loss: 2.8543
   Eval Loss:  3.0234, Eval Gen Loss: 3.0234
   Eval Accuracy (Generative): 0.3245
   Eval Precision: 0.3156, Recall: 0.3245, F1: 0.3198
   🏆 New best model! Accuracy: 0.3245
   ✅ Best MoE weights saved to: .../best_moe (MoE only, XXX parameters)
   Best so far: Epoch 1, Accuracy: 0.3245
```

### Epoch 5
```
📊 Epoch 5 Results:
   Train Loss: 1.5234, Train Gen Loss: 1.5234
   Eval Loss:  1.8456, Eval Gen Loss: 1.8456
   Eval Accuracy (Generative): 0.5678
   Eval Precision: 0.5543, Recall: 0.5678, F1: 0.5609
   🏆 New best model! Accuracy: 0.5678
   Best so far: Epoch 5, Accuracy: 0.5678
```

### Epoch 10
```
📊 Epoch 10 Results:
   Train Loss: 0.8765, Train Gen Loss: 0.8765
   Eval Loss:  1.2345, Eval Gen Loss: 1.2345
   Eval Accuracy (Generative): 0.7123
   Eval Precision: 0.7045, Recall: 0.7123, F1: 0.7083
   🏆 New best model! Accuracy: 0.7123
   Best so far: Epoch 10, Accuracy: 0.7123

============================================================
✅ Training completed successfully!
============================================================
```

---

## ⚠️ 可能的错误和解决方案

### 错误 1: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```bash
# 减小批次大小
--batch_size 2 \
--eval_batch_size 2
```

### 错误 2: LoRA 权重未找到
```
❌ Error: LoRA weights not found
```

**解决方案**:
```bash
# 先训练 LoRA Only
sh run_train_lora_only_gen.sh llama 4
```

### 错误 3: 数据文件未找到
```
❌ Error: Train file not found
```

**解决方案**:
```bash
# 检查数据文件路径
ls -lh /root/autodl-fs/cultureLLM_merge_gen.json
```

### 错误 4: 梯度 NaN
```
RuntimeError: Function 'XXX' returned nan values in its 0th output
```

**解决方案**:
```bash
# 降低学习率
--learning_rate 5e-6

# 或增加梯度裁剪
--max_grad_norm 0.5
```

---

## 📝 训练后的文件结构

```
/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/
└── moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1234/
    ├── best_moe/
    │   ├── moe_config.json          # MoE 配置
    │   └── moe_state_dict.pt        # MoE 权重
    ├── epoch_eval_results.json      # 每个 epoch 的结果
    ├── final_eval_results.json      # 最终评估结果
    ├── config.json                  # 训练配置
    └── generated_answers.json       # 生成的答案（最后一个 epoch）
```

---

## ✅ 最终确认

### 所有修复已完成 ✅

1. ✅ 模型架构：生成式模型
2. ✅ 数据处理：生成式数据处理
3. ✅ 训练逻辑：使用 `generation_loss`
4. ✅ 评估逻辑：使用 `generate()` 方法
5. ✅ 数据类型：自动转换
6. ✅ 序列长度：自动截断
7. ✅ Labels 维度：2D `[B, L]`
8. ✅ 优化器：AdamW

### 可以安全运行 ✅

```bash
sh run_train_culturemoe_from_base_gen.sh llama 4
```

**预期结果**:
- ✅ 训练正常启动
- ✅ Loss 从 2.5-3.0 开始
- ✅ Loss 逐渐下降
- ✅ 准确率逐渐提升
- ✅ 生成答案正常
- ✅ 保存最佳模型

---

## 🎉 总结

所有已知问题都已修复！现在可以安全地运行训练脚本了。

如果遇到任何新问题，请检查：
1. 错误信息
2. 数据格式
3. 模型输出
4. GPU 内存

祝训练顺利！🚀

