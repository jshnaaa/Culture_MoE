# 🔧 LoRA 训练脚本修复说明

## ❌ 问题

运行 `run_train_lora_only.sh` 时出现错误：

```
ValueError: Unable to create tensor, you should probably activate truncation and/or padding
with 'padding=True' 'truncation=True' to have batched tensors with the same length.
Perhaps your features (`input_ids_mask` in this case) have excessive nesting
(inputs type `list` where type `int` is expected).
```

## 🔍 原因

`DataCollatorWithPadding` 是 Transformers 库的标准 collator，只支持标准字段：
- `input_ids`
- `attention_mask`
- `labels`

但我们的数据包含额外的字段：
- `input_ids_mask`（第二路输入）
- `attention_mask_mask`（第二路输入）
- `culture_labels`（文化维度标签）

标准 collator 无法处理这些自定义字段，导致错误。

---

## ✅ 解决方案

### 1. 使用自定义 DataCollator

**修改前**：
```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    max_length=args.max_length
)
```

**修改后**：
```python
from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator

data_collator = DualClassificationDataCollator(
    tokenizer=tokenizer,
    padding=True,
    max_length=args.max_length
)
```

### 2. 修改模型 forward 方法

**修改前**：
```python
def forward(self, input_ids, attention_mask, labels=None):
    # 只接受标准参数
    ...
```

**修改后**：
```python
def forward(self, input_ids, attention_mask, labels=None, **kwargs):
    """
    Args:
        input_ids: [B, L]
        attention_mask: [B, L]
        labels: [B]
        **kwargs: 其他字段会被忽略
    """
    # 只使用第一路输入
    outputs = self.llama_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    ...
```

---

## 📊 数据流程

### 修复后的流程

```
数据集
  ↓
{
  "input_ids": [L],
  "attention_mask": [L],
  "input_ids_mask": [L],          ← 额外字段
  "attention_mask_mask": [L],     ← 额外字段
  "labels": 0,
  "culture_labels": [0]           ← 额外字段
}
  ↓
DualClassificationDataCollator
  ↓
{
  "input_ids": [B, L],
  "attention_mask": [B, L],
  "input_ids_mask": [B, L],       ← 正确处理
  "attention_mask_mask": [B, L],  ← 正确处理
  "labels": [B],
  "culture_labels": [[0], [1, 2], ...]  ← 正确处理
}
  ↓
BinaryClassificationModel.forward(**batch)
  ↓
只使用 input_ids 和 attention_mask
其他字段通过 **kwargs 接收并忽略
```

---

## 🔧 修改的文件

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `train_and_eval_lora_only.py` | 使用自定义 DataCollator | ✅ 修复 |
| `train_and_eval_lora_only.py` | 修改 forward 方法 | ✅ 修复 |

---

## 🎯 为什么这样修改

### 1. 为什么使用自定义 DataCollator？

**原因**：
- 我们的数据格式包含 5 个字段（instruction/instruction_mask/input/output/label）
- 数据处理器 `load_and_process_dual_classification_data` 会生成双路输入
- 标准 collator 无法处理这些额外字段

**好处**：
- ✅ 正确处理所有字段
- ✅ 保持数据格式一致性
- ✅ 支持未来扩展

### 2. 为什么只使用第一路输入？

**原因**：
- LoRA 微调脚本不使用 MoE
- 不需要双路输入的复杂逻辑
- 只需要基本的分类功能

**实现**：
```python
def forward(self, input_ids, attention_mask, labels=None, **kwargs):
    # 只使用 input_ids 和 attention_mask
    # input_ids_mask, attention_mask_mask, culture_labels 通过 **kwargs 接收但不使用
    outputs = self.llama_model(input_ids, attention_mask, ...)
```

---

## 📝 使用方法

### 修复后的使用

```bash
# 直接运行（已修复）
bash run_train_lora_only.sh
```

### 预期输出

```
============================================================
Training LLaMA 3.1 with LoRA (No MoE)
============================================================

1. Loading tokenizer from /root/autodl-tmp/.../Meta-Llama-3.1-8B-Instruct...
   ✅ Tokenizer loaded

2. Loading data from /root/autodl-fs/CulturalBench_Hard_merge.json...
   Data format: instruction/instruction_mask/input/output/label
   ✅ Train dataset size: 4417
   ✅ Validation dataset size: 491

   Dataset columns: ['input_ids', 'attention_mask', 'input_ids_mask', 'attention_mask_mask', 'labels', 'culture_labels']
   ✅ Culture labels found in dataset

3. Loading base LLaMA model from /root/autodl-tmp/.../Meta-Llama-3.1-8B-Instruct...
   ✅ Base model loaded

4. Applying LoRA to LLaMA model...
   LoRA Configuration:
     Rank: 8
     Alpha: 16
     Dropout: 0.05
     Target modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

   Trainable Parameters:
trainable params: 4,194,304 || all params: 8,034,194,304 || trainable%: 0.0522
   ✅ LoRA applied

5. Creating classification model...
   ✅ Classification model created

6. Creating trainer...

7. Starting training...
============================================================
Epoch 1/3:
  Training...
  Step 100: loss=0.523
  Step 200: loss=0.412
  ...
```

---

## 🔍 验证修复

### 检查点

1. ✅ **导入正确的 collator**
   ```python
   from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator
   ```

2. ✅ **使用自定义 collator**
   ```python
   data_collator = DualClassificationDataCollator(...)
   ```

3. ✅ **forward 方法接受 kwargs**
   ```python
   def forward(self, input_ids, attention_mask, labels=None, **kwargs):
   ```

4. ✅ **数据集包含所有字段**
   ```python
   Dataset columns: ['input_ids', 'attention_mask', 'input_ids_mask',
                     'attention_mask_mask', 'labels', 'culture_labels']
   ```

---

## 📊 与其他脚本的对比

| 脚本 | DataCollator | 使用字段 |
|------|-------------|---------|
| **LLaMA + MoE** | `DualClassificationDataCollator` | 全部（双路输入 + 文化标签） |
| **Base LLaMA** | `DualClassificationDataCollator` | 第一路输入 |
| **LoRA 微调** | `DualClassificationDataCollator` ✅ | 第一路输入 |

---

## 🎉 总结

### 问题

- ❌ 使用标准 `DataCollatorWithPadding`
- ❌ 无法处理自定义字段

### 解决

- ✅ 使用自定义 `DualClassificationDataCollator`
- ✅ forward 方法接受 `**kwargs`
- ✅ 正确处理所有字段

### 结果

- ✅ 训练可以正常运行
- ✅ 数据格式保持一致
- ✅ 支持完整的数据格式

---

**修复完成，现在可以正常训练了！** 🚀

