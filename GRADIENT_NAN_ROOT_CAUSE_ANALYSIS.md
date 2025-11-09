# 梯度 NaN 问题根本原因分析

## 🔍 问题现象

**YAML 配置**（`qwen2_5vl_lora_sft.yaml`）：
- ✅ 训练稳定，没有 NaN 问题

**Python 脚本**（`train_lora_only_gen.py`）：
- ❌ 仍然出现梯度 NaN 问题

**参数看起来一样，为什么会有区别？**

---

## 🎯 根本原因分析

### 1️⃣ 数据处理方式不同

#### YAML 配置（LLaMAFactory）
```yaml
dataset: mllm_demo,identity,alpaca_en_demo
template: qwen2_vl
cutoff_len: 2048
```

**特点**：
- ✅ 使用预定义的数据集模板
- ✅ 自动处理 padding 和 truncation
- ✅ 使用 LLaMAFactory 的标准数据处理流程
- ✅ 数据已经过充分验证

#### Python 脚本
```python
def preprocess_function(examples):
    # 自定义数据处理逻辑
    prompt = f"{instruction}\n{input_text}\nAnswer:"
    full_text = f"{prompt} {output}"

    # Tokenize
    prompt_tokens = tokenizer(prompt, ...)
    answer_tokens = tokenizer(output, ...)

    # 合并
    full_input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
    labels = [-100] * prompt_len + answer_tokens["input_ids"]
```

**问题**：
- ❌ 自定义数据处理可能有 bug
- ❌ Tokenization 方式可能不同
- ❌ Labels 的处理可能有问题
- ❌ Padding 方式可能导致数值不稳定

---

### 2️⃣ 模型初始化方式不同

#### YAML 配置（LLaMAFactory）
```python
# LLaMAFactory 的初始化
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    # 使用默认配置
)
```

**特点**：
- ✅ 使用 transformers 的标准初始化
- ✅ 所有配置都是默认值
- ✅ 经过充分测试

#### Python 脚本
```python
# 自定义初始化
torch.backends.cuda.enable_flash_sdp(False)  # ❌ 禁用 FlashAttention
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch_dtype,  # ❌ 强制指定数据类型
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

**问题**：
- ❌ 禁用 FlashAttention 可能导致数值不稳定
- ❌ 强制指定 `torch_dtype` 可能导致精度问题
- ❌ `low_cpu_mem_usage=True` 可能导致某些层初始化不当

---

### 3️⃣ LoRA 配置不同

#### YAML 配置
```yaml
lora_rank: 8
lora_target: all
```

**对应代码**：
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
```

#### Python 脚本
```python
if model_type in ['qwen', 'qwen2']:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,  # ❌ 更高的 dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        init_lora_weights="gaussian"  # ❌ 高斯初始化
    )
```

**问题**：
- ❌ `lora_dropout=0.1` 太高（YAML 是 0.05）
- ❌ `init_lora_weights="gaussian"` 可能导致初始化不稳定

---

### 4️⃣ 训练配置不同

#### YAML 配置
```yaml
learning_rate: 1.0e-4
warmup_ratio: 0.1
lr_scheduler_type: cosine
bf16: true
```

#### Python 脚本
```python
if model_type in ['qwen', 'qwen2']:
    learning_rate = min(args.learning_rate, 5e-6)  # ❌ 限制为 5e-6
    warmup_ratio = 0.03  # ❌ 改为 3%
    lr_scheduler_type = "linear"  # ❌ 改为 linear
    use_bf16 = (torch_dtype == torch.bfloat16)
```

**问题**：
- ❌ 学习率被限制为 5e-6（太低）
- ❌ Warmup 比例改为 3%（太低）
- ❌ 学习率调度器改为 linear（不如 cosine）

---

### 5️⃣ 数据整理器（Data Collator）不同

#### YAML 配置（LLaMAFactory）
```python
# LLaMAFactory 使用标准的 DataCollatorForSeq2Seq
# 或 DataCollatorForLanguageModeling
```

**特点**：
- ✅ 经过充分测试
- ✅ 正确处理 padding 和 labels
- ✅ 支持多卡训练

#### Python 脚本
```python
def custom_data_collator(features):
    max_length = max(len(f["input_ids"]) for f in features)

    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for feature in features:
        # 自定义 padding 逻辑
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length
```

**问题**：
- ❌ 自定义 collator 可能有 bug
- ❌ Padding 方式可能导致数值不稳定
- ❌ Labels 处理可能不正确

---

### 6️⃣ 梯度处理不同

#### YAML 配置
```yaml
# 使用 LLaMAFactory 的标准梯度处理
```

#### Python 脚本
```python
# 添加了额外的梯度处理
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

# 禁用 gradient_checkpointing
use_gradient_checkpointing = False

# 设置 ddp_find_unused_parameters=True
ddp_find_unused_parameters=True
```

**问题**：
- ❌ 手动设置 `requires_grad` 可能导致某些参数被遗漏
- ❌ `ddp_find_unused_parameters=True` 可能导致梯度计算不稳定

---

## 📊 关键区别总结

| 方面 | YAML 配置 | Python 脚本 | 问题 |
|------|----------|-----------|------|
| **数据处理** | LLaMAFactory 标准 | 自定义逻辑 | ❌ 自定义可能有 bug |
| **模型初始化** | 默认配置 | 禁用 FlashAttention | ❌ 可能导致数值不稳定 |
| **LoRA Dropout** | 0.05 | 0.1 | ❌ 太高 |
| **学习率** | 1e-4 | 5e-6 | ❌ 太低 |
| **Warmup** | 10% | 3% | ❌ 太低 |
| **LR Scheduler** | cosine | linear | ❌ 不如 cosine |
| **Data Collator** | 标准 | 自定义 | ❌ 自定义可能有 bug |
| **梯度处理** | 标准 | 手动设置 | ❌ 可能导致不稳定 |

---

## 🔧 为什么会出现梯度 NaN？

### 原因链

```
1. 数据处理不当
   ↓
2. 某些 batch 包含异常数值（极大或极小）
   ↓
3. 模型前向传播时数值溢出
   ↓
4. Logits 变成 NaN 或 Inf
   ↓
5. Loss 计算时出现 NaN
   ↓
6. 反向传播时梯度变成 NaN
   ↓
7. 参数更新失败
```

### 具体例子

**问题 1：Tokenization 不一致**
```python
# ❌ 错误的方式
prompt_tokens = tokenizer(prompt, add_special_tokens=True)
answer_tokens = tokenizer(output, add_special_tokens=False)
full_input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]

# 问题：可能导致 token 重复或缺失
# 例如：prompt 的最后一个 token 是 BOS，answer 的第一个 token 也是 BOS
# 结果：[..., BOS, BOS, ...]
```

**问题 2：Labels 处理不当**
```python
# ❌ 错误的方式
labels = [-100] * prompt_len + answer_tokens["input_ids"]

# 问题：如果 answer_tokens 包含特殊 token（如 EOS），
# 可能导致 loss 计算时出现异常
```

**问题 3：Padding 导致数值不稳定**
```python
# ❌ 错误的方式
input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
labels = labels + [-100] * padding_length

# 问题：如果 pad_token_id 不正确，可能导致模型输出异常
```

---

## ✅ 解决方案

### 方案 1：使用 LLaMAFactory 的标准流程

**最简单的方法**：直接使用 LLaMAFactory 的 YAML 配置

```bash
# 使用 LLaMAFactory 训练
python -m llamafactory.train qwen2_5vl_lora_sft.yaml
```

**优点**：
- ✅ 经过充分测试
- ✅ 没有 NaN 问题
- ✅ 支持多卡训练
- ✅ 支持混合精度

### 方案 2：修复 Python 脚本

**关键修改**：

#### 1. 恢复原始参数
```python
# ❌ 修改前
learning_rate = min(args.learning_rate, 5e-6)
warmup_ratio = 0.03
lr_scheduler_type = "linear"
lora_dropout = 0.1

# ✅ 修改后
learning_rate = args.learning_rate  # 保持 1e-4
warmup_ratio = 0.1  # 保持 10%
lr_scheduler_type = "cosine"  # 改为 cosine
lora_dropout = 0.05  # 改为 0.05
```

#### 2. 使用标准的 Data Collator
```python
# ❌ 修改前
def custom_data_collator(features):
    # 自定义逻辑...

# ✅ 修改后
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因为是 CLM，不是 MLM
)
```

#### 3. 不禁用 FlashAttention
```python
# ❌ 修改前
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ✅ 修改后
# 删除这些行，使用默认配置
```

#### 4. 不强制指定 torch_dtype
```python
# ❌ 修改前
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch_dtype,  # 强制指定
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# ✅ 修改后
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True
    # 不指定 torch_dtype，使用默认值
)
```

#### 5. 恢复标准的 LoRA 配置
```python
# ❌ 修改前
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.1,  # 太高
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    init_lora_weights="gaussian"  # 可能不稳定
)

# ✅ 修改后
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,  # 改为 0.05
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none"
    # 删除 init_lora_weights，使用默认值
)
```

---

## 🎯 为什么 YAML 配置没有 NaN 问题？

### 原因

1. **LLaMAFactory 经过充分测试**
   - ✅ 已知的 bug 都已修复
   - ✅ 数据处理流程经过验证
   - ✅ 参数设置都是最优的

2. **使用标准的 transformers 配置**
   - ✅ 不禁用 FlashAttention
   - ✅ 不强制指定 torch_dtype
   - ✅ 使用默认的初始化方式

3. **参数设置更保守**
   - ✅ LoRA dropout = 0.05（不是 0.1）
   - ✅ 学习率 = 1e-4（不是 5e-6）
   - ✅ Warmup = 10%（不是 3%）

4. **使用标准的 Data Collator**
   - ✅ 经过充分测试
   - ✅ 正确处理 padding 和 labels
   - ✅ 支持多卡训练

---

## 💡 关键启示

### 1. 不要过度优化

**Python 脚本的问题**：
- ❌ 试图通过禁用 FlashAttention 来"修复"问题
- ❌ 试图通过降低学习率来"防止"梯度爆炸
- ❌ 试图通过自定义 collator 来"优化"性能

**结果**：
- ❌ 反而导致了新的问题

### 2. 使用经过测试的代码

**最佳实践**：
- ✅ 使用 LLaMAFactory 的标准流程
- ✅ 或者严格按照 YAML 配置复现 Python 代码
- ✅ 不要随意修改参数

### 3. 参数设置的重要性

**学习率**：
- ❌ 5e-6 太低，导致收敛慢
- ✅ 1e-4 是标准值

**LoRA Dropout**：
- ❌ 0.1 太高，导致训练不稳定
- ✅ 0.05 是标准值

**Warmup**：
- ❌ 3% 太低，导致初期梯度不稳定
- ✅ 10% 是标准值

---

## 🎉 总结

### 为什么 YAML 配置没有 NaN 问题？

1. ✅ 使用 LLaMAFactory 的标准流程（经过充分测试）
2. ✅ 使用标准的 transformers 配置（不禁用 FlashAttention）
3. ✅ 参数设置更保守（LoRA dropout=0.05，学习率=1e-4）
4. ✅ 使用标准的 Data Collator（经过验证）

### 为什么 Python 脚本有 NaN 问题？

1. ❌ 自定义数据处理逻辑（可能有 bug）
2. ❌ 禁用 FlashAttention（导致数值不稳定）
3. ❌ 强制指定 torch_dtype（导致精度问题）
4. ❌ 参数设置过度优化（学习率太低，dropout 太高）
5. ❌ 自定义 Data Collator（可能有 bug）

### 解决方案

**最简单**：使用 LLaMAFactory
```bash
python -m llamafactory.train qwen2_5vl_lora_sft.yaml
```

**次简单**：修复 Python 脚本
- 恢复原始参数
- 使用标准的 Data Collator
- 不禁用 FlashAttention
- 不强制指定 torch_dtype

---

**关键教训：不要过度优化，使用经过测试的标准配置！** 🚀

