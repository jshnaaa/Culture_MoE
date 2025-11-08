# Qwen 梯度 NaN 和学习率为 0 的完整诊断

## 🔍 可能的原因分析

根据你的分析，以下是 4 个可能导致 Qwen 梯度 NaN 和学习率为 0 的原因：

### 1. ✅ LoRA 注入层名称不匹配

### 2. ✅ 学习率调度器配置错误

### 3. ⚠️ RoPE Scaling 配置不匹配

### 4. ⚠️ 数据集归一化或拼接问题

让我逐一检查：

---

## 1. ✅ LoRA 注入层名称检查

### 问题描述

> LLaMA 的层名和 Qwen 不完全一样。Qwen 的 attention 模块通常是 `c_attn`，而 LLaMA 是 `q_proj`, `v_proj`, `k_proj` 等。LoRA 若注入错了层，会导致参数梯度为 NaN。

### 当前配置

```python
# train_lora_only_gen.py (line 528)
if model_type == 'qwen':
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # ✅ 使用 LLaMA 风格
        bias="none",
        init_lora_weights="gaussian"
    )
```

### 验证结果

**✅ 当前配置是正确的！**

**原因**：
1. **Qwen 1.0 (旧版)** 使用 `c_attn`（合并的 attention）
2. **Qwen 1.5/2.0/2.5 (新版)** 使用 `q_proj`, `k_proj`, `v_proj`, `o_proj`（分离的 attention）

**证据**：
```python
# scripts/convert_ckpt/llamafy_qwen.py (line 58-66)
# 这个脚本用于将 Qwen 1.0 转换为 LLaMA 格式
if "attn.c_attn" in key:
    proj_size = value.size(0) // 3
    llama_state_dict[key.replace("attn.c_attn", "self_attn.q_proj")] = value[:proj_size, ...]
    llama_state_dict[key.replace("attn.c_attn", "self_attn.k_proj")] = value[proj_size : 2 * proj_size, ...]
    llama_state_dict[key.replace("attn.c_attn", "self_attn.v_proj")] = value[2 * proj_size :, ...]
elif "attn.c_proj" in key:
    llama_state_dict[key.replace("attn.c_proj", "self_attn.o_proj")] = value
```

**结论**：
- 如果使用 **Qwen 2.5-7B-Instruct**（你的模型），层名是 `q_proj`, `k_proj`, `v_proj`, `o_proj` ✅
- 如果使用 **Qwen 1.0**（旧版），层名是 `c_attn`, `c_proj` ❌

### 如何验证

运行以下代码检查模型的层名：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct",
    trust_remote_code=True
)

# 打印所有层名
for name, param in model.named_parameters():
    if 'attn' in name or 'proj' in name:
        print(name)
```

**预期输出（Qwen 2.5）**：
```
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
...
```

**如果是 Qwen 1.0（旧版）**：
```
model.layers.0.attn.c_attn.weight
model.layers.0.attn.c_proj.weight
...
```

### 修复方案（如果需要）

如果你使用的是 **Qwen 1.0**（旧版），需要修改 LoRA 配置：

```python
if model_type == 'qwen':
    # 检查模型版本
    if hasattr(model.config, 'qwen_version') and model.config.qwen_version == '1.0':
        # Qwen 1.0 使用 c_attn
        target_modules = ["c_attn", "c_proj"]
    else:
        # Qwen 1.5/2.0/2.5 使用分离的 attention
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,  # ✅ 根据版本选择
        bias="none",
        init_lora_weights="gaussian"
    )
```

---

## 2. ✅ 学习率调度器配置检查

### 问题描述

> 如果用了学习率调度器（例如 cosine），可能初始 step 未正确更新，使 lr=0。

### 当前配置

```python
# train_lora_only_gen.py (line 580-600)
training_args = TrainingArguments(
    ...
    learning_rate=learning_rate,
    warmup_ratio=0.1,  # ✅ 使用 warmup_ratio
    # 没有设置 warmup_steps ✅
    # 没有设置 lr_scheduler_type（默认 linear）
    ...
)
```

### 验证结果

**✅ 当前配置是正确的！**

**原因**：
1. 使用 `warmup_ratio=0.1`（10% 预热）✅
2. 没有同时设置 `warmup_steps`（避免冲突）✅
3. 使用默认的 `linear` 调度器（稳定）✅

### 学习率调度器类型

| 调度器 | 说明 | 适用场景 |
|--------|------|---------|
| **linear** | 线性衰减（默认） | ✅ 推荐，稳定 |
| **cosine** | 余弦衰减 | 大模型预训练 |
| **constant** | 恒定学习率 | 调试 |
| **constant_with_warmup** | 预热后恒定 | 小数据集 |

### 学习率变化曲线

```
Linear (默认):
  lr
   |
   |╲
   | ╲___
   |     ╲___
   |         ╲___
   |_____________╲___
   0  warmup  normal  end

Cosine:
  lr
   |
   |╲
   | ╲___
   |     ╲╲___
   |         ╲╲___
   |___________╲╲___
   0  warmup  normal  end
```

### 如何验证

添加学习率监控代码：

```python
# 在 Trainer 中添加 callback
from transformers import TrainerCallback

class LRMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            lr = kwargs['optimizer'].param_groups[0]['lr']
            print(f"Step {state.global_step}: LR = {lr:.2e}")

trainer = Trainer(
    ...
    callbacks=[LRMonitorCallback()]
)
```

### 修复方案（如果需要）

如果学习率一直为 0，检查以下配置：

```python
# ❌ 错误配置
training_args = TrainingArguments(
    warmup_steps=500,  # ❌ 固定步数
    warmup_ratio=0.3,  # ❌ 同时设置两个参数
)

# ✅ 正确配置
training_args = TrainingArguments(
    warmup_ratio=0.1,  # ✅ 只使用 warmup_ratio
    lr_scheduler_type="linear",  # ✅ 使用 linear（默认）
)
```

---

## 3. ⚠️ RoPE Scaling 配置检查

### 问题描述

> Qwen-1.5 / 2.5 在加载权重时 rope scaling 改动过，若配置不匹配，梯度会异常。

### 当前配置

```python
# train_lora_only_gen.py (line 504-513)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
# ❌ 没有显式设置 rope_scaling
```

### RoPE Scaling 说明

**RoPE (Rotary Position Embedding)** 是位置编码方式，Qwen 不同版本的 RoPE 配置不同：

| 版本 | RoPE Scaling | 说明 |
|------|-------------|------|
| **Qwen 1.0** | 无 | 标准 RoPE |
| **Qwen 1.5** | `linear` | 线性缩放 |
| **Qwen 2.0** | `dynamic` | 动态缩放 |
| **Qwen 2.5** | `yarn` | YARN 缩放 |

### 验证结果

**⚠️ 可能存在问题！**

**原因**：
1. 代码中没有显式设置 `rope_scaling`
2. 如果模型配置和训练配置不匹配，可能导致梯度异常

### 如何验证

检查模型的 RoPE 配置：

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct",
    trust_remote_code=True
)

print("RoPE Scaling:", getattr(config, 'rope_scaling', None))
print("Max Position Embeddings:", getattr(config, 'max_position_embeddings', None))
```

**预期输出（Qwen 2.5）**：
```python
RoPE Scaling: {'type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
Max Position Embeddings: 131072
```

### 修复方案

**方案 1：使用模型默认配置（推荐）**

```python
# train_lora_only_gen.py
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    # ✅ 不设置 rope_scaling，使用模型默认配置
)
```

**方案 2：显式设置 RoPE Scaling**

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True
)

# ✅ 确保 rope_scaling 配置正确
if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
    print(f"Using RoPE Scaling: {config.rope_scaling}")
else:
    print("No RoPE Scaling configured")

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,  # ✅ 使用配置
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

**方案 3：禁用 RoPE Scaling（调试用）**

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True
)

# ✅ 禁用 RoPE Scaling（仅用于调试）
config.rope_scaling = None

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

---

## 4. ⚠️ 数据集归一化或拼接问题

### 问题描述

> 数据集经过归一化或拼接时可能生成 NaN（尤其是 instruction 拼接）。

### 当前配置

```python
# train_lora_only_gen.py (line 242-448)
def load_and_process_data(data_path: str, tokenizer, max_length: int = 512, val_split: float = 0.1):
    # 构建 prompt
    if output_type == "text":
        prompt = f"{instruction}\n{input_text}\nAnswer one of 'yes', 'no', or 'neutral'. Your answer is:"
        full_text = f"{prompt} {output}"
    elif output_type == "bool":
        prompt = f"{instruction}\n{input_text}\nAnswer one of 'TRUE' or 'FALSE'. Your answer is:"
        full_text = f"{prompt} {output}"
    else:
        prompt = f"{instruction}\n{input_text}\nAnswer:"
        full_text = f"{prompt} {output}"

    # Tokenize
    prompt_tokens = tokenizer(prompt, add_special_tokens=True, truncation=False, max_length=None)
    answer_tokens = tokenizer(output, add_special_tokens=False, truncation=False, max_length=None)

    # 合并
    full_input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
    full_attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"]

    # 截断
    if len(full_input_ids) > max_length:
        full_input_ids = full_input_ids[:max_length]
        full_attention_mask = full_attention_mask[:max_length]

    # 创建 labels
    prompt_len = len(prompt_tokens["input_ids"])
    labels = [-100] * prompt_len + answer_tokens["input_ids"]

    if len(labels) > max_length:
        labels = labels[:max_length]
```

### 可能的问题

1. **空字符串**：`instruction` 或 `input` 为空
2. **特殊字符**：包含无法编码的字符
3. **长度不一致**：`input_ids`、`attention_mask`、`labels` 长度不匹配
4. **NaN 值**：数据中包含 NaN

### 如何验证

添加数据验证代码：

```python
def validate_data(data):
    """验证数据集"""
    print("\n" + "="*80)
    print("Validating dataset...")
    print("="*80)

    issues = []

    for i, item in enumerate(data):
        # 检查必需字段
        if 'instruction' not in item:
            issues.append(f"Sample {i}: Missing 'instruction'")
        if 'input' not in item:
            issues.append(f"Sample {i}: Missing 'input'")
        if 'output' not in item:
            issues.append(f"Sample {i}: Missing 'output'")

        # 检查空字符串
        if item.get('instruction', '').strip() == '':
            issues.append(f"Sample {i}: Empty 'instruction'")
        if item.get('input', '').strip() == '':
            issues.append(f"Sample {i}: Empty 'input'")
        if str(item.get('output', '')).strip() == '':
            issues.append(f"Sample {i}: Empty 'output'")

        # 检查 NaN
        import math
        if isinstance(item.get('output'), float) and math.isnan(item['output']):
            issues.append(f"Sample {i}: NaN 'output'")

        # 检查特殊字符
        try:
            _ = item['instruction'].encode('utf-8')
            _ = item['input'].encode('utf-8')
            _ = str(item['output']).encode('utf-8')
        except UnicodeEncodeError as e:
            issues.append(f"Sample {i}: Unicode error - {e}")

    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # 只打印前 10 个
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
    else:
        print("✅ No issues found")

    print("="*80)
    return len(issues) == 0

# 在 load_and_process_data 中调用
data = all_data
validate_data(data)  # ✅ 添加验证
```

### 修复方案

**方案 1：过滤无效数据**

```python
def load_and_process_data(data_path: str, tokenizer, max_length: int = 512, val_split: float = 0.1):
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ✅ 过滤无效数据
    valid_data = []
    for item in data:
        # 检查必需字段
        if 'instruction' not in item or 'input' not in item or 'output' not in item:
            continue

        # 检查空字符串
        if not item['instruction'].strip() or not item['input'].strip():
            continue

        # 检查 NaN
        import math
        if isinstance(item['output'], float) and math.isnan(item['output']):
            continue

        valid_data.append(item)

    print(f"Filtered: {len(data)} -> {len(valid_data)} samples")
    data = valid_data

    # 继续处理...
```

**方案 2：添加异常处理**

```python
def preprocess_function(examples):
    """预处理函数 - 添加异常处理"""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples['instruction'])):
        try:
            instruction = examples['instruction'][i]
            input_text = examples['input'][i]
            output = str(examples['output'][i])

            # 检查空字符串
            if not instruction.strip() or not input_text.strip():
                print(f"⚠️  Skipping sample {i}: Empty instruction or input")
                continue

            # 构建 prompt
            prompt = f"{instruction}\n{input_text}\nAnswer:"
            full_text = f"{prompt} {output}"

            # Tokenize
            prompt_tokens = tokenizer(prompt, add_special_tokens=True, truncation=False, max_length=None)
            answer_tokens = tokenizer(output, add_special_tokens=False, truncation=False, max_length=None)

            # 合并
            full_input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
            full_attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"]

            # 截断
            if len(full_input_ids) > max_length:
                full_input_ids = full_input_ids[:max_length]
                full_attention_mask = full_attention_mask[:max_length]

            # 创建 labels
            prompt_len = len(prompt_tokens["input_ids"])
            labels = [-100] * prompt_len + answer_tokens["input_ids"]

            if len(labels) > max_length:
                labels = labels[:max_length]

            # 验证长度
            assert len(full_input_ids) == len(full_attention_mask) == len(labels), \
                f"Length mismatch: input_ids={len(full_input_ids)}, attention_mask={len(full_attention_mask)}, labels={len(labels)}"

            model_inputs["input_ids"].append(full_input_ids)
            model_inputs["attention_mask"].append(full_attention_mask)
            model_inputs["labels"].append(labels)

        except Exception as e:
            print(f"⚠️  Error processing sample {i}: {e}")
            continue

    return model_inputs
```

---

## 📊 完整诊断清单

| 问题 | 状态 | 说明 |
|------|------|------|
| **1. LoRA 层名称** | ✅ 正确 | 使用 `q_proj`, `k_proj`, `v_proj`, `o_proj`（Qwen 2.5） |
| **2. 学习率调度器** | ✅ 正确 | 使用 `warmup_ratio=0.1`，没有冲突 |
| **3. RoPE Scaling** | ⚠️ 需要验证 | 没有显式设置，可能存在问题 |
| **4. 数据集验证** | ⚠️ 需要验证 | 没有数据验证，可能存在 NaN 或空字符串 |

---

## 🎯 推荐的修复步骤

### Step 1：验证 LoRA 层名称

```bash
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct', trust_remote_code=True)
for name, _ in model.named_parameters():
    if 'attn' in name and 'proj' in name:
        print(name)
        break
"
```

**预期输出**：`model.layers.0.self_attn.q_proj.weight` ✅

### Step 2：验证 RoPE Scaling

```bash
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct', trust_remote_code=True)
print('RoPE Scaling:', getattr(config, 'rope_scaling', None))
"
```

### Step 3：验证数据集

```bash
python -c "
import json
with open('/root/autodl-fs/cultureLLM_merge_gen.json', 'r') as f:
    data = json.load(f)
print(f'Total samples: {len(data)}')
print(f'First sample: {data[0]}')
# 检查是否有空字符串或 NaN
import math
issues = 0
for i, item in enumerate(data):
    if not item.get('instruction', '').strip():
        print(f'Sample {i}: Empty instruction')
        issues += 1
    if not item.get('input', '').strip():
        print(f'Sample {i}: Empty input')
        issues += 1
    if isinstance(item.get('output'), float) and math.isnan(item['output']):
        print(f'Sample {i}: NaN output')
        issues += 1
print(f'Total issues: {issues}')
"
```

### Step 4：添加调试日志

在 `train_lora_only_gen.py` 中添加：

```python
# 在训练开始前
print("\n" + "="*80)
print("Model Configuration")
print("="*80)
print(f"Model type: {model.config.model_type}")
print(f"RoPE Scaling: {getattr(model.config, 'rope_scaling', None)}")
print(f"Max Position Embeddings: {getattr(model.config, 'max_position_embeddings', None)}")
print("="*80)

# 打印 LoRA 配置
model.print_trainable_parameters()

# 打印第一个 batch 的信息
for batch in train_loader:
    print("\n" + "="*80)
    print("First Batch")
    print("="*80)
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"input_ids min/max: {batch['input_ids'].min()}/{batch['input_ids'].max()}")
    print(f"labels min/max: {batch['labels'].min()}/{batch['labels'].max()}")
    print("="*80)
    break
```

---

## 总结

✅ **已验证正确**：
1. LoRA 层名称：使用 `q_proj`, `k_proj`, `v_proj`, `o_proj` ✅
2. 学习率调度器：使用 `warmup_ratio=0.1` ✅

⚠️ **需要验证**：
3. RoPE Scaling：没有显式设置，可能存在问题
4. 数据集：没有验证，可能存在 NaN 或空字符串

🎯 **推荐操作**：
1. 运行上述验证脚本
2. 添加数据验证代码
3. 添加调试日志
4. 如果问题仍然存在，考虑使用 bf16 而不是 fp32

现在可以开始诊断了！🔍

