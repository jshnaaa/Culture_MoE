# Qwen LoRA 配置说明

## 📋 问题回答

### 1. **LoRA 注入模块**

对于 Qwen 模型，LoRA 注入的模块是：

```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
```

**说明**：
- `q_proj`: Query 投影层
- `k_proj`: Key 投影层
- `v_proj`: Value 投影层
- `o_proj`: Output 投影层

这些是 **Attention 层的 4 个投影矩阵**，与 LLaMA 使用相同的模块。

### 2. **精度设置**

#### 训练时（train_lora_only_gen.py）

```python
if model_type == 'qwen':
    use_fp32 = True  # ✅ 使用 fp32

training_args = TrainingArguments(
    ...
    fp16=not use_fp32,  # fp16=False
    ...
)
```

**训练精度**：**fp32**（不是 fp16，也不是 bf16）

#### 评估时（eval_lora_only_from_components.py）

```python
# 加载 base 模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  # ✅ 使用 fp16
    device_map="auto",
    trust_remote_code=True
)

# 加载 LoRA 权重
peft_model = PeftModel.from_pretrained(
    base_model,
    lora_weights_path,
    is_trainable=False,
    torch_dtype=torch.float16  # ✅ 使用 fp16
)
```

**评估精度**：**fp16**（不是 bf16，也不是 fp32）

## 📊 完整配置对比

### Qwen vs LLaMA

| 配置项 | Qwen | LLaMA |
|--------|------|-------|
| **LoRA 模块** | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **训练精度** | fp32 | fp16 |
| **评估精度** | fp16 | fp16 |
| **LoRA Dropout** | 0.1 | 0.05 |
| **LoRA 初始化** | gaussian | default |
| **学习率** | 5e-5 (50%) | 1e-4 (100%) |
| **Max Grad Norm** | 0.5 | 1.0 |
| **Warmup Ratio** | 0.1 | 0.1 |

## 🔍 详细配置

### 训练时的 LoRA 配置

```python
# train_lora_only_gen.py

if model_type == 'qwen':
    print("  Using Qwen-specific LoRA configuration")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,                    # 默认 8
        lora_alpha=args.lora_alpha,          # 默认 16
        lora_dropout=0.1,                    # ✅ Qwen 需要更高的 dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # ✅ 注入模块
        bias="none",
        init_lora_weights="gaussian"         # ✅ 使用高斯初始化
    )
else:
    print("  Using LLaMA-specific LoRA configuration")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,                    # 默认 8
        lora_alpha=args.lora_alpha,          # 默认 16
        lora_dropout=args.lora_dropout,      # 默认 0.05
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # ✅ 注入模块
        bias="none"
    )
```

### 训练时的精度配置

```python
# train_lora_only_gen.py

if model_type == 'qwen':
    use_fp32 = True  # ✅ Qwen 使用 fp32
    print("  Using Qwen-specific training configuration")
    print(f"    Using fp32 (not fp16)")
else:
    use_fp32 = False  # LLaMA 使用 fp16

training_args = TrainingArguments(
    ...
    fp16=not use_fp32,  # Qwen: fp16=False, LLaMA: fp16=True
    fp16_full_eval=False,
    fp16_opt_level="O1",
    ...
)
```

### 评估时的精度配置

```python
# eval_lora_only_from_components.py

# 加载 base 模型（统一使用 fp16）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  # ✅ 统一使用 fp16
    device_map="auto",
    trust_remote_code=True
)

# 加载 LoRA 权重（统一使用 fp16）
peft_model = PeftModel.from_pretrained(
    base_model,
    lora_weights_path,
    is_trainable=False,
    torch_dtype=torch.float16  # ✅ 统一使用 fp16
)
```

## ⚠️ 重要说明

### 1. **为什么训练用 fp32，评估用 fp16？**

**训练时（Qwen）**：
- 使用 fp32 是因为 Qwen 在 fp16 下训练不稳定
- 梯度容易出现 NaN
- Loss 不收敛

**评估时**：
- 使用 fp16 是为了节省内存
- 评估时不需要计算梯度，fp16 足够稳定
- 可以加载更大的模型

### 2. **为什么不使用 bf16？**

**bf16 的优势**：
- 动态范围更大，更稳定
- 适合训练大模型

**为什么不用**：
- 代码中没有显式设置 bf16
- 默认使用 fp16（评估）和 fp32（训练）
- 如果 GPU 支持 bf16，可以考虑修改

### 3. **LoRA 模块为什么选择这 4 个？**

```
Attention 层结构：
    Input
      ↓
    Q_proj → Query
    K_proj → Key
    V_proj → Value
      ↓
    Attention
      ↓
    O_proj → Output
```

**选择原因**：
- 这 4 个投影层是 Attention 的核心
- 参数量大，适合 LoRA 微调
- 对模型性能影响最大

**不选择其他层**：
- MLP 层（`gate_proj`, `up_proj`, `down_proj`）：参数量更大，但对性能提升有限
- Embedding 层：通常不需要微调

## 🔧 如何修改配置

### 修改为 bf16（如果 GPU 支持）

#### 训练时

```python
# train_lora_only_gen.py

if model_type == 'qwen':
    use_bf16 = True  # ✅ 改用 bf16

training_args = TrainingArguments(
    ...
    fp16=False,
    bf16=use_bf16,  # ✅ 启用 bf16
    ...
)
```

#### 评估时

```python
# eval_lora_only_from_components.py

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,  # ✅ 改用 bf16
    device_map="auto",
    trust_remote_code=True
)

peft_model = PeftModel.from_pretrained(
    base_model,
    lora_weights_path,
    is_trainable=False,
    torch_dtype=torch.bfloat16  # ✅ 改用 bf16
)
```

### 添加更多 LoRA 模块

```python
lora_config = LoraConfig(
    ...
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # ✅ 添加 MLP 层
    ],
    ...
)
```

**注意**：添加更多模块会增加训练参数和时间。

## 📊 参数量对比

### 只微调 Attention（当前配置）

```
LoRA 参数：约 4M
可训练参数：约 0.05% of total
```

### 微调 Attention + MLP

```
LoRA 参数：约 10M
可训练参数：约 0.13% of total
```

## 总结

✅ **Qwen LoRA 配置**：
- **注入模块**：`q_proj`, `k_proj`, `v_proj`, `o_proj`（Attention 层）
- **训练精度**：fp32（不是 fp16 或 bf16）
- **评估精度**：fp16（不是 bf16 或 fp32）
- **LoRA Dropout**：0.1（比 LLaMA 的 0.05 更高）
- **初始化方式**：gaussian（高斯初始化）

✅ **与 LLaMA 的区别**：
- LoRA 模块：相同
- 训练精度：Qwen 用 fp32，LLaMA 用 fp16
- 评估精度：都用 fp16
- Dropout：Qwen 更高（0.1 vs 0.05）

✅ **为什么这样配置**：
- Qwen 在 fp16 下训练不稳定，改用 fp32
- 评估时用 fp16 节省内存
- 更高的 dropout 防止过拟合
- 高斯初始化提高稳定性

