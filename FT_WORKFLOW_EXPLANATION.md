# 标准语言建模损失微调 - 工作流程详解

## 🎯 核心问题解答

### Q1: 是否需要先训练 LoRA Only 模型？

**答案：是的，必须先训练 LoRA Only 模型。**

### Q2: CultureMoE 是否只训练 MOE 部分？

**答案：是的，CultureMoE 只训练 MOE 部分，保持 LoRA 权重不变。**

### Q3: MOE 是否是生成式的？

**答案：是的，MOE 也是生成式的，使用标准语言建模损失。**

---

## 📊 完整的训练流程

### 第一步：训练 LoRA Only 模型

```bash
sh run_ft_lora_only_from_components.sh llama 4
```

**这一步做什么**：
1. 加载 Base 模型（LLaMA 3.1-8B-Instruct）
2. 添加 LoRA 适配器
3. 使用标准语言建模损失微调 LoRA 权重
4. 保存最好的 LoRA 权重到 `best_lora/`

**输出**：
```
/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/
├── best_lora/                    # ✅ 保存的 LoRA 权重
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── epoch_eval_results.json
├── generated_answers.json
└── config.json
```

**训练时间**：约 1-2 小时

---

### 第二步：训练 CultureMoE 模型

```bash
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5
```

**这一步做什么**：
1. 加载 Base 模型（LLaMA 3.1-8B-Instruct）
2. 加载第一步保存的 LoRA 权重
3. 合并 LoRA 权重到 Base 模型
4. 在合并后的模型基础上添加 MOE 层（Shared Experts + Router + Task-specific Experts）
5. **只训练 MOE 部分**，LoRA 权重保持不变
6. 使用标准语言建模损失 + 文化专注性损失微调 MOE 权重
7. 保存最好的 MOE 权重到 `best_moe/`

**输出**：
```
/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/
├── best_moe/                     # ✅ 保存的 MOE 权重
│   └── pytorch_model.bin
├── epoch_eval_results.json
├── generated_answers.json
└── config.json
```

**训练时间**：约 3-4 小时

---

## 🏗️ 模型架构对比

### LoRA Only 模型

```
Base Model (LLaMA 3.1-8B)
    ↓
LoRA Adapter (微调)
    ↓
Output (生成式)
```

**特点**：
- 只有 LoRA 适配器
- 参数量少（约 4.7M）
- 训练快速
- 性能基础

### CultureMoE 模型

```
Base Model (LLaMA 3.1-8B)
    ↓
LoRA Adapter (来自第一步，冻结)
    ↓
MOE 层 (新增，微调)
    ├── Shared Experts (共享专家)
    ├── Router (路由器)
    └── Task-specific Experts (任务特定专家)
    ↓
Output (生成式)
```

**特点**：
- 包含 LoRA 适配器（冻结）+ MOE 层（微调）
- 参数量多（约 Base + LoRA + MOE）
- 训练较慢
- 性能更好（文化对齐）

---

## 💾 权重管理

### LoRA Only 模型的权重

```python
# 第一步保存的权重
best_lora/
├── adapter_config.json      # LoRA 配置
├── adapter_model.bin        # LoRA 权重
└── ...

# 这些权重在第二步被加载并冻结
```

### CultureMoE 模型的权重

```python
# 第二步保存的权重
best_moe/
└── pytorch_model.bin        # 完整的 MOE 模型权重
                             # 包括：Base + LoRA (冻结) + MOE (微调)
```

---

## 🔄 详细的训练流程

### 第一步：LoRA Only 训练

```python
# ft_lora_only_from_components.py

# 1. 加载 Base 模型
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,  # LLaMA 3.1-8B-Instruct
    torch_dtype=torch.float16,
    device_map='auto'
)

# 2. 添加 LoRA 适配器
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)

# 3. 训练 LoRA 权重
for epoch in range(6):
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels  # 标准语言建模损失
    )
    loss = outputs.loss

    # 反向传播和优化
    loss.backward()
    optimizer.step()

# 4. 保存最好的 LoRA 权重
model.save_pretrained(best_lora_dir)
```

### 第二步：CultureMoE 训练

```python
# ft_culturemoe_from_base_gen.py

# 1. 加载 Base 模型
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,  # LLaMA 3.1-8B-Instruct
    torch_dtype=torch.float16,
    device_map='auto'
)

# 2. 加载第一步保存的 LoRA 权重
model = PeftModel.from_pretrained(
    base_model,
    args.lora_weights_path,  # 第一步保存的 best_lora/
    is_trainable=False  # ✅ 冻结 LoRA 权重
)

# 3. 合并 LoRA 权重到 Base 模型
model = model.merge_and_unload()

# 4. 在合并后的模型基础上添加 MOE 层
# 这里需要创建 CultureMoE 模型
# model = convert_to_culturemoe(model, moe_args)

# 5. 训练 MOE 部分（LoRA 权重已冻结）
for epoch in range(30):
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_ids_mask=input_ids_mask,
        attention_mask_mask=attention_mask_mask,
        labels=labels,
        culture_labels=culture_labels,
        use_culture_loss=True,
        culture_loss_lambda=0.5
    )

    # 总损失 = 生成损失 + 文化损失
    loss = outputs['loss']

    # 反向传播和优化（只更新 MOE 部分）
    loss.backward()
    optimizer.step()

# 6. 保存最好的 MOE 权重
torch.save(model.state_dict(), best_moe_path)
```

---

## 📈 损失函数对比

### LoRA Only 模型

```
Loss = Generation Loss
     = -Σ log P(token_t | token_<t)

这是标准的语言建模损失，对所有 token 计算交叉熵。
```

### CultureMoE 模型

```
Total Loss = Generation Loss + λ * Culture Loss
           = -Σ log P(token_t | token_<t) + λ * Culture Loss

其中：
- Generation Loss：标准语言建模损失
- Culture Loss：基于文化标签的分类损失
- λ：文化损失权重（可调，默认 0.5）
```

---

## 🎯 为什么要分两步训练？

### 原因 1：模块化设计

- **LoRA Only**：验证 LoRA 微调的有效性
- **CultureMoE**：在 LoRA 基础上添加文化对齐能力

### 原因 2：参数效率

- **LoRA Only**：参数少，训练快
- **CultureMoE**：在 LoRA 基础上只添加 MOE 参数

### 原因 3：实验对比

- 可以比较 LoRA Only vs CultureMoE 的性能差异
- 验证 MOE 层的有效性

### 原因 4：消融实验

- 可以独立调整 culture_loss_lambda
- 验证文化损失的影响

---

## 📊 训练时间和资源

### LoRA Only 模型

| 资源 | 需求 |
|------|------|
| GPU 内存 | ~20GB |
| 训练时间 | 1-2 小时 |
| 参数量 | ~4.7M (LoRA) |
| 保存大小 | ~500MB |

### CultureMoE 模型

| 资源 | 需求 |
|------|------|
| GPU 内存 | ~30GB |
| 训练时间 | 3-4 小时 |
| 参数量 | ~Base + LoRA + MOE |
| 保存大小 | ~1GB |

---

## 🔍 关键代码片段

### 冻结 LoRA 权重

```python
# 加载 LoRA 权重时设置 is_trainable=False
model = PeftModel.from_pretrained(
    base_model,
    lora_weights_path,
    is_trainable=False  # ✅ 冻结权重
)

# 合并后，LoRA 权重已经融入 Base 模型
model = model.merge_and_unload()

# 现在只有 MOE 层的参数是可训练的
```

### 只训练 MOE 部分

```python
# 优化器只优化 MOE 层的参数
optimizer = torch.optim.AdamW(
    model.parameters(),  # 这里包括 MOE 层的参数
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

# 由于 LoRA 权重已经合并，优化器会自动只更新 MOE 部分
```

---

## 🚀 完整的工作流程

### 步骤 1：准备数据

```bash
# 确保数据集存在
ls -la /root/autodl-fs/cultureLLM_merge_gen.json
```

### 步骤 2：训练 LoRA Only 模型

```bash
# 运行 LoRA Only 训练
sh run_ft_lora_only_from_components.sh llama 4

# 等待完成（1-2 小时）
# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/epoch_eval_results.json | python -m json.tool
```

### 步骤 3：训练 CultureMoE 模型

```bash
# 运行 CultureMoE 训练
# 脚本会自动找到第一步保存的 LoRA 权重
sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 0.5

# 等待完成（3-4 小时）
# 查看结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_culturemoe_cultureLLM_llama_experts6_CULTURE_LOSS_WEIGHT0.5_*/epoch_eval_results.json | python -m json.tool
```

### 步骤 4：消融实验

```bash
# 测试不同的文化损失权重
for weight in 0.1 0.3 0.5 0.7 0.9; do
    sh run_ft_culturemoe_from_base_gen.sh llama 4 True 6 2 $weight
done

# 比较结果
python -c "
import json
import os

weights = [0.1, 0.3, 0.5, 0.7, 0.9]
for weight in weights:
    # 找到对应的输出目录
    # 加载结果并比较
    pass
"
```

---

## 📊 模型性能对比

### 预期结果

| 模型 | 生成损失 | 文化损失 | 准确率 | 说明 |
|------|---------|---------|--------|------|
| LoRA Only | 低 | N/A | 中等 | 基础性能 |
| CultureMoE (λ=0.1) | 低 | 低 | 中等+ | 轻微文化对齐 |
| CultureMoE (λ=0.5) | 低 | 中等 | 高 | 平衡文化对齐 |
| CultureMoE (λ=0.9) | 低 | 高 | 中等 | 强文化对齐 |

---

## ✅ 常见问题

### Q1: 如果 LoRA Only 训练失败怎么办？

**A**:
```bash
# 检查错误日志
tail -100 /root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_cultureLLM_llama_*/training.log

# 常见原因：
# 1. 磁盘空间不足
# 2. GPU 内存不足
# 3. 数据集路径错误
```

### Q2: CultureMoE 训练时 LoRA 权重会改变吗？

**A**: 不会。LoRA 权重在合并后被冻结，只有 MOE 层的参数会被更新。

### Q3: 可以跳过 LoRA Only 直接训练 CultureMoE 吗？

**A**: 不建议。LoRA Only 是基础，CultureMoE 是在其基础上的增强。跳过会导致：
- 缺少初始的微调
- 性能可能下降
- 无法进行对比实验

### Q4: 如何评估训练效果？

**A**:
```bash
# 查看每个 epoch 的结果
cat /path/to/epoch_eval_results.json | python -m json.tool

# 关键指标：
# - eval_loss：验证损失（越低越好）
# - eval_accuracy：验证准确率（越高越好）
# - eval_culture_loss：文化损失（越低越好）
```

### Q5: 如何使用训练好的模型进行推理？

**A**:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 LoRA Only 模型
tokenizer = AutoTokenizer.from_pretrained('/path/to/best_lora')
model = AutoModelForCausalLM.from_pretrained('/path/to/best_lora')

# 或加载 CultureMoE 模型
state_dict = torch.load('/path/to/best_moe/pytorch_model.bin')
model.load_state_dict(state_dict)

# 推理
inputs = tokenizer("### Question: ...", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0]))
```

---

## 🎉 总结

### 核心要点

✅ **必须先训练 LoRA Only 模型**
- 使用标准语言建模损失
- 保存 LoRA 权重

✅ **然后训练 CultureMoE 模型**
- 加载第一步的 LoRA 权重
- 合并 LoRA 权重到 Base 模型
- 添加 MOE 层
- 只训练 MOE 部分（LoRA 冻结）
- 使用标准语言建模损失 + 文化专注性损失

✅ **MOE 是生成式的**
- 使用标准语言建模损失
- 支持文化专注性损失
- 每个 epoch 生成答案并评估

✅ **支持消融实验**
- 调整 culture_loss_lambda
- 比较不同配置的性能

---

**现在你可以开始两步训练了！** 🚀

