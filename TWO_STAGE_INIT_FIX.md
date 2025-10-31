# 两阶段训练 - 初始化参数修复

## ✅ 问题修复

### 错误信息

```
TypeError: LlamaSharedRouterExpertsModel.__init__() got an unexpected keyword argument 'tokenizer'
```

### 问题原因

在阶段2创建 MoE 模型时，错误地传入了 `tokenizer` 参数：

```python
# ❌ 错误的代码
model = LlamaSharedRouterExpertsModel(
    llama_model=llama_model,
    tokenizer=tokenizer,  # ❌ 错误：不接受 tokenizer 参数
    args=moe_args
)
```

### 正确的初始化

根据 `LlamaSharedRouterExpertsModel` 的定义：

```python
class LlamaSharedRouterExpertsModel(nn.Module):
    def __init__(self, llama_model, config, args: ModelArgs):
        # ...
```

应该传入 `config` 参数：

```python
# ✅ 正确的代码
model = LlamaSharedRouterExpertsModel(
    llama_model=llama_model,
    config=llama_model.config,  # ✅ 正确：传入 config
    args=moe_args
)
```

## 📝 修改内容

### 文件：`train_two_stage_culturemoe.py`

**位置**：阶段2创建 MoE 模型部分

**修改前**：
```python
model = LlamaSharedRouterExpertsModel(
    llama_model=llama_model,
    tokenizer=tokenizer,  # ❌
    args=moe_args
)
```

**修改后**：
```python
model = LlamaSharedRouterExpertsModel(
    llama_model=llama_model,
    config=llama_model.config,  # ✅
    args=moe_args
)
```

## 🔍 为什么需要 config？

### LlamaSharedRouterExpertsModel 的初始化

```python
class LlamaSharedRouterExpertsModel(nn.Module):
    def __init__(self, llama_model, config, args: ModelArgs):
        super().__init__()

        self.llama_model = llama_model
        self.args = args
        self.config = llama_model.config  # 使用 config

        # config 用于获取模型配置信息
        # 例如：hidden_size, num_layers 等
```

### config 的作用

1. **获取模型维度信息**
   - `config.hidden_size`：隐藏层维度
   - `config.num_hidden_layers`：层数
   - `config.vocab_size`：词表大小

2. **模型配置**
   - 用于初始化 MoE 层
   - 确保维度匹配

3. **不需要 tokenizer**
   - MoE 模型不直接使用 tokenizer
   - tokenizer 只在数据处理时使用

## ✅ 验证修复

### 预期输出

```
3. Creating CultureMoE model...
   ✅ CultureMoE model created
   Total parameters: xxx,xxx,xxx
   Trainable parameters: xxx,xxx,xxx
   Trainable ratio: xx.xx%
```

### 不应该出现的错误

```
❌ TypeError: LlamaSharedRouterExpertsModel.__init__() got an unexpected keyword argument 'tokenizer'
```

## 🚀 使用方法（不变）

```bash
sh run_two_stage_culturemoe.sh llama 2 True false
```

## 📊 完整的阶段2流程

```python
# 1. 加载 tokenizer（用于数据处理）
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# 2. 加载合并后的 LLM
llama_model = AutoModelForCausalLM.from_pretrained(merged_model_path)

# 3. 创建 MoE 模型（使用 config，不使用 tokenizer）
model = LlamaSharedRouterExpertsModel(
    llama_model=llama_model,
    config=llama_model.config,  # ✅ 使用 config
    args=moe_args
)

# 4. 加载数据（使用 tokenizer）
datasets = load_and_process_dual_classification_data(
    data_path=args.train_file,
    tokenizer=tokenizer  # tokenizer 在这里使用
)

# 5. 训练
trainer = ClassificationTrainer(...)
trainer.train()
```

## 💡 总结

### 核心修复

- ❌ **错误**：传入 `tokenizer` 参数
- ✅ **正确**：传入 `config` 参数

### 原因

- `LlamaSharedRouterExpertsModel` 需要 `config` 来获取模型配置
- `tokenizer` 只在数据处理时使用，不需要传给模型

### 影响

- 只影响阶段2的 MoE 模型创建
- 阶段1不受影响

现在应该可以正常运行了！🎉

