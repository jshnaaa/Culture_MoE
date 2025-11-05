# VSM13 MoE 模型加载最终修复

## ❌ 问题

运行 `eval_vsm13.py` 时报错：

```
FileNotFoundError: MoE weights not found: /root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2/moe_weights.pt
```

但实际目录中的文件是：
```
model_moe_llama_2/
├── moe_config.json
├── moe_layer_names.json
├── moe_weights.bin          # ✅ 是 .bin 文件，不是 .pt
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

## 🔍 问题分析

### 1. 错误的文件名

```python
# 错误
moe_weights_file = os.path.join(moe_weights_path, "moe_weights.pt")  # ❌

# 正确
moe_weights_file = os.path.join(moe_weights_path, "moe_weights.bin")  # ✅
```

### 2. 缺少正确的加载逻辑

之前的代码没有参考 `load_culturemoe.py` 中的正确加载方式。

## ✅ 修复方案

参考 `load_culturemoe.py` 的加载逻辑，完整的 MoE 模型加载流程：

### 1. 加载 MoE 配置

```python
moe_config_path = os.path.join(moe_weights_path, "moe_config.json")
with open(moe_config_path, 'r') as f:
    moe_config = json.load(f)
```

### 2. 从配置中获取 merged LLM 路径

```python
merged_llm_path = moe_config.get('merged_llm_path', model_path)
```

**关键点**：`moe_config.json` 中保存了 `merged_llm_path`，指向合并后的 LoRA 模型。

### 3. 加载 LLM

```python
llama_model = AutoModelForCausalLM.from_pretrained(
    merged_llm_path,
    torch_dtype=torch_dtype,
    device_map=device_map,
    trust_remote_code=True
)

# 冻结 LLM
for param in llama_model.parameters():
    param.requires_grad = False
```

### 4. 创建 MoE 模型结构

```python
moe_args = ModelArgs(
    num_experts=moe_config['num_experts'],
    shared_hidden_dim=moe_config['shared_hidden_dim'],
    router_hidden_dim=moe_config['router_hidden_dim'],
    experts_hidden_dim=moe_config['experts_hidden_dim'],
    lora_rank=moe_config['moe_lora_rank'],
    num_classes=moe_config['num_classes'],
    classification_hidden_dim=moe_config['classification_hidden_dim'],
    dropout=moe_config['dropout'],
    num_heads=moe_config['num_heads']
)

model = LlamaSharedRouterExpertsModel(
    llama_model=llama_model,
    config=llama_model.config,
    args=moe_args
)
```

### 5. 加载 MoE 权重（.bin 文件）

```python
moe_weights_file = os.path.join(moe_weights_path, "moe_weights.bin")  # ✅ .bin
moe_state_dict = torch.load(moe_weights_file, map_location='cpu')

# 加载权重（strict=False 因为我们只加载 MoE 部分）
missing_keys, unexpected_keys = model.load_state_dict(moe_state_dict, strict=False)

# 验证加载
llama_missing = [k for k in missing_keys if k.startswith('llama_model.')]
non_llama_missing = [k for k in missing_keys if not k.startswith('llama_model.')]
```

### 6. 移动 MoE 层到正确设备

```python
llm_device = next(llama_model.parameters()).device

# 只移动 MoE 层（不移动 llama_model）
if hasattr(model, 'shared'):
    model.shared = model.shared.to(llm_device)
if hasattr(model, 'router'):
    model.router = model.router.to(llm_device)
if hasattr(model, 'experts_layer'):
    model.experts_layer = model.experts_layer.to(llm_device)
if hasattr(model, 'classifier'):
    model.classifier = model.classifier.to(llm_device)
if hasattr(model, 'culture_classifier'):
    model.culture_classifier = model.culture_classifier.to(llm_device)
```

## 📁 MoE 权重目录结构

```
model_moe_llama_2/
├── moe_config.json          # MoE 配置（包含 merged_llm_path）
├── moe_layer_names.json     # MoE 层名称
├── moe_weights.bin          # MoE 权重（.bin 格式）
├── special_tokens_map.json  # Tokenizer 配置
├── tokenizer.json           # Tokenizer
└── tokenizer_config.json    # Tokenizer 配置
```

### moe_config.json 示例

```json
{
  "model_type": "CultureMoE",
  "architecture": "LlamaSharedRouterExpertsModel",
  "merged_llm_path": "/root/autodl-tmp/CultureMoE/Culture_Alignment/llama_merge_2",
  "num_experts": 6,
  "shared_hidden_dim": 2048,
  "router_hidden_dim": 1024,
  "experts_hidden_dim": 2048,
  "moe_lora_rank": 16,
  "num_classes": 2,
  "classification_hidden_dim": 512,
  "dropout": 0.1,
  "num_heads": 8,
  "best_accuracy": 0.8523,
  "training_time": "2024-11-05 14:30:00"
}
```

**关键字段**：
- `merged_llm_path`: 指向合并后的 LoRA 模型
- `num_experts`: 专家数量
- `num_classes`: 分类数量
- 其他 MoE 架构参数

## 🔄 完整的加载流程

```
1. 读取 moe_config.json
   ↓
2. 从配置中获取 merged_llm_path
   ↓
3. 加载 merged LLM（合并后的 LoRA 模型）
   ↓
4. 创建 CultureMoE 模型架构
   ↓
5. 加载 moe_weights.bin（MoE 权重）
   ↓
6. 移动 MoE 层到正确设备
   ↓
7. 设置为评估模式
```

## 🎯 关键点

### 1. 文件名

- ❌ `moe_weights.pt`
- ✅ `moe_weights.bin`

### 2. LLM 路径

- ❌ 硬编码路径
- ✅ 从 `moe_config.json` 中读取 `merged_llm_path`

### 3. 权重加载

- ✅ 使用 `strict=False`（因为只加载 MoE 部分）
- ✅ 验证 missing_keys（LLM 的 keys 应该 missing，MoE 的不应该）

### 4. 设备管理

- ✅ 只移动 MoE 层到 LLM 设备
- ✅ 不移动 llama_model（已经在正确设备上）

## 🚀 使用方法

### 运行 VSM13 测试

```bash
# 测试 MoE 模型
python eval_vsm13.py --model moe --backbone llama --num_classes 2
```

### 预期输出

```
Loading CultureMoE model...
  MoE weights path: /root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2
  ✅ MoE config loaded
     Num experts: 6
     Num classes: 2
  LLM (merged): /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_merge_2
  ✅ Merged LLM loaded and frozen
  ✅ CultureMoE architecture created
  Loading MoE weights from: .../model_moe_llama_2/moe_weights.bin
  ✅ MoE weights loaded
     Missing LLM keys: 291 (expected)
     Missing MoE keys: 0 (should be 0)
✅ CultureMoE model loaded on cuda

✅ Loaded 24 VSM13 questions

Testing 10 countries: China, South Korea, Turkey, ...
```

## ⚠️ 注意事项

### 1. 确保文件存在

```bash
# 检查 MoE 权重目录
ls -la /root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2/

# 应该看到：
# moe_config.json
# moe_weights.bin  ← 注意是 .bin
# tokenizer files...
```

### 2. 检查 moe_config.json

```bash
# 查看配置
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2/moe_config.json | jq

# 确保包含 merged_llm_path
```

### 3. 检查 merged LLM 路径

```bash
# 从配置中获取路径
MERGED_PATH=$(cat model_moe_llama_2/moe_config.json | jq -r '.merged_llm_path')

# 检查是否存在
ls -la $MERGED_PATH
```

## 📝 与 load_culturemoe.py 的一致性

现在 `eval_vsm13.py` 使用与 `load_culturemoe.py` **完全相同**的加载逻辑：

| 步骤 | load_culturemoe.py | eval_vsm13.py |
|------|-------------------|---------------|
| 1. 读取配置 | ✅ | ✅ |
| 2. 获取 LLM 路径 | ✅ | ✅ |
| 3. 加载 LLM | ✅ | ✅ |
| 4. 创建 MoE 架构 | ✅ | ✅ |
| 5. 加载 .bin 权重 | ✅ | ✅ |
| 6. 移动到设备 | ✅ | ✅ |

## 🎉 总结

### 问题
- 文件名错误：`.pt` → `.bin`
- 缺少从配置读取 LLM 路径的逻辑
- 没有参考 `load_culturemoe.py` 的正确实现

### 修复
- ✅ 使用正确的文件名 `moe_weights.bin`
- ✅ 从 `moe_config.json` 读取 `merged_llm_path`
- ✅ 完全参考 `load_culturemoe.py` 的加载逻辑
- ✅ 正确验证权重加载
- ✅ 正确管理设备

### 验证
```bash
python eval_vsm13.py --model moe --backbone llama --num_classes 2
```

现在应该能正确加载 MoE 模型并进行 VSM13 测试了！🎉

