# VSM13 MoE 权重路径修复

## ❌ 问题

运行 `eval_vsm13.py` 评估 MoE 模型时报错：

```
⚠️  Warning: moe_config.json not found, using default config
❌ Error: MoE weights not found: /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_llama_2/moe_weights.pt
FileNotFoundError: MoE weights not found: /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_llama_2/moe_weights.pt
```

## 🔍 问题分析

### 错误的路径

```python
# eval_vsm13.py (第 293 行)
moe_weights_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_{args.backbone}_{args.num_classes}"
```

### 实际的路径

根据训练脚本，MoE 权重实际保存在：

```bash
MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_${BACKBONE}_${NUM_CLASSES}"
```

**对比**：
- ❌ 错误：`culturemoe_llama_2`
- ✅ 正确：`model_moe_llama_2`

## ✅ 修复

### 修改 eval_vsm13.py

```python
# 第 293 行
# 修复前
moe_weights_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_{args.backbone}_{args.num_classes}"

# 修复后
moe_weights_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_{args.backbone}_{args.num_classes}"
```

## 📁 MoE 权重文件结构

### 正确的目录结构

```
/root/autodl-tmp/CultureMoE/Culture_Alignment/
├── llama_merge_2/              # Merged LoRA 模型（LLM）
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   ├── config.json
│   └── tokenizer files...
│
└── model_moe_llama_2/          # MoE 权重
    ├── moe_config.json         # MoE 配置
    ├── moe_weights.pt          # MoE 权重
    └── training_args.json      # 训练参数（可选）
```

### moe_config.json 内容

```json
{
  "num_experts": 6,
  "shared_hidden_dim": 2048,
  "router_hidden_dim": 1024,
  "experts_hidden_dim": 2048,
  "lora_rank": 16,
  "classification_hidden_dim": 512,
  "dropout": 0.1,
  "num_heads": 8
}
```

### moe_weights.pt 内容

包含以下权重：
- `shared_layer.*` - Shared Layer 权重
- `router.*` - Router 权重
- `experts.*` - 所有 Experts 权重
- `classification_head.*` - 分类头权重

## 🔍 验证修复

### 1. 检查文件是否存在

```bash
# 检查 MoE 权重目录
ls -la /root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2/

# 应该看到：
# moe_config.json
# moe_weights.pt
# training_args.json (可选)
```

### 2. 检查文件大小

```bash
# 检查 moe_weights.pt 大小
du -h /root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2/moe_weights.pt

# 应该是几百 MB（取决于 MoE 配置）
```

### 3. 运行 VSM13 测试

```bash
# 测试 MoE 模型
python eval_vsm13.py \
    --model moe \
    --backbone llama \
    --num_classes 2 \
    --device cuda
```

### 4. 预期输出

**修复前**：
```
Loading CultureMoE model...
  LLM (merged): /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_merge_2
  MoE weights: /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_llama_2
⚠️  Warning: moe_config.json not found, using default config
❌ Error: MoE weights not found: .../culturemoe_llama_2/moe_weights.pt
FileNotFoundError: MoE weights not found
```

**修复后**：
```
Loading CultureMoE model...
  LLM (merged): /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_merge_2
  MoE weights: /root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2
  Loading MoE weights from: .../model_moe_llama_2/moe_weights.pt
  ✅ MoE weights loaded
✅ CultureMoE model loaded on cuda

✅ Loaded 24 VSM13 questions

Testing 10 countries: China, South Korea, Turkey, ...
```

## 📝 路径命名规范

### 统一的命名规范

为了避免混淆，建议统一使用以下命名：

| 模型类型 | 目录名 | 说明 |
|---------|--------|------|
| Base 模型 | `Meta-Llama-3.1-8B-Instruct` | 原始基座模型 |
| Merged LoRA | `{backbone}_merge_{num_classes}` | LoRA 合并后的模型 |
| MoE 权重 | `model_moe_{backbone}_{num_classes}` | MoE 部分的权重 |

**示例**：
- Base: `Meta-Llama-3.1-8B-Instruct`
- Merged LoRA: `llama_merge_2`
- MoE: `model_moe_llama_2`

### 其他脚本中的路径

确保所有脚本使用相同的路径命名：

1. **训练脚本** (`run_train_culturemoe_from_merged.sh`)
   ```bash
   MOE_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_${BACKBONE}_${NUM_CLASSES}"
   ```

2. **评估脚本** (`eval_vsm13.py`)
   ```python
   moe_weights_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_{args.backbone}_{args.num_classes}"
   ```

3. **其他评估脚本** (如果有)
   - 确保使用相同的路径格式

## ⚠️ 注意事项

### 1. 如果 MoE 权重不存在

可能原因：
- MoE 训练还没有完成
- 训练时没有保存权重
- 保存路径配置错误

解决方法：
```bash
# 检查训练脚本的保存路径
grep "MOE_WEIGHTS_PATH" run_train_culturemoe_from_merged.sh

# 重新训练 MoE 模型
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 false true
```

### 2. 如果 moe_config.json 不存在

脚本会使用默认配置，但可能与实际训练的配置不一致。

建议：
- 确保训练时保存了 `moe_config.json`
- 或者手动创建配置文件

### 3. 权重加载失败

如果看到 `strict=False` 的警告：
```
Some weights of the model checkpoint were not used when initializing...
```

这是正常的，因为：
- LLM 部分已经加载
- 只需要加载 MoE 部分
- `strict=False` 允许部分权重不匹配

## 🎉 总结

### 问题
- MoE 权重路径错误：`culturemoe_llama_2`
- 实际路径：`model_moe_llama_2`

### 修复
```python
# 修改 eval_vsm13.py 第 293 行
moe_weights_path = f"model_moe_{args.backbone}_{args.num_classes}"
```

### 验证
```bash
# 检查文件
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2/

# 运行测试
python eval_vsm13.py --model moe --backbone llama --num_classes 2
```

现在应该能正确加载 MoE 权重了！🎉

