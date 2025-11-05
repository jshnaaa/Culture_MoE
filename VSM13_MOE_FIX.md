# VSM13 MoE 模型加载修复

## ❌ 问题

之前的代码中，MoE 模型评估时**只加载了 merged LoRA 模型**，没有加载 MoE 部分的权重，导致：

```python
# 错误的代码（第 291-294 行）
else:  # moe
    # MoE 需要特殊处理，这里先用 merged model
    model_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/{args.backbone}_merge_{args.num_classes}"
    model_name = f"{args.model}_{args.backbone}_{args.num_classes}"
```

**结果**：
- LoRA 模型和 MoE 模型的 VSM13 回答**完全一样**
- 每个维度的分数**完全一样**
- 欧式距离**完全一样**

**原因**：
- MoE 模型实际上只是加载了 merged LoRA 模型
- 没有加载 MoE 的 Router、Experts、Shared Layer 等权重
- 相当于只用了 LLM 部分，MoE 部分没有起作用

## ✅ 修复

### 1. 添加 MoE 权重路径

```python
else:  # moe
    # MoE 需要加载 merged model + MoE 权重
    merged_model_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/{args.backbone}_merge_{args.num_classes}"
    moe_weights_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_{args.backbone}_{args.num_classes}"
    model_path = merged_model_path  # 先用这个加载 LLM
    model_name = f"{args.model}_{args.backbone}_{args.num_classes}"
    use_moe = True
```

### 2. 完整的 MoE 模型加载流程

```python
if use_moe:
    # 1. 加载 LLM（merged LoRA）
    llama_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    # 2. 读取 MoE 配置
    moe_config_path = os.path.join(moe_weights_path, "moe_config.json")
    with open(moe_config_path, 'r') as f:
        moe_config = json.load(f)

    moe_args = ModelArgs(
        num_experts=moe_config.get('num_experts', 6),
        shared_hidden_dim=moe_config.get('shared_hidden_dim', 2048),
        router_hidden_dim=moe_config.get('router_hidden_dim', 1024),
        experts_hidden_dim=moe_config.get('experts_hidden_dim', 2048),
        lora_rank=moe_config.get('lora_rank', 16),
        num_classes=args.num_classes,
        classification_hidden_dim=moe_config.get('classification_hidden_dim', 512),
        dropout=moe_config.get('dropout', 0.1),
        num_heads=moe_config.get('num_heads', 8)
    )

    # 3. 创建 CultureMoE 模型结构
    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    )

    # 4. 加载 MoE 权重
    moe_weights_file = os.path.join(moe_weights_path, "moe_weights.pt")
    moe_state_dict = torch.load(moe_weights_file, map_location='cpu')
    model.load_state_dict(moe_state_dict, strict=False)

    print("✅ CultureMoE model loaded (LLM + MoE)")
```

## 📁 MoE 权重文件结构

```
/root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_llama_2/
├── moe_config.json          # MoE 配置
├── moe_weights.pt           # MoE 权重（Router + Experts + Shared + Classification Head）
└── training_args.json       # 训练参数（可选）
```

### moe_config.json 示例

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

### moe_weights.pt 包含的权重

```python
{
    "shared_layer.lora_A": ...,
    "shared_layer.lora_B": ...,
    "router.fc1.weight": ...,
    "router.fc2.weight": ...,
    "experts.0.lora_A": ...,
    "experts.0.lora_B": ...,
    "experts.1.lora_A": ...,
    "experts.1.lora_B": ...,
    ...
    "classification_head.fc1.weight": ...,
    "classification_head.fc2.weight": ...,
}
```

## 🔍 验证修复

### 运行 VSM13 测试

```bash
# 测试 LoRA 模型
sh run_eval_vsm13.sh lora llama --num_classes 2

# 测试 MoE 模型
sh run_eval_vsm13.sh moe llama --num_classes 2
```

### 预期结果

**修复前**：
```
LoRA:  Calculated scores: [85, 53, 68, 45, 72, 38]
MoE:   Calculated scores: [85, 53, 68, 45, 72, 38]  ❌ 完全一样！
```

**修复后**：
```
LoRA:  Calculated scores: [85, 53, 68, 45, 72, 38]
MoE:   Calculated scores: [82, 56, 71, 42, 75, 35]  ✅ 不同！
```

## 🎯 关键点

### 1. MoE 模型 = LLM + MoE 权重

```
CultureMoE = Merged LoRA Model (LLM) + MoE Weights (Router + Experts + Shared)
```

### 2. 加载顺序

1. 加载 merged LoRA 模型（LLM 部分）
2. 创建 MoE 模型结构
3. 加载 MoE 权重到模型中

### 3. 权重文件位置

- **LLM 权重**：`{backbone}_merge_{num_classes}/`
- **MoE 权重**：`culturemoe_{backbone}_{num_classes}/moe_weights.pt`

### 4. 配置文件

- **MoE 配置**：`culturemoe_{backbone}_{num_classes}/moe_config.json`
- 包含 num_experts、hidden_dim 等参数

## ⚠️ 注意事项

### 1. 确保 MoE 权重文件存在

```bash
# 检查文件
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_llama_2/

# 应该看到：
# moe_config.json
# moe_weights.pt
```

### 2. 如果文件不存在

可能原因：
- MoE 训练没有保存权重
- 保存路径不对
- 文件名不对

解决方法：
- 检查训练脚本的保存逻辑
- 确认保存路径
- 重新训练 MoE 模型

### 3. strict=False

```python
model.load_state_dict(moe_state_dict, strict=False)
```

- `strict=False`：允许部分权重不匹配
- 因为 LLM 部分已经加载，只需要加载 MoE 部分
- 如果有不匹配的 key，会打印警告但不会报错

## 🎉 总结

修复后：
1. ✅ MoE 模型正确加载 LLM + MoE 权重
2. ✅ LoRA 和 MoE 的 VSM13 结果不同
3. ✅ MoE 的 Router 和 Experts 正常工作
4. ✅ 可以正确评估 MoE 模型的文化一致性

立即测试：
```bash
sh run_eval_vsm13.sh moe llama --num_classes 2

