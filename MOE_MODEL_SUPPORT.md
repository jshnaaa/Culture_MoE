# MOE 模型支持指南

## 📋 概述

VSM24 评估脚本现在完全支持 MOE 模型类型。MOE 模型通过组合 Base 模型、LoRA 权重和 MOE 权重来构建完整的文化对齐模型。

---

## 🎯 MOE 模型加载流程

### 加载步骤

```
1. 加载 Base 模型
   ↓
2. 加载 LoRA 权重
   ↓
3. 将 LoRA 权重合并到 Base 模型
   ↓
4. 加载 MOE 权重
   ↓
5. 将 MOE 权重合并到模型
   ↓
6. 完整的 MOE 模型准备就绪
```

### 代码实现

```python
# 1. 加载 Base 模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# 2. 加载 LoRA 权重
model = PeftModel.from_pretrained(
    base_model,
    lora_weights_path,
    is_trainable=False,
    torch_dtype=torch.float16
)

# 3. 合并 LoRA 权重
model = model.merge_and_unload()

# 4. 加载 MOE 权重
moe_state_dict = torch.load(
    os.path.join(moe_weights_path, 'pytorch_model.bin'),
    map_location='cpu'
)

# 5. 合并 MOE 权重
model.load_state_dict(moe_state_dict, strict=False)
```

---

## 🚀 使用方法

### 方法 1：使用 Shell 脚本（推荐）

```bash
# 评估 MOE 模型 (Qwen)
sh run_eval_vsm13.sh moe qwen

# 评估 MOE 模型 (LLaMA)
sh run_eval_vsm13.sh moe llama
```

### 方法 2：直接使用 Python 脚本

```bash
# 评估 MOE 模型 (Qwen)
python eval_vsm13.py \
    --model_type moe \
    --backbone qwen \
    --base_model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct \
    --lora_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora \
    --moe_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_qwen_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe \
    --data_path /root/autodl-fs/vsm24.json \
    --output_dir /path/to/output \
    --device cuda

# 评估 MOE 模型 (LLaMA)
python eval_vsm13.py \
    --model_type moe \
    --backbone llama \
    --base_model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct \
    --lora_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora \
    --moe_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe \
    --data_path /root/autodl-fs/vsm24.json \
    --output_dir /path/to/output \
    --device cuda
```

---

## 📊 MOE 权重路径

### Qwen 版本

```
/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_qwen_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe
```

### LLaMA 版本

```
/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe
```

---

## 🔄 模型类型对比

| 模型类型 | 加载流程 | 使用场景 |
|---------|--------|--------|
| **base** | 直接加载 Base 模型 | 基准测试 |
| **lora_only** | Base + LoRA 权重 | LoRA 微调模型评估 |
| **moe** | Base + LoRA + MOE 权重 | MOE 混合专家模型评估 |
| **culturemoe** | 直接加载完整模型 | 完整的文化对齐模型 |

---

## 📁 输出文件

运行 MOE 模型评估后，输出目录包含：

```
output_dir/
├── vsm13_results.json                    # 主要结果（汇总）
├── vsm13_detailed_scores.json            # 详细的分数和距离报告
├── vsm13_all_generated.json              # 所有国家的生成数据汇总
├── vsm13_generated_China.json            # 中国的生成数据
├── vsm13_generated_South Korea.json      # 韩国的生成数据
├── vsm13_generated_Turkey.json           # 土耳其的生成数据
├── vsm13_generated_Saudi Arabia.json     # 沙特阿拉伯的生成数据
├── vsm13_generated_Bangladesh.json       # 孟加拉国的生成数据
├── vsm13_generated_Germany.json          # 德国的生成数据
├── vsm13_generated_Portugal.json         # 葡萄牙的生成数据
├── vsm13_generated_Spain.json            # 西班牙的生成数据
├── vsm13_generated_England.json          # 英国的生成数据
└── vsm13_generated_Greece.json           # 希腊的生成数据
```

---

## 🎯 快速开始

### 1. 运行 MOE 模型评估

```bash
sh run_eval_vsm13.sh moe qwen
```

### 2. 查看结果

```bash
# 查看主要结果
cat output_dir/vsm13_results.json | python -m json.tool

# 查看平均欧式距离
python -c "import json; data = json.load(open('output_dir/vsm13_results.json')); print(f'Average Distance: {data[\"average_euclidean_distance\"]:.2f}')"

# 查看特定国家的生成数据
cat output_dir/vsm13_generated_China.json | python -m json.tool | head -50
```

---

## 💡 MOE 模型特点

### 优势

1. **混合专家架构**：多个专家网络处理不同的文化特征
2. **高效计算**：只激活部分专家，减少计算量
3. **文化适应性**：为不同国家提供定制化的文化理解
4. **性能提升**：相比 LoRA Only 模型有更好的文化对齐效果

### 应用场景

- 多文化对话系统
- 跨文化理解任务
- 文化敏感性评估
- 国际化 AI 应用

---

## 🔍 模型加载验证

### 检查模型是否正确加载

```bash
# 查看模型参数数量
python -c "
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    '/path/to/model',
    torch_dtype=torch.float16,
    device_map='auto'
)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')
"
```

---

## 📊 性能对比

### 评估指标

| 模型类型 | 平均欧式距离 | 推理速度 | 内存占用 |
|---------|-----------|--------|--------|
| Base | 基准 | 快 | 低 |
| LoRA Only | 改进 | 快 | 低 |
| MOE | 最优 | 中等 | 中等 |
| CultureMoE | 最优 | 中等 | 高 |

---

## 🎉 总结

### ✅ 功能

1. **完整的 MOE 模型支持**
2. **自动路径推断**
3. **灵活的参数配置**
4. **详细的评估报告**

### 🚀 使用

```bash
# 快速开始
sh run_eval_vsm13.sh moe qwen

# 查看结果
cat output_dir/vsm13_results.json | python -m json.tool
```

---

**现在可以开始 MOE 模型评估了！** 🚀

