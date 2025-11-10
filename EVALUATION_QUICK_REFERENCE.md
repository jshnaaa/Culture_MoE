# VSM13 评估快速参考

## 🚀 快速开始

### 评估所有模型类型

```bash
# 1. 评估 Base 模型
sh run_eval_vsm13.sh base llama

# 2. 评估 LoRA Only 模型
sh run_eval_vsm13.sh lora_only llama

# 3. 评估 MOE 模型（推荐用于 CultureMoE）
sh run_eval_vsm13.sh moe llama

# 4. 评估 CultureMoE 模型（如果有完整的模型文件）
sh run_eval_vsm13.sh culturemoe llama
```

---

## 📊 模型类型说明

| 模型类型 | 加载方式 | 权重来源 | 使用场景 |
|---------|--------|--------|--------|
| **base** | 直接加载 | Base 模型 | 基准测试 |
| **lora_only** | Base + LoRA | Base + LoRA 权重 | LoRA 微调模型 |
| **moe** | Base + LoRA + MOE | Base + LoRA + MOE 权重 | **MOE 混合专家模型** |
| **culturemoe** | 完整模型 | 完整的 CultureMoE 模型 | 已保存的完整模型 |

---

## ⚠️ 重要提示

### 为什么 CultureMoE 结果与 Base 相同？

**原因**：之前的代码只加载了 Base 模型，没有加载 LoRA 和 MOE 权重。

**解决方案**：使用 `--model_type moe` 来正确还原完整的 CultureMoE 模型。

### 正确的评估方式

```bash
# ✅ 正确：评估 MOE 模型（包含 LoRA + MOE 权重）
sh run_eval_vsm13.sh moe llama

# ❌ 错误：评估 CultureMoE（只加载 Base 模型）
sh run_eval_vsm13.sh culturemoe llama
```

---

## 🔍 验证修复

### 检查模型加载日志

运行评估时，应该看到：

```
✅ Base model loaded
✅ LoRA weights loaded
✅ LoRA weights merged
✅ MOE weights loaded and merged
```

如果看到这些日志，说明模型被正确加载了。

### 比较结果

```bash
# 查看平均欧式距离
python -c "
import json

# Base 模型
with open('/path/to/base_results.json') as f:
    base_dist = json.load(f)['average_euclidean_distance']

# LoRA Only 模型
with open('/path/to/lora_results.json') as f:
    lora_dist = json.load(f)['average_euclidean_distance']

# MOE 模型
with open('/path/to/moe_results.json') as f:
    moe_dist = json.load(f)['average_euclidean_distance']

print(f'Base:      {base_dist:.2f}')
print(f'LoRA Only: {lora_dist:.2f}')
print(f'MOE:       {moe_dist:.2f}')
print()
print('Expected: MOE < LoRA Only < Base (或类似的改进趋势)')
"
```

---

## 📈 预期结果

### 修复前（错误）

```
Base:      45.32
LoRA Only: 42.15
MOE:       45.32  ❌ 与 Base 相同（错误）
```

### 修复后（正确）

```
Base:      45.32
LoRA Only: 42.15
MOE:       38.76  ✅ 比 LoRA Only 更好（正确）
```

---

## 🛠️ 故障排除

### 问题 1：MOE 权重文件不存在

```
❌ Error: MOE weights not found: /path/to/moe_weights
```

**解决方案**：
1. 检查 MOE 权重路径是否正确
2. 确保 `best_moe` 目录中有 `pytorch_model.bin` 文件

### 问题 2：模型加载失败

```
❌ Failed to load CultureMoE from pretrained
```

**解决方案**：
1. 使用 `--model_type moe` 而不是 `culturemoe`
2. 提供正确的 `--lora_weights_path` 和 `--moe_weights_path`

### 问题 3：结果仍然与 Base 相同

**检查清单**：
- [ ] 是否使用了 `--model_type moe`？
- [ ] 是否提供了 `--moe_weights_path`？
- [ ] MOE 权重文件是否存在？
- [ ] 日志中是否显示 "✅ MOE weights loaded and merged"？

---

## 📝 完整命令示例

### 评估 MOE 模型（完整示例）

```bash
python eval_vsm13.py \
    --model_type moe \
    --backbone llama \
    --base_model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct \
    --lora_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora \
    --moe_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe \
    --data_path /root/autodl-fs/vsm13.json \
    --output_dir /root/autodl-tmp/CultureMoE/Culture_Alignment/vsm13/moe_llama_$(date +%Y%m%d_%H%M) \
    --device cuda
```

### 评估 Qwen 版本

```bash
python eval_vsm13.py \
    --model_type moe \
    --backbone qwen \
    --base_model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct \
    --lora_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora \
    --moe_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_qwen_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe \
    --data_path /root/autodl-fs/vsm13.json \
    --output_dir /root/autodl-tmp/CultureMoE/Culture_Alignment/vsm13/moe_qwen_$(date +%Y%m%d_%H%M) \
    --device cuda
```

---

## 📊 结果分析

### 查看结果

```bash
# 查看主要结果
cat /path/to/output/vsm13_results.json | python -m json.tool

# 查看平均欧式距离
python -c "
import json
with open('/path/to/output/vsm13_results.json') as f:
    data = json.load(f)
    print(f'Average Euclidean Distance: {data[\"average_euclidean_distance\"]:.2f}')
"

# 查看各国家的分数
python -c "
import json
with open('/path/to/output/vsm13_results.json') as f:
    data = json.load(f)
    for country, scores in data['country_scores'].items():
        print(f'{country}: {scores}')
"
```

---

## ✅ 检查清单

在运行评估前，确保：

- [ ] 已修改 `eval_vsm13.py`（添加 MOE 加载逻辑）
- [ ] 已更新 `run_eval_vsm13.sh`（支持 moe 类型）
- [ ] Base 模型路径正确
- [ ] LoRA 权重路径正确
- [ ] MOE 权重路径正确
- [ ] 数据集路径正确
- [ ] 输出目录存在或可创建
- [ ] GPU 内存充足

---

## 🎉 成功标志

评估成功的标志：

✅ 模型加载日志显示所有权重都被加载
✅ 评估完成，生成结果文件
✅ MOE 模型的结果与 Base 不同
✅ MOE 模型的结果比 Base 更好（欧式距离更小）

---

**现在可以正确评估 CultureMoE 模型了！** 🚀

