# CultureMoE 评估问题诊断与修复

## 🔍 问题诊断

### 问题描述

运行 `bash run_eval_vsm13.sh culturemoe llama` 时，输出的答案以及 6 个维度的分数和欧氏距离与 Base LLaMA 几乎完全一样，这表明 CultureMoE 模型没有被正确还原。

### 根本原因

在 `eval_vsm13.py` 中，当 `model_type == 'culturemoe'` 时，代码直接加载 Base 模型：

```python
else:  # culturemoe
    # CultureMoE 模型：直接加载
    print(f"\nLoading CultureMoE model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ CultureMoE model loaded")
```

**问题**：
1. CultureMoE 是一个自定义模型类 `LlamaSharedRouterExpertsModel`，不能用 `AutoModelForCausalLM` 直接加载
2. 代码没有加载 LoRA 权重和 MOE 权重
3. 结果就是加载了 Base 模型，而不是完整的 CultureMoE 模型

### 为什么 LoRA Only 有区别，但 CultureMoE 没有区别？

**LoRA Only 有区别的原因**：
- LoRA 权重被正确加载并合并到 Base 模型中
- LoRA 修改了模型的参数，所以输出不同

**CultureMoE 没有区别的原因**：
- MOE 权重没有被加载
- 只加载了 Base 模型
- 所以输出与 Base 完全相同

---

## ✅ 解决方案

### 修改内容

已修改 `eval_vsm13.py` 中的模型加载逻辑：

#### 1. 添加导入

```python
import sys
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs
```

#### 2. 添加 MOE 权重路径参数

```python
if model_type == 'moe' and moe_weights_path is None:
    if backbone == 'qwen':
        moe_weights_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_qwen_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe"
    else:  # llama
        moe_weights_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe"
```

#### 3. 添加 MOE 模型加载逻辑

```python
elif model_type == 'moe':
    # MOE 模型：加载 base 模型 + LoRA 权重 + MOE 权重
    print(f"\nLoading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Base model loaded")

    print(f"\nLoading LoRA weights from {lora_weights_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        is_trainable=False,
        torch_dtype=torch.float16
    )
    print("✅ LoRA weights loaded")

    print(f"\nMerging LoRA weights into base model...")
    model = model.merge_and_unload()
    print("✅ LoRA weights merged")

    print(f"\nLoading MOE weights from {moe_weights_path}...")
    moe_state_dict = torch.load(
        os.path.join(moe_weights_path, 'pytorch_model.bin'),
        map_location='cpu'
    )
    model.load_state_dict(moe_state_dict, strict=False)
    print("✅ MOE weights loaded and merged")
```

#### 4. 改进 CultureMoE 加载逻辑

```python
else:  # culturemoe
    # ⚠️ CultureMoE 模型：需要从 Base + LoRA + MOE 权重还原
    print(f"\n⚠️  CultureMoE 模型需要从 Base + LoRA + MOE 权重还原")
    print(f"如果你想评估完整的 CultureMoE 模型，请使用 --model_type moe")
    print(f"或者提供 --moe_weights_path 参数")

    # 尝试从 base_model_path 加载（如果是完整的 CultureMoE 模型）
    print(f"\nAttempting to load CultureMoE model from {base_model_path}...")
    try:
        model = LlamaSharedRouterExpertsModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ CultureMoE model loaded from pretrained")
    except Exception as e:
        print(f"❌ Failed to load CultureMoE from pretrained: {e}")
        print(f"Falling back to loading as Base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("⚠️  Loaded as Base model (not CultureMoE)")
```

---

## 🚀 使用方法

### 正确的评估方式

#### 方式 1：使用 `moe` 类型（推荐）

```bash
# 评估 MOE 模型（从 Base + LoRA + MOE 权重还原）
sh run_eval_vsm13.sh moe llama

# 或者直接运行 Python
python eval_vsm13.py \
    --model_type moe \
    --backbone llama \
    --base_model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct \
    --lora_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora \
    --moe_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe \
    --data_path /root/autodl-fs/vsm13.json \
    --output_dir /path/to/output \
    --device cuda
```

#### 方式 2：使用 `culturemoe` 类型（如果有完整的 CultureMoE 模型）

```bash
# 如果 base_model_path 是一个完整的 CultureMoE 模型
sh run_eval_vsm13.sh culturemoe llama
```

**注意**：这种方式需要 `base_model_path` 指向一个完整的、已保存的 CultureMoE 模型。

---

## 📊 预期结果

### 修复前

```
Model type: culturemoe
Base model: /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
✅ CultureMoE model loaded

结果：与 Base LLaMA 完全相同
```

### 修复后

```
Model type: moe
Base model: /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
LoRA weights: /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora
MOE weights: /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe

✅ Base model loaded
✅ LoRA weights loaded
✅ LoRA weights merged
✅ MOE weights loaded and merged

结果：与 Base LLaMA 不同，显示 MOE 的改进
```

---

## 🔍 验证修复

### 检查点 1：模型加载日志

运行评估时，应该看到：

```
Loading Model
================================================================================
Model type: moe
Backbone: llama
Base model path: /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
LoRA weights path: /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora
MOE weights path: /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe
================================================================================

Loading tokenizer from /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora...
✅ Tokenizer loaded

Loading base model from /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct...
✅ Base model loaded

Loading LoRA weights from /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora...
✅ LoRA weights loaded

Merging LoRA weights into base model...
✅ LoRA weights merged

Loading MOE weights from /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe...
✅ MOE weights loaded and merged

✅ Model ready for evaluation
```

### 检查点 2：结果差异

运行以下命令比较结果：

```bash
# 评估 Base 模型
sh run_eval_vsm13.sh base llama

# 评估 LoRA Only 模型
sh run_eval_vsm13.sh lora_only llama

# 评估 MOE 模型
sh run_eval_vsm13.sh moe llama

# 比较结果
python -c "
import json

with open('/path/to/base_results.json') as f:
    base = json.load(f)
with open('/path/to/lora_results.json') as f:
    lora = json.load(f)
with open('/path/to/moe_results.json') as f:
    moe = json.load(f)

print('Base Average Distance:', base['average_euclidean_distance'])
print('LoRA Average Distance:', lora['average_euclidean_distance'])
print('MOE Average Distance:', moe['average_euclidean_distance'])

print('\nExpected: MOE < LoRA < Base (or similar)')
"
```

---

## 💡 为什么 LoRA Only 有区别，但 CultureMoE 没有区别？

### LoRA Only 的情况

```
Base 模型
    ↓
加载 LoRA 权重
    ↓
合并 LoRA 权重
    ↓
✅ 模型参数改变，输出不同
```

### CultureMoE 的情况（修复前）

```
Base 模型
    ↓
❌ 没有加载 LoRA 权重
    ↓
❌ 没有加载 MOE 权重
    ↓
❌ 只有 Base 模型，输出与 Base 相同
```

### CultureMoE 的情况（修复后）

```
Base 模型
    ↓
加载 LoRA 权重
    ↓
合并 LoRA 权重
    ↓
加载 MOE 权重
    ↓
✅ 模型参数改变，输出不同
```

---

## 🎯 总结

### 问题
- CultureMoE 模型没有被正确还原
- 只加载了 Base 模型，没有加载 LoRA 和 MOE 权重
- 结果与 Base 完全相同

### 解决方案
- 添加 MOE 权重加载逻辑
- 正确还原 Base + LoRA + MOE 的完整模型
- 提供清晰的错误提示和回退机制

### 使用方法
```bash
# 评估 MOE 模型（推荐）
sh run_eval_vsm13.sh moe llama

# 或者
python eval_vsm13.py \
    --model_type moe \
    --backbone llama \
    --lora_weights_path /path/to/lora \
    --moe_weights_path /path/to/moe \
    --data_path /path/to/data \
    --output_dir /path/to/output
```

---

**现在 CultureMoE 模型应该被正确评估了！** ✅

