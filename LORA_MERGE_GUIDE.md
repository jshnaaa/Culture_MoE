# LoRA 训练和合并指南

## 📋 完整流程

### 步骤 1：训练 LoRA Only 模型（保存最佳 checkpoint）

```bash
# 训练并保存模型（保存验证准确率最高的 checkpoint）
sh run_train_lora_only.sh llama 2 true

# 或者指定其他配置
sh run_train_lora_only.sh qwen 3 true
```

**关键参数**：
- 第3个参数 `true`：启用模型保存
- 自动保存验证准确率（accuracy）最高的 checkpoint
- 只保留最近3个 checkpoint（节省空间）

**输出目录**：
```
/root/autodl-fs/output/lora_only_llama_2class_YYYYMMDD_HHMM/
├── adapter_config.json          # LoRA 配置
├── adapter_model.bin            # LoRA 权重（最佳）
├── tokenizer_config.json        # Tokenizer 配置
├── special_tokens_map.json
├── tokenizer.json
├── eval_results.json            # 评估结果
├── trainer_state.json           # 训练状态
└── checkpoint-xxx/              # 其他 checkpoints
```

### 步骤 2：合并 LoRA 权重到 base 模型

```bash
# 基本用法
sh run_merge_lora.sh llama \
    /root/autodl-fs/output/lora_only_llama_2class_YYYYMMDD_HHMM \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best

# Qwen 模型
sh run_merge_lora.sh qwen \
    /root/autodl-fs/output/lora_only_qwen_3class_YYYYMMDD_HHMM \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/qwen_3class_best
```

**参数说明**：
1. `backbone`：llama 或 qwen
2. `lora_model_dir`：LoRA 模型目录（步骤1的输出）
3. `output_dir`：合并后模型的输出目录

**输出目录**：
```
/root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best/
├── config.json                  # 模型配置
├── pytorch_model.bin            # 合并后的完整模型权重
├── tokenizer_config.json        # Tokenizer 配置
├── special_tokens_map.json
└── tokenizer.json
```

### 步骤 3：使用合并后的模型训练 CultureMoE

```bash
# 使用合并后的模型作为 frozen LLM
python train_culturemoe.py \
    --model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best \
    --train_file /root/autodl-fs/CulturalBench_Hard_merge.json \
    --output_dir /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_output \
    --num_classes 2 \
    --use_culture_loss True \
    --culture_loss_lambda 0.5
```

## 🎯 完整示例

### 示例 1：LLaMA + 2分类

```bash
# 1. 训练 LoRA Only（保存最佳模型）
sh run_train_lora_only.sh llama 2 true

# 2. 合并权重
sh run_merge_lora.sh llama \
    /root/autodl-fs/output/lora_only_llama_2class_20251031_1430 \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best

# 3. 训练 CultureMoE
python train_culturemoe.py \
    --model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best \
    --train_file /root/autodl-fs/CulturalBench_Hard_merge.json \
    --output_dir /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_llama_2class \
    --num_classes 2
```

### 示例 2：Qwen + 3分类

```bash
# 1. 训练 LoRA Only（保存最佳模型）
sh run_train_lora_only.sh qwen 3 true

# 2. 合并权重
sh run_merge_lora.sh qwen \
    /root/autodl-fs/output/lora_only_qwen_3class_20251031_1500 \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/qwen_3class_best

# 3. 训练 CultureMoE
python train_culturemoe.py \
    --model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/qwen_3class_best \
    --train_file /root/autodl-fs/normad_ed_merge.json \
    --output_dir /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_qwen_3class \
    --num_classes 3
```

## 📊 关键改进

### 1. **保存最佳模型（基于验证准确率）**

**修改前**：
```python
metric_for_best_model="f1"  # 使用 F1 作为最佳模型指标
```

**修改后**：
```python
metric_for_best_model="accuracy"  # ✅ 使用 accuracy 作为最佳模型指标
```

### 2. **自动加载最佳模型**

```python
load_best_model_at_end=True  # 训练结束后自动加载最佳模型
save_total_limit=3           # 只保留最近3个 checkpoint
```

### 3. **合并脚本**

新增 `merge_lora_weights.py` 和 `run_merge_lora.sh`：
- 自动合并 LoRA 权重到 base 模型
- 生成完整的模型文件
- 可直接用于 CultureMoE 训练

## 🔍 验证合并结果

### 检查合并后的模型

```bash
# 查看模型文件
ls -lh /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best/

# 预期输出：
# config.json
# pytorch_model.bin (约 16GB for LLaMA 3.1-8B)
# tokenizer_config.json
# ...
```

### 测试合并后的模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

print("✅ Model loaded successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 💡 使用建议

### 1. **快速实验（不保存模型）**

```bash
# 只保存评估结果，不保存模型
sh run_train_lora_only.sh llama 2 false
```

### 2. **正式训练（保存最佳模型）**

```bash
# 保存验证准确率最高的模型
sh run_train_lora_only.sh llama 2 true
```

### 3. **合并后用于 CultureMoE**

```bash
# 1. 训练 LoRA
sh run_train_lora_only.sh llama 2 true

# 2. 合并权重
sh run_merge_lora.sh llama \
    /root/autodl-fs/output/lora_only_llama_2class_xxx \
    /root/autodl-tmp/merged_llama_2class

# 3. 训练 CultureMoE（使用合并后的模型）
python train_culturemoe.py \
    --model_path /root/autodl-tmp/merged_llama_2class \
    --train_file /root/autodl-fs/CulturalBench_Hard_merge.json \
    --output_dir /root/autodl-tmp/culturemoe_output \
    --num_classes 2
```

## 📁 目录结构建议

```
/root/autodl-fs/output/                    # LoRA 训练输出
├── lora_only_llama_2class_xxx/
├── lora_only_llama_3class_xxx/
└── lora_only_qwen_2class_xxx/

/root/autodl-tmp/CultureMoE/Culture_Alignment/
├── merged_models/                         # 合并后的模型
│   ├── llama_2class_best/
│   ├── llama_3class_best/
│   └── qwen_2class_best/
│
└── culturemoe_output/                     # CultureMoE 训练输出
    ├── llama_2class_moe/
    └── qwen_3class_moe/
```

## ⚠️ 注意事项

### 1. **磁盘空间**

- LoRA 模型：~200 MB
- 合并后的模型：~16 GB (LLaMA 3.1-8B) 或 ~14 GB (Qwen 2.5-7B)
- 确保有足够的磁盘空间

### 2. **显存要求**

- 合并过程：需要加载完整的 base 模型（~16 GB 显存）
- 如果显存不足，可以使用 CPU 合并（较慢）：
  ```bash
  python merge_lora_weights.py \
      --base_model_path /path/to/base \
      --lora_model_path /path/to/lora \
      --output_path /path/to/output \
      --device cpu
  ```

### 3. **验证最佳模型**

训练结束后，检查 `eval_results.json` 确认最佳模型的准确率：

```bash
cat /root/autodl-fs/output/lora_only_llama_2class_xxx/eval_results.json | jq
```

## 🎉 总结

### 完整流程

1. ✅ **训练 LoRA Only**：`sh run_train_lora_only.sh llama 2 true`
2. ✅ **合并权重**：`sh run_merge_lora.sh llama /path/to/lora /path/to/output`
3. ✅ **训练 CultureMoE**：使用合并后的模型作为 frozen LLM

### 核心改进

- ✅ 自动保存验证准确率最高的 checkpoint
- ✅ 提供便捷的合并脚本
- ✅ 合并后的模型可直接用于 CultureMoE 训练

现在你可以轻松地训练 LoRA 模型，合并权重，并用于 CultureMoE 训练了！🚀

