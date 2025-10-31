# 手动两阶段训练 CultureMoE 指南

## 📋 完整流程

你需要**手动执行3个步骤**：

### 步骤 1：训练 LoRA Only（保存最佳模型）

```bash
sh run_train_lora_only.sh llama 2 true
```

**输出**：
```
/root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only/
├── adapter_config.json
├── adapter_model.bin          # LoRA 权重（最佳）
├── tokenizer_config.json
└── eval_results.json
```

### 步骤 2：合并 LoRA 权重到 base 模型

```bash
sh run_merge_lora.sh llama \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best
```

**输出**：
```
/root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best/
├── config.json
├── pytorch_model.bin          # 完整模型权重
├── tokenizer_config.json
└── ...
```

### 步骤 3：从合并后的模型训练 CultureMoE

```bash
sh run_train_culturemoe_from_merged.sh \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/merged_models/llama_2class_best \
    2 \
    True \
    6
```

**输出**：
```
/root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_output/culturemoe_2class_experts6_YYYYMMDD_HHMM/
├── config.json
├── epoch_eval_results.json
├── final_eval_results.json
└── logs/
```

## 🎯 完整示例

### 示例 1：LLaMA + 2分类

```bash
# 步骤1：训练 LoRA Only
sh run_train_lora_only.sh llama 2 true

# 步骤2：合并权重
sh run_merge_lora.sh llama \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only \
    /root/autodl-tmp/merged_llama_2class

# 步骤3：训练 CultureMoE
sh run_train_culturemoe_from_merged.sh \
    /root/autodl-tmp/merged_llama_2class \
    2 \
    True \
    6
```

### 示例 2：Qwen + 3分类

```bash
# 步骤1：训练 LoRA Only
sh run_train_lora_only.sh qwen 3 true

# 步骤2：合并权重
sh run_merge_lora.sh qwen \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/qwen_lora_only \
    /root/autodl-tmp/merged_qwen_3class

# 步骤3：训练 CultureMoE
sh run_train_culturemoe_from_merged.sh \
    /root/autodl-tmp/merged_qwen_3class \
    3 \
    True \
    6
```

### 示例 3：自定义专家数量

```bash
# 步骤1：训练 LoRA Only
sh run_train_lora_only.sh llama 2 true

# 步骤2：合并权重
sh run_merge_lora.sh llama \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only \
    /root/autodl-tmp/merged_llama_2class

# 步骤3：训练 CultureMoE（4个专家）
sh run_train_culturemoe_from_merged.sh \
    /root/autodl-tmp/merged_llama_2class \
    2 \
    True \
    4
```

## 📝 脚本参数说明

### 步骤 1：`run_train_lora_only.sh`

```bash
sh run_train_lora_only.sh [backbone] [num_classes] [save_model]
```

- `backbone`: llama 或 qwen
- `num_classes`: 2/3/4/5
- `save_model`: **必须是 true**（保存最佳模型）

### 步骤 2：`run_merge_lora.sh`

```bash
sh run_merge_lora.sh [backbone] [lora_model_dir] [output_dir]
```

- `backbone`: llama 或 qwen
- `lora_model_dir`: LoRA 模型目录（步骤1的输出）
- `output_dir`: 合并后模型的输出目录

### 步骤 3：`run_train_culturemoe_from_merged.sh`

```bash
sh run_train_culturemoe_from_merged.sh [merged_model_path] [num_classes] [use_culture_loss] [num_experts]
```

- `merged_model_path`: 合并后的模型路径（步骤2的输出）
- `num_classes`: 2/3/4/5
- `use_culture_loss`: True/False
- `num_experts`: 专家数量（默认：6）

## 🔍 验证每个步骤

### 验证步骤 1：LoRA Only 训练

```bash
# 检查输出文件
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only/

# 预期文件：
# adapter_config.json
# adapter_model.bin
# eval_results.json

# 查看评估结果
cat /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only/eval_results.json | jq
```

### 验证步骤 2：权重合并

```bash
# 检查输出文件
ls /root/autodl-tmp/merged_llama_2class/

# 预期文件：
# config.json
# pytorch_model.bin (约 16GB)
# tokenizer_config.json

# 测试加载模型
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = '/root/autodl-tmp/merged_llama_2class'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
print('✅ Model loaded successfully!')
"
```

### 验证步骤 3：CultureMoE 训练

```bash
# 检查输出文件
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/culturemoe_output/culturemoe_2class_experts6_xxx/

# 预期文件：
# config.json
# epoch_eval_results.json
# final_eval_results.json

# 查看最终结果
cat .../final_eval_results.json | jq
```

## 💡 为什么要手动执行？

### 优势

1. **灵活性**：可以在每个步骤之间检查结果
2. **可控性**：可以使用不同的 LoRA 模型或合并配置
3. **调试方便**：每个步骤独立，容易定位问题
4. **资源管理**：可以在不同时间或不同机器上执行

### 适用场景

- ✅ 需要尝试不同的 LoRA 配置
- ✅ 需要使用已有的 LoRA 模型
- ✅ 需要在不同机器上执行不同步骤
- ✅ 需要详细检查每个步骤的输出

## 📊 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1：训练 LoRA Only                                       │
│ sh run_train_lora_only.sh llama 2 true                     │
│                                                             │
│ 输入：Base Model                                            │
│ 输出：LoRA 权重（最佳）                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2：合并权重                                             │
│ sh run_merge_lora.sh llama /path/to/lora /path/to/output   │
│                                                             │
│ 输入：Base Model + LoRA 权重                                │
│ 输出：合并后的完整模型                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3：训练 CultureMoE                                      │
│ sh run_train_culturemoe_from_merged.sh /path/to/merged ... │
│                                                             │
│ 输入：合并后的模型                                           │
│ 输出：CultureMoE 模型                                        │
└─────────────────────────────────────────────────────────────┘
```

## ⚠️ 注意事项

### 1. **步骤1必须保存模型**

```bash
# ❌ 错误：不保存模型
sh run_train_lora_only.sh llama 2 false

# ✅ 正确：保存模型
sh run_train_lora_only.sh llama 2 true
```

### 2. **路径必须正确**

确保步骤2使用步骤1的输出路径：

```bash
# 步骤1输出
/root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only

# 步骤2必须使用这个路径
sh run_merge_lora.sh llama \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only \  # ✅ 正确
    /root/autodl-tmp/merged_llama_2class
```

### 3. **分类数必须一致**

```bash
# 步骤1：2分类
sh run_train_lora_only.sh llama 2 true

# 步骤3：必须也是2分类
sh run_train_culturemoe_from_merged.sh /path/to/merged 2 True 6  # ✅ 正确
sh run_train_culturemoe_from_merged.sh /path/to/merged 3 True 6  # ❌ 错误
```

## 🎉 总结

### 完整流程

```bash
# 1. 训练 LoRA Only（保存最佳模型）
sh run_train_lora_only.sh llama 2 true

# 2. 合并权重
sh run_merge_lora.sh llama \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_lora_only \
    /root/autodl-tmp/merged_llama_2class

# 3. 训练 CultureMoE
sh run_train_culturemoe_from_merged.sh \
    /root/autodl-tmp/merged_llama_2class \
    2 \
    True \
    6
```

### 核心文件

1. ✅ `train_culturemoe_from_merged.py` - 从合并后的模型训练 MoE
2. ✅ `run_train_culturemoe_from_merged.sh` - Shell 脚本

### 优势

- ✅ 完全手动控制每个步骤
- ✅ 可以在每个步骤之间检查结果
- ✅ 灵活性高，适合实验和调试

现在你可以完全手动控制两阶段训练流程了！🚀

