# 使用指南

## ✅ 完整的参数支持

### 参数说明

1. **backbone** - 基座模型类型
   - `llama` - LLaMA 3.1-8B-Instruct（默认）
   - `qwen` - Qwen 2.5-7B-Instruct

2. **num_classes** - 分类类别数
   - `2` - 二分类，使用 `CulturalBench_Hard_merge.json`
   - `3` - 三分类，使用 `normad_ed_merge.json`
   - `4` - 四分类，使用 `wvs_all_llama_merge_4.json`
   - `5` - 五分类，使用 `wvs_all_llama_merge_5.json`

3. **save_model** - 是否保存模型权重
   - `false` - 不保存模型权重，只保存评估结果（默认）
   - `true` - 保存模型权重和评估结果

## 📝 使用示例

### 1. LoRA Only 
#### 训练+验证

```bash
# LLaMA + 2分类 + 保存模型
sh run_train_lora_only.sh llama 3 true

# Qwen + 4分类 + 不保存模型
sh run_train_lora_only.sh qwen 4 false
```

#### 测试

```bash
# LLaMA + 2分类
sh run_train_lora_only.sh llama 3 true

# Qwen + 4分类
sh run_train_lora_only.sh qwen 4 false
```

#### 生成式的训练+验证

```bash
# sh run_train_lora_only_gen.sh <BACKBONE> <DATA_ID>

sh run_train_lora_only_gen.sh llama 4
sh run_train_lora_only_gen.sh llama 3
sh run_train_lora_only_gen.sh qwen 2
```

#### 生成式的测试

```bash
# 使用 LLaMA 模型
sh run_eval_lora_only_from_components.sh llama

# 使用 Qwen 模型
sh run_eval_lora_only_from_components.sh qwen
```

### 2. CultureMoE

#### 端到端训练

```bash
# 命令格式
sh run_train_ddp_lora_dual.sh [backbone] [num_classes] [use_culture_loss] [save_model]

# 不保存模型（默认）
sh run_train_ddp_lora_dual.sh llama 3 True false

# 保存模型
sh run_train_ddp_lora_dual.sh llama 3 True true
```

#### 两阶段训练

```bash
# 先完成lora only训练

# 再合并lora
sh run_merge_lora.sh llama 2

# 最后训练MoE
# 参数：BACKBONE, NUM_CLASSES, USE_CULTURE_LOSS, NUM_EXPERTS, SAVE_MODEL, NUM_GPUS, MASK_USE, LORA_USE
# MASK_USE: true=使用instruction_mask字段, false=两路都用instruction字段
# LORA_USE: true=从合并后的LoRA开始训练, false=从Base模型开始训练
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 true true

# 消融实验：不使用 instruction_mask
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 false true

# 消融实验：不使用 LoRA（从 Base 模型开始）
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1 true false
```

#### 测试

```bash
# sh run_load_culturemoe.sh <BACKBONE> <NUM_CLASSES>
sh run_load_culturemoe.sh llama 2
sh run_load_culturemoe.sh qwen 4
```
#### 生成式
```bash

```

### 3. Base 模型评估+测试
```bash
# 最后一个参数指定数据集，最后一个数字为1，前面的数字：2和4表示验证集/22和44表示测试集，选择：21/41/221/441
# LLaMA + 4分类
sh run_eval_base_llama.sh llama 41

# Qwen + 2分类
sh run_eval_base_llama.sh qwen 221

# sh run_eval_base_gen.sh <BACKBONE> <NUM_CLASSES>
sh run_eval_base_gen.sh llama 11
sh run_eval_base_gen.sh qwen 21
```


### 4. Role-Play 模型评估+测试
```bash
# 最后一个参数指定数据集，最后一个数字为2，前面的数字：2和4表示验证集/22和44表示测试集，选择：22/42/222/442
# LLaMA + 4分类
sh run_eval_base_llama.sh llama 42

# Qwen + 2分类
sh run_eval_base_llama.sh qwen 222

# sh run_eval_base_gen.sh <BACKBONE> <NUM_CLASSES>
sh run_eval_base_gen.sh llama 12
sh run_eval_base_gen.sh qwen 22
```


## 🎯 分析实验

### 1：VSM13 文化一致性测试

```bash
# 使用方法：
   sh run_eval_vsm13.sh <MODEL> <BACKBONE> [NUM_CLASSES]

# 示例：
   sh run_eval_vsm13.sh base llama
   sh run_eval_vsm13.sh lora qwen 2
   sh run_eval_vsm13.sh moe llama 4
```

### 2：消融实验
```bash
# 使用方法：


# 示例：
```

### 3：实验
```bash
# 使用方法：
   

# 示例：
```