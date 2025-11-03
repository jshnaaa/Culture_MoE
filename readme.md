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

### 1. LoRA Only 训练+验证

```bash
# LLaMA + 2分类 + 保存模型
sh run_train_lora_only.sh llama 3 true

# Qwen + 4分类 + 不保存模型
sh run_train_lora_only.sh qwen 4 false
```

### 2. CultureMoE 训练+验证

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
# 参数：BACKBONE, NUM_CLASSES, USE_CULTURE_LOSS, NUM_EXPERTS, SAVE_MODEL, NUM_GPUS
sh run_train_culturemoe_from_merged.sh llama 2 True 6 false 1

```

### 3. Base 模型评估

```bash
# LLaMA + 4分类
sh run_eval_base_llama.sh llama 4

# Qwen + 2分类
sh run_eval_base_llama.sh qwen 2
```

## 🎯 分析实验

### 场景 1：VSM13 文化一致性测试

# 使用方法：
   sh run_eval_vsm13.sh <MODEL> <BACKBONE> [NUM_CLASSES]

# 示例：
   sh run_eval_vsm13.sh base llama
   sh run_eval_vsm13.sh lora qwen 2
   sh run_eval_vsm13.sh moe llama 4

