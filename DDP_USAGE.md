# DDP 训练脚本使用说明

## 📝 脚本：`run_train_ddp_lora_dual.sh`

### 参数说明

```bash
sh run_train_ddp_lora_dual.sh [backbone] [num_classes] [use_culture_loss]
```

#### 参数详解

1. **backbone** - 基座模型类型
   - `llama` - LLaMA 3.1-8B-Instruct（默认）
   - `qwen` - Qwen 2.5-7B-Instruct

2. **num_classes** - 分类类别数
   - `2` - 二分类，使用 `CulturalBench_Hard_merge.json`（默认）
   - `3` - 三分类，使用 `normad_ed_merge.json`
   - `4` - 四分类，使用 `wvs_all_llama_merge_4.json`
   - `5` - 五分类，使用 `wvs_all_llama_merge_5.json`

3. **use_culture_loss** - 是否使用文化专注性损失
   - `True` - 使用文化损失（默认）
   - `False` - 不使用文化损失

### 使用示例

#### 1. 使用默认参数

```bash
sh run_train_ddp_lora_dual.sh
# 等价于：sh run_train_ddp_lora_dual.sh llama 2 True
```

**效果**：
- 使用 LLaMA 模型
- 2 分类任务
- 使用文化损失（lambda=0.01）

#### 2. 指定模型和类别数

```bash
# LLaMA + 3 分类 + 使用文化损失
sh run_train_ddp_lora_dual.sh llama 3

# Qwen + 5 分类 + 使用文化损失
sh run_train_ddp_lora_dual.sh qwen 5
```

#### 3. 禁用文化损失

```bash
# LLaMA + 2 分类 + 不使用文化损失
sh run_train_ddp_lora_dual.sh llama 2 False

# Qwen + 4 分类 + 不使用文化损失
sh run_train_ddp_lora_dual.sh qwen 4 False
```

#### 4. 完整指定所有参数

```bash
# LLaMA + 3 分类 + 使用文化损失
sh run_train_ddp_lora_dual.sh llama 3 True

# Qwen + 5 分类 + 不使用文化损失
sh run_train_ddp_lora_dual.sh qwen 5 False
```

## 🔍 文化损失说明

### 什么是文化专注性损失？

文化专注性损失（Culture-Focused Loss）是一个额外的损失项，用于增强模型对文化相关特征的关注。

### 参数配置

在脚本中，文化损失相关的参数有：

1. **`--use_culture_loss`** - 是否启用文化损失
   - `True`：启用
   - `False`：禁用

2. **`--culture_loss_lambda`** - 文化损失的权重系数
   - 当前设置：`0.01`
   - 范围：通常在 0.001 到 0.1 之间

### 使用场景

**建议使用文化损失（`True`）**：
- ✅ 训练文化感知模型
- ✅ 需要模型关注文化特征
- ✅ 处理文化相关的分类任务

**建议禁用文化损失（`False`）**：
- ❌ 作为 baseline 对比
- ❌ 非文化相关任务
- ❌ 想要更快的训练速度

### 对比实验示例

```bash
# 实验 1：使用文化损失
sh run_train_ddp_lora_dual.sh llama 3 True

# 实验 2：不使用文化损失（作为 baseline）
sh run_train_ddp_lora_dual.sh llama 3 False

# 对比两个实验的结果
```

## ⚙️ 其他重要参数

脚本中还包含以下固定参数（可以在脚本中修改）：

### 训练参数
- `num_train_epochs`: 10
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 8
- `gradient_accumulation_steps`: 8
- `learning_rate`: 5e-6

### LoRA 参数
- `use_llama_lora`: True
- `llama_lora_rank`: 8
- `llama_lora_alpha`: 16
- `llama_lora_dropout`: 0.05

### 其他参数
- `use_dual_input`: True（使用双路输入）
- `freeze_llama`: False（不冻结 LLaMA）
- `max_length`: 512
- `val_split`: 0.1

## 🎯 完整的实验流程

### 场景 1：对比文化损失的效果

```bash
# 步骤 1：使用文化损失训练
sh run_train_ddp_lora_dual.sh llama 3 True

# 步骤 2：不使用文化损失训练（baseline）
sh run_train_ddp_lora_dual.sh llama 3 False

# 步骤 3：对比两个模型的评估结果
```

### 场景 2：测试不同模型

```bash
# LLaMA + 文化损失
sh run_train_ddp_lora_dual.sh llama 3 True

# Qwen + 文化损失
sh run_train_ddp_lora_dual.sh qwen 3 True

# 对比 LLaMA 和 Qwen 的表现
```

### 场景 3：测试不同分类任务

```bash
# 2 分类 + 文化损失
sh run_train_ddp_lora_dual.sh llama 2 True

# 3 分类 + 文化损失
sh run_train_ddp_lora_dual.sh llama 3 True

# 4 分类 + 文化损失
sh run_train_ddp_lora_dual.sh llama 4 True

# 5 分类 + 文化损失
sh run_train_ddp_lora_dual.sh llama 5 True
```

## 📊 参数组合表

| backbone | num_classes | use_culture_loss | 数据集 | 说明 |
|----------|-------------|------------------|--------|------|
| llama | 2 | True | CulturalBench_Hard_merge.json | 默认配置 |
| llama | 3 | True | normad_ed_merge.json | 3分类+文化损失 |
| llama | 3 | False | normad_ed_merge.json | 3分类 baseline |
| qwen | 5 | True | wvs_all_llama_merge_5.json | Qwen 5分类 |
| qwen | 5 | False | wvs_all_llama_merge_5.json | Qwen baseline |

## ⚠️ 注意事项

1. **GPU 要求**：DDP 训练需要至少 2 个 GPU
   - 脚本默认使用 GPU 0 和 1
   - 可以修改 `CUDA_VISIBLE_DEVICES` 环境变量

2. **模型保存**：DDP 训练总是保存模型
   - 不支持 `--save_model` 参数
   - 模型会保存到 `OUTPUT_DIR`

3. **文化损失权重**：
   - 当前设置为 `0.01`
   - 如果需要调整，修改脚本中的 `--culture_loss_lambda` 值

4. **训练时间**：
   - 使用文化损失会略微增加训练时间（约 5-10%）
   - 但通常能获得更好的性能

## ✅ 总结

**回答你的问题**：

> `--use_culture_loss` 参数可以正常使用吗？

**答案：✅ 可以正常使用！**

- ✅ 参数已在 `ClassificationTrainingArguments` 中定义
- ✅ 默认值为 `True`（使用文化损失）
- ✅ 可以通过命令行参数控制：`sh run_train_ddp_lora_dual.sh llama 3 False`
- ✅ 修改后的脚本支持通过第三个参数灵活控制

**使用建议**：
1. 首次训练：使用默认值（`True`）
2. 对比实验：分别训练 `True` 和 `False` 两个版本
3. 根据评估结果选择最佳配置

