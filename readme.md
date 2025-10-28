# 最终使用指南

## ✅ 完整的参数支持

所有训练脚本现在都支持三个参数：

```bash
sh <script_name>.sh [backbone] [num_classes] [save_model]
```

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

### 1. LoRA Only 训练

```bash
# 默认：LLaMA + 2分类 + 不保存模型
sh run_train_lora_only.sh

# LLaMA + 3分类 + 不保存模型
sh run_train_lora_only.sh llama 3

# LLaMA + 3分类 + 保存模型
sh run_train_lora_only.sh llama 3 true

# Qwen + 4分类 + 不保存模型
sh run_train_lora_only.sh qwen 4 false
```

### 2. CultureMoE 训练

```bash
# 命令格式
sh run_train_ddp_lora_dual.sh [backbone] [num_classes] [use_culture_loss] [save_model]

# 不保存模型（默认）
sh run_train_ddp_lora_dual.sh llama 3 True false
sh run_train_ddp_lora_dual.sh llama 3 True  # 等价于上面

# 保存模型
sh run_train_ddp_lora_dual.sh llama 3 True true
```

#### 禁用文化损失

```bash
# LLaMA + 2 分类 + 不使用文化损失
sh run_train_ddp_lora_dual.sh llama 2 False

# Qwen + 4 分类 + 不使用文化损失
sh run_train_ddp_lora_dual.sh qwen 4 False

# LLaMA + 3 分类 + 使用文化损失
sh run_train_ddp_lora_dual.sh llama 3 True

# Qwen + 5 分类 + 不使用文化损失
sh run_train_ddp_lora_dual.sh qwen 5 False
```

**注意**：DDP 训练脚本目前总是保存模型，不支持 `save_model` 参数。

### 4. Base 模型评估

```bash
# 默认：LLaMA + 4分类
sh run_eval_base_llama.sh

# LLaMA + 3分类
sh run_eval_base_llama.sh llama 3

# Qwen + 2分类
sh run_eval_base_llama.sh qwen 2
```

**注意**：评估脚本只保存评估结果，不保存模型。

## 🎯 典型使用场景

### 场景 1：快速实验（不保存模型）

当你只想快速测试不同配置的效果时：

```bash
# 测试 LLaMA 在不同分类任务上的表现
sh run_train_lora_only.sh llama 2 false
sh run_train_lora_only.sh llama 3 false
sh run_train_lora_only.sh llama 4 false
sh run_train_lora_only.sh llama 5 false

# 测试 Qwen 在不同分类任务上的表现
sh run_train_lora_only.sh qwen 2 false
sh run_train_lora_only.sh qwen 3 false
```

这样可以节省大量磁盘空间，只保留评估结果用于对比。

### 场景 2：正式训练（保存模型）

当你找到最佳配置，需要保存模型用于后续使用时：

```bash
# 训练并保存最佳模型
sh run_train_lora_only.sh llama 3 true
sh run_train_culturemoe.sh llama 3 true
```

## 📊 输出说明

### 不保存模型时（save_model=false）

输出目录包含：
- `eval_results.json` - 评估结果
- `trainer_log.jsonl` - 训练日志
- `running_log.txt` - 运行日志
- TensorBoard 日志文件

**不包含**：
- 模型权重文件（`pytorch_model.bin` 或 `adapter_model.bin`）
- 模型配置文件
- Tokenizer 文件

### 保存模型时（save_model=true）

输出目录包含：
- 所有上述文件
- **加上**：模型权重文件、配置文件、Tokenizer 文件等

## 💾 磁盘空间对比

以 LLaMA 3.1-8B 为例：

| 配置 | 磁盘占用 | 说明 |
|------|---------|------|
| 不保存模型 | ~100 MB | 只有日志和评估结果 |
| 保存 LoRA 模型 | ~200 MB | LoRA 权重较小 |
| 保存 CultureMoE 模型 | ~500 MB | 包含 MoE 层权重 |

**建议**：
- 实验阶段：使用 `save_model=false`，节省空间
- 最终模型：使用 `save_model=true`，保存最佳模型

## 🔍 查看评估结果

无论是否保存模型，评估结果都会保存在 `eval_results.json` 中：

```bash
# 查看评估结果
cat /root/autodl-fs/output/lora_only_llama_3class_*/eval_results.json

# 或使用 jq 格式化输出
cat /root/autodl-fs/output/lora_only_llama_3class_*/eval_results.json | jq
```

## ⚙️ 默认值总结

| 脚本 | 默认 backbone | 默认 num_classes | 默认 save_model |
|------|--------------|-----------------|----------------|
| `run_train_lora_only.sh` | llama | 2 | false |
| `run_train_culturemoe.sh` | llama | 4 | false |
| `run_train_ddp_lora_dual.sh` | llama | 2 | N/A (总是保存) |
| `run_eval_base_llama.sh` | llama | 4 | N/A (不保存) |

## 📝 完整命令示例

```bash
# 示例 1：使用所有默认值
sh run_train_lora_only.sh
# 等价于：sh run_train_lora_only.sh llama 2 false

# 示例 2：只指定 backbone
sh run_train_lora_only.sh qwen
# 等价于：sh run_train_lora_only.sh qwen 2 false

# 示例 3：指定 backbone 和 num_classes
sh run_train_lora_only.sh qwen 5
# 等价于：sh run_train_lora_only.sh qwen 5 false

# 示例 4：指定所有参数
sh run_train5 true

# 示例 5：使用默认 backbone，指定其他参数
sh run_train_lora_only.sh llama 3 true
```

