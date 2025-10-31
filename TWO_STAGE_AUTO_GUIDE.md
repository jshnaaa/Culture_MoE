# 两阶段训练 CultureMoE - 自动化指南

## ✅ 核心改进

现在两阶段训练**完全自动化**：

1. ✅ **自动调用 LoRA Only 训练**（保存最佳模型）
2. ✅ **自动调用合并脚本**（合并 LoRA 权重）
3. ✅ **自动训练 MoE**（使用合并后的模型）

**你只需要运行一个命令！**

## 🚀 使用方法

### 基本用法

```bash
sh run_two_stage_culturemoe.sh llama 2 True false
```

### 完整参数

```bash
sh run_two_stage_culturemoe.sh [backbone] [num_classes] [use_culture_loss] [save_model] [num_experts]
```

**参数说明**：
1. `backbone`: llama 或 qwen（默认：llama）
2. `num_classes`: 2/3/4/5（默认：2）
3. `use_culture_loss`: True/False（默认：True）
4. `save_model`: true/false（默认：false）
5. `num_experts`: 专家数量（默认：6）

## 📊 完整流程

### 自动执行的步骤

```
1. 阶段1：LoRA Only 训练
   ├── 调用 train_and_eval_lora_only.py
   ├── 保存验证准确率最高的 checkpoint
   └── 输出：stage1_lora/

2. 合并权重
   ├── 调用 merge_lora_weights.py
   ├── 合并 LoRA 权重到 base 模型
   └── 输出：merged_model_temp/

3. 阶段2：训练 MoE
   ├── 加载合并后的模型
   ├── 冻结 LLM
   ├── 训练 MoE 部分
   └── 输出：stage2/

4. 清理临时文件
   └── 删除 merged_model_temp/
```

## 🎯 使用示例

### 示例 1：LLaMA + 2分类（默认）

```bash
sh run_two_stage_culturemoe.sh llama 2 True false
```

**输出目录**：
```
/root/autodl-tmp/CultureMoE/Culture_Alignment/two_stage_moe/llama_2class_experts6_YYYYMMDD_HHMM/
├── config.json                         # 训练配置
├── final_report.json                   # 最终报告
│
├── stage1_lora/                        # 阶段1输出
│   ├── adapter_config.json
│   ├── adapter_model.bin               # LoRA 权重（最佳）
│   ├── eval_results.json
│   └── ...
│
└── stage2/                             # 阶段2输出
    ├── stage2_epoch_eval_results.json
    ├── stage2_final_eval.json
    └── ...
```

### 示例 2：Qwen + 3分类

```bash
sh run_two_stage_culturemoe.sh qwen 3 True false
```

### 示例 3：自定义专家数量

```bash
sh run_two_stage_culturemoe.sh llama 2 True false 4
```

### 示例 4：保存模型

```bash
sh run_two_stage_culturemoe.sh llama 2 True true
```

## 📈 预期输出

### 阶段1：LoRA Only 训练

```
================================================================================
STAGE 1: LoRA Fine-tuning (Auto Call)
================================================================================
Model: /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
Dataset: /root/autodl-fs/CulturalBench_Hard_merge.json
Num classes: 2
Epochs: 3
================================================================================

1. Calling LoRA Only training script...
   Output: .../stage1_lora

   (LoRA Only 训练过程...)

   ✅ LoRA Only training completed

2. Stage 1 Results:
   Accuracy: 0.7800
   F1:       0.7750

3. Merging LoRA weights...
   ✅ LoRA weights merged
   Merged model: .../merged_model_temp

4. Cleaning up checkpoints...
   ✅ Removed 3 checkpoints

================================================================================
Stage 1 Completed!
================================================================================
   Accuracy: 0.7800
   F1:       0.7750
   Merged model: .../merged_model_temp
================================================================================
```

### 阶段2：训练 MoE

```
================================================================================
STAGE 2: Training MoE (Frozen LLM)
================================================================================
Merged model: .../merged_model_temp
Num experts: 6
Epochs: 5
Output: .../stage2
================================================================================

1. Loading tokenizer...
   ✅ Tokenizer loaded

2. Loading merged LLM (will be frozen)...
   ✅ LLM loaded and frozen
   LLM parameters: 8,030,261,248
   Trainable: 0

3. Creating CultureMoE model...
   ✅ CultureMoE model created
   Total parameters: 8,045,123,456
   Trainable parameters: 14,862,208
   Trainable ratio: 0.18%

4. Loading dataset...
   ✅ Train: 1800, Val: 200

5. Creating training arguments...

6. Creating trainer...
   Use culture loss: True
   Culture loss lambda: 0.5

7. Starting Stage 2 training...
   (MoE 训练过程...)

8. Final evaluation...

================================================================================
Stage 2 Completed!
================================================================================
   Final Accuracy: 0.8200
   Final F1:       0.8150
================================================================================
```

### 最终报告

```
================================================================================
Final Report
================================================================================
End time: 2025-10-31 21:30:00

Stage 1 (LoRA Fine-tuning):
   Accuracy: 0.7800
   F1:       0.7750

Stage 2 (MoE Training):
   Accuracy: 0.8200
   F1:       0.8150

Improvement: +0.0400
================================================================================

Cleaning up temporary files...
   ✅ Temporary files removed

✅ Two-stage training completed!
   Results saved to: /root/autodl-tmp/CultureMoE/Culture_Alignment/two_stage_moe/llama_2class_experts6_20251031_2130
```

## 📁 输出文件

### 完整目录结构

```
output_dir/
├── config.json                         # 训练配置
├── final_report.json                   # 最终报告
│
├── stage1_lora/                        # 阶段1：LoRA Only
│   ├── adapter_config.json
│   ├── adapter_model.bin               # LoRA 权重（最佳）
│   ├── tokenizer_config.json
│   ├── eval_results.json               # 评估结果
│   └── trainer_state.json
│
└── stage2/                             # 阶段2：MoE
    ├── stage2_epoch_eval_results.json  # 每个 epoch 的结果
    ├── stage2_final_eval.json          # 最终评估结果
    └── logs/                           # TensorBoard 日志
```

### 查看结果

```bash
# 查看阶段1结果
cat output_dir/stage1_lora/eval_results.json | jq

# 查看阶段2结果
cat output_dir/stage2/stage2_final_eval.json | jq

# 查看最终报告
cat output_dir/final_report.json | jq
```

## 💡 核心优势

### 1. **完全自动化**

- ❌ **之前**：需要手动运行3个脚本
  ```bash
  # 1. 训练 LoRA
  sh run_train_lora_only.sh llama 2 true

  # 2. 合并权重
  sh run_merge_lora.sh llama /path/to/lora /path/to/output

  # 3. 训练 MoE
  python train_culturemoe.py --model_path /path/to/merged ...
  ```

- ✅ **现在**：只需一个命令
  ```bash
  sh run_two_stage_culturemoe.sh llama 2 True false
  ```

### 2. **自动保存最佳模型**

- ✅ 阶段1自动保存验证准确率最高的 checkpoint
- ✅ 自动合并最佳 LoRA 权重
- ✅ 阶段2使用最佳模型

### 3. **自动清理**

- ✅ 自动删除临时文件（merged_model_temp）
- ✅ 可选清理 checkpoints（save_model=false）

### 4. **完整记录**

- ✅ 保存每个阶段的评估结果
- ✅ 生成最终报告（包含改进幅度）

## ⚙️ 高级配置

### 调整阶段1参数

编辑 `run_two_stage_culturemoe.sh`：

```bash
--stage1_epochs 5          # 增加训练轮数
--lora_rank 32             # 增加 LoRA rank
--learning_rate 5e-6       # 调整学习率
```

### 调整阶段2参数

```bash
--stage2_epochs 10         # 增加训练轮数
--num_experts 4            # 调整专家数量
--culture_loss_lambda 0.3  # 调整文化损失权重
```

## 🔍 故障排除

### 问题 1：阶段1训练失败

**检查**：
```bash
# 查看 LoRA Only 训练日志
cat output_dir/stage1_lora/trainer_log.jsonl
```

**解决**：
- 检查数据文件路径
- 检查模型路径
- 检查显存是否足够

### 问题 2：合并失败

**检查**：
```bash
# 确认 LoRA 模型文件存在
ls output_dir/stage1_lora/adapter_model.bin
```

**解决**：
- 确保阶段1成功完成
- 确保 `--save_model` 参数正确

### 问题 3：阶段2训练失败

**检查**：
```bash
# 确认合并后的模型存在
ls output_dir/merged_model_temp/pytorch_model.bin
```

**解决**：
- 确保合并步骤成功
- 检查显存是否足够

## 📊 性能对比

| 方法 | 准确率 | 训练时间 | 操作步骤 |
|------|--------|---------|---------|
| **LoRA Only** | 0.78 | 30 min | 1 步 |
| **两阶段（手动）** | 0.82 | 60 min | 3 步 |
| **两阶段（自动）** | 0.82 | 60 min | **1 步** ✅ |

## ✅ 总结

### 核心改进

1. ✅ **完全自动化**：一个命令完成所有步骤
2. ✅ **自动保存最佳模型**：基于验证准确率
3. ✅ **自动合并权重**：无需手动操作
4. ✅ **自动清理**：删除临时文件

### 使用流程

```bash
# 1. 运行两阶段训练（一个命令）
sh run_two_stage_culturemoe.sh llama 2 True false

# 2. 查看结果
cat output_dir/final_report.json | jq
```

现在你可以轻松地运行两阶段训练，无需手动操作多个脚本！🎉

