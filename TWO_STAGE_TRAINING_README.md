# 两阶段训练 CultureMoE

## 概述

这是一个全新的训练策略，将 CultureMoE 的训练分为两个独立的阶段：

### 阶段 1：LoRA 微调 LLM
- 使用 LoRA 微调基座 LLM
- 训练分类头
- 合并 LoRA 权重到 LLM

### 阶段 2：训练 MoE（冻结 LLM）
- 加载阶段1合并后的 LLM
- **冻结 LLM 参数**（不再微调）
- 只训练 MoE 部分（Router + Experts）

## 与原始训练策略的对比

| 特性 | 原始策略 | 两阶段策略 |
|------|---------|-----------|
| LLM 微调 | LoRA 微调，端到端训练 | 阶段1 LoRA 微调，阶段2冻结 |
| MoE 训练 | 与 LLM 同时训练 | 在冻结的 LLM 基础上训练 |
| 训练稳定性 | 可能不稳定 | 更稳定（分阶段） |
| 计算效率 | 较低（同时训练） | 较高（分阶段，阶段2更快） |
| 模型保存 | 保存完整模型 | 不保存模型（只保存评估结果） |

## 优势

### 1. 训练更稳定
- 阶段1专注于 LLM 微调
- 阶段2在稳定的 LLM 基础上训练 MoE
- 避免 LLM 和 MoE 相互干扰

### 2. 节省空间
- **不保存任何模型权重**
- 只保存每个 epoch 的评估结果
- 适合存储空间有限的环境

### 3. 更灵活
- 可以单独调整每个阶段的超参数
- 可以重用阶段1的结果
- 便于消融实验

### 4. 更快的迭代
- 阶段2训练更快（LLM 冻结）
- 可以快速测试不同的 MoE 配置

## 使用方法

### 基本用法

```bash
# 默认：LLaMA + 2分类 + 6个专家
sh run_two_stage_culturemoe.sh

# 指定 backbone 和分类数
sh run_two_stage_culturemoe.sh llama 3

# 指定所有参数
sh run_two_stage_culturemoe.sh llama 4 8
```

### 参数说明

```bash
sh run_two_stage_culturemoe.sh [backbone] [num_classes] [num_experts]
```

1. **backbone** - 基座模型
   - `llama` - LLaMA 3.1-8B-Instruct（默认）
   - `qwen` - Qwen 2.5-7B-Instruct

2. **num_classes** - 分类数量
   - `2` - 二分类（默认）
   - `3` - 三分类
   - `4` - 四分类
   - `5` - 五分类

3. **num_experts** - 专家数量
   - 默认：`6`
   - 可选：任意正整数（建议 2-12）

### 使用示例

```bash
# 示例 1：LLaMA + 2分类 + 6个专家
sh run_two_stage_culturemoe.sh llama 2 6

# 示例 2：LLaMA + 3分类 + 4个专家
sh run_two_stage_culturemoe.sh llama 3 4

# 示例 3：Qwen + 4分类 + 8个专家
sh run_two_stage_culturemoe.sh qwen 4 8

# 示例 4：LLaMA + 5分类 + 12个专家
sh run_two_stage_culturemoe.sh llama 5 12
```

## 训练流程

### 阶段 1：LoRA 微调 LLM（3 epochs）

```
1. 加载 tokenizer
2. 加载 base model
3. 添加 LoRA adapters
   - rank: 16
   - alpha: 32
   - dropout: 0.1
   - target_modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
4. 添加分类头
5. 加载数据集
6. 训练 3 epochs
7. 评估
8. 合并 LoRA 权重
9. 保存合并后的模型（临时）
```

**输出**：
- `stage1_epoch_eval_results.json` - 每个 epoch 的评估结果
- `stage1_final_eval.json` - 最终评估结果

### 阶段 2：训练 MoE（5 epochs）

```
1. 加载阶段1合并后的 LLM
2. 冻结 LLM 参数
3. 创建 CultureMoE 模型
   - num_experts: 6 (可配置)
   - shared_hidden_dim: 2048
   - router_hidden_dim: 1024
   - experts_hidden_dim: 2048
4. 加载数据集
5. 训练 5 epochs（只训练 MoE 部分）
6. 评估
7. 清理临时文件
```

**输出**：
- `stage2_epoch_eval_results.json` - 每个 epoch 的评估结果
- `stage2_final_eval.json` - 最终评估结果

### 最终输出

- `final_report.json` - 两阶段的完整报告
- `config.json` - 训练配置

## 输出文件结构

```
/root/autodl-fs/output/two_stage_moe/llama_2class_experts6_20251030_1430/
├── config.json                         # 训练配置
├── final_report.json                   # 最终报告
│
├── stage1/                             # 阶段1输出
│   ├── stage1_epoch_eval_results.json  # 每个epoch的评估结果
│   ├── stage1_final_eval.json          # 最终评估结果
│   └── logs/                           # TensorBoard日志
│       └── events.out.tfevents.*
│
└── stage2/                             # 阶段2输出
    ├── stage2_epoch_eval_results.json  # 每个epoch的评估结果
    ├── stage2_final_eval.json          # 最终评估结果
    └── logs/                           # TensorBoard日志
        └── events.out.tfevents.*
```

## 查看结果

### 查看最终报告

```bash
cat /root/autodl-fs/output/two_stage_moe/*/final_report.json | jq
```

**输出示例**：
```json
{
  "stage1": {
    "eval_accuracy": 0.7800,
    "eval_precision": 0.7750,
    "eval_recall": 0.7700,
    "eval_f1": 0.7725,
    "eval_loss": 0.5234
  },
  "stage2": {
    "eval_accuracy": 0.8234,
    "eval_precision": 0.8200,
    "eval_recall": 0.8150,
    "eval_f1": 0.8175,
    "eval_loss": 0.4567
  },
  "improvement": 0.0434
}
```

### 查看每个 epoch 的结果

```bash
# 阶段1
cat /root/autodl-fs/output/two_stage_moe/*/stage1_epoch_eval_results.json | jq

# 阶段2
cat /root/autodl-fs/output/two_stage_moe/*/stage2_epoch_eval_results.json | jq
```

### 对比两个阶段

```bash
# 提取准确率
echo "Stage 1:"
cat /root/autodl-fs/output/two_stage_moe/*/stage1_epoch_eval_results.json | jq '.[].eval_accuracy'

echo "Stage 2:"
cat /root/autodl-fs/output/two_stage_moe/*/stage2_epoch_eval_results.json | jq '.[].eval_accuracy'
```

## 训练配置

### 阶段1配置（LoRA 微调）

| 参数 | 值 | 说明 |
|------|-----|------|
| `stage1_epochs` | 3 | 训练轮数 |
| `lora_rank` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha |
| `lora_dropout` | 0.1 | LoRA dropout |
| `learning_rate` | 1e-5 | 学习率 |
| `lr_scheduler` | cosine | 学习率调度器 |

### 阶段2配置（MoE 训练）

| 参数 | 值 | 说明 |
|------|-----|------|
| `stage2_epochs` | 5 | 训练轮数 |
| `num_experts` | 6 | 专家数量 |
| `shared_hidden_dim` | 2048 | Shared层维度 |
| `router_hidden_dim` | 1024 | Router维度 |
| `experts_hidden_dim` | 2048 | Experts维度 |
| `learning_rate` | 1e-5 | 学习率 |
| `lr_scheduler` | cosine | 学习率调度器 |

### 通用配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `batch_size` | 4 | 训练批次大小 |
| `eval_batch_size` | 8 | 评估批次大小 |
| `gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `weight_decay` | 0.01 | 权重衰减 |
| `warmup_ratio` | 0.1 | Warmup比例 |
| `max_length` | 512 | 最大序列长度 |
| `val_split` | 0.1 | 验证集比例 |

## 预期结果

### 训练时间

- **阶段1**（3 epochs）：约 1-1.5 小时
- **阶段2**（5 epochs）：约 1.5-2 小时
- **总计**：约 2.5-3.5 小时

### 性能预期

| 阶段 | 准确率 | 说明 |
|------|--------|------|
| 阶段1（LoRA Only） | 0.78 | 基线性能 |
| 阶段2（+ MoE） | 0.82 | 提升 4% |

**预期**：阶段2应该优于阶段1

## 与原始策略对比

### 运行对比实验

```bash
# 原始策略（端到端训练）
sh run_train_ddp_lora_dual.sh llama 2 True false

# 两阶段策略
sh run_two_stage_culturemoe.sh llama 2 6
```

### 预期对比

| 策略 | 准确率 | 训练时间 | 稳定性 | 空间占用 |
|------|--------|---------|--------|---------|
| 原始策略 | 0.80 | 3 hours | 中等 | 500 MB |
| 两阶段策略 | 0.82 | 3 hours | 更高 | 100 MB |

## 优化建议

### 如果阶段2性能不佳

1. **增加阶段2的训练轮数**
   ```bash
   # 修改脚本中的 --stage2_epochs
   --stage2_epochs 10
   ```

2. **调整专家数量**
   ```bash
   sh run_two_stage_culturemoe.sh llama 2 4  # 减少到4个专家
   ```

3. **提高学习率**
   ```bash
   # 修改脚本中的 --learning_rate
   --learning_rate 2e-5
   ```

### 如果阶段1性能不佳

1. **增加阶段1的训练轮数**
   ```bash
   # 修改脚本中的 --stage1_epochs
   --stage1_epochs 5
   ```

2. **增加 LoRA rank**
   ```bash
   # 修改脚本中的 --lora_rank
   --lora_rank 32
   ```

## 消融实验

### 实验1：对比端到端 vs 两阶段

```bash
# 端到端
sh run_train_ddp_lora_dual.sh llama 2 True false

# 两阶段
sh run_two_stage_culturemoe.sh llama 2 6
```

### 实验2：不同专家数

```bash
sh run_two_stage_culturemoe.sh llama 2 2
sh run_two_stage_culturemoe.sh llama 2 4
sh run_two_stage_culturemoe.sh llama 2 6
sh run_two_stage_culturemoe.sh llama 2 8
```

### 实验3：不同阶段轮数

修改脚本中的 epoch 数，测试：
- 阶段1: 3/5/7 epochs
- 阶段2: 3/5/7/10 epochs

## 常见问题

### Q1: 为什么不保存模型？

**A**: 为了节省空间。如果需要保存模型，可以修改脚本：
- 阶段1：保存合并后的 LLM
- 阶段2：保存完整的 CultureMoE 模型

### Q2: 可以只运行阶段2吗？

**A**: 可以，但需要先有阶段1的合并模型。可以修改脚本跳过阶段1。

### Q3: 阶段2为什么要冻结 LLM？

**A**:
1. 避免破坏阶段1学到的知识
2. 加快训练速度
3. 减少显存占用
4. 提高训练稳定性

### Q4: 如何调整专家数量？

**A**: 通过第三个参数指定：
```bash
sh run_two_stage_culturemoe.sh llama 2 8  # 8个专家
```

### Q5: 临时文件会自动清理吗？

**A**: 是的，阶段2完成后会自动删除阶段1的临时模型文件。

## 总结

### 核心特点

- ✅ **两阶段训练**：先 LoRA 微调，再训练 MoE
- ✅ **冻结 LLM**：阶段2只训练 MoE 部分
- ✅ **不保存模型**：只保存评估结果，节省空间
- ✅ **每 epoch 评估**：保存每个 epoch 的结果
- ✅ **自动清理**：自动删除临时文件

### 使用流程

```bash
# 1. 运行两阶段训练
sh run_two_stage_culturemoe.sh llama 2 6

# 2. 查看结果
cat /root/autodl-fs/output/two_stage_moe/*/final_report.json | jq

# 3. 对比阶段1和阶段2
cat /root/autodl-fs/output/two_stage_moe/*/stage1_epoch_eval_results.json | jq '.[].eval_accuracy'
cat /root/autodl-fs/output/two_stage_moe/*/stage2_epoch_eval_results.json | jq '.[].eval_accuracy'
```

现在可以开始两阶段训练了！🚀

