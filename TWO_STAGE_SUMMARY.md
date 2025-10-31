# 两阶段训练 CultureMoE - 完整总结

## 🎯 核心思想

将 CultureMoE 的训练分为两个独立阶段：

1. **阶段 1**：LoRA 微调 LLM → 合并权重
2. **阶段 2**：冻结 LLM → 训练 MoE

## 📦 创建的文件

### 1. 核心脚本

**`train_two_stage_culturemoe.py`** - Python 训练脚本
- 实现两阶段训练逻辑
- 自动合并 LoRA 权重
- 自动清理临时文件
- 保存每个 epoch 的评估结果

**`run_two_stage_culturemoe.sh`** - Shell 启动脚本
- 一键启动两阶段训练
- 支持参数配置
- 自动选择数据集和模型

**`compare_training_strategies.sh`** - 策略对比脚本
- 自动运行两种策略
- 生成对比报告
- 输出性能差异

### 2. 文档

**`TWO_STAGE_TRAINING_README.md`** - 详细使用文档
**`TWO_STAGE_SUMMARY.md`** - 总结文档（本文件）

## 🚀 使用方法

### 快速开始

```bash
# 默认配置（LLaMA + 2分类 + 6个专家）
sh run_two_stage_culturemoe.sh

# 自定义配置
sh run_two_stage_culturemoe.sh llama 3 4  # LLaMA + 3分类 + 4个专家
sh run_two_stage_culturemoe.sh qwen 4 8   # Qwen + 4分类 + 8个专家
```

### 对比两种策略

```bash
# 自动对比原始策略 vs 两阶段策略
sh compare_training_strategies.sh llama 2 6
```

## 📊 训练流程

### 阶段 1：LoRA 微调（3 epochs）

```
加载模型 → 添加 LoRA → 训练 → 合并权重 → 保存临时模型
```

**特点**：
- 使用 LoRA 微调 LLM
- Rank=16, Alpha=32, Dropout=0.1
- 训练分类头
- 合并 LoRA 权重到 LLM

**输出**：
- `stage1_epoch_eval_results.json`
- `stage1_final_eval.json`

### 阶段 2：训练 MoE（5 epochs）

```
加载合并模型 → 冻结 LLM → 创建 MoE → 训练 → 清理临时文件
```

**特点**：
- 加载阶段1的合并模型
- **冻结 LLM 参数**
- 只训练 MoE 部分
- 更快、更稳定

**输出**：
- `stage2_epoch_eval_results.json`
- `stage2_final_eval.json`

### 最终输出

- `final_report.json` - 完整报告
- `config.json` - 训练配置

## 💡 核心优势

### 1. 训练更稳定

| 原始策略 | 两阶段策略 |
|---------|-----------|
| LLM 和 MoE 同时训练 | 分阶段训练 |
| 可能相互干扰 | 互不干扰 |
| 训练不稳定 | 训练稳定 |

### 2. 节省空间

| 原始策略 | 两阶段策略 |
|---------|-----------|
| 保存完整模型 | 不保存模型 |
| ~500 MB | ~100 MB |
| 需要大量空间 | 适合空间有限 |

### 3. 更灵活

- 可以单独调整每个阶段的超参数
- 可以重用阶段1的结果
- 便于消融实验

### 4. 更快的迭代

- 阶段2训练更快（LLM 冻结）
- 可以快速测试不同的 MoE 配置
- 便于专家数搜索

## 📈 预期结果

### 性能对比

| 策略 | 阶段1准确率 | 最终准确率 | 训练时间 |
|------|-----------|-----------|---------|
| 原始策略 | N/A | 0.80 | 3 hours |
| 两阶段策略 | 0.78 | 0.82 | 3 hours |

**预期**：两阶段策略应该优于原始策略

### 训练时间

- **阶段1**：1-1.5 小时
- **阶段2**：1.5-2 小时
- **总计**：2.5-3.5 小时

## 🔍 查看结果

### 查看最终报告

```bash
cat /root/autodl-fs/output/two_stage_moe/*/final_report.json | jq
```

**输出示例**：
```json
{
  "stage1": {
    "eval_accuracy": 0.7800,
    "eval_f1": 0.7725
  },
  "stage2": {
    "eval_accuracy": 0.8234,
    "eval_f1": 0.8175
  },
  "improvement": 0.0434
}
```

### 查看每个 epoch

```bash
# 阶段1
cat /root/autodl-fs/output/two_stage_moe/*/stage1_epoch_eval_results.json | jq '.[].eval_accuracy'

# 阶段2
cat /root/autodl-fs/output/two_stage_moe/*/stage2_epoch_eval_results.json | jq '.[].eval_accuracy'
```

### 查看对比报告

```bash
cat /root/autodl-fs/output/strategy_comparison_*/comparison_report.json | jq
```

## 🎨 使用场景

### 场景 1：空间有限

```bash
# 不保存模型，只保存评估结果
sh run_two_stage_culturemoe.sh llama 2 6
```

### 场景 2：快速实验

```bash
# 测试不同专家数
sh run_two_stage_culturemoe.sh llama 2 2
sh run_two_stage_culturemoe.sh llama 2 4
sh run_two_stage_culturemoe.sh llama 2 6
sh run_two_stage_culturemoe.sh llama 2 8
```

### 场景 3：策略对比

```bash
# 自动对比两种策略
sh compare_training_strategies.sh llama 2 6
```

### 场景 4：消融实验

```bash
# 测试不同配置
sh run_two_stage_culturemoe.sh llama 2 4  # 4个专家
sh run_two_stage_culturemoe.sh llama 3 4  # 3分类
sh run_two_stage_culturemoe.sh qwen 2 4   # Qwen模型
```

## 🔧 参数调优

### 调整阶段1（LoRA 微调）

编辑 `run_two_stage_culturemoe.sh`：

```bash
--stage1_epochs 5      # 增加训练轮数
--lora_rank 32         # 增加 LoRA rank
--learning_rate 2e-5   # 提高学习率
```

### 调整阶段2（MoE 训练）

```bash
--stage2_epochs 10     # 增加训练轮数
--num_experts 8        # 增加专家数
--learning_rate 2e-5   # 提高学习率
```

## 📝 与原始策略的详细对比

### 训练方式

| 特性 | 原始策略 | 两阶段策略 |
|------|---------|-----------|
| LLM 微调 | LoRA，端到端 | 阶段1 LoRA，阶段2冻结 |
| MoE 训练 | 与 LLM 同时 | 在冻结 LLM 上训练 |
| 权重合并 | 训练结束后 | 阶段1结束后 |
| 训练稳定性 | 中等 | 更高 |

### 资源占用

| 资源 | 原始策略 | 两阶段策略 |
|------|---------|-----------|
| 显存 | 较高 | 较低（阶段2） |
| 磁盘 | 500 MB | 100 MB |
| 训练时间 | 3 hours | 3 hours |

### 灵活性

| 特性 | 原始策略 | 两阶段策略 |
|------|---------|-----------|
| 超参数调整 | 一次性 | 分阶段调整 |
| 重用结果 | 不可 | 可重用阶段1 |
| 消融实验 | 困难 | 容易 |

## 🎯 最佳实践

### 1. 首次使用

```bash
# 使用默认配置
sh run_two_stage_culturemoe.sh
```

### 2. 寻找最佳专家数

```bash
# 测试不同专家数
for num_experts in 2 4 6 8; do
    sh run_two_stage_culturemoe.sh llama 2 $num_experts
done
```

### 3. 对比策略

```bash
# 运行对比实验
sh compare_training_strategies.sh llama 2 6
```

### 4. 生产环境

```bash
# 使用最佳配置
sh run_two_stage_culturemoe.sh llama 2 4  # 假设4个专家最好
```

## 🐛 故障排除

### 问题1：阶段2性能不如阶段1

**原因**：MoE 训练不足

**解决**：
```bash
# 增加阶段2训练轮数
--stage2_epochs 10
```

### 问题2：显存不足

**原因**：批次大小太大

**解决**：
```bash
# 减小批次大小
--batch_size 2
--gradient_accumulation_steps 16
```

### 问题3：训练太慢

**原因**：数据加载慢

**解决**：
```bash
# 增加数据加载线程
--num_workers 8
```

## 📚 相关文档

- `TWO_STAGE_TRAINING_README.md` - 详细使用文档
- `CULTUREMOE_OPTIMIZATION.md` - 参数优化说明
- `EXPERT_SEARCH_README.md` - 专家数搜索
- `BATCH_TRAINING_README.md` - 批量训练

## ✅ 总结

### 创建的文件

1. ✅ `train_two_stage_culturemoe.py` - Python 训练脚本
2. ✅ `run_two_stage_culturemoe.sh` - Shell 启动脚本
3. ✅ `compare_training_strategies.sh` - 策略对比脚本
4. ✅ `TWO_STAGE_TRAINING_README.md` - 详细文档
5. ✅ `TWO_STAGE_SUMMARY.md` - 总结文档

### 核心特点

- ✅ **两阶段训练**：LoRA 微调 → 冻结 LLM → 训练 MoE
- ✅ **不保存模型**：只保存评估结果，节省空间
- ✅ **每 epoch 评估**：保存每个 epoch 的结果
- ✅ **自动清理**：自动删除临时文件
- ✅ **策略对比**：自动对比两种策略

### 使用流程

```bash
# 1. 运行两阶段训练
sh run_two_stage_culturemoe.sh llama 2 6

# 2. 查看结果
cat /root/autodl-fs/output/two_stage_moe/*/final_report.json | jq

# 3. 对比策略（可选）
sh compare_training_strategies.sh llama 2 6
```

现在你有了一个完整的两阶段训练系统！🎉

