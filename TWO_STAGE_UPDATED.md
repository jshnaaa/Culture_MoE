# 两阶段训练 CultureMoE - 更新说明

## ✅ 最新更新（选择最佳 Checkpoint）

### 核心改进

现在阶段1会：
1. ✅ 保存每个 epoch 的 checkpoint
2. ✅ 跟踪每个 epoch 的评估准确率
3. ✅ **自动选择准确率最高的 epoch**
4. ✅ 加载最佳 checkpoint 进入阶段2
5. ✅ 自动清理 checkpoints（如果不保存模型）

### 工作流程

```
阶段1：LoRA 微调
├── Epoch 1 → 评估 → 保存 checkpoint → 准确率: 0.75
├── Epoch 2 → 评估 → 保存 checkpoint → 准确率: 0.78 🏆 (最佳)
├── Epoch 3 → 评估 → 保存 checkpoint → 准确率: 0.76
└── 选择 Epoch 2 的 checkpoint → 合并权重 → 进入阶段2

阶段2：训练 MoE（使用阶段1最佳模型）
├── 加载阶段1最佳模型（Epoch 2）
├── 冻结 LLM
└── 训练 MoE
```

## 🚀 使用方法

### 命令格式（不变）

```bash
sh run_two_stage_culturemoe.sh [backbone] [num_classes] [use_culture_loss] [save_model] [num_experts]
```

### 使用示例

```bash
# 默认配置
sh run_two_stage_culturemoe.sh llama 2 True false

# 自定义配置
sh run_two_stage_culturemoe.sh llama 3 True false 4
```

## 📊 输出文件

### 新增文件

```
output_dir/
├── stage1_epoch_eval_results.json      # 阶段1每个epoch的评估结果
├── stage1_best_eval.json               # 阶段1最佳epoch的评估结果 ⭐ 新增
├── stage2_epoch_eval_results.json      # 阶段2每个epoch的评估结果
├── stage2_final_eval.json              # 阶段2最终评估结果
└── final_report.json                   # 完整报告
```

### 查看最佳 Epoch

```bash
# 查看阶段1所有epoch的结果
cat /root/autodl-fs/output/two_stage_moe/*/stage1_epoch_eval_results.json | jq

# 查看阶段1最佳epoch的结果
cat /root/autodl-fs/output/two_stage_moe/*/stage1_best_eval.json | jq

# 提取最佳epoch编号
cat /root/autodl-fs/output/two_stage_moe/*/stage1_epoch_eval_results.json | jq 'max_by(.eval_accuracy) | .epoch'
```

## 💡 核心特点

### 1. 自动选择最佳模型

- ✅ 不再使用最后一个 epoch 的模型
- ✅ 自动选择准确率最高的 epoch
- ✅ 避免过拟合问题

### 2. 节省空间

- ✅ 训练时保存 checkpoints
- ✅ 训练后自动清理（如果 `save_model=false`）
- ✅ 只保留评估结果

### 3. 完整记录

- ✅ 记录每个 epoch 的评估结果
- ✅ 标记最佳 epoch
- ✅ 保存最佳模型的评估结果

## 📈 预期效果

### 示例输出

```
阶段1训练：
📊 [STAGE1] Epoch 1 Evaluation Results:
   Accuracy:  0.7500

📊 [STAGE1] Epoch 2 Evaluation Results:
   Accuracy:  0.7800
   🏆 New best accuracy!

📊 [STAGE1] Epoch 3 Evaluation Results:
   Accuracy:  0.7600

9. Finding best checkpoint...
   🏆 Best epoch: 2
   🏆 Best accuracy: 0.7800
   Loading best checkpoint from: .../checkpoint-xxx

10. Loading best checkpoint...
   ✅ Best checkpoint loaded
   ✅ Best classification head loaded

11. Evaluating best model...

Stage 1 Completed!
   Best Epoch:     2
   Best Accuracy:  0.7800
   Best F1:        0.7750
```

## 🔍 对比：旧版 vs 新版

| 特性 | 旧版 | 新版 |
|------|------|------|
| 进入阶段2的模型 | 最后一个 epoch | 准确率最高的 epoch ⭐ |
| Checkpoint 保存 | 不保存 | 保存所有 epoch |
| 最佳 epoch 跟踪 | 无 | 自动跟踪 ⭐ |
| Checkpoint 清理 | N/A | 自动清理 |
| 评估结果 | 只有最终结果 | 每个 epoch + 最佳 epoch ⭐ |

## ⚙️ 技术细节

### 最佳 Checkpoint 选择逻辑

```python
# 1. 训练时跟踪最佳准确率
class EpochEvalCallback:
    def on_evaluate(self, metrics):
        if metrics['eval_accuracy'] > self.best_accuracy:
            self.best_accuracy = metrics['eval_accuracy']
            self.best_epoch = current_epoch

# 2. 训练后加载最佳 checkpoint
best_checkpoint_dir = f"checkpoint-{best_epoch}"
best_model.load_state_dict(checkpoint)

# 3. 合并权重
merged_model = best_model.merge_and_unload()

# 4. 进入阶段2
stage2_train_moe(merged_model)
```

### Checkpoint 清理

```python
# 如果不保存模型，自动清理 checkpoints
if not args.save_model:
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)
```

## 🎯 使用建议

### 1. 快速实验

```bash
# 不保存模型，自动清理 checkpoints
sh run_two_stage_culturemoe.sh llama 2 True false
```

### 2. 保存最佳模型

```bash
# 保存模型，保留 checkpoints
sh run_two_stage_culturemoe.sh llama 2 True true
```

### 3. 调整阶段1轮数

如果发现最佳 epoch 总是在最后，可以增加阶段1轮数：

```bash
# 编辑 run_two_stage_culturemoe.sh
--stage1_epochs 5  # 从 3 增加到 5
```

## 📝 常见问题

### Q1: 如何查看哪个 epoch 是最佳的？

**A**: 查看日志或 `stage1_epoch_eval_results.json`：
```bash
cat output/stage1_epoch_eval_results.json | jq 'max_by(.eval_accuracy)'
```

### Q2: 最佳 epoch 总是最后一个怎么办？

**A**: 说明模型还在提升，建议增加阶段1的训练轮数。

### Q3: Checkpoints 会占用多少空间？

**A**:
- 训练时：每个 checkpoint 约 200-300 MB
- 训练后：如果 `save_model=false`，自动清理，只保留评估结果

### Q4: 可以手动指定使用哪个 epoch 吗？

**A**: 目前自动选择最佳 epoch。如需手动指定，可以修改代码中的 `best_epoch` 变量。

## ✅ 总结

### 核心改进

1. ✅ **自动选择最佳 epoch**：不再使用最后一个 epoch
2. ✅ **完整记录**：保存每个 epoch 的评估结果
3. ✅ **自动清理**：训练后自动删除 checkpoints
4. ✅ **更好的性能**：使用最佳模型进入阶段2

### 使用流程

```bash
# 1. 运行训练
sh run_two_stage_culturemoe.sh llama 2 True false

# 2. 查看最佳 epoch
cat output/stage1_best_eval.json | jq

# 3. 查看最终结果
cat output/final_report.json | jq
```

现在阶段1会自动选择最佳 epoch 进入阶段2！🎉

