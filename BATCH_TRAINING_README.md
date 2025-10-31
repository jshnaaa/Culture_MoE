# 批量训练和每 Epoch 评估功能

## 完成的修改

### 1. 修改训练轮数为 3 epoch

**文件**：`run_train_ddp_lora_dual.sh`

```bash
--num_train_epochs 3  # 从 5 改为 3
```

### 2. 每个 epoch 评估一次

**文件**：`src/llamafactory/train/classification/workflow.py`

**修改前**：
```python
eval_strategy=args.evaluation_strategy if val_dataset else "no"
eval_steps=args.eval_steps
```

**修改后**：
```python
eval_strategy="epoch" if val_dataset else "no"  # ✅ 每个 epoch 评估一次
# 移除 eval_steps（不再需要）
```

### 3. 保存每个 epoch 的评估结果

**新增**：`EpochEvalCallback` 类

```python
class EpochEvalCallback(TrainerCallback):
    """每个 epoch 结束后保存评估结果的回调"""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 保存到 epoch_eval_results.json
        # 打印每个 epoch 的评估结果
```

**输出文件**：`epoch_eval_results.json`

格式：
```json
[
  {
    "epoch": 1,
    "step": 100,
    "eval_loss": 1.234,
    "eval_accuracy": 0.7500,
    "eval_precision": 0.7400,
    "eval_recall": 0.7300,
    "eval_f1": 0.7350
  },
  {
    "epoch": 2,
    "step": 200,
    "eval_loss": 1.123,
    "eval_accuracy": 0.7800,
    "eval_precision": 0.7700,
    "eval_recall": 0.7600,
    "eval_f1": 0.7650
  },
  {
    "epoch": 3,
    "step": 300,
    "eval_loss": 1.089,
    "eval_accuracy": 0.7900,
    "eval_precision": 0.7850,
    "eval_recall": 0.7750,
    "eval_f1": 0.7800
  }
]
```

### 4. 批量执行脚本

**文件**：`run_batch_culturemoe.sh`

依次执行：
1. `sh run_train_ddp_lora_dual.sh qwen 2 True false`
2. `sh run_train_ddp_lora_dual.sh qwen 4 True false`
3. `sh run_train_ddp_lora_dual.sh qwen 5 True false`

## 使用方法

### 方式 1：批量执行（推荐）

```bash
sh run_batch_culturemoe.sh
```

**功能**：
- ✅ 依次执行 3 个训练任务
- ✅ 记录每个任务的执行时间
- ✅ 保存完整的批量日志
- ✅ 如果某个任务失败，停止后续任务
- ✅ 显示所有输出目录

**输出**：
```
============================================================
Batch Training: CultureMoE with Qwen
============================================================
Tasks:
  1. Qwen + 2-class + Culture Loss
  2. Qwen + 4-class + Culture Loss
  3. Qwen + 5-class + Culture Loss
============================================================

[Task 1/3] Training: Qwen + 2-class + Culture Loss
...
✅ Task 1 completed successfully!
Duration: 45 minutes 30 seconds

[Task 2/3] Training: Qwen + 4-class + Culture Loss
...
✅ Task 2 completed successfully!
Duration: 50 minutes 15 seconds

[Task 3/3] Training: Qwen + 5-class + Culture Loss
...
✅ Task 3 completed successfully!
Duration: 52 minutes 40 seconds

============================================================
Batch Training Summary
============================================================
Task 1 (Qwen 2-class): 45m 30s
Task 2 (Qwen 4-class): 50m 15s
Task 3 (Qwen 5-class): 52m 40s
------------------------------------------------------------
Total time: 148 minutes 25 seconds
============================================================
✅ All tasks completed successfully!
```

### 方式 2：单独执行

```bash
# 单独执行某个任务
sh run_train_ddp_lora_dual.sh qwen 2 True false
sh run_train_ddp_lora_dual.sh qwen 4 True false
sh run_train_ddp_lora_dual.sh qwen 5 True false
```

## 查看每个 Epoch 的评估结果

### 方法 1：查看 JSON 文件

```bash
# 查看某个训练任务的 epoch 评估结果
cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_*/epoch_eval_results.json

# 使用 jq 格式化输出
cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_*/epoch_eval_results.json | jq

# 提取特定指标
cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_*/epoch_eval_results.json | jq '.[].eval_accuracy'
```

### 方法 2：对比不同 Epoch

```bash
# 查看所有 epoch 的准确率
cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_*/epoch_eval_results.json | jq '.[] | {epoch: .epoch, accuracy: .eval_accuracy, f1: .eval_f1}'
```

**输出示例**：
```json
{
  "epoch": 1,
  "accuracy": 0.75,
  "f1": 0.735
}
{
  "epoch": 2,
  "accuracy": 0.78,
  "f1": 0.765
}
{
  "epoch": 3,
  "accuracy": 0.79,
  "f1": 0.78
}
```

### 方法 3：判断是否过拟合

```bash
# 查看 loss 和 accuracy 的变化趋势
cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_*/epoch_eval_results.json | jq '.[] | {epoch: .epoch, loss: .eval_loss, acc: .eval_accuracy}'
```

**判断标准**：
- **正常训练**：loss 下降，accuracy 上升
  ```
  Epoch 1: loss=1.234, acc=0.75
  Epoch 2: loss=1.123, acc=0.78  ✅
  Epoch 3: loss=1.089, acc=0.79  ✅
  ```

- **过拟合**：loss 上升，accuracy 下降或不变
  ```
  Epoch 1: loss=1.234, acc=0.75
  Epoch 2: loss=1.123, acc=0.78  ✅
  Epoch 3: loss=1.156, acc=0.77  ❌ 过拟合！
  ```

- **欠拟合**：loss 和 accuracy 都不再变化
  ```
  Epoch 1: loss=1.234, acc=0.75
  Epoch 2: loss=1.230, acc=0.75  ⚠️ 欠拟合
  Epoch 3: loss=1.228, acc=0.75  ⚠️ 欠拟合
  ```

## 输出文件说明

### 每个训练任务的输出目录

```
/root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_20251030_1430/
├── epoch_eval_results.json      # ✅ 每个 epoch 的评估结果（新增）
├── eval_results.json             # 最终评估结果
├── train_results.json            # 训练结果
├── trainer_state.json            # 训练状态
├── training_args.bin             # 训练参数
└── logs/                         # TensorBoard 日志
    └── events.out.tfevents.*
```

### 批量训练日志

```
/root/autodl-fs/output/batch_logs/batch_training_20251030_1430.log
```

包含所有 3 个任务的完整输出。

## 训练过程中的输出

### 每个 Epoch 结束后

```
📊 Epoch 1 Evaluation Results:
   Accuracy:  0.7500
   Precision: 0.7400
   Recall:    0.7300
   F1:        0.7350
   Loss:      1.2340
   Saved to: /root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_20251030_1430/epoch_eval_results.json

📊 Epoch 2 Evaluation Results:
   Accuracy:  0.7800
   Precision: 0.7700
   Recall:    0.7600
   F1:        0.7650
   Loss:      1.1230
   Saved to: /root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_20251030_1430/epoch_eval_results.json

📊 Epoch 3 Evaluation Results:
   Accuracy:  0.7900
   Precision: 0.7850
   Recall:    0.7750
   F1:        0.7800
   Loss:      1.0890
   Saved to: /root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_20251030_1430/epoch_eval_results.json
```

## 对比不同任务的结果

```bash
# 对比 2分类、4分类、5分类的最终结果
echo "=== 2-class ==="
cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_2class_*/epoch_eval_results.json | jq '.[-1]'

echo "=== 4-class ==="
cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_4class_*/epoch_eval_results.json | jq '.[-1]'

echo "=== 5-class ==="
cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_5class_*/epoch_eval_results.json | jq '.[-1]'
```

## 可视化训练曲线

### 使用 TensorBoard

```bash
tensorboard --logdir /root/autodl-fs/output/CultureMoE/
```

### 使用 Python 绘图

```python
import json
import matplotlib.pyplot as plt

# 读取数据
with open('epoch_eval_results.json') as f:
    data = json.load(f)

epochs = [d['epoch'] for d in data]
accuracy = [d['eval_accuracy'] for d in data]
loss = [d['eval_loss'] for d in data]

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, accuracy, marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs Epoch')
ax1.grid(True)

ax2.plot(epochs, loss, marker='o', color='red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss vs Epoch')
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
```

## 常见问题

### Q1: 如何判断是否过拟合？

**A**: 查看 `epoch_eval_results.json`：
- 如果 loss 上升，accuracy 下降 → 过拟合
- 如果 loss 和 accuracy 都不再变化 → 欠拟合
- 如果 loss 下降，accuracy 上升 → 正常训练

### Q2: 如果发现过拟合怎么办？

**A**:
1. 减少训练轮数（从 3 改为 2）
2. 增加 dropout（从 0.1 改为 0.15）
3. 增加 weight decay（从 0.01 改为 0.05）

### Q3: 批量训练中途失败怎么办？

**A**:
- 脚本会自动停止后续任务
- 查看批量日志：`cat /root/autodl-fs/output/batch_logs/batch_training_*.log`
- 从失败的任务开始重新执行

### Q4: 如何只执行某几个任务？

**A**: 编辑 `run_batch_culturemoe.sh`，注释掉不需要的任务。

## 总结

### 关键改进

1. ✅ **训练轮数改为 3 epoch** - 防止过拟合
2. ✅ **每个 epoch 评估一次** - 及时发现过拟合
3. ✅ **保存每个 epoch 的评估结果** - 方便分析训练过程
4. ✅ **批量执行脚本** - 自动依次执行多个任务

### 使用流程

```bash
# 1. 运行批量训练
sh run_batch_culturemoe.sh

# 2. 查看每个 epoch 的评估结果
cat /root/autodl-fs/output/CultureMoE/culturemoe_qwen_*class_*/epoch_eval_results.json | jq

# 3. 判断是否过拟合
# 如果 epoch 3 的 loss 比 epoch 2 高 → 过拟合

# 4. 对比不同任务的结果
# 查看 2分类、4分类、5分类的最终性能
```

现在可以轻松判断模型是否过拟合了！🎉

