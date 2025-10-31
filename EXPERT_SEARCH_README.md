# CultureMoE 最佳专家数搜索

## 概述

这个脚本自动搜索 CultureMoE 模型的最佳专家数（num_experts），通过三个阶段的实验找到最优配置。

## 脚本功能

### 阶段 1：粗粒度搜索
- **测试专家数**：2, 4, 6, 8, 12
- **训练轮数**：5 epochs
- **数据集**：CulturalBench（2分类）
- **保存模型**：否（只保存评估结果）
- **输出**：每个专家数下每个 epoch 的评估结果

### 阶段 2：细粒度搜索
- **测试专家数**：在阶段1最佳值附近（±1）
- **训练轮数**：5 epochs
- **其他配置**：与阶段1相同

### 阶段 3：最终验证
- **专家数**：使用搜索到的最佳值
- **训练轮数**：10 epochs
- **输出**：完整的训练曲线和评估结果

## 使用方法

### 运行搜索

```bash
sh search_best_num_experts.sh
```

### 预计时间

- **阶段 1**：5 个专家数 × 5 epochs ≈ 4-5 小时
- **阶段 2**：1-2 个专家数 × 5 epochs ≈ 1-2 小时
- **阶段 3**：1 个专家数 × 10 epochs ≈ 2 小时
- **总计**：约 7-9 小时

## 输出文件

### 目录结构

```
/root/autodl-fs/output/expert_search_20251030_1430/
├── search_log.txt                          # 完整的搜索日志
├── final_report.json                       # 最终报告（JSON格式）
├── phase1_results.json                     # 阶段1结果汇总
│
├── phase1_experts_2/                       # 阶段1：2个专家
│   └── epoch_eval_results.json             # 每个epoch的评估结果
├── phase1_experts_4/                       # 阶段1：4个专家
│   └── epoch_eval_results.json
├── phase1_experts_6/                       # 阶段1：6个专家
│   └── epoch_eval_results.json
├── phase1_experts_8/                       # 阶段1：8个专家
│   └── epoch_eval_results.json
├── phase1_experts_12/                      # 阶段1：12个专家
│   └── epoch_eval_results.json
│
├── phase2_experts_5/                       # 阶段2：细粒度搜索
│   └── epoch_eval_results.json             # （如果最佳值是6，则测试5和7）
├── phase2_experts_7/
│   └── epoch_eval_results.json
│
└── final_model_experts_6/                  # 阶段3：最终模型
    └── epoch_eval_results.json             # 10个epoch的评估结果
```

### 关键文件说明

#### 1. `search_log.txt`

完整的搜索日志，包含：
- 每个阶段的开始/结束时间
- 每个专家数的训练输出
- 每个专家数的最佳准确率
- 最终的最佳专家数

#### 2. `final_report.json`

最终报告，格式：
```json
{
  "search_timestamp": "20251030_1430",
  "best_num_experts": 6,
  "best_accuracy_5epochs": 0.8234,
  "final_model": {
    "num_experts": 6,
    "epochs": 10,
    "best_accuracy": 0.8456,
    "best_epoch": 7,
    "all_epochs": [
      {
        "epoch": 1,
        "eval_accuracy": 0.7800,
        "eval_precision": 0.7750,
        "eval_recall": 0.7700,
        "eval_f1": 0.7725,
        "eval_loss": 0.5234
      },
      ...
    ]
  },
  "search_results": {
    "2": {
      "phase": "coarse",
      "best_accuracy": 0.7800,
      "best_epoch": 4
    },
    "4": {
      "phase": "coarse",
      "best_accuracy": 0.8100,
      "best_epoch": 5
    },
    "6": {
      "phase": "coarse",
      "best_accuracy": 0.8234,
      "best_epoch": 4
    },
    ...
  },
  "output_directory": "/root/autodl-fs/output/expert_search_20251030_1430"
}
```

#### 3. `epoch_eval_results.json`

每个专家数的每个 epoch 评估结果：
```json
[
  {
    "epoch": 1,
    "step": 100,
    "eval_loss": 0.6234,
    "eval_accuracy": 0.7500,
    "eval_precision": 0.7400,
    "eval_recall": 0.7300,
    "eval_f1": 0.7350
  },
  {
    "epoch": 2,
    "step": 200,
    "eval_loss": 0.5456,
    "eval_accuracy": 0.7800,
    "eval_precision": 0.7750,
    "eval_recall": 0.7650,
    "eval_f1": 0.7700
  },
  ...
]
```

## 查看结果

### 查看最终报告

```bash
# 查看完整报告
cat /root/autodl-fs/output/expert_search_*/final_report.json | jq

# 只查看最佳专家数
cat /root/autodl-fs/output/expert_search_*/final_report.json | jq '.best_num_experts'

# 查看所有专家数的结果
cat /root/autodl-fs/output/expert_search_*/final_report.json | jq '.search_results'
```

### 查看搜索日志

```bash
# 查看完整日志
cat /root/autodl-fs/output/expert_search_*/search_log.txt

# 只查看结果摘要
grep "Best accuracy" /root/autodl-fs/output/expert_search_*/search_log.txt
```

### 查看某个专家数的详细结果

```bash
# 查看 6 个专家的每个 epoch 结果
cat /root/autodl-fs/output/expert_search_*/phase1_experts_6/epoch_eval_results.json | jq

# 提取准确率
cat /root/autodl-fs/output/expert_search_*/phase1_experts_6/epoch_eval_results.json | jq '.[].eval_accuracy'
```

### 对比不同专家数

```bash
# 对比所有专家数的最佳准确率
for dir in /root/autodl-fs/output/expert_search_*/phase1_experts_*/; do
    num=$(basename $dir | grep -o '[0-9]*')
    acc=$(cat $dir/epoch_eval_results.json | jq 'max_by(.eval_accuracy).eval_accuracy')
    echo "Experts: $num, Best Accuracy: $acc"
done
```

## 结果分析

### 示例输出

```
============================================================
Phase 1 Results Summary
============================================================
Num Experts | Best Accuracy | Best Epoch
------------|---------------|------------
          2 |        0.7800 |          4
          4 |        0.8100 |          5
          6 |        0.8234 |          4
          8 |        0.8150 |          5
         12 |        0.7950 |          3
============================================================

🏆 Best num_experts from Phase 1: 6 (accuracy: 0.8234)
```

### 解读

1. **单峰曲线**：如果准确率先升后降，说明存在最优专家数
2. **递增曲线**：如果准确率持续上升，可能需要测试更多专家
3. **递减曲线**：如果准确率持续下降，说明专家数越少越好

### 可视化

使用 Python 绘制曲线：

```python
import json
import matplotlib.pyplot as plt

# 读取报告
with open('final_report.json') as f:
    report = json.load(f)

# 提取数据
experts = []
accuracies = []
for num_experts, result in sorted(report['search_results'].items(), key=lambda x: int(x[0])):
    experts.append(int(num_experts))
    accuracies.append(result['best_accuracy'])

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(experts, accuracies, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Experts', fontsize=12)
plt.ylabel('Best Accuracy', fontsize=12)
plt.title('CultureMoE: Accuracy vs Number of Experts', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(experts)

# 标注最佳点
best_idx = accuracies.index(max(accuracies))
plt.scatter(experts[best_idx], accuracies[best_idx],
            color='red', s=200, zorder=5, label='Best')
plt.legend()

plt.tight_layout()
plt.savefig('expert_search_results.png', dpi=300)
print("Plot saved to: expert_search_results.png")
```

## 常见问题

### Q1: 搜索过程中断了怎么办？

**A**: 脚本会在每个阶段保存结果。可以：
1. 查看已完成的结果
2. 手动运行未完成的部分
3. 或者重新运行整个脚本

### Q2: 如何只运行某个阶段？

**A**: 编辑脚本，注释掉不需要的阶段。

### Q3: 如何修改搜索范围？

**A**: 编辑脚本中的 `COARSE_EXPERTS` 数组：
```bash
COARSE_EXPERTS=(2 4 6 8 12)  # 修改为你想要的值
```

### Q4: 如何使用不同的数据集？

**A**: 修改脚本中的配置：
```bash
NUM_CLASSES=4  # 改为 3, 4, 或 5
TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge_4.json"  # 改为对应数据集
```

### Q5: 最终模型的 epoch 数可以修改吗？

**A**: 可以，修改阶段3中的 `--num_train_epochs 10` 为你想要的值。

## 预期结果

### 假设 1：专家数 = 分类数最优

如果 2 分类任务，可能 2-3 个专家最好。

### 假设 2：专家数 4-6 最优

基于经验，4-6 个专家通常是一个好的平衡点。

### 假设 3：专家数越多越好

如果数据量足够大，更多专家可能带来更好的性能。

## 后续步骤

找到最佳专家数后：

1. **更新配置**：在 `run_train_ddp_lora_dual.sh` 中设置最佳专家数
2. **测试其他数据集**：在 3/4/5 分类任务上验证
3. **对比 LoRA Only**：确认 CultureMoE 确实优于 LoRA Only
4. **消融实验**：测试文化损失的影响

## 总结

这个脚本会：
1. ✅ 自动搜索最佳专家数（粗粒度 + 细粒度）
2. ✅ 保存每个专家数的每个 epoch 评估结果
3. ✅ 使用最佳专家数训练完整模型（10 epochs）
4. ✅ 生成详细的报告和日志
5. ✅ 输出最佳专家数和对应的准确率

运行后，你将得到：
- 📊 最佳专家数
- 📈 准确率 vs 专家数曲线
- 📁 所有实验的详细结果
- 🏆 使用最佳配置的完整模型

开始搜索吧！🚀

