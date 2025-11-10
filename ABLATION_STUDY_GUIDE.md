# CultureMoE 消融实验指南

## 📋 概述

为了验证 CultureMoE 模型中各个组件的有效性，我们添加了两个关键的消融实验参数：

1. **`culture_loss_weight`** - 文化专注性损失的权重
2. **`use_shared`** - 是否使用共享专家层

---

## 🎯 参数说明

### 1. `culture_loss_weight` (文化损失权重)

**含义**：控制文化专注性损失在总损失中的比例

**默认值**：`0.5`

**取值范围**：`0.0 - 1.0`

**说明**：
- `0.0` - 不使用文化损失（只有生成损失）
- `0.1` - 文化损失权重较低（10%）
- `0.5` - 文化损失权重中等（50%，默认）
- `1.0` - 文化损失权重较高（100%）

**作用**：
- 验证文化损失对模型性能的影响
- 找到最优的文化损失权重
- 理解文化对齐的重要性

### 2. `use_shared` (是否使用共享专家)

**含义**：控制是否使用共享专家层

**默认值**：`true`

**取值**：`true` 或 `false`

**说明**：
- `true` - 使用共享专家层（完整的 CultureMoE 模型）
- `false` - 不使用共享专家层（只有路由专家）

**作用**：
- 验证共享专家层的必要性
- 理解共享知识与文化特定知识的平衡
- 评估模型复杂度与性能的权衡

---

## 🚀 使用方法

### 基础训练（默认参数）

```bash
# 使用所有默认参数
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true

# 等价于
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 true
```

### 消融实验 1：调整文化损失权重

```bash
# 实验 1a：不使用文化损失
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.0 true

# 实验 1b：低文化损失权重
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.1 true

# 实验 1c：中等文化损失权重（默认）
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 true

# 实验 1d：高文化损失权重
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.8 true

# 实验 1e：仅使用文化损失
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 1.0 true
```

### 消融实验 2：是否使用共享专家

```bash
# 实验 2a：使用共享专家（默认）
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 true

# 实验 2b：不使用共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 false
```

### 消融实验 3：组合实验

```bash
# 实验 3a：低文化损失 + 无共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.1 false

# 实验 3b：高文化损失 + 无共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.8 false

# 实验 3c：无文化损失 + 无共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.0 false
```

---

## 📊 实验设计

### 完整的消融实验矩阵

| 实验编号 | culture_loss_weight | use_shared | 说明 |
|---------|-------------------|-----------|------|
| Exp-1 | 0.0 | true | 无文化损失 + 共享专家 |
| Exp-2 | 0.1 | true | 低文化损失 + 共享专家 |
| Exp-3 | 0.5 | true | 中等文化损失 + 共享专家（默认） |
| Exp-4 | 0.8 | true | 高文化损失 + 共享专家 |
| Exp-5 | 1.0 | true | 仅文化损失 + 共享专家 |
| Exp-6 | 0.0 | false | 无文化损失 + 无共享专家 |
| Exp-7 | 0.1 | false | 低文化损失 + 无共享专家 |
| Exp-8 | 0.5 | false | 中等文化损失 + 无共享专家 |
| Exp-9 | 0.8 | false | 高文化损失 + 无共享专家 |
| Exp-10 | 1.0 | false | 仅文化损失 + 无共享专家 |

---

## 📈 预期结果

### 文化损失权重的影响

```
准确率 (%)
    |
100 |                    ╱╲
    |                  ╱    ╲
 90 |                ╱        ╲
    |              ╱            ╲
 80 |            ╱                ╲
    |          ╱                    ╲
 70 |        ╱                        ╲
    |      ╱                            ╲
 60 |    ╱                                ╲
    |  ╱                                    ╲
 50 |_______________________________________|
    0.0   0.2   0.4   0.6   0.8   1.0
         culture_loss_weight

预期：存在最优的文化损失权重（通常在 0.3-0.7 之间）
```

### 共享专家的影响

```
准确率 (%)
    |
100 |  ┌─────────────────────────────────┐
    |  │ 使用共享专家                      │
 90 |  │ (use_shared=true)                │
    |  │                                 │
 80 |  │                                 │
    |  │                                 │
 70 |  ├─────────────────────────────────┤
    |  │ 不使用共享专家                    │
 60 |  │ (use_shared=false)               │
    |  │                                 │
 50 |  └─────────────────────────────────┘
    0.0   0.2   0.4   0.6   0.8   1.0
         culture_loss_weight

预期：共享专家应该提升性能（约 5-10%）
```

---

## 🔍 分析指标

### 关键指标

1. **准确率 (Accuracy)**
   - 生成式任务的主要评估指标
   - 越高越好

2. **F1 分数**
   - 综合考虑精确率和召回率
   - 越高越好

3. **欧式距离 (Euclidean Distance)**
   - 用于 VSM13 评估
   - 越小越好（更接近霍夫斯泰德标准分数）

4. **训练损失**
   - 应该稳定下降
   - 反映模型学习进度

5. **文化损失**
   - 反映文化对齐程度
   - 应该随着训练逐步下降

### 监控命令

```bash
# 查看训练结果
python -c "
import json

# 加载结果
with open('/path/to/epoch_eval_results.json') as f:
    results = json.load(f)

# 打印关键指标
for epoch in results:
    print(f\"Epoch {epoch['epoch']}:\")
    print(f\"  Train Loss: {epoch['train_loss']:.4f}\")
    print(f\"  Eval Accuracy: {epoch['eval_accuracy']:.4f}\")
    print(f\"  Eval F1: {epoch['eval_f1']:.4f}\")
    print()
"

# 比较不同实验的结果
python -c "
import json
import os

experiments = {
    'Exp-1': '/path/to/exp1/epoch_eval_results.json',
    'Exp-3': '/path/to/exp3/epoch_eval_results.json',
    'Exp-5': '/path/to/exp5/epoch_eval_results.json',
}

for name, path in experiments.items():
    with open(path) as f:
        results = json.load(f)
        final = results[-1]  # 最后一个 epoch
        print(f'{name}: Accuracy={final[\"eval_accuracy\"]:.4f}, F1={final[\"eval_f1\"]:.4f}')
"
```

---

## 📝 实验记录模板

```
实验编号：Exp-3
实验日期：2025-11-10
实验名称：中等文化损失 + 共享专家

参数设置：
- culture_loss_weight: 0.5
- use_shared: true
- backbone: llama
- num_experts: 6
- 其他参数：默认

结果：
- 最终准确率：68.5%
- 最终 F1 分数：0.685
- 最终欧式距离：38.2
- 训练时间：2.5 小时

观察：
- 训练损失稳定下降
- 文化损失逐步减小
- 模型收敛良好

对比：
- vs Exp-1 (无文化损失)：+5.2%
- vs Exp-5 (仅文化损失)：-2.1%

结论：
- 文化损失对性能有正面影响
- 中等权重（0.5）是较好的选择
- 共享专家有助于提升性能
```

---

## 🎯 实验建议

### 第一阶段：文化损失权重优化

1. 运行 Exp-1, Exp-2, Exp-3, Exp-4, Exp-5
2. 比较准确率和 F1 分数
3. 找到最优的文化损失权重

### 第二阶段：共享专家评估

1. 运行 Exp-3 和 Exp-8（最优权重的两个版本）
2. 比较性能差异
3. 评估共享专家的必要性

### 第三阶段：深度分析

1. 分析不同实验的路由分布
2. 检查专家利用率
3. 理解文化对齐的机制

---

## 💡 常见问题

### Q1: 为什么要做消融实验？

**A**: 消融实验可以帮助我们：
- 验证每个组件的有效性
- 找到最优的超参数
- 理解模型的工作机制
- 为论文提供有力的证据

### Q2: 应该运行多少个实验？

**A**: 建议至少运行：
- 基础实验：Exp-1, Exp-3, Exp-5（文化损失权重）
- 对比实验：Exp-3, Exp-8（共享专家）
- 总共：4-5 个实验

### Q3: 如何快速运行实验？

**A**:
- 使用较小的 batch size（4 → 2）
- 减少 epoch 数（30 → 10）
- 使用更少的数据进行初步测试

### Q4: 实验结果不符合预期怎么办？

**A**:
- 检查参数是否正确传递
- 查看训练日志中的损失值
- 确保数据加载正确
- 尝试调整学习率

---

## 📊 结果汇总表

创建一个 Excel 或 CSV 文件来汇总所有实验结果：

```csv
Experiment,culture_loss_weight,use_shared,Final_Accuracy,Final_F1,Final_Distance,Training_Time
Exp-1,0.0,true,0.6234,0.6145,42.3,2.5h
Exp-2,0.1,true,0.6512,0.6423,40.1,2.5h
Exp-3,0.5,true,0.6850,0.6785,38.2,2.5h
Exp-4,0.8,true,0.6723,0.6654,39.1,2.5h
Exp-5,1.0,true,0.6645,0.6576,39.8,2.5h
Exp-6,0.0,false,0.5890,0.5801,45.2,2.3h
Exp-7,0.1,false,0.6123,0.6034,43.5,2.3h
Exp-8,0.5,false,0.6234,0.6145,42.3,2.3h
Exp-9,0.8,false,0.6156,0.6067,42.9,2.3h
Exp-10,1.0,false,0.6089,0.6000,43.6,2.3h
```

---

## 🎉 总结

通过这两个消融实验参数，你可以：

✅ 验证文化损失的有效性
✅ 找到最优的文化损失权重
✅ 评估共享专家的必要性
✅ 理解 CultureMoE 的工作机制
✅ 为论文提供有力的证据

**现在可以开始消融实验了！** 🚀

