# 消融实验快速参考

## 🚀 快速开始

### 基础命令格式

```bash
sh run_train_culturemoe_from_base_gen.sh <BACKBONE> <DATA_ID> <USE_CULTURE_LOSS> <NUM_EXPERTS> <NUM_GPUS> <MASK_USE> <CULTURE_LOSS_WEIGHT> <USE_SHARED>
```

### 最常用的命令

```bash
# 1. 默认配置（推荐）
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true

# 2. 不使用文化损失
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.0 true

# 3. 低文化损失权重
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.1 true

# 4. 高文化损失权重
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.8 true

# 5. 不使用共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 false

# 6. 低文化损失 + 无共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.1 false
```

---

## 📊 参数速查表

### culture_loss_weight（文化损失权重）

| 值 | 说明 | 用途 |
|----|------|------|
| 0.0 | 无文化损失 | 基准测试 |
| 0.1 | 低权重 | 验证文化损失的必要性 |
| 0.5 | 中等权重 | **推荐默认值** |
| 0.8 | 高权重 | 强调文化对齐 |
| 1.0 | 仅文化损失 | 极端情况 |

### use_shared（是否使用共享专家）

| 值 | 说明 | 用途 |
|----|------|------|
| true | 使用共享专家 | **推荐默认值** |
| false | 不使用共享专家 | 评估共享专家的必要性 |

---

## 🎯 常见实验场景

### 场景 1：验证文化损失的有效性

```bash
# 基准：无文化损失
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.0 true

# 对比：有文化损失
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 true
```

**预期**：有文化损失的模型准确率更高

### 场景 2：找到最优的文化损失权重

```bash
# 运行多个权重
for weight in 0.1 0.3 0.5 0.7 0.9; do
    sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true $weight true
done
```

**预期**：存在最优权重，通常在 0.3-0.7 之间

### 场景 3：评估共享专家的必要性

```bash
# 有共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 true

# 无共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 false
```

**预期**：有共享专家的模型性能更好（约 5-10%）

### 场景 4：完整的消融实验

```bash
# 创建实验脚本
cat > run_ablation_study.sh << 'EOF'
#!/bin/bash

# 实验 1-5：文化损失权重（有共享专家）
for weight in 0.0 0.1 0.5 0.8 1.0; do
    echo "Running Exp with culture_loss_weight=$weight, use_shared=true"
    sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true $weight true
done

# 实验 6-10：文化损失权重（无共享专家）
for weight in 0.0 0.1 0.5 0.8 1.0; do
    echo "Running Exp with culture_loss_weight=$weight, use_shared=false"
    sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true $weight false
done
EOF

chmod +x run_ablation_study.sh
./run_ablation_study.sh
```

---

## 📈 结果对比

### 快速查看结果

```bash
# 查看最终准确率
python -c "
import json
import os

# 列出所有实验目录
exp_dirs = [d for d in os.listdir('/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/')
            if d.startswith('moe_')]

for exp_dir in sorted(exp_dirs):
    path = f'/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/{exp_dir}/epoch_eval_results.json'
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
            final = results[-1]
            print(f'{exp_dir}:')
            print(f'  Accuracy: {final[\"eval_accuracy\"]:.4f}')
            print(f'  F1: {final[\"eval_f1\"]:.4f}')
            print()
"
```

### 绘制对比图

```bash
python -c "
import json
import matplotlib.pyplot as plt
import os

# 收集数据
weights = []
accuracies_with_shared = []
accuracies_without_shared = []

for weight in [0.0, 0.1, 0.5, 0.8, 1.0]:
    weights.append(weight)

    # 有共享专家
    path1 = f'/path/to/exp_weight{weight}_shared_true/epoch_eval_results.json'
    with open(path1) as f:
        results = json.load(f)
        accuracies_with_shared.append(results[-1]['eval_accuracy'])

    # 无共享专家
    path2 = f'/path/to/exp_weight{weight}_shared_false/epoch_eval_results.json'
    with open(path2) as f:
        results = json.load(f)
        accuracies_without_shared.append(results[-1]['eval_accuracy'])

# 绘制
plt.figure(figsize=(10, 6))
plt.plot(weights, accuracies_with_shared, 'o-', label='With Shared Experts', linewidth=2)
plt.plot(weights, accuracies_without_shared, 's-', label='Without Shared Experts', linewidth=2)
plt.xlabel('Culture Loss Weight')
plt.ylabel('Accuracy')
plt.title('Ablation Study: Impact of Culture Loss Weight and Shared Experts')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
print('Saved to ablation_study_results.png')
"
```

---

## ✅ 检查清单

在运行消融实验前，确保：

- [ ] 已修改 `run_train_culturemoe_from_base_gen.sh`（添加新参数）
- [ ] 已修改 `train_culturemoe_from_base_gen.py`（添加参数处理）
- [ ] 已修改 `CultureMoE.py`（支持 use_shared_experts）
- [ ] 数据集路径正确
- [ ] 输出目录存在或可创建
- [ ] GPU 内存充足
- [ ] 有足够的磁盘空间存储实验结果

---

## 🎯 实验计划示例

### 第一周：基础实验

```
Day 1: Exp-1 (无文化损失)
Day 2: Exp-3 (默认配置)
Day 3: Exp-5 (仅文化损失)
Day 4: 分析结果，调整参数
Day 5: Exp-2, Exp-4 (中间权重)
```

### 第二周：深度实验

```
Day 1: Exp-8 (最优权重 + 无共享专家)
Day 2: Exp-6, Exp-7, Exp-9, Exp-10 (其他组合)
Day 3: 数据分析和可视化
Day 4: 撰写实验报告
Day 5: 论文修改和提交
```

---

## 📝 实验日志模板

```
实验编号：Exp-3
日期：2025-11-10
参数：culture_loss_weight=0.5, use_shared=true

开始时间：10:00
预计完成时间：12:30
实际完成时间：12:25

结果：
- 最终准确率：68.5%
- 最终 F1：0.685
- 训练时间：2h 25m

备注：
- 训练过程稳定
- 没有 NaN 损失
- 模型收敛良好
```

---

## 🔗 相关文件

- **完整指南**：`ABLATION_STUDY_GUIDE.md`
- **训练脚本**：`run_train_culturemoe_from_base_gen.sh`
- **训练代码**：`train_culturemoe_from_base_gen.py`
- **模型代码**：`src/llamafactory/model/CultureMoE.py`

---

## 💡 提示

1. **并行运行实验**：使用不同的 GPU 并行运行多个实验
2. **监控资源**：使用 `nvidia-smi` 监控 GPU 使用情况
3. **保存日志**：将输出重定向到文件以便后续分析
4. **定期备份**：定期备份实验结果

---

**现在可以开始消融实验了！** 🚀

