# MOE 优化快速开始指南

## 🚀 快速开始（3 步）

### 第 1 步：运行训练（使用默认参数）

```bash
# 使用默认的正则化参数（推荐）
sh run_train_culturemoe_from_base_gen.sh qwen 2
```

**默认参数**：
- `lambda_entropy = 0.05`（熵正则化）
- `lambda_load = 0.01`（负载均衡）

### 第 2 步：监控训练过程

```bash
# 查看实时日志
tail -f /path/to/output/training.log

# 关键指标：
# - Train Loss: 应该稳定下降
# - Train Entropy Loss: 应该在 0.01-0.1 范围内
# - Train Load Loss: 应该在 0.001-0.01 范围内
# - Eval Accuracy: 应该逐步上升
```

### 第 3 步：评估结果

```bash
# 查看最终结果
cat /path/to/output/epoch_eval_results.json | python -m json.tool

# 关键指标：
# - 最终准确率（应该比基准提升 5-15%）
# - 最终 F1 分数
```

---

## 📊 参数调优速查表

### 如果准确率没有改进

```bash
# 增加正则化强度
python train_culturemoe_from_base_gen.py \
    ... \
    --lambda_entropy 0.1 \
    --lambda_load 0.02
```

### 如果损失 NaN

```bash
# 降低正则化强度
python train_culturemoe_from_base_gen.py \
    ... \
    --lambda_entropy 0.01 \
    --lambda_load 0.005
```

### 如果专家利用率不均

```bash
# 增加负载均衡
python train_culturemoe_from_base_gen.py \
    ... \
    --lambda_entropy 0.05 \
    --lambda_load 0.05
```

### 如果路由过度置信

```bash
# 增加熵正则化
python train_culturemoe_from_base_gen.py \
    ... \
    --lambda_entropy 0.15 \
    --lambda_load 0.01
```

---

## 🎯 推荐参数组合

### 保守方案（最安全）
```bash
--lambda_entropy 0.01
--lambda_load 0.005
```

### 平衡方案（推荐）✅
```bash
--lambda_entropy 0.05
--lambda_load 0.01
```

### 激进方案（最强）
```bash
--lambda_entropy 0.1
--lambda_load 0.02
```

---

## 📈 预期改进

| 指标 | 基准 | 优化后 | 改进 |
|------|------|--------|------|
| 准确率 | 60% | 68-72% | +8-12% |
| F1 分数 | 0.58 | 0.66-0.70 | +0.08-0.12 |
| 训练稳定性 | 波动大 | 稳定 | ✅ |
| 专家利用率 | 不均 | 均衡 | ✅ |

---

## 🔍 关键指标解释

### Entropy Loss（熵损失）

```
含义：Router 的路由分布熵
范围：0.01-0.1（正常）
过小：路由过度置信（某些专家被过度使用）
过大：路由过度分散（所有专家都被均匀使用）
```

### Load Loss（负载损失）

```
含义：专家利用率的不均衡程度
范围：0.001-0.01（正常）
过小：专家利用率均衡
过大：某些专家被严重闲置
```

### Generation Loss（生成损失）

```
含义：主要的任务损失（生成式预测）
范围：取决于任务
趋势：应该稳定下降
```

---

## 💡 常见问题

### Q1: 应该先调整哪个参数？

**A**: 先调整 `lambda_entropy`，因为它对路由稳定性影响最大。

### Q2: 参数应该同时调整还是逐个调整？

**A**: 建议逐个调整，这样可以看到每个参数的单独效果。

### Q3: 多少个 epoch 后能看到效果？

**A**: 通常 2-3 个 epoch 就能看到初步效果，5+ 个 epoch 能看到最终效果。

### Q4: 如果准确率反而下降了怎么办？

**A**: 说明正则化系数过大，降低参数值重新训练。

### Q5: 可以同时使用知识蒸馏吗？

**A**: 可以，但建议先用这两个方法，如果效果不理想再加知识蒸馏。

---

## 📝 实验记录模板

```
实验日期：2025-11-10
实验编号：Exp-001

参数设置：
- lambda_entropy: 0.05
- lambda_load: 0.01
- 其他参数：[保持默认]

结果：
- 基准准确率：60.0%
- 优化后准确率：68.5%
- 改进：+8.5%
- 训练稳定性：✅ 改进

观察：
- [记录训练过程中的观察]
- [记录损失曲线的变化]
- [记录专家权重的分布]

下一步：
- [根据结果决定下一步行动]
```

---

## 🎯 优化路线图

```
第 1 阶段：基准测试
├─ 运行不使用正则化的训练
├─ 记录基准准确率和损失曲线
└─ 目标：建立基准

第 2 阶段：添加熵正则化
├─ 运行 lambda_entropy=0.05, lambda_load=0.0
├─ 观察准确率是否改进
└─ 目标：改进 3-5%

第 3 阶段：添加负载均衡
├─ 运行 lambda_entropy=0.05, lambda_load=0.01
├─ 观察最终效果
└─ 目标：再改进 3-5%

第 4 阶段：微调参数
├─ 根据第 3 阶段结果调整参数
├─ 尝试不同的参数组合
└─ 目标：达到最优效果

第 5 阶段：考虑知识蒸馏（可选）
├─ 如果效果仍不理想
├─ 添加知识蒸馏
└─ 目标：进一步改进
```

---

## 📊 监控命令

```bash
# 实时监控训练日志
tail -f /path/to/output/training.log | grep -E "loss|accuracy|entropy|load"

# 查看 epoch 结果
python -c "
import json
with open('/path/to/output/epoch_eval_results.json') as f:
    data = json.load(f)
    for epoch in data:
        print(f\"Epoch {epoch['epoch']}: Acc={epoch['eval_accuracy']:.4f}, Loss={epoch['eval_loss']:.4f}, Entropy={epoch['train_entropy_loss']:.4f}, Load={epoch['train_load_loss']:.4f}\")
"

# 绘制损失曲线
python -c "
import json
import matplotlib.pyplot as plt

with open('/path/to/output/epoch_eval_results.json') as f:
    data = json.load(f)

epochs = [e['epoch'] for e in data]
losses = [e['train_loss'] for e in data]
accuracies = [e['eval_accuracy'] for e in data]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evaluation Accuracy')

plt.tight_layout()
plt.savefig('/path/to/output/training_curves.png')
print('Saved to /path/to/output/training_curves.png')
"
```

---

## ✅ 检查清单

在开始训练前，确保：

- [ ] 已修改 `train_culturemoe_from_base_gen.py`
- [ ] 已添加 `--lambda_entropy` 和 `--lambda_load` 参数
- [ ] 已更新 `train_epoch()` 函数
- [ ] 已更新训练循环调用
- [ ] 已更新日志输出
- [ ] 数据集路径正确
- [ ] 输出目录存在或可创建
- [ ] GPU 内存充足

---

## 🎉 成功标志

训练成功的标志：

✅ 训练损失稳定下降
✅ 熵损失在 0.01-0.1 范围内
✅ 负载损失在 0.001-0.01 范围内
✅ 验证准确率逐步上升
✅ 没有 NaN 或 Inf 损失
✅ 最终准确率比基准提升 5-15%

---

**现在开始优化你的 MOE 模型吧！** 🚀

