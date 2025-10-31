# 快速开始：专家数搜索

## 一键运行

```bash
sh search_best_num_experts.sh
```

就这么简单！脚本会自动完成所有工作。

## 预计时间

⏱️ **总计：7-9 小时**

- 阶段 1（粗粒度）：4-5 小时
- 阶段 2（细粒度）：1-2 小时
- 阶段 3（最终验证）：2 小时

## 查看结果

### 方法 1：查看最终报告

```bash
# 找到最新的搜索结果
cd /root/autodl-fs/output/expert_search_*

# 查看报告
cat final_report.json | jq

# 只看最佳专家数
cat final_report.json | jq '.best_num_experts'
```

### 方法 2：使用可视化脚本

```bash
# 生成图表和摘要
python visualize_expert_search.py /root/autodl-fs/output/expert_search_*/final_report.json
```

**输出**：
- 📊 `expert_search_plot.png` - 准确率 vs 专家数曲线
- 📈 `epoch_curves.png` - 不同专家数的训练曲线
- 📝 控制台输出详细摘要

### 方法 3：查看日志

```bash
# 查看完整日志
tail -f /root/autodl-fs/output/expert_search_*/search_log.txt

# 只看结果摘要
grep "Best" /root/autodl-fs/output/expert_search_*/search_log.txt
```

## 预期输出示例

```
============================================================
CultureMoE Expert Number Search Results
============================================================
Search Timestamp: 20251030_1430
Best Num Experts: 4
Best Accuracy (5 epochs): 0.8234
============================================================

----------------------------------------------------------------------
Search Results (All Tested Values)
----------------------------------------------------------------------
Num Experts     Phase      Best Accuracy   Best Epoch
----------------------------------------------------------------------
2               coarse     0.7800          4
4               coarse     0.8234          5
6               coarse     0.8150          4
8               coarse     0.8050          3
12              coarse     0.7900          4
3               fine       0.8100          5
5               fine       0.8200          4
----------------------------------------------------------------------

----------------------------------------------------------------------
Final Model (10 epochs)
----------------------------------------------------------------------
Num Experts: 4
Best Accuracy: 0.8456
Best Epoch: 7
----------------------------------------------------------------------
```

## 结果解读

### 情况 1：单峰曲线（最常见）

```
准确率
  ^
  |     *
  |   *   *
  | *       *
  +-----------> 专家数
  2  4  6  8  12
```

**结论**：存在最优专家数（如 4 或 6）

### 情况 2：递增后平稳

```
准确率
  ^
  |       * * *
  |     *
  |   *
  +-----------> 专家数
  2  4  6  8  12
```

**结论**：专家数 6-12 都可以，选择较小的（节省计算）

### 情况 3：递减

```
准确率
  ^
  | *
  |   *
  |     *
  +-----------> 专家数
  2  4  6  8  12
```

**结论**：专家数越少越好（数据量可能不足）

## 下一步

### 1. 使用最佳专家数训练其他数据集

```bash
# 假设最佳专家数是 4
# 修改 run_train_ddp_lora_dual.sh 中的 num_experts 参数

# 测试 3 分类
sh run_train_ddp_lora_dual.sh llama 3 True false

# 测试 4 分类
sh run_train_ddp_lora_dual.sh llama 4 True false

# 测试 5 分类
sh run_train_ddp_lora_dual.sh llama 5 True false
```

### 2. 对比 LoRA Only

```bash
# 训练 LoRA Only（相同数据集）
sh run_train_lora_only.sh llama 2 false

# 对比准确率
# CultureMoE (4 experts): 0.8456
# LoRA Only:              0.8200
# 提升：+2.56%
```

### 3. 消融实验

```bash
# 测试文化损失的影响
sh run_train_ddp_lora_dual.sh llama 2 False false  # 无文化损失
sh run_train_ddp_lora_dual.sh llama 2 True false   # 有文化损失
```

## 常见问题

### Q: 搜索过程中可以中断吗？

**A**: 可以。每个阶段的结果都会保存。中断后可以查看已完成的结果。

### Q: 如何只测试特定的专家数？

**A**: 编辑脚本，修改 `COARSE_EXPERTS` 数组：
```bash
COARSE_EXPERTS=(3 4 5)  # 只测试 3, 4, 5
```

### Q: 如何加快搜索速度？

**A**:
1. 减少 epoch 数（从 5 改为 3）
2. 减少测试的专家数（只测试 2, 4, 6）
3. 跳过细粒度搜索

### Q: 结果不理想怎么办？

**A**: 可能的原因：
1. 数据量不足 → 尝试更少的专家数
2. 学习率不合适 → 调整学习率
3. 训练轮数不够 → 增加 epoch 数

## 文件清单

创建的文件：
- ✅ `search_best_num_experts.sh` - 主搜索脚本
- ✅ `visualize_expert_search.py` - 可视化脚本
- ✅ `EXPERT_SEARCH_README.md` - 详细文档
- ✅ `QUICK_START_EXPERT_SEARCH.md` - 快速指南（本文件）

## 总结

### 运行搜索

```bash
sh search_best_num_experts.sh
```

### 查看结果

```bash
python visualize_expert_search.py /root/autodl-fs/output/expert_search_*/final_report.json
```

### 使用最佳配置

```bash
# 在 run_train_ddp_lora_dual.sh 中设置 num_experts
# 然后正常训练
sh run_train_ddp_lora_dual.sh llama 2 True false
```

就这么简单！🚀

---

**提示**：搜索过程较长，建议在后台运行：

```bash
nohup sh search_best_num_experts.sh > search.log 2>&1 &

# 查看进度
tail -f search.log

