# 专家数搜索系统 - 完整总结

## 🎯 目标

自动搜索 CultureMoE 模型的最佳专家数（num_experts），通过系统性实验找到最优配置。

## 📦 创建的文件

### 1. 核心脚本

#### `search_best_num_experts.sh` ⭐
**主搜索脚本**

**功能**：
- ✅ 阶段 1：粗粒度搜索（2, 4, 6, 8, 12 个专家）
- ✅ 阶段 2：细粒度搜索（在最佳值附近）
- ✅ 阶段 3：使用最佳专家数训练完整模型（10 epochs）
- ✅ 自动保存每个 epoch 的评估结果
- ✅ 生成详细的日志和报告

**使用方法**：
```bash
sh search_best_num_experts.sh
```

**预计时间**：7-9 小时

---

#### `visualize_expert_search.py` 📊
**结果可视化脚本**

**功能**：
- ✅ 打印详细的结果摘要
- ✅ 生成准确率 vs 专家数曲线图
- ✅ 生成不同专家数的训练曲线对比图
- ✅ 输出最佳专家数和对应准确率

**使用方法**：
```bash
python visualize_expert_search.py /root/autodl-fs/output/expert_search_*/final_report.json
```

**输出文件**：
- `expert_search_plot.png` - 主结果图
- `epoch_curves.png` - 训练曲线对比图

---

### 2. 文档

#### `EXPERT_SEARCH_README.md` 📖
**详细使用文档**

**内容**：
- 脚本功能详解
- 三个阶段的详细说明
- 输出文件结构
- 结果查看方法
- 结果分析指南
- 常见问题解答

---

#### `QUICK_START_EXPERT_SEARCH.md` 🚀
**快速开始指南**

**内容**：
- 一键运行命令
- 快速查看结果
- 结果解读
- 下一步建议

---

#### `EXPERT_SEARCH_SUMMARY.md` 📝
**总结文档（本文件）**

**内容**：
- 所有文件的概览
- 完整的工作流程
- 使用示例

---

## 🔄 完整工作流程

### 步骤 1：运行搜索

```bash
sh search_best_num_experts.sh
```

**过程**：
1. 测试 5 个专家数（2, 4, 6, 8, 12），每个训练 5 epochs
2. 找到最佳值，在其附近细化搜索
3. 使用最佳专家数训练 10 epochs

**输出**：
- 每个专家数的评估结果文件
- 完整的搜索日志
- 最终报告（JSON 格式）

---

### 步骤 2：查看结果

```bash
# 方法 1：查看 JSON 报告
cat /root/autodl-fs/output/expert_search_*/final_report.json | jq

# 方法 2：使用可视化脚本
python visualize_expert_search.py /root/autodl-fs/output/expert_search_*/final_report.json

# 方法 3：查看日志
cat /root/autodl-fs/output/expert_search_*/search_log.txt
```

---

### 步骤 3：应用最佳配置

假设搜索结果显示 4 个专家最好：

```bash
# 修改训练脚本中的 num_experts 参数
# 或者在命令行中指定
sh run_train_ddp_lora_dual.sh llama 2 True false --num_experts 4
```

---

## 📊 输出文件结构

```
/root/autodl-fs/output/expert_search_20251030_1430/
│
├── search_log.txt                          # 完整日志
├── final_report.json                       # 最终报告
├── phase1_results.json                     # 阶段1汇总
├── expert_search_plot.png                  # 结果图表
├── epoch_curves.png                        # 训练曲线图
│
├── phase1_experts_2/                       # 阶段1：2个专家
│   ├── epoch_eval_results.json             # 5个epoch的评估结果
│   ├── eval_results.json                   # 最终评估结果
│   └── trainer_log.jsonl                   # 训练日志
│
├── phase1_experts_4/                       # 阶段1：4个专家
│   └── ...
│
├── phase1_experts_6/                       # 阶段1：6个专家
│   └── ...
│
├── phase1_experts_8/                       # 阶段1：8个专家
│   └── ...
│
├── phase1_experts_12/                      # 阶段1：12个专家
│   └── ...
│
├── phase2_experts_3/                       # 阶段2：细粒度搜索
│   └── ...                                 # （如果需要）
│
├── phase2_experts_5/
│   └── ...
│
└── final_model_experts_4/                  # 阶段3：最终模型
    ├── epoch_eval_results.json             # 10个epoch的评估结果
    ├── eval_results.json
    └── trainer_log.jsonl
```

---

## 📈 预期结果示例

### 控制台输出

```
============================================================
CultureMoE Expert Number Search
============================================================
Start time: 2025-10-30 14:30:00
Backbone: llama
Num classes: 2
Dataset: /root/autodl-fs/CulturalBench_Hard_merge.json
============================================================

============================================================
Phase 1: Coarse-grained Search
============================================================
Testing num_experts: 2, 4, 6, 8, 12
Training: 5 epochs, no model saving
============================================================

[Phase 1] Testing num_experts=2
...
✅ Completed: num_experts=2
   Best accuracy: 0.7800 (at epoch 4)

[Phase 1] Testing num_experts=4
...
✅ Completed: num_experts=4
   Best accuracy: 0.8234 (at epoch 5)

[Phase 1] Testing num_experts=6
...
✅ Completed: num_experts=6
   Best accuracy: 0.8150 (at epoch 4)

[Phase 1] Testing num_experts=8
...
✅ Completed: num_experts=8
   Best accuracy: 0.8050 (at epoch 3)

[Phase 1] Testing num_experts=12
...
✅ Completed: num_experts=12
   Best accuracy: 0.7900 (at epoch 4)

============================================================
Phase 1 Results Summary
============================================================
Num Experts | Best Accuracy | Best Epoch
------------|---------------|------------
          2 |        0.7800 |          4
          4 |        0.8234 |          5
          6 |        0.8150 |          4
          8 |        0.8050 |          3
         12 |        0.7900 |          4
============================================================

🏆 Best num_experts from Phase 1: 4 (accuracy: 0.8234)

============================================================
Phase 2: Fine-grained Search
============================================================
Searching around best value: 4
Testing num_experts: 3, 5
============================================================

[Phase 2] Testing num_experts=3
...
✅ Completed: num_experts=3
   Best accuracy: 0.8100 (at epoch 5)

[Phase 2] Testing num_experts=5
...
✅ Completed: num_experts=5
   Best accuracy: 0.8200 (at epoch 4)

============================================================
Phase 2 Results Summary
============================================================
Num Experts | Best Accuracy | Best Epoch
------------|---------------|------------
          3 |        0.8100 |          5
          5 |        0.8200 |          4
============================================================

🏆 Final best num_experts: 4 (accuracy: 0.8234)

============================================================
Phase 3: Training Final Model
============================================================
Using best num_experts: 4
Training: 10 epochs
============================================================

Running final training with 4 experts...
...
✅ Final model training completed!

============================================================
Final Report
============================================================
Search completed at: 2025-10-30 22:15:00

🏆 Best num_experts: 4
📊 Best accuracy (5 epochs): 0.8234

📁 Output directory: /root/autodl-fs/output/expert_search_20251030_1430
📄 Search log: /root/autodl-fs/output/expert_search_20251030_1430/search_log.txt
📊 Final model results: /root/autodl-fs/output/expert_search_20251030_1430/final_model_experts_4/epoch_eval_results.json
============================================================

✅ All tasks completed successfully!
```

---

### 可视化输出

运行 `visualize_expert_search.py` 后：

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
3               fine       0.8100          5
4               coarse     0.8234          5
5               fine       0.8200          4
6               coarse     0.8150          4
8               coarse     0.8050          3
12              coarse     0.7900          4
----------------------------------------------------------------------

----------------------------------------------------------------------
Final Model (10 epochs)
----------------------------------------------------------------------
Num Experts: 4
Best Accuracy: 0.8456
Best Epoch: 7
----------------------------------------------------------------------

✅ Plot saved to: expert_search_plot.png
✅ Epoch curves saved to: epoch_curves.png
```

---

## 🎨 生成的图表

### 1. `expert_search_plot.png`

准确率 vs 专家数曲线：
- 蓝色圆点：粗粒度搜索
- 绿色方块：细粒度搜索
- 红色星星：最佳值
- 曲线连接所有测试点

### 2. `epoch_curves.png`

两个子图：
- 左图：不同专家数的准确率 vs epoch
- 右图：不同专家数的损失 vs epoch

---

## 💡 使用建议

### 1. 首次运行

```bash
# 使用默认配置运行
sh search_best_num_experts.sh
```

### 2. 后台运行（推荐）

```bash
# 在后台运行，避免 SSH 断开
nohup sh search_best_num_experts.sh > search.log 2>&1 &

# 查看进度
tail -f search.log

# 或者使用 screen/tmux
screen -S expert_search
sh search_best_num_experts.sh
# Ctrl+A, D 分离会话
```

### 3. 自定义配置

编辑 `search_best_num_experts.sh`：

```bash
# 修改测试的专家数
COARSE_EXPERTS=(2 3 4 5 6)  # 自定义范围

# 修改数据集
NUM_CLASSES=4
TRAIN_FILE="/root/autodl-fs/wvs_all_llama_merge_4.json"

# 修改训练轮数
--num_train_epochs 3  # 阶段1和2
--num_train_epochs 7  # 阶段3
```

---

## 🔍 结果分析

### 判断标准

1. **单峰曲线**：存在明确的最优专家数
2. **递增曲线**：专家数越多越好（但收益递减）
3. **递减曲线**：专家数越少越好（数据量不足）
4. **平坦曲线**：专家数影响不大

### 决策建议

- **如果有明确最优值**：使用该值
- **如果多个值接近**：选择较小的（节省计算）
- **如果结果不理想**：检查数据质量和训练配置

---

## 🚀 下一步

找到最佳专家数后：

### 1. 验证其他数据集

```bash
# 测试 3 分类
sh run_train_ddp_lora_dual.sh llama 3 True false --num_experts 4

# 测试 4 分类
sh run_train_ddp_lora_dual.sh llama 4 True false --num_experts 4

# 测试 5 分类
sh run_train_ddp_lora_dual.sh llama 5 True false --num_experts 4
```

### 2. 对比 LoRA Only

```bash
# 训练 LoRA Only
sh run_train_lora_only.sh llama 2 false

# 对比性能
# 如果 CultureMoE > LoRA Only，说明 MoE 架构有效
```

### 3. 消融实验

```bash
# 测试文化损失的影响
sh run_train_ddp_lora_dual.sh llama 2 False false  # 无文化损失
sh run_train_ddp_lora_dual.sh llama 2 True false   # 有文化损失
```

### 4. 超参数调优

基于最佳专家数，进一步优化：
- 学习率
- LoRA rank
- 文化损失权重
- 训练轮数

---

## 📚 相关文档

- `EXPERT_SEARCH_README.md` - 详细文档
- `QUICK_START_EXPERT_SEARCH.md` - 快速指南
- `CULTUREMOE_OPTIMIZATION.md` - 参数优化说明
- `BATCH_TRAINING_README.md` - 批量训练说明

---

## ✅ 总结

### 创建的文件

1. ✅ `search_best_num_experts.sh` - 主搜索脚本
2. ✅ `visualize_expert_search.py` - 可视化脚本
3. ✅ `EXPERT_SEARCH_README.md` - 详细文档
4. ✅ `QUICK_START_EXPERT_SEARCH.md` - 快速指南
5. ✅ `EXPERT_SEARCH_SUMMARY.md` - 总结文档

### 核心功能

- ✅ 自动搜索最佳专家数（粗粒度 + 细粒度）
- ✅ 保存每个专家数的每个 epoch 评估结果
- ✅ 使用最佳专家数训练完整模型（10 epochs）
- ✅ 生成详细的报告和日志
- ✅ 可视化结果（图表 + 摘要）

### 使用流程

```bash
# 1. 运行搜索
sh search_best_num_experts.sh

# 2. 查看结果
python visualize_expert_search.py /root/autodl-fs/output/expert_search_*/final_report.json

# 3. 应用最佳配置
# 在训练脚本中使用找到的最佳专家数
```

现在你有了一个完整的专家数搜索系统！🎉

