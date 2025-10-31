# 两阶段训练 CultureMoE - 快速开始

## ✅ 已修复的问题

1. ✅ 修复了 `load_and_process_dual_classification_data` 参数错误
2. ✅ 添加了文化损失支持
3. ✅ 添加了保存模型选项
4. ✅ 更新了参数顺序

## 🚀 使用方法

### 命令格式

```bash
sh run_two_stage_culturemoe.sh [backbone] [num_classes] [use_culture_loss] [save_model] [num_experts]
```

### 参数说明

1. **backbone** - 基座模型
   - `llama` - LLaMA 3.1-8B-Instruct（默认）
   - `qwen` - Qwen 2.5-7B-Instruct

2. **num_classes** - 分类数量
   - `2` - 二分类（默认）
   - `3` - 三分类
   - `4` - 四分类
   - `5` - 五分类

3. **use_culture_loss** - 是否使用文化损失
   - `True` - 使用文化损失（默认）
   - `False` - 不使用文化损失

4. **save_model** - 是否保存模型
   - `false` - 不保存模型（默认）
   - `true` - 保存模型

5. **num_experts** - 专家数量
   - 默认：`6`
   - 可选：任意正整数

## 📝 使用示例

### 示例 1：默认配置

```bash
sh run_two_stage_culturemoe.sh
# 等价于：sh run_two_stage_culturemoe.sh llama 2 True false 6
```

### 示例 2：LLaMA + 2分类 + 使用文化损失 + 不保存模型

```bash
sh run_two_stage_culturemoe.sh llama 2 True false
```

### 示例 3：LLaMA + 4分类 + 不使用文化损失 + 不保存模型

```bash
sh run_two_stage_culturemoe.sh llama 4 False false
```

### 示例 4：Qwen + 3分类 + 使用文化损失 + 保存模型

```bash
sh run_two_stage_culturemoe.sh qwen 3 True true
```

### 示例 5：自定义专家数

```bash
sh run_two_stage_culturemoe.sh llama 2 True false 4  # 4个专家
sh run_two_stage_culturemoe.sh llama 2 True false 8  # 8个专家
```

## 🎯 典型场景

### 场景 1：快速实验（不保存模型）

```bash
# 测试不同配置
sh run_two_stage_culturemoe.sh llama 2 True false
sh run_two_stage_culturemoe.sh llama 3 True false
sh run_two_stage_culturemoe.sh llama 4 True false
```

### 场景 2：消融实验（文化损失）

```bash
# 有文化损失
sh run_two_stage_culturemoe.sh llama 2 True false

# 无文化损失
sh run_two_stage_culturemoe.sh llama 2 False false
```

### 场景 3：保存最佳模型

```bash
# 找到最佳配置后保存模型
sh run_two_stage_culturemoe.sh llama 2 True true
```

## 📊 输出文件

### 不保存模型时（save_model=false）

```
output_dir/
├── config.json                         # 训练配置
├── final_report.json                   # 最终报告
├── stage1_epoch_eval_results.json      # 阶段1每个epoch结果
├── stage1_final_eval.json              # 阶段1最终结果
├── stage2_epoch_eval_results.json      # 阶段2每个epoch结果
├── stage2_final_eval.json              # 阶段2最终结果
└── stage1/logs/                        # TensorBoard日志
└── stage2/logs/                        # TensorBoard日志
```

### 保存模型时（save_model=true）

除了上述文件，还包括：
- 阶段1合并后的模型
- 阶段2的 CultureMoE 模型

## 🔍 查看结果

```bash
# 查看最终报告
cat /root/autodl-fs/output/two_stage_moe/*/final_report.json | jq

# 查看阶段1每个epoch
cat /root/autodl-fs/output/two_stage_moe/*/stage1_epoch_eval_results.json | jq '.[].eval_accuracy'

# 查看阶段2每个epoch
cat /root/autodl-fs/output/two_stage_moe/*/stage2_epoch_eval_results.json | jq '.[].eval_accuracy'
```

## ⚙️ 默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| backbone | llama | LLaMA 3.1-8B |
| num_classes | 2 | 二分类 |
| use_culture_loss | True | 使用文化损失 |
| save_model | false | 不保存模型 |
| num_experts | 6 | 6个专家 |
| stage1_epochs | 3 | 阶段1训练3轮 |
| stage2_epochs | 5 | 阶段2训练5轮 |
| culture_loss_lambda | 0.05 | 文化损失权重 |

## ⏱️ 预计时间

- **阶段1**（LoRA 微调）：1-1.5 小时
- **阶段2**（MoE 训练）：1.5-2 小时
- **总计**：2.5-3.5 小时

## 💡 提示

1. **不保存模型**：适合快速实验，节省空间
2. **使用文化损失**：通常能提升性能
3. **专家数**：建议从 4-6 开始尝试
4. **后台运行**：使用 `nohup` 或 `screen`

```bash
nohup sh run_two_stage_culturemoe.sh llama 2 True false > train.log 2>&1 &
```

## 🐛 常见问题

### Q1: 如何修改训练轮数？

**A**: 编辑 `run_two_stage_culturemoe.sh`：
```bash
--stage1_epochs 5  # 修改阶段1轮数
--stage2_epochs 10 # 修改阶段2轮数
```

### Q2: 如何修改文化损失权重？

**A**: 编辑 `run_two_stage_culturemoe.sh`：
```bash
--culture_loss_lambda 0.1  # 修改权重
```

### Q3: 显存不足怎么办？

**A**: 减小批次大小：
```bash
--batch_size 2
--gradient_accumulation_steps 16
```

## ✅ 总结

### 核心命令

```bash
# 基本用法
sh run_two_stage_culturemoe.sh [backbone] [num_classes] [use_culture_loss] [save_model]

# 示例
sh run_two_stage_culturemoe.sh llama 2 True false
```

### 核心特点

- ✅ 两阶段训练（LoRA → MoE）
- ✅ 支持文化损失
- ✅ 可选保存模型
- ✅ 每个epoch评估
- ✅ 自动清理临时文件

现在可以开始训练了！🚀

