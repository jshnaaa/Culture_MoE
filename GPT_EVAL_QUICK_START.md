# GPT评测脚本快速使用指南

## 🚀 快速开始

### 最简单的使用方式（使用所有默认值）

```bash
bash run_eval_gpt.sh
```

这将使用默认配置：
- 数据集: CulturalBench (DATA_ID=2)
- 模型: GPT-4o
- 样本数: 20个（快速测试）

## 📝 参数说明

```bash
bash run_eval_gpt.sh [data_id] [gpt_model] [max_samples]
```

所有参数都是**可选的**，有默认值：

| 参数 | 默认值 | 说明 | 可选值 |
|------|--------|------|--------|
| data_id | 2 | 数据集ID | 2=CulturalBench, 3=normad, 4=cultureLLM, 5=cultureAtlas |
| gpt_model | 4o | GPT模型简化名 | 3.5=gpt-3.5-turbo, 4o=gpt-4o, 4omini=gpt-4o-mini |
| max_samples | 20 | 最大样本数 | 任意正整数，0或留空=全部样本 |

## 💡 使用示例

### 1. 使用默认配置
```bash
bash run_eval_gpt.sh
# 等同于: bash run_eval_gpt.sh 2 4o 20
```

### 2. 只改变数据集
```bash
bash run_eval_gpt.sh 3          # 评测normad数据集
bash run_eval_gpt.sh 4          # 评测cultureLLM数据集
bash run_eval_gpt.sh 5          # 评测cultureAtlas数据集
```

### 3. 改变模型
```bash
bash run_eval_gpt.sh 2 3.5      # 使用GPT-3.5-turbo
bash run_eval_gpt.sh 2 4omini   # 使用GPT-4o-mini（更便宜）
```

### 4. 改变样本数
```bash
bash run_eval_gpt.sh 2 4o 100   # 评测100个样本
bash run_eval_gpt.sh 2 4o 0     # 评测全部样本
```

### 5. 组合使用
```bash
# 使用GPT-3.5评测normad数据集的前50个样本
bash run_eval_gpt.sh 3 3.5 50

# 使用GPT-4o-mini评测CulturalBench全部样本
bash run_eval_gpt.sh 2 4omini 0
```

## 🔑 API KEY配置

脚本已内置API KEY，无需额外配置。

如果需要使用自己的API KEY，可以通过环境变量设置（优先级更高）：

```bash
export OPENAI_API_KEY='your-api-key-here'
bash run_eval_gpt.sh
```

## 📊 输出结果

评测完成后，结果保存在 `/root/autodl-fs/gpt_eval_results/` 目录：

```
gpt4o_CulturalBench_20250120_143022/
├── eval_config.json      # 评测配置
├── eval_results.json     # 汇总结果（准确率等）
├── detailed_results.json # 详细结果（每个样本）
└── eval.log             # 评测日志
```

### 查看结果摘要

脚本运行完成后会自动显示：

```
📊 评测结果摘要:
  模型: gpt-4o
  准确率: 0.8500 (85.00%)
  总样本数: 20
  正确预测数: 17
  API调用失败数: 0
```

## 🎯 常见使用场景

### 快速测试（默认配置）
```bash
bash run_eval_gpt.sh
```

### 完整评测（全部样本）
```bash
bash run_eval_gpt.sh 2 4o 0
```

### 成本优化（使用便宜的模型）
```bash
bash run_eval_gpt.sh 2 3.5      # GPT-3.5最便宜
bash run_eval_gpt.sh 2 4omini   # GPT-4o-mini性价比高
```

### 对比不同模型
```bash
# 依次评测三个模型
bash run_eval_gpt.sh 2 3.5 100
bash run_eval_gpt.sh 2 4omini 100
bash run_eval_gpt.sh 2 4o 100
```

### 对比不同数据集
```bash
# 使用同一模型评测所有数据集
bash run_eval_gpt.sh 2 4o 50    # CulturalBench
bash run_eval_gpt.sh 3 4o 50    # normad
bash run_eval_gpt.sh 4 4o 50    # cultureLLM
bash run_eval_gpt.sh 5 4o 50    # cultureAtlas
```

## 🔧 查看帮助信息

```bash
bash run_eval_gpt.sh -h
# 或
bash run_eval_gpt.sh --help
```

## ⚡ 性能提示

- **快速测试**: 使用默认的20个样本（几分钟完成）
- **中等测试**: 100个样本（约10-15分钟）
- **完整评测**: 全部样本（可能需要1-2小时，取决于数据集大小）

## 💰 成本估算

- **GPT-3.5-turbo**: 最便宜，约$0.001/1K tokens
- **GPT-4o-mini**: 性价比高，约$0.15/1M tokens
- **GPT-4o**: 最强大，约$2.5/1M tokens

每个样本约消耗100-200 tokens，20个样本的成本：
- GPT-3.5: < $0.01
- GPT-4o-mini: < $0.01
- GPT-4o: 约$0.01-0.02

## 🐛 故障排查

### 问题1: API调用失败

检查网络连接和API配额：
```bash
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

### 问题2: 数据文件不存在

确认数据文件路径正确（脚本默认使用 `/root/autodl-fs/` 路径）

### 问题3: 权限问题

确保脚本有执行权限：
```bash
chmod +x run_eval_gpt.sh
```

## 📚 更多信息

详细文档请参考: `GPT_EVAL_README.md`
