# GPT评测脚本快速使用指南

## 🚀 快速开始

### 基本使用方式

```bash
bash run_eval_gpt.sh <your-api-key>
```

这将使用默认配置：
- 数据集: CulturalBench (DATA_ID=2)
- 模型: GPT-4o
- 样本数: 20个（快速测试）

## 📝 参数说明

```bash
bash run_eval_gpt.sh <api_key> [data_id] [gpt_model] [max_samples]
```

| 参数 | 是否必需 | 默认值 | 说明 | 可选值 |
|------|----------|--------|------|--------|
| api_key | **必需** | - | OpenAI API KEY | sk-proj-开头的密钥 |
| data_id | 可选 | 2 | 数据集ID | 2=CulturalBench, 3=normad, 4=cultureLLM, 5=cultureAtlas |
| gpt_model | 可选 | 4o | GPT模型简化名 | 3.5=gpt-3.5-turbo, 4o=gpt-4o, 4omini=gpt-4o-mini |
| max_samples | 可选 | 20 | 最大样本数 | 任意正整数，0或留空=全部样本 |

## 💡 使用示例

### 1. 使用默认配置
```bash
bash run_eval_gpt.sh sk-proj-your-api-key-here
# 等同于: bash run_eval_gpt.sh sk-proj-xxx 2 4o 20
```

### 2. 只改变数据集
```bash
bash run_eval_gpt.sh sk-proj-xxx 3          # 评测normad数据集
bash run_eval_gpt.sh sk-proj-xxx 4          # 评测cultureLLM数据集
bash run_eval_gpt.sh sk-proj-xxx 5          # 评测cultureAtlas数据集
```

### 3. 改变模型
```bash
bash run_eval_gpt.sh sk-proj-xxx 2 3.5      # 使用GPT-3.5-turbo
bash run_eval_gpt.sh sk-proj-xxx 2 4omini   # 使用GPT-4o-mini（更便宜）
```

### 4. 改变样本数
```bash
bash run_eval_gpt.sh sk-proj-xxx 2 4o 100   # 评测100个样本
bash run_eval_gpt.sh sk-proj-xxx 2 4o 0     # 评测全部样本
```

### 5. 组合使用
```bash
# 使用GPT-3.5评测normad数据集的前50个样本
bash run_eval_gpt.sh sk-proj-xxx 3 3.5 50

# 使用GPT-4o-mini评测CulturalBench全部样本
bash run_eval_gpt.sh sk-proj-xxx 2 4omini 0
```

## 🔑 API KEY安全说明

### ✅ 推荐做法（命令行传参）

```bash
# 方式1: 直接传参（推荐）
bash run_eval_gpt.sh sk-proj-your-api-key-here 2 4o 20

# 方式2: 使用变量（避免历史记录）
API_KEY="sk-proj-your-api-key-here"
bash run_eval_gpt.sh "$API_KEY" 2 4o 20
unset API_KEY
```

### ⚠️ 安全提示

- ✅ API KEY作为参数传入，不会被记录到Git历史
- ✅ 配置输出中只显示前20个字符（如: `sk-proj-vGsST9hHdXBt...`）
- ⚠️ 注意：命令行历史可能记录API KEY，使用后建议清理：
  ```bash
  history -d $(history 1)  # 删除上一条命令
  # 或者
  history -c               # 清空所有历史
  ```

### 🔒 更安全的方式（使用环境变量）

```bash
# 临时设置（当前会话有效）
export OPENAI_API_KEY='sk-proj-your-api-key-here'
bash run_eval_gpt.sh dummy-key 2 4o 20  # 第一个参数会被环境变量覆盖

# 或者从文件读取（不要提交.env文件到Git）
echo 'sk-proj-your-api-key' > .api_key
export OPENAI_API_KEY=$(cat .api_key)
bash run_eval_gpt.sh dummy-key 2 4o 20
rm .api_key
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
bash run_eval_gpt.sh sk-proj-xxx
```

### 完整评测（全部样本）
```bash
bash run_eval_gpt.sh sk-proj-xxx 2 4o 0
```

### 成本优化（使用便宜的模型）
```bash
bash run_eval_gpt.sh sk-proj-xxx 2 3.5      # GPT-3.5最便宜
bash run_eval_gpt.sh sk-proj-xxx 2 4omini   # GPT-4o-mini性价比高
```

### 对比不同模型
```bash
# 依次评测三个模型
bash run_eval_gpt.sh sk-proj-xxx 2 3.5 100
bash run_eval_gpt.sh sk-proj-xxx 2 4omini 100
bash run_eval_gpt.sh sk-proj-xxx 2 4o 100
```

### 对比不同数据集
```bash
# 使用同一模型评测所有数据集
bash run_eval_gpt.sh sk-proj-xxx 2 4o 50    # CulturalBench
bash run_eval_gpt.sh sk-proj-xxx 3 4o 50    # normad
bash run_eval_gpt.sh sk-proj-xxx 4 4o 50    # cultureLLM
bash run_eval_gpt.sh sk-proj-xxx 5 4o 50    # cultureAtlas
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

### 问题1: 未提供API KEY

```
❌ 错误: 未提供OpenAI API KEY
```

**解决方案**: 确保第一个参数是有效的API KEY
```bash
bash run_eval_gpt.sh sk-proj-your-api-key-here
```

### 问题2: API调用失败

检查API KEY是否有效：
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer sk-proj-your-api-key"
```

### 问题3: 数据文件不存在

确认数据文件路径正确（脚本默认使用 `/root/autodl-fs/` 路径）

### 问题4: 权限问题

确保脚本有执行权限：
```bash
chmod +x run_eval_gpt.sh
```

## 📚 更多信息

详细文档请参考: `GPT_EVAL_README.md`
