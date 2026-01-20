# GPT API 评测脚本使用指南

## 概述

本脚本使用OpenAI GPT模型对文化数据集进行评测，可作为baseline对比。支持对CulturalBench、normad、cultureLLM、cultureAtlas等数据集进行自动化评测。

## 文件说明

- `eval_gpt.py`: Python评测脚本，负责调用GPT API并计算评测指标
- `run_eval_gpt.sh`: Shell启动脚本，负责参数配置和流程控制

## 依赖安装

```bash
# 安装OpenAI Python库
pip install openai tqdm
```

## 使用方法

### 1. 设置API KEY

```bash
export OPENAI_API_KEY='your-api-key-here'
```

获取API KEY: https://platform.openai.com/api-keys

### 2. 运行评测

#### 基础用法

```bash
# 评测CulturalBench数据集（使用默认的gpt-3.5-turbo）
bash run_eval_gpt.sh 2

# 评测normad数据集
bash run_eval_gpt.sh 3

# 评测cultureLLM数据集
bash run_eval_gpt.sh 4

# 评测cultureAtlas数据集
bash run_eval_gpt.sh 5
```

#### 指定GPT模型

```bash
# 使用GPT-4评测CulturalBench
bash run_eval_gpt.sh 2 gpt-4

# 使用GPT-4 Turbo评测
bash run_eval_gpt.sh 2 gpt-4-turbo-preview
```

#### 测试模式（限制样本数）

```bash
# 仅评测前100个样本（用于快速测试）
bash run_eval_gpt.sh 2 gpt-3.5-turbo 100

# 使用GPT-4评测前50个样本
bash run_eval_gpt.sh 2 gpt-4 50
```

## 参数说明

### Shell脚本参数

```bash
bash run_eval_gpt.sh <data_id> [gpt_model] [max_samples]
```

- `data_id` (必需): 数据集ID
  - `2`: CulturalBench
  - `3`: normad
  - `4`: cultureLLM
  - `5`: cultureAtlas

- `gpt_model` (可选): GPT模型名称，默认`gpt-3.5-turbo`
  - 支持: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo-preview` 等

- `max_samples` (可选): 最大评测样本数，默认评测全部

### Python脚本参数

```bash
python eval_gpt.py \
    --data_file <数据集文件路径> \
    --output_dir <输出目录> \
    --model_name <GPT模型名称> \
    [--max_samples <最大样本数>] \
    [--temperature <温度参数>] \
    [--max_retries <最大重试次数>] \
    [--retry_delay <重试延迟秒数>]
```

## 输出文件

评测完成后，会在 `/root/autodl-fs/gpt_eval_results/` 目录下生成：

```
<model>_<dataset>_<timestamp>/
├── eval_config.json      # 评测配置
├── eval_results.json     # 汇总结果（准确率等）
├── detailed_results.json # 详细结果（每个样本的预测和对比）
└── eval.log             # 评测日志
```

### 结果文件格式

#### eval_results.json
```json
{
  "model": "gpt-3.5-turbo",
  "temperature": 0.0,
  "total_samples": 1000,
  "correct_predictions": 850,
  "failed_api_calls": 5,
  "accuracy": 0.85
}
```

#### detailed_results.json
```json
[
  {
    "sample_id": 0,
    "instruction": "Give me the answer from 1 to 4: ...",
    "input": "This question is for Arabic.",
    "expected_answer": "2",
    "gpt_response": "2",
    "extracted_answer": "2",
    "is_correct": true,
    "label": "0"
  },
  ...
]
```

## 评测流程

1. **数据加载**: 读取JSON格式的数据集文件
2. **Prompt构造**: 将instruction和input组合成完整问题
3. **GPT调用**: 使用OpenAI API获取GPT回答
4. **答案提取**: 从GPT响应中提取数字答案（1-4）
5. **结果对比**: 与标准答案(output字段)进行对比
6. **指标计算**: 计算准确率等评测指标
7. **结果保存**: 保存详细结果和汇总指标

## 特性

- ✅ **自动重试**: API调用失败时自动重试（默认3次）
- ✅ **进度显示**: 使用tqdm显示评测进度
- ✅ **答案提取**: 智能提取GPT响应中的数字答案
- ✅ **错误处理**: 完善的错误处理和日志记录
- ✅ **批量评测**: 支持对整个数据集进行批量评测
- ✅ **测试模式**: 支持限制样本数进行快速测试

## 常见问题

### 1. API调用失败

**症状**: 出现 "API调用失败" 错误

**解决方案**:
- 检查OPENAI_API_KEY是否正确设置
- 检查网络连接是否正常
- 检查API配额是否充足
- 查看详细错误信息: `cat <output_dir>/eval.log`

### 2. 答案提取失败

**症状**: `extracted_answer` 显示 "EXTRACTION_FAILED"

**原因**: GPT返回的答案格式不符合预期（不是1-4的数字）

**解决方案**:
- 检查 `detailed_results.json` 中的 `gpt_response` 字段
- 可能需要调整prompt或答案提取逻辑

### 3. 速率限制

**症状**: 频繁出现速率限制错误

**解决方案**:
- 增加重试延迟: 修改 `--retry_delay` 参数
- 使用更高级别的API账户
- 分批评测数据集

## 性能优化

### 成本优化

- 使用 `gpt-3.5-turbo` 而非 `gpt-4` 可显著降低成本
- 使用 `--max_samples` 参数先在小样本上测试

### 速度优化

- 脚本已设置 `max_tokens=50` 限制输出长度
- 温度设为0.0确保确定性输出，减少不必要的随机性

## 与其他评测脚本的对比

| 脚本 | 评测对象 | 用途 |
|------|---------|------|
| `run_eval_gpt.sh` | GPT模型 | Baseline对比 |
| `run_eval_joint_culturemoe.sh` | CultureMoE模型 | 主要评测 |
| `run_eval_lora_only_from_components.sh` | LoRA-only模型 | 消融实验 |

## 示例输出

```
======================================
GPT API 数据集评测
使用OpenAI GPT作为baseline对比
======================================

配置信息:
  GPT模型: gpt-3.5-turbo
  数据集: CulturalBench
  数据文件: /root/autodl-fs/CulturalBench_merge_gen.json
  最大样本数: 全部 (完整评测)
  输出目录: /root/autodl-fs/gpt_eval_results/gpt_3_5_turbo_CulturalBench_20250120_143022

开始GPT API评测...

📂 加载数据集: /root/autodl-fs/CulturalBench_merge_gen.json
  - 总样本数: 1000

🚀 开始评测，共 1000 个样本...
评测进度: 100%|████████████████████| 1000/1000 [10:23<00:00,  1.60it/s]

✅ 详细结果已保存: .../detailed_results.json
✅ 汇总结果已保存: .../eval_results.json

==================================================
📊 评测结果摘要
==================================================
模型: gpt-3.5-turbo
总样本数: 1000
正确预测数: 850
API调用失败数: 5
准确率: 0.8500 (85.00%)
==================================================
```

## 技术细节

### Prompt设计

脚本使用以下prompt模板：

```
{instruction}
{input}

Please provide your answer as a single number (1, 2, 3, or 4) without any explanation.
```

### 答案提取策略

1. 检查响应是否为单个数字（1-4）
2. 使用正则表达式匹配"答案是X"或"Answer: X"等格式
3. 匹配以数字开头的响应
4. 提取响应中的任何独立数字

### API配置

- **temperature**: 0.0（确定性输出）
- **max_tokens**: 50（限制输出长度）
- **system prompt**: "You are a helpful assistant that answers cultural survey questions."

## 许可证

本脚本遵循项目主仓库的许可证。
