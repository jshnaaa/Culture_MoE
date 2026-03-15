# 5-Agent Cultural Multi-Agent Debate (MAD) System

## 概述

这是一个基于5个文化视角的多代理辩论系统,用于文化理解任务的评估。不同于传统的2-Agent对抗式辩论,本系统采用多元文化协作的方式,从5个主要文化区域的视角进行讨论和决策。

## 系统架构

### 5个文化Agent

1. **Asia Agent (亚洲)**
   - 文化特征: 集体主义、和谐、传统、间接沟通
   - 代表区域: 东亚、东南亚、南亚

2. **Western Agent (西方)**
   - 文化特征: 个人主义、理性、自由、直接沟通
   - 代表区域: 北美、欧洲

3. **South America Agent (南美)**
   - 文化特征: 社区纽带、热情、家庭、表达力
   - 代表区域: 拉丁美洲

4. **Oceania Agent (大洋洲)**
   - 文化特征: 自然和谐、多元包容、平等、实用
   - 代表区域: 澳大利亚、新西兰、太平洋岛屿

5. **Africa Agent (非洲)**
   - 文化特征: Ubuntu哲学、社区、口述传统、智慧
   - 代表区域: 非洲大陆

### 讨论流程

```
Round 0 (初始回答)
├─ 5个Agent独立回答问题
└─ 每个Agent基于自己的文化背景给出答案

Round 1 (第一轮讨论)
├─ 每个Agent看到其他4个Agent的Round 0答案
├─ 基于多元文化视角反思和更新答案
└─ 可以改变或保持原答案

Round 2 (第二轮讨论)
├─ 每个Agent看到其他4个Agent的Round 1答案
├─ 做出最终决策
└─ 综合所有文化视角

最终决策
├─ 检查是否达成多数共识 (≥3/5 Agent同意)
├─ 如果有共识 → 使用共识答案
└─ 如果无共识 → Summarizer Agent综合所有视角做决策
```

### 决策机制

**多数投票 (Consensus)**
- 如果≥3个Agent在Round 2给出相同答案
- 直接采用这个答案作为最终答案
- 体现多数文化的共同理解

**Summarizer后备**
- 如果没有达成多数共识(每个答案都<3票)
- 召唤Summarizer Agent
- Summarizer综合所有5个Agent的完整讨论
- 做出最终决策

## 使用方法

### 基本命令

```bash
bash run_eval_mad_5cultural.sh MODEL_TYPE DATA_ID [MAX_SAMPLES] [RANDOM_P]
```

### 参数说明

- **MODEL_TYPE**: 模型类型
  - `1` = LLaMA 3.1-8B-Instruct
  - `2` = Qwen 2.5-7B-Instruct

- **DATA_ID**: 数据集编号
  - `2` = CulturalBench
  - `3` = NORMAD
  - `4` = CultureLLM
  - `5` = CultureAtlas

- **MAX_SAMPLES** (可选,默认10):
  - `> 0`: 取前N个样本(忽略RANDOM_P)
  - `= 0`: 使用RANDOM_P随机采样

- **RANDOM_P** (可选,默认0.1):
  - 仅当MAX_SAMPLES=0时生效
  - `1.0`: 使用全部数据
  - `0.1`: 随机采样10%数据

### 使用示例

**快速测试 (10样本)**
```bash
bash run_eval_mad_5cultural.sh 1 2
# LLaMA在CulturalBench上测试10个样本
```

**中等规模测试 (50样本)**
```bash
bash run_eval_mad_5cultural.sh 2 3 50
# Qwen在NORMAD上测试50个样本
```

**随机采样 (10%数据)**
```bash
bash run_eval_mad_5cultural.sh 1 4 0
# LLaMA在CultureLLM上随机采样10%数据
```

**随机采样 (20%数据)**
```bash
bash run_eval_mad_5cultural.sh 2 5 0 0.2
# Qwen在CultureAtlas上随机采样20%数据
```

**完整评估 (全部数据)**
```bash
bash run_eval_mad_5cultural.sh 1 2 0 1.0
# LLaMA在CulturalBench上使用全部数据
```

## 输出文件

评估完成后会在 `/root/autodl-fs/mad_5cultural_results/` 目录下生成:

### 1. detailed_results.json
完整的辩论历史和结果
```json
{
  "question": "问题内容",
  "context": "上下文",
  "true_answer": "1",
  "debate_history": {
    "round_0": {
      "asia": {"answer": "1", "reasoning": "...", "confidence": 80},
      "western": {"answer": "2", ...},
      ...
    },
    "round_1": {...},
    "round_2": {...}
  },
  "final_answer": "1",
  "decision_method": "consensus",  // 或 "summarizer"
  "has_consensus": true,
  "correct": true
}
```

### 2. summary_statistics.json
汇总统计指标
```json
{
  "accuracy": 0.85,
  "total_samples": 100,
  "correct_predictions": 85,
  "consensus_rate": 0.72,           // 多数共识率
  "summarizer_usage_rate": 0.28,   // Summarizer使用率
  "asia_r0_accuracy": 0.65,         // 亚洲Agent初始准确率
  "western_r0_accuracy": 0.70,
  "south_america_r0_accuracy": 0.60,
  "oceania_r0_accuracy": 0.68,
  "africa_r0_accuracy": 0.62,
  "mad_gain": 0.15                  // MAD增益
}
```

### 3. generated_answers.json
每个样本的答案报告
```json
[
  {
    "question": "...",
    "cultural_answers": {
      "asia": {"round_0": "1", "round_1": "1", "round_2": "1"},
      "western": {"round_0": "2", "round_1": "1", "round_2": "1"},
      ...
    },
    "final_answer": "1",
    "decision_method": "consensus",
    "has_consensus": true,
    "true_answer": "1",
    "is_correct": true
  }
]
```

### 4. eval_results.json
标准评估指标
```json
{
  "accuracy": 0.85,
  "precision": 0.85,
  "recall": 0.85,
  "f1_score": 0.85,
  "total_samples": 100,
  "correct_predictions": 85
}
```

### 5. eval.log
完整的运行日志

## 性能特性

### 并行化
- 同一轮次的5个Agent并行生成
- 使用ThreadPoolExecutor(max_workers=5)
- 线程锁保护CUDA操作确保安全

### 显存优化
- 使用float16精度
- 禁用fast tokenizer确保多线程安全
- 环境变量优化: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`

### 时间估算
- 每个样本约需5-10分钟(取决于模型和硬件)
- Round 0: ~2分钟(5个Agent并行)
- Round 1: ~2分钟(5个Agent并行)
- Round 2: ~2分钟(5个Agent并行)
- Summarizer(如需): ~1分钟

## 评估指标

### 核心指标

1. **Final Accuracy**: MAD系统的最终准确率
2. **Consensus Rate**: 达成多数共识的样本比例
3. **Summarizer Usage**: 需要Summarizer决策的样本比例

### 文化Agent指标

每个文化Agent的Round 0准确率:
- Asia R0 Accuracy
- Western R0 Accuracy
- South America R0 Accuracy
- Oceania R0 Accuracy
- Africa R0 Accuracy

### MAD增益

```
MAD Gain = MAD Accuracy - Best_Agent_R0_Accuracy
```

正值表示MAD系统优于最好的单个Agent。

## 与2-Agent MAD的对比

| 特性 | 2-Agent MAD | 5-Agent Cultural MAD |
|------|-------------|----------------------|
| **Agent数量** | 2 (正方/反方) | 5 (5个文化) |
| **角色定位** | 对抗辩论 | 多元协作 |
| **决策机制** | Judge选择 | 多数投票+Summarizer |
| **文化视角** | 无特定文化 | 5个主要文化区域 |
| **共识机制** | 无 | 多数共识(≥3/5) |
| **生成次数** | 7次(2×3+Judge) | 15次(5×3)+可能的Summarizer |
| **并行化** | 2-way | 5-way |
| **适用场景** | 通用辩论 | 文化理解任务 |

## 优势

1. **多元文化视角**: 从5个主要文化区域分析问题
2. **协作而非对抗**: 强调理解和学习,而非赢得辩论
3. **共识机制**: 多数投票体现跨文化共同理解
4. **灵活决策**: 无共识时Summarizer综合判断
5. **更高准确率**: 理论上能利用5个Agent的互补优势

## 注意事项

1. **计算成本**: 5-Agent系统比2-Agent系统计算量更大
2. **显存需求**: 确保GPU有足够显存(推荐≥24GB)
3. **时间消耗**: 每个样本需要更长时间处理
4. **样本量选择**: 建议先用小样本(10-50)测试,再扩大规模

## 故障排查

### 显存不足
- 减少并行度(修改max_workers)
- 使用更小的batch size
- 减少max_new_tokens

### 程序卡住
- 检查CUDA_LAUNCH_BLOCKING设置
- 查看eval.log日志
- 确认模型路径正确

### 准确率异常
- 检查数据集是否正确加载
- 验证答案解析逻辑
- 查看detailed_results.json分析具体案例

## 未来改进方向

1. **更多文化区域**: 增加中东、南亚等区域
2. **动态Agent数量**: 根据问题类型选择相关文化Agent
3. **加权投票**: 根据Agent置信度加权
4. **层次化讨论**: 先区域内讨论,再跨区域讨论
5. **文化特征学习**: 让Agent学习更准确的文化特征

## 引用

如果使用本系统,请引用:
```
5-Agent Cultural Multi-Agent Debate System for Cultural Understanding Evaluation
```

## 联系方式

如有问题或建议,请通过项目issue反馈。
