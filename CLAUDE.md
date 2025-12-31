# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Culture_Moe 是一个基于 Mixture of Experts (MoE) 架构的文化感知语言模型研究项目。该项目实现了 CultureMoE 联合训练模型，通过端到端的LoRA+MoE联合优化、专家路由机制和CSL文化损失函数来提升模型对不同文化背景的理解和生成能力。

## 核心架构组件

### 1. CultureMoE 联合训练模型架构
- **联合训练架构**: 同时优化预训练LoRA适配器和新增MoE专家层
- **端到端优化**: 避免预训练LoRA权重冻结，实现真正的联合优化
- **CSL文化损失**: 创新的Culture Similarity Loss (CSL)，包含三个互补的损失组件
- **MASK机制**: 双路输入处理，促进专家表示分化
- **文化感知路由**: 通过文化损失引导专家学习文化特定模式
- **内存高效**: 基于LoRA的专家设计，显著降低参数量和内存需求
- **主要组件**:
  - LoRA专家 (`LoRAExpert`): 参数高效的专家网络
  - MoE路由器 (`MoERouter`): 动态专家选择和权重分配
  - MoE FFN层 (`MoEFFNLoRA`): 完整的专家混合前馈网络
  - 联合训练模型 (`JointLoRAMoEModel`): 端到端联合训练架构

### 2. CSL损失函数体系
- **总损失**: `L_total = L_generation + λ × (α × L_aux + β × L_culture)`
- **生成损失**: 标准自回归语言建模损失
- **负载均衡损失**: 防止专家使用不均衡
- **CSL文化损失**: 三组件文化相似性损失
  - `L_culture_router`: 路由专家文化相似性损失
  - `L_culture_share`: 共享专家文化无关损失
  - `L_culture_sr`: 共享-路由专家解耦损失

### 3. 支持的基础模型
- LLaMA 3.1-8B-Instruct (默认)
- Qwen 2.5-7B-Instruct

## 项目结构

```
Culture_Moe/
├── src/llamafactory/          # 核心模型实现
│   └── model/
│       ├── joint_lora_moe_model.py      # 联合训练模型类
│       ├── CultureMoE.py                # 主要MoE模型类
│       ├── simplified_culturemoe.py     # 简化版配置
│       └── simplified_culturemoe_adapter.py # 适配器实现
├── train_joint_lora_moe.py     # 联合训练主脚本
├── train_*.py                  # 其他训练脚本
├── eval_*.py                   # 各种评估脚本
├── run_joint_lora_moe_training.sh     # 联合训练执行脚本
├── run_eval_joint_culturemoe.sh       # 联合模型评估脚本
├── run_*.sh                    # 其他训练/评估执行脚本
├── ft_*.py                     # 微调相关脚本
├── requirements.txt            # Python依赖
├── culturemoe.md              # 详细架构文档
├── readme.md                  # 使用指南
└── examples/                  # 示例和教程
```

## 配置参数 (基于run_joint_lora_moe_training.sh)

### 模型参数 (默认值)
- `backbone`: 基础模型类型 (默认: llama)
- `data_id`: 数据集编号 (默认: 2)
- `use_shared`: 是否使用共享专家 (默认: true)
- `use_mask`: 是否启用MASK机制双路输入处理 (默认: true)
- `use_gate`: 是否使用门控网络 (默认: true)
- `use_culture_loss`: 文化损失模式 (默认: csl)
- `num_moe_experts`: MoE专家数量 (默认: 4)
- `num_activated_experts`: 激活的专家数量 (默认: 2)
- `lora_rank`: LoRA rank (默认: 16)
- `lora_alpha`: LoRA alpha (默认: 32)
- `use_lora`: 是否启用基础LoRA训练 (默认: true)
- `num_gpus`: 使用的GPU数量 (默认: 2)

### 训练参数
- `learning_rate_base`: 基础LoRA学习率 (默认: 2e-4 for LLaMA / 1e-4 for Qwen)
- `learning_rate_moe`: MoE学习率 (默认: 8e-5 for LLaMA / 2e-5 for Qwen)
- `batch_size`: 批次大小 (默认: 2, 支持CSL多样本计算)
- `gradient_accumulation`: 梯度累积步数 (默认: 16)
- `num_epochs`: 训练轮数 (默认: 8)
- `max_seq_len`: 最大序列长度 (动态设置: 384/850)

### CSL损失函数权重
- `lambda`: 辅助损失总权重 (默认: 1.0)
- `alpha`: 负载均衡损失权重 (默认: 0.1)
- `beta`: 文化损失权重 (默认: 0.5)
  - `L_culture_router`: β/3 (路由专家文化相似性损失)
  - `L_culture_share`: β/3 (共享专家文化无关损失)
  - `L_culture_sr`: β/3 (共享-路由解耦损失)

## 数据集配置

支持的数据集 (通过 DATA_ID 参数指定):
- `1`: unified_all_datasets.json
- `2`: CulturalBench_merge_gen.json (默认)
- `3`: normad_merge_gen.json
- `4`: cultureLLM_merge_gen.json

## MASK机制与双路输入处理

### 数据格式支持
```json
{
    "instruction": "Give me the answer from 1 to 4: Do you agree with ...",
    "instruction_mask": "Give me the answer from 1 to 4: Do you agree with ... [MASK]",
    "input": "This question is for a country or language that is Arabic.",
    "output": "2",
    "label": "0"
}
```

### 工作模式
- **USE_MASK=true (默认)**: 双路处理模式
  - 路由专家接收原始输入 (`instruction + input`)
  - 共享专家接收MASK版本输入 (`instruction_mask + input`)
- **USE_MASK=false**: 单路处理模式
  - 所有专家接收相同输入

## 内存优化策略

### GPU内存配置
- **单卡训练**: 7-9GB显存
- **双卡训练**: 4-5GB per GPU
- **推理阶段**: 6-8GB显存

### 环境变量优化
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
```

### 分布式训练配置
- 启用 `find_unused_parameters` 处理top-k路由的动态计算图
- 禁用 `broadcast_buffers` 减少同步开销
- 启用 `gradient_as_bucket_view` 优化梯度存储
- 启用静态图模式优化

## 代码质量标准

### Linting 配置 (ruff)
- 目标Python版本: 3.9+
- 行长度限制: 119字符
- 缩进宽度: 4空格
- 引用风格: 双引号

### Pre-commit Hooks
- AST语法检查
- 大文件检查 (最大25MB)
- 合并冲突检查
- YAML格式检查
- 调试语句检查
- 文件末尾换行符
- 行尾空格清理

## 关键文件位置

### 联合训练脚本 (主要)
- `train_joint_lora_moe.py`: 联合LoRA+MoE训练主脚本
- `run_joint_lora_moe_training.sh`: 联合训练执行脚本
- `run_eval_joint_culturemoe.sh`: 联合模型评估脚本

### 其他训练脚本
- `train_simplified_culturemoe.py`: 简化版CultureMoE训练
- `ft_lora_only_gen.py`: LoRA生成式微调

### 评估脚本
- `eval_simplified_culturemoe.py`: 简化版模型评估
- `eval_culturemoe_from_components.py`: 组件式模型评估
- `eval_base_gen.py`: 基础模型生成式评估

### 核心模型实现
- `src/llamafactory/model/joint_lora_moe_model.py`: 联合训练模型架构
- `src/llamafactory/model/CultureMoE.py`: 主要MoE模型类
- `src/llamafactory/model/simplified_culturemoe_adapter.py`: 简化版适配器

## 使用示例

### 联合训练基础命令
```bash
# 使用默认CSL配置启动联合训练
./run_joint_lora_moe_training.sh llama 2 true true true csl 4 2 16 32 true 2
```

### 参数说明 (按脚本顺序)
1. backbone: 基础模型类型 (llama/qwen)
2. data_id: 数据集编号
3. use_shared: 是否启用共享专家
4. use_mask: 是否启用MASK机制
5. use_gate: 是否启用门控网络
6. use_culture_loss: 文化损失模式 (ori/new/kl/csl/false)
7. num_moe_experts: MoE专家总数
8. num_activated_experts: 激活专家数量
9. lora_rank: LoRA秩
10. lora_alpha: LoRA缩放参数
11. use_lora: 是否启用基础LoRA训练
12. num_gpus: 使用的GPU数量

### 评估命令
```bash
./run_eval_joint_culturemoe.sh /path/to/model llama 2 true true true csl 4 2 2
```

## 故障排查

### 常见问题
1. **显存不足**:
   - 减少batch_size
   - 增加gradient_accumulation_steps

2. **训练不稳定**:
   - 检查learning_rate_base和learning_rate_moe设置
   - 验证CSL损失函数权重配置
   - 确认数据预处理正确性
   - 检查MASK机制配置

3. **分布式训练问题**:
   - 确认CUDA_VISIBLE_DEVICES设置
   - 检查master_port是否被占用
   - 验证nproc_per_node参数

4. **CSL损失相关**:
   - 确保batch_size >= 2 (CSL需要多样本计算)
   - 检查文化标签是否正确传递
   - 验证专家输出是否正常

## 技术创新点

### 1. 联合训练架构
- 端到端同时优化基础LoRA适配器和MoE专家层
- 分层学习率策略优化不同组件的学习过程
- 避免预训练权重冻结，实现真正的联合优化

### 2. CSL文化相似性损失
- 三组件设计：路由专家专业化、共享专家无关性、专家解耦
- 基于余弦相似度的对比学习框架
- 数值稳定的实现保证训练稳定性

### 3. MASK机制双路处理
- 共享专家和路由专家接收不同版本的输入
- 促进专家学习互补表示
- 增强文化感知和专业化能力

## 开发最佳实践
1. **开发要求**: 不要创建debug或fix或test版本的代码，未经允许不可新增任何md文档
2. **测试要求**: 本地环境中没有运行环境，不要试图在本地测试
3. **参数配置**: 严格遵循run_joint_lora_moe_training.sh中的参数顺序和默认值
4. **架构选择**: 优先使用联合训练架构，避免使用简化版本进行新开发