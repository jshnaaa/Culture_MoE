# CultureMoE 联合训练模型架构文档

## 概述

CultureMoE是一个基于Mixture of Experts (MoE)架构的文化感知语言模型，通过联合训练实现基础LoRA适配器和MoE专家层的端到端优化。本文档描述的是联合训练版本（Joint LoRA+MoE Training），该版本同时训练预训练LoRA适配器和新增MoE专家层，实现真正的端到端文化感知训练。

### 主要特点

- **联合训练架构**: 同时优化预训练LoRA适配器和新增MoE专家层
- **端到端优化**: 避免预训练LoRA权重冻结，实现真正的联合优化
- **CSL文化损失**: 创新的Culture Similarity Loss (CSL)，包含三个互补的损失组件
- **MASK机制**: 双路输入处理，促进专家表示分化
- **文化感知路由**: 通过文化损失引导专家学习文化特定模式
- **内存高效**: 基于LoRA的专家设计，显著降低参数量和内存需求
- **分层学习率**: 基础LoRA和MoE组件使用不同学习率的精细化优化

## 整体架构

CultureMoE联合训练架构包含以下主要组件：

1. **基础模型层**: LLaMA 3.1-8B-Instruct 或 Qwen 2.5-7B-Instruct 作为骨干网络
2. **联合LoRA适配器**: 同时训练的基础模型LoRA适配器，实现端到端优化
3. **MoE专家层**: 新增的专家混合层，包含：
   - **路由专家群**: 多个基于LoRA的文化专家网络
   - **共享专家**: 学习文化无关表示的共享专家（默认启用）
   - **MoE路由器**: 负责专家选择和权重分配的路由网络
   - **门控网络**: 专家输出融合机制（默认启用）
4. **CSL文化损失函数**: 三组件文化相似性损失，促进专家文化专业化
5. **MASK机制**: 双路输入处理，增强文化感知能力
6. **分层优化器**: 基础LoRA和MoE组件的差异化学习率优化

## 核心组件详细说明

### 1. LoRA专家 (LoRAExpert)

LoRA专家是CultureMoE的核心计算单元，采用低秩适应技术实现参数高效的专家设计。

#### 设计理念
- 为FFN的每个线性层添加LoRA分支，包括gate_proj、up_proj和down_proj
- 保持原始FFN权重冻结，只训练LoRA参数
- 通过低秩分解实现参数压缩，显著降低训练开销

#### 网络结构
每个LoRA专家包含三组LoRA分支：
- **gate_proj LoRA**: 从hidden_dim到intermediate_dim的门控投影
- **up_proj LoRA**: 从hidden_dim到intermediate_dim的上投影
- **down_proj LoRA**: 从intermediate_dim到hidden_dim的下投影

每个LoRA分支由两个线性层组成：A矩阵(rank降维)和B矩阵(rank升维)，缩放因子为α/r。

#### 前向传播流程
1. **原始FFN计算**: 使用冻结的原始权重计算gate和up分支输出
2. **LoRA分支计算**: 通过A-B矩阵链计算LoRA增量，应用缩放因子
3. **组合激活**: 将原始输出与LoRA增量相加，进行SiLU激活和逐元素乘法
4. **输出投影**: 通过down_proj原始权重和LoRA增量生成最终输出

#### 参数配置
- **LoRA rank**: 16 (默认)，控制低秩分解的维度
- **LoRA alpha**: 32 (默认)，控制LoRA贡献的缩放
- **缩放因子**: alpha / rank = 2.0
- **参数量**: 约0.26M per expert，相比原始FFN的约50M大幅减少

### 2. MoE路由器 (MoERouter)

MoE路由器负责根据输入特征动态选择和加权专家，是实现文化感知的关键组件。

#### 设计理念
- 基于输入隐藏状态计算专家激活权重
- 使用数值稳定的softmax避免梯度爆炸
- 支持文化感知的路由决策学习

#### 网络结构
路由器由单个线性层构成，将hidden_dim映射到num_experts维度。采用保守的正态分布初始化(std=0.01)避免训练初期的不稳定。

#### 前向传播流程
1. **路由logits计算**: 通过线性变换计算每个专家的原始分数
2. **数值稳定化**: 使用shifted softmax技术防止数值溢出
3. **专家权重归一化**: 应用softmax得到归一化的专家权重分布
4. **异常检测**: 检测NaN/Inf并提供均匀分布fallback

#### 输入输出规格
- **输入**: hidden_states [B, L, H] - 批次大小×序列长度×隐藏维度
- **输出**: expert_weights [B, L, num_experts], router_logits [B, L, num_experts]

### 3. MoE FFN层 (MoEFFNLoRA)

MoE FFN层是完整的专家混合前馈网络，整合了路由、专家计算和融合机制。

#### 设计理念
- 完全替换原始FFN为MoE结构
- 支持Top-k稀疏激活和Dense全激活两种模式
- 集成共享专家和门控网络

#### 核心组件
- **路由器**: 负责专家选择的MoERouter实例
- **专家池**: 包含num_experts个LoRAExpert的模块列表
- **共享专家**: 始终激活的LoRA专家，学习文化无关表示
- **门控网络**: 专家输出融合网络

#### 专家激活模式

## Simplified MoE

Simplified MoE是CultureMoE的简化版本，采用纯LoRA MoE架构，将所有Transformer层的FFN组件替换为LoRA MoE专家网络。相比联合训练版本，Simplified MoE专注于MoE专家层的训练，提供了更轻量级的文化感知解决方案。

### 整体架构设计

Simplified MoE采用全层替换策略，将基础模型的所有FFN层替换为LoRA MoE专家网络。这种设计理念基于以下考虑：

**全层MoE化**: 不同于联合训练版本仅在特定层添加MoE，Simplified版本将所有层的FFN都替换为MoE结构。这种设计使得模型在每一层都具备专家选择能力，能够在不同抽象层次上学习文化特定的表示。

**纯LoRA专家**: 所有专家均采用LoRA技术实现，保持原始FFN权重冻结状态，仅训练低秩适应参数。这种设计显著减少了可训练参数数量，同时保持了模型的表达能力。

**简化训练流程**: 相比联合训练的复杂优化策略，Simplified版本使用统一的学习率和简化的训练配置，降低了训练复杂度和调参难度。

### 核心组件功能

**LoRA MoE专家网络**: 每个专家继承了联合训练版本的LoRA专家设计，包含gate_proj、up_proj和down_proj三个LoRA分支。专家数量可配置，默认为4个专家，支持1到8个专家的灵活配置。

**专家激活机制**: 支持Top-k稀疏激活和Dense全激活两种模式。当激活专家数等于总专家数时自动切换为Dense模式，所有专家参与计算；否则采用Top-k机制选择最相关的专家。

**可选共享专家**: 通过USE_SHARED参数控制是否启用共享专家。共享专家始终激活，负责学习文化无关的通用表示，与路由专家形成互补。默认配置为关闭状态，简化模型结构。

**可选门控融合**: 通过USE_GATE参数控制是否启用MoE内部的门控网络。门控网络负责融合多个专家的输出，提供额外的表示学习能力。默认配置为关闭状态，减少参数复杂度。

### 文化损失配置

Simplified MoE支持多种文化损失模式，通过USE_CULTURE_LOSS参数进行控制：

**原始文化损失模式**: 使用基础的文化对比损失，促进专家学习文化特异性表示。

**增强文化损失模式**: 采用改进的文化感知损失函数，增强专家之间的文化分化效果。

**KL散度损失模式**: 基于KL散度的文化相似性损失，提供不同的优化目标。

**无文化损失模式**: 仅使用负载均衡损失，专注于专家负载分布的优化。

### 参数配置策略

**LoRA参数设置**: 默认使用rank=16和alpha=32的配置，在参数效率和表达能力之间取得平衡。支持1到64的rank范围调整，以适应不同的计算资源约束。

**专家配置**: MoE专家数量默认为4个，激活专家数默认为2个，实现Top-2稀疏激活。这种配置在计算效率和模型容量之间提供良好平衡。

**损失权重配置**: Lambda默认设置为1.0，确保辅助损失的有效性；Alpha设置为0.1，控制负载均衡损失的强度；Beta设置为0.5，调节文化损失的影响程度。

**训练参数优化**: 批次大小设置为2以支持文化损失的对比学习，梯度累积步数为8，有效批次大小为32。学习率统一设置为1e-4，简化优化过程。

### 内存和计算优化

**动态序列长度**: 根据数据集特性动态设置最大序列长度，NormAD数据集使用769，长文本数据集使用512，其他数据集使用256，实现显存使用的精细化控制。

**内存优化环境**: 配置PYTORCH_CUDA_ALLOC_CONF参数，设置内存分片和垃圾回收阈值，提高GPU内存利用效率。

**多GPU支持**: 支持1到8卡的分布式训练，自动检测可用GPU数量并进行合理分配。使用torchrun进行多卡协调，确保训练稳定性。

**层级替换策略**: 对于LLaMA模型替换全部32层FFN，对于Qwen模型替换全部28层FFN，实现全面的MoE化改造。

Simplified MoE版本为研究者提供了一个轻量级的文化感知MoE实现方案，在保持核心功能的同时简化了训练流程，适合快速原型开发和资源受限环境下的实验。

**Dense模式** (num_activated_experts == num_moe_experts):
- 激活所有专家并使用路由权重加权组合
- 计算复杂度高但表达能力最强
- 适合需要充分利用所有专家知识的场景

**Sparse模式** (Top-k激活):
- 仅激活权重最高的k个专家
- 对top-k权重重新归一化确保和为1
- 计算效率高但可能损失部分信息

### 4. CSL文化相似性损失函数 (Culture Similarity Loss)

CSL是CultureMoE联合训练的核心创新，通过三个互补的损失组件实现精细的文化感知训练。

#### 4.1 总损失函数 (Total Loss)

**总损失公式**:
$$L_{total} = L_{generation} + \alpha \times L_{aux} + \beta \times L_{culture}$$

其中：
- $L_{generation}$: 主要生成损失（语言建模损失）
- $L_{aux}$: 负载均衡损失（辅助损失）
- $L_{culture}$: 文化损失（CSL两组件损失）
- $\alpha$: 负载均衡损失系数 (默认0.01)
- $\beta$: 文化损失权重 (默认0.01)

当USE_CULTURE_LOSS=csl时，文化损失采用创新的CSL设计：
$$L_{culture} = L_{culture\_router} + L_{culture\_sr}$$

#### 4.2 生成损失 (Generation Loss)

**生成损失公式**:
$$L_{generation} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i | x_i, \theta)$$

其中：
- $N$: 训练样本数量
- $x_i$: 第$i$个输入序列（instruction + input）
- $y_i$: 第$i$个目标序列（output）
- $\theta$: 模型参数
- $P(y_i | x_i, \theta)$: 给定输入下目标序列的条件概率

#### 4.3 负载均衡损失 (Load Balancing Loss)

**负载均衡损失公式**:
$$L_{balance} = \frac{1}{L_{layers}} \sum_{l=1}^{L_{layers}} L_{balance}^{(l)}$$

其中每层的负载均衡损失为：
$$L_{balance}^{(l)} = \text{MSE}(\bar{u}^{(l)}, \frac{1}{E} \mathbf{1})$$

**组成部分**:
- $L_{layers}$: MoE层总数
- $\bar{u}^{(l)} = \frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} w_{b,t}^{(l)}$: 第$l$层的平均专家使用率
- $w_{b,t}^{(l)} \in \mathbb{R}^E$: 第$l$层在样本$b$时间步$t$的专家权重
- $E$: 专家数量
- $\mathbf{1}$: 全1向量

#### 4.4 CSL两组件损失

##### 4.4.1 路由专家文化相似性损失 (L_culture_router)

**设计目标**: 同文化样本应激活相似的专家组合，不同文化样本应激活不同的专家组合。

**数学公式**:
$$L_{culture\_router} = \frac{1}{|P|} \sum_{(i,j) \in P} L_{router}^{(i,j)}$$

其中：
$$L_{router}^{(i,j)} = \begin{cases}
1 - sim(wr_i, wr_j) & \text{if } cul_i = cul_j \text{ (鼓励相似)} \\
sim(wr_i, wr_j) & \text{if } cul_i \neq cul_j \text{ (惩罚相似)}
\end{cases}$$

##### 4.4.2 共享-路由专家解耦损失 (L_culture_sr)

**设计目标**: 对于同一个样本，共享专家和路由专家的输出表示应尽可能不同，促进互补学习。

**数学公式**:
$$L_{culture\_sr} = \frac{1}{B} \sum_{i=1}^{B} sim(es_i, er_i)$$

**重要说明**: 已移除L_culture_share（共享专家文化无关损失）组件，CSL现在只包含两个组件：L_culture_router和L_culture_sr。

### 5. MASK机制与双路输入处理

MASK机制是CultureMoE联合训练的创新特性，通过双路输入处理实现更精细的文化感知训练。

#### 设计理念
- **双路输入**: 共享专家接收带MASK的输入，路由专家接收原始输入
- **表示分化**: 促进共享专家和路由专家学习不同的语义表示
- **文化解耦**: 通过输入差异化增强文化感知专业化能力
- **灵活配置**: 支持启用/禁用MASK机制的灵活切换（默认启用）

#### 核心机制

**双路输入处理**:
- **路由专家输入**: `instruction + input` (原始完整输入)
- **共享专家输入**: `instruction_mask + input` (带MASK标记的输入)
- **输出融合**: 通过门控网络融合两路专家的输出表示

**数据格式支持**:
```json
{
    "instruction": "Give me the answer from 1 to 4: Do you agree with ...",
    "instruction_mask": "Give me the answer from 1 to 4: Do you agree with ... [MASK]",
    "input": "This question is for a country or language that is Arabic.",
    "output": "2",
    "label": "0"
}
```

#### 工作流程

**USE_MASK=true时** (双路处理模式，默认):
1. **数据预处理**: 同时tokenize instruction和instruction_mask版本的输入
2. **路由专家**: 接收`instruction + input`，学习完整的语义理解
3. **共享专家**: 接收`instruction_mask + input`，学习文化无关的基础表示
4. **输出融合**: 通过门控网络加权融合两路专家输出
5. **损失计算**: CSL损失基于不同输入版本计算专家表示差异

**USE_MASK=false时** (单路处理模式):
1. **统一输入**: 共享专家和路由专家都接收相同的`instruction + input`
2. **传统MoE**: 回退到标准的专家混合架构
3. **兼容性**: 保持与现有训练流程的完全兼容

### 6. 联合训练模型架构 (JointLoRAMoEModel)

JointLoRAMoEModel是联合训练版本的核心架构，实现了基础LoRA适配器和MoE专家层的同时优化。

#### 设计理念
- **端到端优化**: 同时训练基础模型LoRA适配器和新增MoE专家层
- **分层学习率**: 基础LoRA和MoE组件使用不同学习率的精细化优化
- **统一架构**: 集成LoRA微调和MoE专家训练于单一模型框架
- **内存高效**: 优化的参数管理和梯度计算，支持大规模联合训练

#### 核心特性
- **联合优化**: 避免预训练LoRA权重冻结，实现基础适配器和专家层的协同学习
- **差异化学习率**: 基础LoRA使用较小学习率保持稳定性，MoE使用较大学习率促进专业化
- **CSL损失集成**: 原生支持三组件CSL文化损失，优化文化感知能力
- **灵活配置**: 支持不同的专家数量、激活模式和共享专家配置

## 联合训练配置

### 模型参数（默认值基于run_joint_lora_moe_training.sh）
| 参数 | 默认值 | 描述 |
|------|--------|------|
| backbone | llama | 基础模型类型 (llama/qwen) |
| data_id | 2 | 数据集编号 |
| use_shared | true | 是否使用共享专家 |
| use_mask | true | 是否启用MASK机制双路输入处理 |
| use_gate | true | 是否使用门控网络 |
| use_culture_loss | csl | 文化损失模式 (ori/new/kl/csl/false) |
| alpha | 0.01 | 负载均衡损失系数 |
| beta | 0.01 | 文化专注性损失CSL系数 |
| num_moe_experts | 4 | MoE专家数量 |
| num_activated_experts | 2 | 激活的专家数量 |
| lora_rank | 16 | LoRA rank |
| lora_alpha | 32 | LoRA alpha |
| use_lora | true | 是否启用基础LoRA训练 |
| num_gpus | 2 | 使用的GPU数量 |

### 训练参数
| 参数 | 默认值 | 描述 |
|------|--------|------|
| learning_rate_base | 2e-4 (LLaMA) / 1e-4 (Qwen) | 基础LoRA学习率 |
| learning_rate_moe | 8e-5 (LLaMA) / 2e-5 (Qwen) | MoE学习率 |
| batch_size | 2 | 批次大小 (支持CSL多样本计算) |
| gradient_accumulation | 16 | 梯度累积步数 |
| num_epochs | 8 | 训练轮数 |
| max_seq_len | 384/850 | 最大序列长度 (动态设置) |

### CSL损失函数权重
| 损失类型 | 权重 | 描述 |
|----------|------|------|
| generation_loss | 1.0 | 主要生成损失 |
| L_culture_router | BETA | 路由专家文化相似性损失 |
| L_culture_sr | BETA | 共享-路由解耦损失 |
| L_culture_total | BETA | 文化专注性损失CSL（两组件） |
| aux_loss | ALPHA | 辅助损失（负载均衡） |

其中ALPHA=0.01（负载均衡损失系数），BETA=0.01（文化专注性损失CSL系数）为默认权重配置。

## 内存优化策略

### 1. 数据类型优化
- **模型权重**: float16精度减少显存占用
- **损失计算**: float16计算保持数值稳定
- **梯度**: 自动混合精度训练

### 2. 内存管理
- **定期清理**: 每N步清理GPU缓存避免碎片化
- **梯度同步**: DDP模式下优化同步减少通信开销
- **设备一致性**: 确保所有参数在正确设备避免数据传输

### 3. 分布式训练
DDP配置针对MoE特点优化：
- 启用find_unused_parameters处理top-k路由的动态计算图
- 禁用broadcast_buffers减少同步开销
- 启用gradient_as_bucket_view优化梯度存储

## 性能特点

### 参数效率
- **基础模型**: 约8B参数（完全冻结）
- **可训练参数**: 约100M（约1.25%的总参数量）
- **LoRA参数**: 注意力层低秩适应参数
- **MoE参数**: 专家网络和路由器参数

### 计算复杂度
- **Dense模式**: O(B×L×H×E)，其中E为专家总数
- **Sparse模式**: O(B×L×H×K)，其中K为激活专家数
- **路由开销**: O(B×L×H×E)，路由计算的额外开销

### 内存占用
- **单卡训练**: 7-9GB显存
- **双卡训练**: 4-5GB per GPU
- **推理阶段**: 6-8GB显存

## 使用示例

### 联合训练基础命令
使用默认CSL配置启动联合训练：
```bash
./run_joint_lora_moe_training.sh llama 2 true true true csl 4 2 16 32 true 2
```

### 联合训练参数说明（按脚本顺序）
1. backbone: 基础模型类型 (llama/qwen)
2. data_id: 数据集编号
3. use_shared: 是否启用共享专家
4. use_mask: 是否启用MASK机制
5. use_gate: 是否启用门控网络
6. use_culture_loss: 文化损失模式 (ori/new/kl/csl/false)
7. alpha: 负载均衡损失系数 (默认0.01)
8. beta: 文化专注性损失CSL系数 (默认0.01)
9. num_moe_experts: MoE专家总数
10. num_activated_experts: 激活专家数量
11. lora_rank: LoRA秩
12. lora_alpha: LoRA缩放参数
13. use_lora: 是否启用基础LoRA训练
14. num_gpus: 使用的GPU数量

### 配置实验示例

**启用CSL文化损失**:
```bash
./run_joint_lora_moe_training.sh llama 2 true true true csl 0.01 0.01 4 2 16 32 true 2
```

**使用传统文化损失**:
```bash
./run_joint_lora_moe_training.sh llama 2 true true true new 4 2 16 32 true 2
```

**禁用文化损失**:
```bash
./run_joint_lora_moe_training.sh llama 2 true true true false 4 2 16 32 true 2
```

**禁用MASK机制**:
```bash
./run_joint_lora_moe_training.sh llama 2 true false true csl 4 2 16 32 true 2
```

**Dense模式（激活所有专家）**:
```bash
./run_joint_lora_moe_training.sh llama 2 true true true csl 4 4 16 32 true 2
```

### 评估命令
```bash
./run_eval_joint_culturemoe.sh /path/to/model llama 2 true true true csl 4 2 2
```

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

### 4. LoRA专家设计
- 参数高效的专家实现方案
- 保持原始FFN权重作为稳定基础
- 仅训练少量LoRA参数实现专家差异化

### 5. 内存优化技术
- float16精度计算减少显存需求
- 分布式训练支持大规模部署
- 动态内存管理避免显存碎片化

## 总结

CultureMoE联合训练通过端到端的LoRA+MoE架构和创新的CSL损失函数，在保持模型性能的同时实现了文化专业化。其创新的联合训练设计为大规模语言模型的文化适应提供了高效的解决方案。

### 核心优势
1. **参数高效**: 仅训练1.25%的参数实现文化适应
2. **文化感知**: 通过CSL损失和MASK机制实现专家的文化专业化
3. **内存友好**: 针对48GB×2卡环境优化设计
4. **端到端**: 联合训练避免权重冻结，实现真正的优化
5. **训练稳定**: 数值稳定的实现和完善的异常处理

该架构为跨文化语言理解和生成任务提供了一个强大、高效、可解释的解决方案，在文化多样性建模方面具有重要的理论价值和实用意义。

## 推理评估脚本功能说明

### run_eval_joint_culturemoe.sh 脚本功能

`run_eval_joint_culturemoe.sh` 脚本提供了完整的联合训练模型推理评估功能，支持所有训练时的配置参数，包括关键的MASK机制配置。

#### 脚本参数说明（按顺序）

```bash
./run_eval_joint_culturemoe.sh MODEL_PATH BACKBONE DATA_ID USE_SHARED USE_MASK USE_GATE USE_CULTURE_LOSS NUM_MOE_EXPERTS NUM_ACTIVATED_EXPERTS NUM_GPUS
```

1. **MODEL_PATH**: 训练好的联合模型路径
2. **BACKBONE**: 基础模型类型 (llama/qwen)
3. **DATA_ID**: 数据集编号 (0-5, 24)
4. **USE_SHARED**: 是否使用共享专家 (true/false)
5. **USE_MASK**: 是否启用MASK机制双路输入处理 (true/false)
6. **USE_GATE**: 是否使用门控网络 (true/false)
7. **USE_CULTURE_LOSS**: 文化损失模式 (ori/new/kl/csl/false)
8. **NUM_MOE_EXPERTS**: MoE专家数量
9. **NUM_ACTIVATED_EXPERTS**: 激活的专家数量
10. **NUM_GPUS**: 使用的GPU数量

#### USE_MASK参数功能详解

**USE_MASK=true (推荐默认值)**:
- **双路推理模式**: 模型在推理时使用与训练时一致的MASK机制
- **输入处理**:
  - 路由专家接收原始输入 (`instruction + input`)
  - 共享专家接收MASK版本输入 (`instruction_mask + input`)
- **专家分化**: 奇数专家使用原始隐藏状态，偶数专家使用MASK隐藏状态
- **性能优势**: 充分利用训练时学到的双路表示分化能力
- **适用场景**: 模型使用MASK机制训练时的标准推理模式

**USE_MASK=false**:
- **单路推理模式**: 所有专家使用相同的原始输入
- **兼容性**: 适用于未使用MASK机制训练的模型
- **简化处理**: 回退到标准MoE推理流程
- **性能影响**: 可能无法充分发挥MASK训练模型的性能优势

**重要说明**: 当USE_SHARED=false（消融模式）时，MASK机制会自动禁用，因为MASK主要用于shared专家和routing专家的表示分化。在消融模式下，所有routing专家都使用原始隐藏状态，确保消融研究的一致性。

通过USE_MASK参数的灵活配置，可以验证MASK机制对模型性能的具体影响，为模型优化和部署提供重要参考。

#### USE_SHARED参数功能详解

USE_SHARED参数控制推理时是否使用共享专家，支持对训练好的模型进行消融研究：

**USE_SHARED=true (默认推荐值)**:
- **完整推理模式**: 使用模型训练时的完整架构
- **专家组合**: 路由专家 + 共享专家双重输出
- **MASK分化**: 启用专家间的输入表示分化机制
- **融合策略**: 通过门控网络或简单平均融合专家输出
- **性能表现**: 充分发挥模型的完整性能潜力
- **适用场景**:
  - 标准模型推理和性能评估
  - 生产环境部署
  - 模型能力的完整展示

**USE_SHARED=false (消融研究模式)**:
- **消融推理模式**: 推理时禁用共享专家组件
- **专家组合**: 仅使用路由专家输出
- **MASK分化**: 自动禁用，所有路由专家使用统一输入
- **计算优化**: 跳过共享专家计算，节省约25%的专家层开销
- **研究价值**: 量化共享专家对模型性能的具体贡献
- **适用场景**:
  - 消融实验：测量共享专家的性能贡献
  - 架构分析：理解不同专家组件的作用
  - 计算优化：在资源受限环境下的推理加速 

**技术实现机制**:
- **参数传递**: 推理时的use_shared参数覆盖训练配置
- **条件跳过**: 完全绕过共享专家的前向计算
- **输入统一**: 消融模式下所有路由专家使用原始隐藏状态
- **架构保持**: 模型结构完整，仅选择性禁用特定组件

**消融研究意义**:
通过对比USE_SHARED=true和USE_SHARED=false的性能差异，可以：
1. **量化贡献**: 精确测量共享专家对模型性能的提升幅度
2. **验证设计**: 验证共享专家架构设计的有效性
3. **优化部署**: 为不同部署场景选择合适的模型配置
4. **理论分析**: 深入理解MoE架构中不同组件的作用机制

这种灵活的消融控制机制为CultureMoE模型的深入分析和优化部署提供了重要工具。

## 脚本依赖关系

### 脚本到Python文件映射

- **base**: run_ft_base.sh
  - 调用文件: ft_base_gen.py

- **lora only**: run_ft_lora_only_gen.sh, run_eval_lora_only_from_components.sh
  - run_ft_lora_only_gen.sh → ft_lora_only_gen.py
  - run_eval_lora_only_from_components.sh → eval_lora_only_from_components.py

- **simplified culturemoe**: run_simplified_culturemoe.sh, run_ablation_study.sh
  - run_simplified_culturemoe.sh → train_simplified_culturemoe.py
  - run_ablation_study.sh → eval_simplified_culturemoe.py

- **joint culturemoe**: run_joint_lora_moe_training.sh, run_eval_joint_culturemoe.sh
  - run_joint_lora_moe_training.sh → train_joint_lora_moe.py
  - run_eval_joint_culturemoe.sh → eval_joint_culturemoe.py

- **vsm**: run_eval_vsm13.sh
  - 调用文件: eval_vsm13.py

- **milora**: run_ft_milora_gen.sh
  - 调用文件: ft_milora_gen.py

- **mixlora**: run_ft_mixlora.sh
  - 调用文件: ft_mixlora.py

### src/llamafactory/model依赖关系

被使用的核心模型文件：
- **joint_lora_moe_model.py**: 联合训练模型架构 (JointLoRAMoEModel, JointLoRAMoEConfig)
- **CultureMoE.py**: 主要MoE模型类 (LlamaSharedRouterExpertsModel)
- **simplified_culturemoe.py**: 简化版配置 (SimplifiedCultureMoEConfig)
- **simplified_culturemoe_adapter.py**: 简化版适配器 (create_simplified_culturemoe_model)
- **mixlora.py**: MixLoRA配置 (MixLoRAConfig)
- **mixlora_adapter.py**: MixLoRA适配器 (create_mixlora_model)
- **moe_args.py**: MoE参数配置 (ModelArgs)

