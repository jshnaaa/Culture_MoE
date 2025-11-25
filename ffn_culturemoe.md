# FFN集成CultureMoE模型架构文档

## 📋 目录
1. [模型概述](#模型概述)
2. [整体架构](#整体架构)
3. [核心组件详解](#核心组件详解)
4. [训练策略](#训练策略)
5. [消融实验](#消融实验)
6. [配置参数](#配置参数)
7. [使用指南](#使用指南)

## 模型概述

FFN集成CultureMoE是一个基于LoRA (Low-Rank Adaptation) 的文化感知混合专家模型，专门设计用于处理多文化语言理解任务。该模型在保持参数效率的同时，通过层级专家分配和文化感知机制实现了对不同文化背景的精准建模。

### 核心特点
- **层级专家分配**：Layer 1-16保持原始FFN，Layer 17-24使用3个专家，Layer 25-32使用5个专家
- **LoRA增强**：在Attention和FFN层添加低秩适配器，实现参数高效微调
- **文化感知注意力**：可选的文化条件注意力机制
- **共享-专门专家分离**：共享专家处理通用知识，文化专家处理特定文化知识
- **Mask机制**：支持不同输入版本的差异化处理

## 整体架构

```
LLaMA/Qwen Base Model
├── Embedding Layer (冻结)
├── Layer 1-16: 原始FFN + LoRA增强Attention
├── Layer 17-24: MoE FFN (3专家) + LoRA增强Attention
├── Layer 25-32: MoE FFN (5专家) + LoRA增强Attention
├── Layer Norm (冻结)
└── LM Head (冻结)

总计专家数：(3×8) + (5×8) = 64个专家
可训练参数：仅LoRA参数 (~1-5% of total parameters)
```

### 架构设计原理

1. **渐进式专家增加**：从简单到复杂的层级设计，前16层处理基础语言理解，后16层处理复杂的文化推理
2. **内存优化**：针对48GB×2GPU环境优化，通过层级分配减少显存占用
3. **参数效率**：只训练LoRA参数，基础模型权重保持冻结

## 核心组件详解

### 1. LoRA增强注意力模块 (LoRAEnhancedAttention)

#### 标准LoRA注意力
```python
class LoRAEnhancedAttention:
    def __init__(self, original_attention, lora_config):
        # Q, K, V, O投影层的LoRA包装
        self.q_proj = LoRALinear(original_q_proj, rank, alpha, dropout)
        self.k_proj = LoRALinear(original_k_proj, rank, alpha, dropout)
        self.v_proj = LoRALinear(original_v_proj, rank, alpha, dropout)
        self.o_proj = LoRALinear(original_o_proj, rank, alpha, dropout)
```

#### 文化感知注意力（可选）
```python
# 文化嵌入层
self.culture_embeddings = nn.Embedding(num_cultures, culture_dim)

# 文化感知QKV生成器
self.cultural_qkv_generator = CulturalQKVGenerator(...)

# 文化偏置生成器
self.cultural_bias_generator = CulturalBiasGenerator(...)

# 自适应融合网络
self.adaptive_fusion = AdaptiveFusion(...)
```

**特点**：
- 支持标准LoRA和文化感知两种模式
- 文化感知模式通过文化ID动态调整注意力权重
- 自适应融合机制平衡文化特异性和通用性

### 2. 向量化CultureMoE FFN模块

#### 层级专家分配逻辑
```python
def get_layer_expert_count(self, layer_idx):
    if layer_idx < 17:
        return 0  # 原始FFN
    elif 17 <= layer_idx <= 24:
        return 3  # 3个专家
    elif 25 <= layer_idx <= 32:
        return 5  # 5个专家
    else:
        return 0  # 原始FFN
```

#### 专家类型设计

**共享专家 (SharedExpertWithLoRA)**
- **作用**：处理通用知识，文化无关的语言理解
- **输入**：Mask版本的hidden states（如果启用mask机制）
- **结构**：标准SwiGLU FFN + LoRA适配器
```python
class SharedExpertWithLoRA:
    def __init__(self, hidden_size, intermediate_size, lora_config):
        # 创建LoRA增强的FFN层
        self.gate_proj = LoRALinear(base_gate_proj, ...)
        self.up_proj = LoRALinear(base_up_proj, ...)
        self.down_proj = LoRALinear(base_down_proj, ...)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**文化专家 (CulturalExpertWithLoRA)**
- **作用**：处理特定文化的知识和推理模式
- **输入**：原始版本的hidden states
- **特点**：每个专家有特定的文化分配和文化条件向量
```python
class CulturalExpertWithLoRA:
    def __init__(self, expert_id, primary_culture_ids, ...):
        # LoRA增强的FFN层
        self.gate_proj = LoRALinear(...)
        self.up_proj = LoRALinear(...)
        self.down_proj = LoRALinear(...)

        # 文化条件向量
        self.culture_prompt = nn.Parameter(...)
        self.culture_condition = nn.Linear(culture_dim, hidden_size)

    def forward(self, hidden_states, culture_ids):
        # 获取文化条件
        culture_condition = self.culture_condition(culture_prompt)
        conditioned_input = hidden_states + culture_condition

        # SwiGLU FFN
        return self.down_proj(self.act_fn(self.gate_proj(conditioned_input)) * self.up_proj(conditioned_input))
```

### 3. 文化感知路由器 (VectorizedCulturalRouterWithLoRA)

#### 多路径路由决策
```python
# 1. 内容路由器：基于输入内容
content_logits = self.content_router(pooled_states)

# 2. 文化路由器：基于文化嵌入
culture_emb = self.culture_embeddings(culture_ids)
culture_logits = self.culture_router(culture_emb)

# 3. 亲和性路由器：文化-专家预定义亲和性
affinity_logits = self.culture_expert_affinity[culture_ids]

# 4. 融合决策
combined_logits = (
    α * content_logits +
    β * culture_logits +
    γ * affinity_logits
)
```

#### 增强池化机制
```python
class EnhancedPoolingWithLoRA:
    def forward(self, hidden_states):
        # 1. Mean pooling
        mean_pooled = hidden_states.mean(dim=1)

        # 2. Attention pooling
        attention_pooled, _ = self.attention_pool(
            query=self.cls_token,
            key=hidden_states,
            value=hidden_states
        )

        # 3. 动态加权组合（通过LoRA）
        pool_weights = F.softmax(self.pool_weight_net(mean_pooled), dim=-1)
        return pool_weights[0] * mean_pooled + pool_weights[1] * attention_pooled
```

### 4. 文化信息注入器 (CulturalInjectorWithLoRA)

#### FiLM调制机制
```python
class CulturalInjectorWithLoRA:
    def forward(self, hidden_states, culture_ids):
        # 获取文化嵌入
        culture_emb = self.layer_culture_embedding(culture_ids)

        # 计算FiLM参数（通过LoRA增强）
        gamma = 1.0 + self.gamma_up(self.gamma_down(culture_emb))
        beta = self.beta_up(self.beta_down(culture_emb))

        # FiLM调制
        modulated = gamma * hidden_states + beta

        # 门控融合
        culture_strength = self.culture_gate(culture_emb)
        return self.layer_norm(
            hidden_states + culture_strength * (modulated - hidden_states)
        )
```

### 5. 损失函数设计

#### 综合损失计算
```python
class LoRACultureMoELoss:
    def forward(self, logits, labels, moe_aux_info, culture_labels):
        # 1. 语言模型损失
        lm_loss = CrossEntropyLoss(logits, labels)

        # 2. 负载均衡损失
        load_balance_loss = compute_load_balancing_loss(expert_weights)

        # 3. 熵正则化损失
        entropy_loss = entropy_regularization(expert_weights)

        # 4. 文化对齐损失
        culture_loss = culture_alignment_loss(expert_weights, culture_labels)

        # 5. LoRA正则化
        lora_regularization = l2_regularization(lora_parameters)

        total_loss = (
            lm_loss +
            λ₁ * load_balance_loss +
            λ₂ * entropy_loss +
            λ₃ * culture_loss +
            λ₄ * lora_regularization
        )
```

## 训练策略

### 1. 参数冻结策略
```python
# 冻结基础模型参数
for param in [embedding, layer_norms, lm_head]:
    param.requires_grad = False

# 只训练LoRA参数
for lora_module in [attention_lora, expert_lora]:
    lora_module.lora_A.weight.requires_grad = True
    lora_module.lora_B.weight.requires_grad = True
```

### 2. 分层学习率策略
```python
param_groups = [
    {
        'params': attention_lora_params,
        'lr': base_lr * 0.5,  # 注意力LoRA使用较小学习率
        'weight_decay': weight_decay * 0.5
    },
    {
        'params': expert_lora_params,
        'lr': base_lr,  # 专家LoRA使用标准学习率
        'weight_decay': weight_decay
    },
    {
        'params': cultural_lora_params,
        'lr': base_lr * 0.8,  # 文化LoRA使用中等学习率
        'weight_decay': weight_decay * 0.5
    }
]
```

### 3. 内存优化策略

#### 梯度累积
```bash
batch_size=1              # 最小batch size
gradient_accumulation=4   # 梯度累积
effective_batch_size=8    # 1 × 4 × 2GPU = 8
```

#### 激活检查点
```python
model.gradient_checkpointing_enable()  # 节省内存
```

#### 内存管理
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
torch.cuda.set_per_process_memory_fraction(0.90)
```

### 4. 数据处理策略

#### Mask机制
```python
# 原始输入（文化专家使用）
input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"

# Mask输入（共享专家使用）
input_text_mask = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction_mask}<|eot_id|>"
```

#### 数据集划分
- 训练集：80%
- 验证集：10%
- 测试集：10%

### 5. 评估策略

#### Post Evaluation方式
1. **生成阶段**：模型生成回答，保存到JSON文件
2. **提取阶段**：从生成文本中提取阿拉伯数字(0-5)
3. **评估阶段**：计算accuracy/precision/recall/F1指标

```python
def extract_number_from_text(text):
    # 查找0-5的数字
    numbers = re.findall(r'\b[0-5]\b', text)
    return int(numbers[0]) if numbers else None

def post_eval_from_generated_answers(file_path):
    # 从JSON文件计算指标
    correct_count = sum(1 for item in data if item['is_correct'])
    accuracy = correct_count / total_count
    # 计算其他指标...
```

## 消融实验

### 1. 架构消融实验

#### 专家配置消融
```python
# 实验组1：均匀专家分配
layer_expert_config = {i: 4 for i in range(17, 33)}

# 实验组2：层级专家分配（推荐）
layer_expert_config = {
    **{i: 0 for i in range(1, 17)},   # Layer 1-16: 原始FFN
    **{i: 3 for i in range(17, 25)},  # Layer 17-24: 3专家
    **{i: 5 for i in range(25, 33)}   # Layer 25-32: 5专家
}

# 实验组3：稀疏专家分配
layer_expert_config = {i: 2 for i in range(20, 30)}
```

#### 注意力机制消融
```python
# 控制变量
enable_cultural_attention = True/False

# 实验组合
configs = [
    {"enable_cultural_attention": False, "name": "标准LoRA注意力"},
    {"enable_cultural_attention": True, "name": "文化感知注意力"}
]
```

### 2. 功能组件消融实验

#### 共享专家消融
```bash
# 实验1：启用共享专家
./run_ffn_culturemoe.sh qwen 2 true true true true 2

# 实验2：禁用共享专家
./run_ffn_culturemoe.sh qwen 2 false true true true 2
```

#### Mask机制消融
```bash
# 实验1：启用mask机制
./run_ffn_culturemoe.sh qwen 2 true true true true 2

# 实验2：禁用mask机制
./run_ffn_culturemoe.sh qwen 2 true false true true 2
```

#### 门控融合消融
```bash
# 实验1：启用门控融合
./run_ffn_culturemoe.sh qwen 2 true true true true 2

# 实验2：禁用门控融合
./run_ffn_culturemoe.sh qwen 2 true true false true 2
```

#### 文化损失消融
```bash
# 实验1：启用文化损失
./run_ffn_culturemoe.sh qwen 2 true true true true 2

# 实验2：禁用文化损失
./run_ffn_culturemoe.sh qwen 2 true true true false 2
```

### 3. LoRA参数消融实验

#### Rank消融
```python
lora_ranks = [4, 8, 16, 32, 64]
for rank in lora_ranks:
    config.lora_rank = rank
    # 训练并评估
```

#### Alpha消融
```python
lora_alphas = [8, 16, 32, 64, 128]
for alpha in lora_alphas:
    config.lora_alpha = alpha
    # 训练并评估
```

#### Dropout消融
```python
lora_dropouts = [0.0, 0.05, 0.1, 0.2, 0.3]
for dropout in lora_dropouts:
    config.lora_dropout = dropout
    # 训练并评估
```

### 4. 损失权重消融实验

#### 负载均衡权重
```python
load_balance_weights = [0.001, 0.01, 0.1, 0.5]
```

#### 熵正则化权重
```python
entropy_weights = [0.01, 0.1, 0.5, 1.0]
```

#### 文化损失权重
```python
culture_loss_weights = [0.01, 0.05, 0.1, 0.2]
```

### 5. 完整消融实验矩阵

| 实验ID | 共享专家 | Mask机制 | 门控融合 | 文化损失 | 文化注意力 | 预期效果 |
|--------|----------|----------|----------|----------|------------|----------|
| E1     | ✓        | ✓        | ✓        | ✓        | ✓          | 完整模型（基线） |
| E2     | ✗        | ✓        | ✓        | ✓        | ✓          | 测试共享专家作用 |
| E3     | ✓        | ✗        | ✓        | ✓        | ✓          | 测试mask机制作用 |
| E4     | ✓        | ✓        | ✗        | ✓        | ✓          | 测试门控融合作用 |
| E5     | ✓        | ✓        | ✓        | ✗        | ✓          | 测试文化损失作用 |
| E6     | ✓        | ✓        | ✓        | ✓        | ✗          | 测试文化注意力作用 |
| E7     | ✗        | ✗        | ✗        | ✗        | ✗          | 最简配置 |

## 配置参数

### 模型配置
```python
@dataclass
class LoRACultureMoEConfig:
    # 层级专家分配
    layer_expert_config: Dict[int, int] = field(default_factory=dict)
    moe_start_layer: int = 17

    # MoE基础配置
    num_experts: int = 8  # 默认专家数
    top_k: int = 2
    capacity_factor: float = 1.25

    # 文化配置
    num_cultures: int = 6
    culture_dim: int = 256

    # LoRA配置
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1

    # 功能开关
    enable_cultural_attention: bool = True
    use_shared_expert: bool = True
    use_gate_fusion: bool = True
    use_mask_mechanism: bool = True

    # 损失权重
    load_balance_weight: float = 0.01
    entropy_weight: float = 0.1
    culture_loss_weight: float = 0.05
```

### 训练配置
```bash
# 基础参数
BASE_MODEL="/path/to/base/model"
DATA_FILE="/path/to/training/data.json"
OUTPUT_DIR="/path/to/output"

# 训练参数
NUM_EPOCHS=5
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
LEARNING_RATE=2e-4
MAX_SEQ_LEN=512

# 硬件配置
NUM_GPUS=2
ENABLE_ACTIVATION_CHECKPOINTING=true
```

## 使用指南

### 1. 快速开始
```bash
# 使用Qwen2.5-7B，CulturalBench数据集，启用所有功能
./run_ffn_culturemoe.sh qwen 2 true true true true 2
```

### 2. 自定义配置
```bash
# 参数说明：backbone data_id use_shared use_mask use_gate use_culture_loss num_gpus
./run_ffn_culturemoe.sh llama 3 false true false false 1
```

### 3. 输出文件说明
```
output_dir/
├── best_moe/best_lora_weights.pt     # 最佳模型权重
├── config.json                      # 完整配置
├── training.log                     # 训练日志
├── eval_result_per_epoch.json       # 每轮结果
├── eval_generated_answers_epoch_*.json  # 验证集回答
├── test_generated_answers.json      # 测试集回答
└── test_result.json                 # 最终测试结果
```

### 4. 推理使用
```python
# 加载最佳模型
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
lora_weights = torch.load("best_moe/best_lora_weights.pt")

# 应用LoRA权重
model = create_lora_culturemoe_model(base_model_path, lora_config)
# 加载权重逻辑...

# 推理
outputs = model.generate(input_ids, culture_ids=culture_ids)
```

### 5. 评估分析
```python
# 加载生成的回答
with open("test_generated_answers.json") as f:
    results = json.load(f)

# 分析结果
for item in results:
    print(f"Question: {item['instruction']}")
    print(f"Correct: {item['correct_label']}")
    print(f"Predicted: {item['extracted_number']}")
    print(f"Correct: {item['is_correct']}")
    print("-" * 50)
```

## 技术优势

1. **参数效率**：仅训练1-5%的参数，大幅降低计算成本
2. **内存优化**：层级专家分配，适配48GB×2GPU环境
3. **文化感知**：多层次文化建模，提升跨文化理解能力
4. **模块化设计**：支持灵活的消融实验和功能组合
5. **可扩展性**：支持不同基础模型和专家配置

## 局限性

1. **专家利用率**：可能存在专家利用不均衡问题
2. **文化标注**：依赖高质量的文化标注数据
3. **计算复杂度**：MoE结构增加了推理复杂度
4. **超参敏感性**：多个损失权重需要仔细调优

## 未来改进方向

1. **动态专家分配**：根据输入动态调整专家数量
2. **跨语言扩展**：支持多语言文化理解
3. **知识蒸馏**：将专家知识蒸馏到更小的模型
4. **在线学习**：支持增量学习新的文化知识