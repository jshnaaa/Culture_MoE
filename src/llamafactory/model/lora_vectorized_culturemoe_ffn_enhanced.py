# src/llamafactory/model/lora_vectorized_culturemoe_ffn_enhanced.py
"""
增强版LoRA向量化CultureMoE FFN层
支持共享专家和文化专家分离，以及mask机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .lora_enhanced_culturemoe import (
    LoRACultureMoEConfig,
    CulturalInjectorWithLoRA,
    VectorizedLoRAExpertLayer,
    LoRALinear
)


class EnhancedPoolingWithLoRA(nn.Module):
    """LoRA增强的池化层"""
    def __init__(self, hidden_dim: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 可学习的注意力池化
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # 组合权重网络（使用LoRA）
        base_weight_net = nn.Linear(hidden_dim, 2)  # [mean_pool, attention_pool]
        self.pool_weight_net = LoRALinear(
            base_weight_net,
            rank=lora_config.lora_rank // 4,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """增强池化前向传播"""
        batch_size, seq_len = hidden_states.shape[:2]

        # 1. 简单mean pooling
        mean_pooled = hidden_states.mean(dim=1)

        # 2. 注意力池化
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        attention_pooled, _ = self.attention_pool(
            query=cls_tokens,
            key=hidden_states,
            value=hidden_states
        )
        attention_pooled = attention_pooled.squeeze(1)

        # 3. 动态加权组合（通过LoRA）
        pool_weights = F.softmax(self.pool_weight_net(mean_pooled), dim=-1)
        final_pooled = (
            pool_weights[:, 0:1] * mean_pooled +
            pool_weights[:, 1:2] * attention_pooled
        )

        return final_pooled


class VectorizedCulturalRouterWithLoRA(nn.Module):
    """LoRA增强的文化感知路由器"""
    def __init__(self, hidden_dim: int, num_experts: int, num_cultures: int,
                 culture_dim: int, capacity_factor: float, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_cultures = num_cultures
        self.capacity_factor = capacity_factor

        # 增强的池化层
        self.enhanced_pooler = EnhancedPoolingWithLoRA(hidden_dim, lora_config)

        # 内容路由器（使用LoRA）
        base_content_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        # 只对最后一层使用LoRA
        base_content_final = base_content_router[-1]
        lora_content_final = LoRALinear(
            base_content_final,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )
        self.content_router = nn.Sequential(
            base_content_router[0],  # Linear
            base_content_router[1],  # GELU
            base_content_router[2],  # Dropout
            lora_content_final       # LoRA enhanced final layer
        )

        # 文化路由器（使用LoRA）
        base_culture_router = nn.Sequential(
            nn.Linear(culture_dim, culture_dim // 2),
            nn.GELU(),
            nn.Linear(culture_dim // 2, num_experts)
        )
        base_culture_final = base_culture_router[-1]
        lora_culture_final = LoRALinear(
            base_culture_final,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )
        self.culture_router = nn.Sequential(
            base_culture_router[0],  # Linear
            base_culture_router[1],  # GELU
            lora_culture_final       # LoRA enhanced final layer
        )

        # 文化-专家亲和性矩阵
        self.culture_expert_affinity = nn.Parameter(
            torch.randn(num_cultures, num_experts) * 0.1
        )

        # 融合权重（可学习）
        self.routing_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))

        # 文化嵌入
        self.culture_embeddings = nn.Embedding(num_cultures, culture_dim)

        # 噪声注入
        self.noise_epsilon = 1e-2

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.normal_(self.culture_embeddings.weight, mean=0, std=0.02)

    def enhanced_pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """增强的池化方法"""
        return self.enhanced_pooler(hidden_states)

    def forward(self, pooled_states: torch.Tensor, culture_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """向量化路由前向传播"""
        batch_size = pooled_states.shape[0]
        device = pooled_states.device

        # 确保culture_ids在正确的设备上
        if culture_ids.device != device:
            culture_ids = culture_ids.to(device)

        # 1. 多路径路由决策（通过LoRA增强）
        content_logits = self.content_router(pooled_states)

        culture_emb = self.culture_embeddings(culture_ids)
        culture_logits = self.culture_router(culture_emb)

        affinity_logits = self.culture_expert_affinity[culture_ids]

        # 2. 融合路由决策
        routing_weights = F.softmax(self.routing_weights, dim=0)
        combined_logits = (
            routing_weights[0] * content_logits +
            routing_weights[1] * culture_logits +
            routing_weights[2] * affinity_logits
        )

        # 3. 添加噪声（仅训练时）
        if self.training:
            noise = torch.randn_like(combined_logits) * self.noise_epsilon
            combined_logits = combined_logits + noise

        # 4. 数值稳定的softmax
        router_logits = torch.clamp(combined_logits, min=-10.0, max=10.0)
        expert_weights = F.softmax(router_logits, dim=-1)

        # 5. 检查数值稳定性
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            expert_weights = torch.ones_like(expert_weights) / self.num_experts
            logging.warning("Router weights contain NaN/Inf, using uniform distribution")

        return {
            'expert_weights': expert_weights,
            'router_logits': router_logits,
            'content_logits': content_logits,
            'culture_logits': culture_logits,
            'affinity_logits': affinity_logits
        }

    def compute_load_balancing_loss(self, expert_weights: torch.Tensor, router_probs: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失"""
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)
        if torch.isnan(router_probs).any() or torch.isinf(router_probs).any():
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)

        f_i = expert_weights.mean(dim=0)
        P_i = router_probs.mean(dim=0)
        N = self.num_experts
        balance_loss = N * torch.sum(f_i * P_i)
        balance_loss = torch.clamp(balance_loss, min=1e-8, max=10.0)

        return balance_loss

    def entropy_regularization(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """计算熵正则化损失"""
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)

        expert_weights_normalized = torch.clamp(expert_weights, min=1e-8, max=1.0)
        expert_weights_normalized = expert_weights_normalized / expert_weights_normalized.sum(dim=-1, keepdim=True)

        mean_weight = expert_weights_normalized.mean(dim=-1, keepdim=True)
        variance = ((expert_weights_normalized - mean_weight) ** 2).mean(dim=-1)

        num_experts = expert_weights.size(-1)
        ideal_variance = (1.0 / num_experts) * (1.0 - 1.0 / num_experts)

        entropy_loss = torch.clamp(ideal_variance - variance, min=0.0).mean()
        entropy_loss = torch.clamp(entropy_loss, min=1e-8, max=1.0)

        return entropy_loss


class SharedExpertWithLoRA(nn.Module):
    """
    共享专家（LoRA增强）
    处理通用知识，使用mask字段的输入
    """
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # 创建基础FFN层
        base_gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        base_up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        base_down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 用LoRA包装
        self.gate_proj = LoRALinear(
            base_gate_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )
        self.up_proj = LoRALinear(
            base_up_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )
        self.down_proj = LoRALinear(
            base_down_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

        # 激活函数
        if activation == "silu":
            self.act_fn = F.silu
        elif activation == "relu":
            self.act_fn = F.relu
        elif activation == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        """共享专家前向传播（SwiGLU架构）"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class CulturalExpertsWithLoRA(nn.Module):
    """
    文化专家组（LoRA增强）
    处理文化特定知识，使用未mask的输入
    """
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int,
                 activation: str, culture_assignments: List[List[int]],
                 culture_dim: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.culture_assignments = culture_assignments

        # 创建LoRA增强的文化专家
        self.experts = nn.ModuleList([
            CulturalExpertWithLoRA(
                expert_id=i,
                primary_culture_ids=culture_assignments[i],
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                culture_dim=culture_dim,
                lora_config=lora_config
            ) for i in range(num_experts)
        ])

    def forward_with_dispatch(self, hidden_states: torch.Tensor, expert_weights: torch.Tensor,
                            culture_ids: torch.Tensor, top_k: int, capacity_factor: float) -> Tuple[torch.Tensor, Dict]:
        """
        向量化的专家调度和处理
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Top-K专家选择
        top_k_weights, top_k_indices = torch.topk(expert_weights, k=top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # 向量化专家处理
        expert_outputs = []
        culture_relevances = []

        for i in range(top_k):
            batch_expert_output = torch.zeros_like(hidden_states)
            batch_relevance = torch.zeros(batch_size, device=hidden_states.device)

            for b in range(batch_size):
                expert_idx = top_k_indices[b, i].item()
                weight = top_k_weights[b, i]

                expert = self.experts[expert_idx]
                expert_out, relevance = expert(hidden_states[b:b+1], culture_ids[b:b+1])

                batch_expert_output[b] = expert_out[0] * weight
                batch_relevance[b] = relevance[0]

            expert_outputs.append(batch_expert_output)
            culture_relevances.append(batch_relevance)

        # 专家输出融合
        final_output = sum(expert_outputs)

        dispatch_info = {
            'top_k_indices': top_k_indices,
            'top_k_weights': top_k_weights,
            'culture_relevances': torch.stack(culture_relevances, dim=1)
        }

        return final_output, dispatch_info


class CulturalExpertWithLoRA(nn.Module):
    """
    单个文化专家（LoRA增强）
    """
    def __init__(self, expert_id: int, primary_culture_ids: List[int],
                 hidden_size: int, intermediate_size: int, activation: str,
                 culture_dim: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.expert_id = expert_id
        self.primary_culture_ids = primary_culture_ids
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # 创建基础FFN层
        base_gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        base_up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        base_down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 用LoRA包装FFN层
        self.gate_proj = LoRALinear(
            base_gate_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )
        self.up_proj = LoRALinear(
            base_up_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )
        self.down_proj = LoRALinear(
            base_down_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

        # 激活函数
        if activation == "silu":
            self.act_fn = F.silu
        elif activation == "relu":
            self.act_fn = F.relu
        elif activation == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 文化条件向量
        if len(primary_culture_ids) > 0:
            self.culture_prompt = nn.Parameter(
                torch.randn(len(primary_culture_ids), culture_dim) * 0.01
            )
        else:
            self.culture_prompt = nn.Parameter(
                torch.randn(1, culture_dim) * 0.01
            )

        # 文化条件层
        self.culture_condition = nn.Linear(culture_dim, hidden_size)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.culture_condition.weight, mean=0, std=0.001)
        if self.culture_condition.bias is not None:
            nn.init.zeros_(self.culture_condition.bias)

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 获取文化提示
        culture_prompt = self._get_culture_prompt(culture_ids)
        culture_condition = self.culture_condition(culture_prompt).unsqueeze(1)

        # 文化条件的输入
        conditioned_input = hidden_states + culture_condition

        # SwiGLU FFN前向传播（通过LoRA增强）
        gate_output = self.act_fn(self.gate_proj(conditioned_input))
        up_output = self.up_proj(conditioned_input)
        ffn_output = self.down_proj(gate_output * up_output)

        # 计算文化相关性
        culture_relevance = self._compute_culture_relevance(culture_ids)

        return ffn_output, culture_relevance

    def _get_culture_prompt(self, culture_ids: torch.Tensor) -> torch.Tensor:
        """获取文化提示向量"""
        prompts = []
        for culture_id in culture_ids:
            if len(self.primary_culture_ids) == 0:
                prompt = self.culture_prompt[0]
            elif culture_id.item() in self.primary_culture_ids:
                idx = self.primary_culture_ids.index(culture_id.item())
                prompt = self.culture_prompt[idx]
            else:
                prompt = self.culture_prompt.mean(dim=0)
            prompts.append(prompt)
        return torch.stack(prompts)

    def _compute_culture_relevance(self, culture_ids: torch.Tensor) -> torch.Tensor:
        """计算文化相关性"""
        relevance_scores = []
        for culture_id in culture_ids:
            if len(self.primary_culture_ids) == 0:
                relevance_scores.append(0.5)  # 冲突处理专家
            elif culture_id.item() in self.primary_culture_ids:
                relevance_scores.append(1.0)  # 高相关性
            else:
                relevance_scores.append(0.1)  # 低相关性
        return torch.tensor(relevance_scores, device=culture_ids.device, dtype=torch.float32)


class VectorizedCultureMoE_FFN_WithLoRA_Enhanced(nn.Module):
    """
    增强版LoRA向量化文化感知FFN层
    支持共享专家和文化专家分离，以及mask机制
    """
    def __init__(self, config, layer_idx: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = lora_config.num_experts
        self.top_k = lora_config.top_k

        # 文化信息注入层（每层都有，使用LoRA）
        self.cultural_injector = CulturalInjectorWithLoRA(
            hidden_dim=self.hidden_size,
            culture_dim=lora_config.culture_dim,
            layer_idx=layer_idx,
            lora_config=lora_config
        )

        # LoRA增强的路由器（只用于文化专家）
        self.cultural_router = VectorizedCulturalRouterWithLoRA(
            hidden_dim=self.hidden_size,
            num_experts=self.num_experts,
            num_cultures=lora_config.num_cultures,
            culture_dim=lora_config.culture_dim,
            capacity_factor=lora_config.capacity_factor,
            lora_config=lora_config
        )

        # 创建文化分配
        from .cultural_components import create_culture_assignments
        culture_assignments = create_culture_assignments(self.num_experts, lora_config.num_cultures)

        # LoRA增强的文化专家组
        self.cultural_experts = CulturalExpertsWithLoRA(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=config.hidden_act,
            culture_assignments=culture_assignments,
            culture_dim=lora_config.culture_dim,
            lora_config=lora_config
        )

        # LoRA增强的共享专家
        self.shared_expert = SharedExpertWithLoRA(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=config.hidden_act,
            lora_config=lora_config
        )

        # 融合参数
        self.moe_fusion_alpha = nn.Parameter(torch.tensor(lora_config.moe_fusion_alpha_init))

        # 损失权重
        self.load_balance_weight = lora_config.load_balance_weight
        self.entropy_weight = lora_config.entropy_weight

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor,
                hidden_states_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        增强版前向传播，支持mask机制

        Args:
            hidden_states: [B, L, H] 原始输入（文化专家使用）
            culture_ids: [B] 文化标识
            hidden_states_mask: [B, L, H] mask版本的hidden states（共享专家使用）

        Returns:
            output: [B, L, H]
            aux_info: Dict 辅助信息
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 如果没有提供culture_ids，使用默认值
        if culture_ids is None:
            culture_ids = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)

        # 1. 文化信息注入（使用原始输入）
        culturally_enhanced_states = self.cultural_injector(hidden_states, culture_ids)

        # 2. 共享专家处理（使用mask输入，如果提供的话）
        if hidden_states_mask is not None:
            shared_input = hidden_states_mask
        else:
            shared_input = hidden_states  # 如果没有mask，使用原始输入

        shared_output = self.shared_expert(shared_input)

        # 3. 文化专家处理（使用未mask的输入）
        # 增强的池化表示
        pooled_representation = self.cultural_router.enhanced_pooling(culturally_enhanced_states)

        # LoRA增强的路由决策
        routing_result = self.cultural_router(pooled_representation, culture_ids)
        expert_weights = routing_result['expert_weights']
        router_logits = routing_result['router_logits']

        # LoRA增强的文化专家调度和处理
        cultural_expert_output, dispatch_info = self.cultural_experts.forward_with_dispatch(
            hidden_states=culturally_enhanced_states,
            expert_weights=expert_weights,
            culture_ids=culture_ids,
            top_k=self.top_k,
            capacity_factor=self.cultural_router.capacity_factor
        )

        # 4. 共享专家和文化专家融合
        moe_alpha = torch.sigmoid(self.moe_fusion_alpha)
        final_output = (1 - moe_alpha) * shared_output + moe_alpha * cultural_expert_output

        # 5. 计算辅助损失
        load_balance_loss = self.cultural_router.compute_load_balancing_loss(
            expert_weights, F.softmax(router_logits, dim=-1)
        )
        entropy_loss = self.cultural_router.entropy_regularization(expert_weights)

        # 6. 辅助信息收集
        aux_info = {
            'expert_weights': expert_weights,
            'router_logits': router_logits,
            'load_balance_loss': load_balance_loss,
            'entropy_loss': entropy_loss,
            'dispatch_info': dispatch_info,
            'moe_alpha': moe_alpha,
            'routing_result': routing_result,
            'shared_expert_output': shared_output,
            'cultural_expert_output': cultural_expert_output
        }

        return final_output, aux_info

    def get_expert_type_info(self) -> Dict[str, Any]:
        """获取专家类型信息"""
        return {
            'shared_expert': {
                'type': 'shared',
                'input_type': 'masked',
                'param_count': sum(p.numel() for p in self.shared_expert.parameters()),
                'description': 'Handles general knowledge using masked input'
            },
            'cultural_experts': {
                'type': 'cultural',
                'input_type': 'unmasked',
                'num_experts': self.num_experts,
                'param_count': sum(p.numel() for p in self.cultural_experts.parameters()),
                'description': 'Handle culture-specific knowledge using unmasked input',
                'culture_assignments': [expert.primary_culture_ids for expert in self.cultural_experts.experts]
            }
        }

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """获取所有LoRA参数"""
        lora_params = []
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.lora_A.weight, module.lora_B.weight])
        return lora_params

    def freeze_base_parameters(self):
        """冻结所有基础参数，只训练LoRA"""
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                # 冻结基础层
                for param in module.base_layer.parameters():
                    param.requires_grad = False
                # 解冻LoRA层
                module.lora_A.weight.requires_grad = True
                module.lora_B.weight.requires_grad = True
            elif not any(isinstance(child, LoRALinear) for child in module.children()):
                # 对于不包含LoRA的模块，冻结所有参数
                for param in module.parameters():
                    param.requires_grad = False

    def print_expert_info(self):
        """打印专家信息"""
        expert_info = self.get_expert_type_info()

        print(f"Layer {self.layer_idx} Expert Information:")
        print("-" * 50)

        # 共享专家信息
        shared_info = expert_info['shared_expert']
        print(f"Shared Expert:")
        print(f"  Type: {shared_info['type']}")
        print(f"  Input: {shared_info['input_type']}")
        print(f"  Parameters: {shared_info['param_count']:,}")
        print(f"  Description: {shared_info['description']}")

        # 文化专家信息
        cultural_info = expert_info['cultural_experts']
        print(f"\nCultural Experts:")
        print(f"  Type: {cultural_info['type']}")
        print(f"  Input: {cultural_info['input_type']}")
        print(f"  Count: {cultural_info['num_experts']}")
        print(f"  Parameters: {cultural_info['param_count']:,}")
        print(f"  Description: {cultural_info['description']}")

        # 文化分配
        print(f"\nCulture Assignments:")
        for i, assignment in enumerate(cultural_info['culture_assignments']):
            if len(assignment) == 0:
                role = "Conflict Resolution Specialist"
            elif len(assignment) == 6:
                role = "Cross-Cultural Generalist"
            else:
                role = f"Cultures {assignment}"
            print(f"  Expert {i}: {role}")

        print("-" * 50)