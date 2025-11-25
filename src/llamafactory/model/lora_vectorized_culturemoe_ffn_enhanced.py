# src/llamafactory/model/lora_vectorized_culturemoe_ffn_enhanced.py
"""
增强版LoRA向量化CultureMoE FFN层
支持共享专家和文化专家分离，以及mask机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
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


class SharedFFNWithLoRA(nn.Module):
    """
    MixLoRA优化：共享基础FFN层
    所有专家共享gate_proj和up_proj的计算，只有down_proj和适配器不同
    """
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # 共享的基础FFN层（所有专家共用）
        base_gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        base_up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # 用LoRA包装共享层
        self.shared_gate_proj = LoRALinear(
            base_gate_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )
        self.shared_up_proj = LoRALinear(
            base_up_proj,
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
        """共享FFN前向传播，返回中间结果供专家使用"""
        gate_output = self.act_fn(self.shared_gate_proj(x))
        up_output = self.shared_up_proj(x)
        return gate_output, up_output  # 返回中间结果而不是最终输出


class ExpertAdapter(nn.Module):
    """
    MixLoRA优化：轻量级专家适配器
    每个专家只需要小的适配器层，大幅减少参数量
    """
    def __init__(self, intermediate_size: int, hidden_size: int, lora_config: LoRACultureMoEConfig):
        super().__init__()

        # 使用更小的适配器rank来减少参数
        adapter_rank = max(4, lora_config.lora_rank // 4)

        # 中间层适配器（调整共享计算结果）
        self.gate_adapter = nn.Sequential(
            nn.Linear(intermediate_size, adapter_rank, bias=False),
            nn.ReLU(),
            nn.Linear(adapter_rank, intermediate_size, bias=False)
        )

        self.up_adapter = nn.Sequential(
            nn.Linear(intermediate_size, adapter_rank, bias=False),
            nn.ReLU(),
            nn.Linear(adapter_rank, intermediate_size, bias=False)
        )

        # 输出投影层（每个专家独有）
        base_down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.down_proj = LoRALinear(
            base_down_proj,
            rank=adapter_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

        # 初始化适配器权重为接近零，保持初始时接近原始行为
        for module in [self.gate_adapter, self.up_adapter]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)

    def forward(self, shared_gate, shared_up):
        """专家适配器前向传播"""
        # 应用轻量级适配器调整
        adapted_gate = shared_gate + self.gate_adapter(shared_gate) * 0.1  # 小的调整幅度
        adapted_up = shared_up + self.up_adapter(shared_up) * 0.1

        # 专家特化的输出投影
        expert_output = self.down_proj(adapted_gate * adapted_up)
        return expert_output


class SharedExpertWithLoRA(nn.Module):
    """
    MixLoRA优化：共享专家（处理通用知识）
    使用LoRA增强的标准FFN结构
    """
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str, lora_config: LoRACultureMoEConfig):
        super().__init__()
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """共享专家前向传播"""
        # SwiGLU FFN前向传播（通过LoRA增强）
        gate_output = self.act_fn(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        return self.down_proj(gate_output * up_output)


class CulturalExpertsWithLoRA(nn.Module):
    """
    MixLoRA优化：文化专家组（共享计算架构）
    使用共享FFN + 轻量级适配器的架构
    """
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int,
                 activation: str, culture_assignments: List[List[int]],
                 culture_dim: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.culture_assignments = culture_assignments

        # MixLoRA优化：共享基础FFN计算
        self.shared_ffn = SharedFFNWithLoRA(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            lora_config=lora_config
        )

        # MixLoRA优化：轻量级专家适配器
        self.expert_adapters = nn.ModuleList([
            ExpertAdapter(
                intermediate_size=intermediate_size,
                hidden_size=hidden_size,
                lora_config=lora_config
            ) for _ in range(num_experts)
        ])

        # 文化条件向量（保持原有的文化感知能力）
        self.culture_prompts = nn.ParameterList([
            nn.Parameter(
                torch.randn(len(culture_assignments[i]) if culture_assignments[i] else 1, culture_dim) * 0.01
            ) for i in range(num_experts)
        ])

        # 文化条件层
        self.culture_condition = nn.Linear(culture_dim, hidden_size)

        # MixLoRA优化：稀疏激活参数
        self.activation_threshold = 0.01  # 专家激活阈值
        self.max_active_experts = min(2, num_experts)  # 最大激活专家数

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.culture_condition.weight, mean=0, std=0.001)
        if self.culture_condition.bias is not None:
            nn.init.zeros_(self.culture_condition.bias)

    def forward_with_dispatch(self, hidden_states: torch.Tensor, expert_weights: torch.Tensor,
                            culture_ids: torch.Tensor, top_k: int, capacity_factor: float) -> Tuple[torch.Tensor, Dict]:
        """
        MixLoRA优化：稀疏激活的专家调度和处理
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # MixLoRA优化1：稀疏激活 - 只选择权重超过阈值的专家
        active_mask = expert_weights > self.activation_threshold
        expert_weights = expert_weights * active_mask.float()

        # MixLoRA优化2：限制最大激活专家数
        effective_top_k = min(top_k, self.max_active_experts)
        top_k_weights, top_k_indices = torch.topk(expert_weights, k=effective_top_k, dim=-1)

        # 重新归一化权重
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # MixLoRA优化3：共享基础计算 - 所有专家共享gate_proj和up_proj
        shared_gate, shared_up = self.shared_ffn(hidden_states)

        # 只对激活的专家进行计算
        expert_outputs = []
        culture_relevances = []
        active_expert_count = 0

        for i in range(effective_top_k):
            batch_expert_output = torch.zeros_like(hidden_states)
            batch_relevance = torch.zeros(batch_size, device=hidden_states.device)

            for b in range(batch_size):
                expert_idx = top_k_indices[b, i].item()
                weight = top_k_weights[b, i]

                # 跳过权重过小的专家（稀疏激活）
                if weight < self.activation_threshold:
                    continue

                active_expert_count += 1

                # 获取文化条件
                culture_prompt = self._get_culture_prompt(expert_idx, culture_ids[b:b+1])
                culture_condition = self.culture_condition(culture_prompt).unsqueeze(1)

                # 应用文化条件到输入
                conditioned_input = hidden_states[b:b+1] + culture_condition

                # 重新计算该样本的共享基础（考虑文化条件）
                sample_gate, sample_up = self.shared_ffn(conditioned_input)

                # 使用专家适配器进行特化处理
                expert_out = self.expert_adapters[expert_idx](
                    sample_gate[0], sample_up[0]
                ).unsqueeze(0)

                batch_expert_output[b] = expert_out[0] * weight

                # 计算文化相关性
                relevance = self._compute_culture_relevance(expert_idx, culture_ids[b:b+1])
                batch_relevance[b] = relevance[0]

            expert_outputs.append(batch_expert_output)
            culture_relevances.append(batch_relevance)

        # 专家输出融合
        final_output = sum(expert_outputs) if expert_outputs else torch.zeros_like(hidden_states)

        dispatch_info = {
            'top_k_indices': top_k_indices,
            'top_k_weights': top_k_weights,
            'culture_relevances': torch.stack(culture_relevances, dim=1) if culture_relevances else torch.zeros(batch_size, effective_top_k, device=hidden_states.device),
            'active_expert_count': active_expert_count,
            'activation_rate': active_expert_count / (batch_size * effective_top_k)
        }

        return final_output, dispatch_info

    def _get_culture_prompt(self, expert_idx: int, culture_ids: torch.Tensor) -> torch.Tensor:
        """获取专家的文化提示向量"""
        prompts = []
        culture_assignments = self.culture_assignments[expert_idx]

        for culture_id in culture_ids:
            if len(culture_assignments) == 0:
                prompt = self.culture_prompts[expert_idx][0]
            elif culture_id.item() in culture_assignments:
                idx = culture_assignments.index(culture_id.item())
                prompt = self.culture_prompts[expert_idx][idx]
            else:
                prompt = self.culture_prompts[expert_idx].mean(dim=0)
            prompts.append(prompt)

        return torch.stack(prompts)

    def _compute_culture_relevance(self, expert_idx: int, culture_ids: torch.Tensor) -> torch.Tensor:
        """计算专家对文化的相关性"""
        relevance_scores = []
        culture_assignments = self.culture_assignments[expert_idx]

        for culture_id in culture_ids:
            if len(culture_assignments) == 0:
                relevance_scores.append(0.5)  # 冲突处理专家
            elif culture_id.item() in culture_assignments:
                relevance_scores.append(1.0)  # 高相关性
            else:
                relevance_scores.append(0.1)  # 低相关性

        return torch.tensor(relevance_scores, device=culture_ids.device, dtype=torch.float32)


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
        # 支持层级专家分配
        if hasattr(lora_config, 'get_layer_expert_count'):
            self.num_experts = lora_config.get_layer_expert_count(layer_idx)
        else:
            self.num_experts = lora_config.num_experts
        self.top_k = lora_config.top_k

        # 消融实验配置
        self.use_shared_expert = lora_config.use_shared_expert
        self.use_gate_fusion = lora_config.use_gate_fusion
        self.use_mask_mechanism = lora_config.use_mask_mechanism

        # 检查是否为MoE层
        self.is_moe_layer = self.num_experts > 0

        if self.is_moe_layer:
            # MoE层：创建文化信息注入层（使用LoRA）
            self.cultural_injector = CulturalInjectorWithLoRA(
                hidden_dim=self.hidden_size,
                culture_dim=lora_config.culture_dim,
                layer_idx=layer_idx,
                lora_config=lora_config
            )
        else:
            # 原始FFN层：创建带LoRA的标准FFN
            from transformers.models.llama.modeling_llama import LlamaMLP
            original_ffn = LlamaMLP(config)

            # 为原始FFN添加LoRA适配器
            self.gate_proj = LoRALinear(
                original_ffn.gate_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
            self.up_proj = LoRALinear(
                original_ffn.up_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
            self.down_proj = LoRALinear(
                original_ffn.down_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
            self.act_fn = original_ffn.act_fn
            return  # 早期返回，不需要创建MoE组件

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
        def create_culture_assignments(num_experts: int, num_cultures: int) -> List[List[int]]:
            """创建专家的文化分配"""
            assignments = []
            for i in range(num_experts):
                if i == 0:
                    # 第一个专家处理所有文化（通用专家）
                    assignments.append(list(range(num_cultures)))
                elif i == num_experts - 1:
                    # 最后一个专家处理文化冲突（空分配表示冲突处理专家）
                    assignments.append([])
                else:
                    # 其他专家分配特定文化
                    cultures_per_expert = max(1, num_cultures // (num_experts - 2))
                    start_culture = ((i - 1) * cultures_per_expert) % num_cultures
                    expert_cultures = []
                    for j in range(cultures_per_expert):
                        culture_id = (start_culture + j) % num_cultures
                        expert_cultures.append(culture_id)
                    assignments.append(expert_cultures)
            return assignments

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

        # LoRA增强的共享专家（根据配置决定是否创建）
        if self.use_shared_expert:
            self.shared_expert = SharedExpertWithLoRA(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                activation=config.hidden_act,
                lora_config=lora_config
            )
        else:
            self.shared_expert = None

        # 融合参数
        self.moe_fusion_alpha = nn.Parameter(torch.tensor(lora_config.moe_fusion_alpha_init))

        # 损失权重
        self.load_balance_weight = lora_config.load_balance_weight
        self.entropy_weight = lora_config.entropy_weight

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor = None,
                hidden_states_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        增强版前向传播，支持层级专家分配和mask机制

        Args:
            hidden_states: [B, L, H] 原始输入（文化专家使用）
            culture_ids: [B] 文化标识（可选）
            hidden_states_mask: [B, L, H] mask版本的hidden states（共享专家使用）

        Returns:
            output: [B, L, H]
            aux_info: Dict 辅助信息
        """
        # 如果不是MoE层，使用原始FFN
        if not self.is_moe_layer:
            # 标准FFN前向传播 (SwiGLU)
            gate_output = self.gate_proj(hidden_states)
            up_output = self.up_proj(hidden_states)
            activated = self.act_fn(gate_output) * up_output
            output = self.down_proj(activated)

            # 返回空的辅助信息以保持接口一致
            aux_info = {
                'load_balance_loss': torch.tensor(0.0, device=hidden_states.device),
                'entropy_loss': torch.tensor(0.0, device=hidden_states.device),
                'expert_weights': torch.zeros(hidden_states.shape[0], 1, device=hidden_states.device)  # 假的权重
            }
            return output, aux_info

        # MoE层的处理逻辑
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 如果没有提供culture_ids，使用默认值
        if culture_ids is None:
            culture_ids = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)

        # 1. 文化信息注入（使用原始输入）
        culturally_enhanced_states = self.cultural_injector(hidden_states, culture_ids)

        # 2. 共享专家处理（根据配置决定是否使用和如何处理）
        if self.use_shared_expert:
            # 决定共享专家的输入
            if self.use_mask_mechanism and hidden_states_mask is not None:
                shared_input = hidden_states_mask  # 使用mask版本
            else:
                shared_input = hidden_states  # 使用原始输入

            shared_output = self.shared_expert(shared_input)
        else:
            # 不使用共享专家
            shared_output = torch.zeros_like(hidden_states)

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

        # 4. 共享专家和文化专家融合（根据配置决定融合方式）
        moe_alpha = torch.tensor(0.5, device=hidden_states.device)  # 默认值
        if self.use_gate_fusion and self.use_shared_expert:
            # 使用门控融合
            moe_alpha = torch.sigmoid(self.moe_fusion_alpha)
            final_output = (1 - moe_alpha) * shared_output + moe_alpha * cultural_expert_output
        elif self.use_shared_expert:
            # 直接相加
            final_output = shared_output + cultural_expert_output
        else:
            # 只使用文化专家输出
            final_output = cultural_expert_output

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
        expert_info = {}

        # 共享专家信息（如果存在）
        if self.use_shared_expert and self.shared_expert is not None:
            expert_info['shared_expert'] = {
                'type': 'shared',
                'input_type': 'masked' if self.use_mask_mechanism else 'unmasked',
                'param_count': sum(p.numel() for p in self.shared_expert.parameters()),
                'description': 'Handles general knowledge using masked input' if self.use_mask_mechanism else 'Handles general knowledge'
            }

        # 文化专家信息
        if hasattr(self, 'cultural_experts'):
            expert_info['cultural_experts'] = {
                'type': 'cultural',
                'input_type': 'unmasked',
                'num_experts': self.num_experts,
                'param_count': sum(p.numel() for p in self.cultural_experts.parameters()),
                'description': 'Handle culture-specific knowledge using unmasked input',
                'culture_assignments': getattr(self.cultural_experts, 'culture_assignments', [])
            }

        return expert_info

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

        # 共享专家信息（如果存在）
        if 'shared_expert' in expert_info:
            shared_info = expert_info['shared_expert']
            print(f"Shared Expert:")
            print(f"  Type: {shared_info['type']}")
            print(f"  Input: {shared_info['input_type']}")
            print(f"  Parameters: {shared_info['param_count']:,}")
            print(f"  Description: {shared_info['description']}")
            print()

        # 文化专家信息
        if 'cultural_experts' in expert_info:
            cultural_info = expert_info['cultural_experts']
            print(f"Cultural Experts:")
            print(f"  Type: {cultural_info['type']}")
            print(f"  Input: {cultural_info['input_type']}")
            print(f"  Count: {cultural_info['num_experts']}")
            print(f"  Parameters: {cultural_info['param_count']:,}")
            print(f"  Description: {cultural_info['description']}")

            # 文化分配（如果有）
            culture_assignments = cultural_info.get('culture_assignments', [])
            if culture_assignments:
                print(f"\nCulture Assignments:")
                for i, assignment in enumerate(culture_assignments):
                    if len(assignment) == 0:
                        role = "Conflict Resolution Specialist"
                    elif len(assignment) == 6:
                        role = "Cross-Cultural Generalist"
                    else:
                        role = f"Cultures {assignment}"
                    print(f"  Expert {i}: {role}")

        if not expert_info:
            print("  This layer uses standard FFN (no MoE)")

        print("-" * 50)