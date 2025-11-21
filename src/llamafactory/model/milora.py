#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiLoRA (Mixture of LoRA) Implementation

MiLoRA 的核心思想：
1. 每个 LoRA 模块视为一个专家（而非将一个 LoRA 分解为多个专家）
2. 提示感知路由机制：只在生成第一个新词前计算一次路由，后续复用
3. 每个 Transformer 层有 7 个线性模块专家：Q, K, V, O, G, U, D

架构特点：
- LoRA 专家定义：每个线性模块对应一个 LoRA 专家
- LoRA 路由器：包含池化器、可学习激活函数、MoE 路由器
- 训练机制：负载均衡损失 + 可学习激活函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class RationalActivation(nn.Module):
    """
    可学习的理性激活函数
    Ra(x) = (∑(j=0~m) a_j * x^j) / (1 + ||∑(i=1~n) b_i * x^i||)
    """

    def __init__(self, input_dim: int, numerator_degree: int = 3, denominator_degree: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.numerator_degree = numerator_degree
        self.denominator_degree = denominator_degree

        # 分子系数 a_j (j=0 to m)
        self.numerator_coeffs = nn.Parameter(torch.randn(numerator_degree + 1, input_dim))

        # 分母系数 b_i (i=1 to n)
        self.denominator_coeffs = nn.Parameter(torch.randn(denominator_degree, input_dim))

        self._initialize_parameters()

    def _initialize_parameters(self):
        """初始化参数，使激活函数接近 ReLU"""
        with torch.no_grad():
            # 初始化分子系数，使其接近 max(0, x)
            self.numerator_coeffs[0].fill_(0.0)  # 常数项
            self.numerator_coeffs[1].fill_(1.0)  # 一次项
            self.numerator_coeffs[2:].fill_(0.0)  # 高次项

            # 更保守的分母系数初始化，避免数值不稳定
            self.denominator_coeffs.fill_(0.01)  # 更小的初始值

            # 限制参数范围避免极端值
            self.numerator_coeffs.data.clamp_(-2.0, 2.0)
            self.denominator_coeffs.data.clamp_(0.001, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
        Returns:
            激活后的张量
        """
        target_device = x.device

        # 确保系数在正确的设备和数据类型上
        if (self.numerator_coeffs.device != target_device or
            self.numerator_coeffs.dtype != x.dtype):
            self.numerator_coeffs = self.numerator_coeffs.to(device=target_device, dtype=x.dtype)
        if (self.denominator_coeffs.device != target_device or
            self.denominator_coeffs.dtype != x.dtype):
            self.denominator_coeffs = self.denominator_coeffs.to(device=target_device, dtype=x.dtype)

        # 计算分子：∑(j=0~m) a_j * x^j
        numerator = self.numerator_coeffs[0]  # 常数项
        x_power = x

        for j in range(1, self.numerator_degree + 1):
            numerator = numerator + self.numerator_coeffs[j] * x_power
            if j < self.numerator_degree:
                x_power = x_power * x

        # 计算分母：1 + ||∑(i=1~n) b_i * x^i||
        denominator_sum = torch.zeros_like(x)
        x_power = x

        for i in range(self.denominator_degree):
            denominator_sum = denominator_sum + self.denominator_coeffs[i] * x_power
            if i < self.denominator_degree - 1:
                x_power = x_power * x

        # 数值稳定的分母计算
        denominator_norm = torch.norm(denominator_sum, dim=-1, keepdim=True)
        denominator = 1.0 + denominator_norm

        # 避免除零和数值不稳定
        denominator = torch.clamp(denominator, min=1e-6)

        result = numerator / denominator

        # 检查和修复NaN/Inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN/Inf in RationalActivation, using ReLU fallback")
            result = torch.relu(x)

        return result


class LoRAPooler(nn.Module):
    """
    LoRA 路由器的池化组件
    支持多种池化方式：last-token, average, max, self-attention
    """

    def __init__(self, hidden_dim: int, pooling_type: str = "self_attention"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type

        if pooling_type == "self_attention":
            self.attention_weights = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        池化操作
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] 可选的注意力掩码
        Returns:
            pooled_output: [batch_size, hidden_dim]
        """
        if self.pooling_type == "last_token":
            # 使用最后一个 token
            if attention_mask is not None:
                # 找到每个序列的最后一个有效 token
                last_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                return hidden_states[batch_indices, last_indices]
            else:
                return hidden_states[:, -1]

        elif self.pooling_type == "average":
            # 平均池化
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(hidden_states, dim=1)

        elif self.pooling_type == "max":
            # 最大池化
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                hidden_states = hidden_states.masked_fill(mask_expanded == 0, float('-inf'))
            return torch.max(hidden_states, dim=1)[0]

        elif self.pooling_type == "self_attention":
            # 自注意力池化（默认）
            target_device = hidden_states.device

            # 确保attention_weights在正确的设备和数据类型上
            if (self.attention_weights.weight.device != target_device or
                self.attention_weights.weight.dtype != hidden_states.dtype):
                self.attention_weights = self.attention_weights.to(device=target_device, dtype=hidden_states.dtype)

            attention_scores = self.attention_weights(hidden_states).squeeze(-1)  # [batch_size, seq_len]

            if attention_mask is not None:
                # 确保attention_mask在正确的设备上
                if attention_mask.device != target_device:
                    attention_mask = attention_mask.to(target_device)
                attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

            attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
            attention_weights = self.dropout(attention_weights)

            # 加权平均
            pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
            return pooled_output

        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")


class LoRARouter(nn.Module):
    """
    LoRA 路由器
    包含：池化器 + 可学习激活函数 + MoE 路由器
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 7,  # Q, K, V, O, G, U, D
        top_k: int = 3,
        pooling_type: str = "self_attention"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # 池化器
        self.pooler = LoRAPooler(hidden_dim, pooling_type)

        # 可学习的激活函数
        self.activation = RationalActivation(hidden_dim)

        # MoE 路由器权重
        self.router_weights = nn.Linear(hidden_dim, num_experts)

        # 用于负载均衡损失的统计
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('expert_probs', torch.zeros(num_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            training: 是否为训练模式
        Returns:
            expert_weights: [batch_size, top_k] 选中专家的权重
            expert_indices: [batch_size, top_k] 选中专家的索引
            load_balance_loss: 负载均衡损失
        """
        batch_size = hidden_states.size(0)
        target_device = hidden_states.device

        # 确保池化器在正确的设备和数据类型上
        if (next(self.pooler.parameters()).device != target_device or
            next(self.pooler.parameters()).dtype != hidden_states.dtype):
            self.pooler = self.pooler.to(device=target_device, dtype=hidden_states.dtype)

        # 确保激活函数在正确的设备和数据类型上
        if (next(self.activation.parameters()).device != target_device or
            next(self.activation.parameters()).dtype != hidden_states.dtype):
            self.activation = self.activation.to(device=target_device, dtype=hidden_states.dtype)

        # 确保路由器权重在正确的设备和数据类型上
        if (self.router_weights.weight.device != target_device or
            self.router_weights.weight.dtype != hidden_states.dtype):
            self.router_weights = self.router_weights.to(device=target_device, dtype=hidden_states.dtype)

        # 1. 池化：将隐藏状态聚合为固定长度向量
        pooled_hidden = self.pooler(hidden_states, attention_mask)  # [batch_size, hidden_dim]

        # 2. 可学习激活函数
        activated_hidden = self.activation(pooled_hidden)  # [batch_size, hidden_dim]

        # 3. MoE 路由器：计算专家概率
        router_logits = self.router_weights(activated_hidden)  # [batch_size, num_experts]

        # 检查和修复router_logits中的异常值
        if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
            print("Warning: NaN/Inf detected in router_logits, applying clipping")
            router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)

        # 使用数值稳定的softmax
        expert_probs = F.softmax(router_logits, dim=-1)  # [batch_size, num_experts]

        # 再次检查expert_probs
        if torch.isnan(expert_probs).any():
            print("Warning: NaN detected in expert_probs, using uniform distribution")
            expert_probs = torch.ones_like(expert_probs) / self.num_experts

        # 4. Top-k 选择
        top_k_probs, top_k_indices = torch.topk(expert_probs, self.top_k, dim=-1)

        # 重新归一化 top-k 权重（数值稳定版本）
        top_k_probs_stable = top_k_probs + 1e-8  # 避免全零
        expert_weights = F.softmax(top_k_probs_stable, dim=-1)  # [batch_size, top_k]

        # 5. 计算负载均衡损失
        load_balance_loss = torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True).sum()

        if training:
            # 更新专家使用统计
            expert_mask = torch.zeros(batch_size, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype)
            expert_mask.scatter_(1, top_k_indices, 1.0)

            # 频率统计
            expert_freq = expert_mask.mean(dim=0)  # [num_experts]

            # 概率统计
            expert_avg_prob = expert_probs.mean(dim=0)  # [num_experts]

            # 检查是否有NaN
            if torch.isnan(expert_freq).any() or torch.isnan(expert_avg_prob).any():
                print("Warning: NaN detected in expert statistics, using zero load balance loss")
                # 创建一个有梯度的零张量
                load_balance_loss = torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True).sum()
            else:
                # 负载均衡损失：L_lb = N_mod * ∑(f_i * p_i)
                balance_product = expert_freq * expert_avg_prob

                # 再次检查乘积结果
                if torch.isnan(balance_product).any():
                    print("Warning: NaN detected in balance product, using zero load balance loss")
                    # 创建一个有梯度的零张量
                    load_balance_loss = torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True).sum()
                else:
                    load_balance_loss = self.num_experts * torch.sum(balance_product)

                    # 最终检查负载均衡损失
                    if torch.isnan(load_balance_loss) or torch.isinf(load_balance_loss):
                        print("Warning: NaN/Inf in final load balance loss, setting to zero")
                        # 创建一个有梯度的零张量
                        load_balance_loss = torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True).sum()

            # 更新全局统计（用于监控）
            with torch.no_grad():
                # 确保统计张量在正确的设备上
                expert_mask_sum = expert_mask.sum(dim=0)
                expert_probs_sum = expert_probs.sum(dim=0)

                if self.expert_counts.device != expert_mask_sum.device:
                    self.expert_counts = self.expert_counts.to(expert_mask_sum.device)
                if self.expert_probs.device != expert_probs_sum.device:
                    self.expert_probs = self.expert_probs.to(expert_probs_sum.device)

                self.expert_counts += expert_mask_sum
                self.expert_probs += expert_probs_sum
                self.total_samples += batch_size

        return expert_weights, top_k_indices, load_balance_loss


class LoRAExpert(nn.Module):
    """
    单个 LoRA 专家
    对应 Transformer 中的一个线性模块（Q, K, V, O, G, U, D）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA 低秩矩阵
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 原始线性层（冻结）
        self.base_layer = nn.Linear(in_features, out_features)

        self._initialize_parameters()

    def _initialize_parameters(self):
        """初始化 LoRA 参数"""
        # LoRA A 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # LoRA B 初始化为零，确保训练开始时 LoRA 输出为零
        nn.init.zeros_(self.lora_B.weight)

        # 冻结基础层
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：x' = x * W_m + x * W_m^A * W_m^B + b_m
        """
        target_device = x.device
        target_dtype = x.dtype

        # 确保基础层在正确的设备和数据类型上
        if (next(self.base_layer.parameters()).device != target_device or
            next(self.base_layer.parameters()).dtype != target_dtype):
            self.base_layer = self.base_layer.to(device=target_device, dtype=target_dtype)

        # 确保LoRA层在正确的设备和数据类型上
        if (self.lora_A.weight.device != target_device or
            self.lora_A.weight.dtype != target_dtype):
            self.lora_A = self.lora_A.to(device=target_device, dtype=target_dtype)
            self.lora_B = self.lora_B.to(device=target_device, dtype=target_dtype)

        # 确保dropout层在正确的设备上（如果有参数）
        if hasattr(self.dropout, 'weight') and self.dropout.weight is not None:
            if self.dropout.weight.device != target_device:
                self.dropout = self.dropout.to(target_device)

        # 基础层输出
        base_output = self.base_layer(x)

        # 确保基础层输出在正确的数据类型上
        if base_output.dtype != target_dtype:
            base_output = base_output.to(target_dtype)

        # LoRA 输出
        lora_input = self.dropout(x)
        lora_a_output = self.lora_A(lora_input)
        lora_output = self.lora_B(lora_a_output) * self.scaling

        # 确保LoRA输出在正确的数据类型上
        if lora_output.dtype != target_dtype:
            lora_output = lora_output.to(target_dtype)

        return base_output + lora_output


class MiLoRALayer(nn.Module):
    """
    MiLoRA 层
    包含 7 个 LoRA 专家（Q, K, V, O, G, U, D）和一个路由器
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_attention_heads: int,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        top_k: int = 3,
        pooling_type: str = "self_attention",
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_dim // num_attention_heads
        self.top_k = top_k

        # LoRA 路由器
        self.router = LoRARouter(
            hidden_dim=hidden_dim,
            num_experts=7,  # Q, K, V, O, G, U, D
            top_k=top_k,
            pooling_type=pooling_type
        )

        # 7 个 LoRA 专家
        self.experts = nn.ModuleDict({
            'q_proj': LoRAExpert(hidden_dim, hidden_dim, lora_rank, lora_alpha, dropout),
            'k_proj': LoRAExpert(hidden_dim, hidden_dim, lora_rank, lora_alpha, dropout),
            'v_proj': LoRAExpert(hidden_dim, hidden_dim, lora_rank, lora_alpha, dropout),
            'o_proj': LoRAExpert(hidden_dim, hidden_dim, lora_rank, lora_alpha, dropout),
            'gate_proj': LoRAExpert(hidden_dim, intermediate_dim, lora_rank, lora_alpha, dropout),
            'up_proj': LoRAExpert(hidden_dim, intermediate_dim, lora_rank, lora_alpha, dropout),
            'down_proj': LoRAExpert(intermediate_dim, hidden_dim, lora_rank, lora_alpha, dropout),
        })

        self.expert_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

        # 缓存路由结果（用于提示感知机制）
        self.cached_expert_weights = None
        self.cached_expert_indices = None
        self.use_cached_routing = False

    def set_prompt_routing_mode(self, use_cached: bool = True):
        """
        设置提示感知路由模式
        Args:
            use_cached: 是否使用缓存的路由结果
        """
        self.use_cached_routing = use_cached
        if not use_cached:
            self.cached_expert_weights = None
            self.cached_expert_indices = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_name: str = 'q_proj',
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            expert_name: 要使用的专家名称
            training: 是否为训练模式
        Returns:
            output: 专家输出
            load_balance_loss: 负载均衡损失
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        target_device = hidden_states.device
        target_dtype = hidden_states.dtype

        # 确保路由器在正确的设备和数据类型上
        if (next(self.router.parameters()).device != target_device or
            next(self.router.parameters()).dtype != target_dtype):
            self.router = self.router.to(device=target_device, dtype=target_dtype)

        # 确保所有专家在正确的设备和数据类型上
        for expert_key, expert in self.experts.items():
            if (next(expert.parameters()).device != target_device or
                next(expert.parameters()).dtype != target_dtype):
                self.experts[expert_key] = expert.to(device=target_device, dtype=target_dtype)

        # 1. 路由决策（提示感知机制）
        if self.use_cached_routing and self.cached_expert_weights is not None:
            # 使用缓存的路由结果，确保在正确设备和数据类型上
            expert_weights = self.cached_expert_weights.to(device=target_device, dtype=target_dtype)
            expert_indices = self.cached_expert_indices.to(target_device)
            load_balance_loss = torch.zeros(1, device=target_device, dtype=target_dtype, requires_grad=True).sum()
        else:
            # 计算新的路由结果
            expert_weights, expert_indices, load_balance_loss = self.router(
                hidden_states, attention_mask, training
            )

            # 确保路由结果在正确的设备和数据类型上
            if expert_weights.device != target_device or expert_weights.dtype != target_dtype:
                expert_weights = expert_weights.to(device=target_device, dtype=target_dtype)
            if expert_indices.device != target_device:
                expert_indices = expert_indices.to(target_device)
            if load_balance_loss.device != target_device or load_balance_loss.dtype != target_dtype:
                load_balance_loss = load_balance_loss.to(device=target_device, dtype=target_dtype)

            # 缓存路由结果（用于后续生成步骤）
            if self.use_cached_routing:
                self.cached_expert_weights = expert_weights.detach()
                self.cached_expert_indices = expert_indices.detach()

        # 2. 获取目标专家索引
        if expert_name not in self.expert_names:
            raise ValueError(f"Unknown expert name: {expert_name}")

        target_expert_idx = self.expert_names.index(expert_name)

        # 3. 检查目标专家是否在 top-k 中
        expert_mask = (expert_indices == target_expert_idx).float()  # [batch_size, top_k]
        expert_in_topk = expert_mask.sum(dim=-1) > 0  # [batch_size]

        # 4. 计算专家输出
        if expert_in_topk.any():
            # 有些样本的目标专家在 top-k 中
            expert_output = torch.zeros_like(hidden_states)

            for batch_idx in range(batch_size):
                if expert_in_topk[batch_idx]:
                    # 找到目标专家在 top-k 中的位置
                    topk_positions = (expert_indices[batch_idx] == target_expert_idx).nonzero(as_tuple=True)[0]
                    if len(topk_positions) > 0:
                        weight = expert_weights[batch_idx, topk_positions[0]]
                        expert_out = self.experts[expert_name](hidden_states[batch_idx:batch_idx+1])

                        # 确保专家输出在正确的设备和数据类型上
                        if expert_out.device != target_device or expert_out.dtype != target_dtype:
                            expert_out = expert_out.to(device=target_device, dtype=target_dtype)

                        expert_output[batch_idx] = expert_out.squeeze(0) * weight
                else:
                    # 目标专家不在 top-k 中，使用基础层
                    base_out = self.experts[expert_name].base_layer(hidden_states[batch_idx])

                    # 确保基础层输出在正确的设备和数据类型上
                    if base_out.device != target_device or base_out.dtype != target_dtype:
                        base_out = base_out.to(device=target_device, dtype=target_dtype)

                    expert_output[batch_idx] = base_out
        else:
            # 所有样本的目标专家都不在 top-k 中，使用基础层
            expert_output = self.experts[expert_name].base_layer(hidden_states)

            # 确保基础层输出在正确的设备和数据类型上
            if expert_output.device != target_device or expert_output.dtype != target_dtype:
                expert_output = expert_output.to(device=target_device, dtype=target_dtype)

        # 最终检查expert_output和load_balance_loss
        if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
            print("Warning: NaN/Inf in expert_output, using zero output")
            expert_output = torch.zeros_like(hidden_states)

        if torch.isnan(load_balance_loss) or torch.isinf(load_balance_loss):
            print("Warning: NaN/Inf in load_balance_loss, setting to zero")
            # 创建一个有梯度的零张量
            load_balance_loss = torch.zeros(1, device=target_device, dtype=target_dtype, requires_grad=True).sum()

        return expert_output, load_balance_loss

    def get_expert_statistics(self) -> Dict[str, torch.Tensor]:
        """获取专家使用统计"""
        if self.router.total_samples > 0:
            expert_usage_freq = self.router.expert_counts / self.router.total_samples
            expert_avg_prob = self.router.expert_probs / self.router.total_samples

            return {
                'expert_usage_frequency': expert_usage_freq,
                'expert_average_probability': expert_avg_prob,
                'total_samples': self.router.total_samples,
                'expert_names': self.expert_names
            }
        else:
            return {}


class MiLoRAModel(nn.Module):
    """
    完整的 MiLoRA 模型
    在每个 Transformer 层中集成 MiLoRA 层
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_layers: int,
        hidden_dim: int,
        intermediate_dim: int,
        num_attention_heads: int,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        top_k: int = 3,
        pooling_type: str = "self_attention",
        dropout: float = 0.1,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        self.base_model = base_model
        self.num_layers = num_layers
        self.load_balance_weight = load_balance_weight

        # 为每一层创建 MiLoRA 层
        self.milora_layers = nn.ModuleList([
            MiLoRALayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_attention_heads=num_attention_heads,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                top_k=top_k,
                pooling_type=pooling_type,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False

    def set_prompt_routing_mode(self, use_cached: bool = True):
        """
        设置所有层的提示感知路由模式
        """
        for layer in self.milora_layers:
            layer.set_prompt_routing_mode(use_cached)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        # 确保所有MiLoRA层在正确的设备上
        target_device = input_ids.device
        base_dtype = next(self.base_model.parameters()).dtype

        # 将所有MiLoRA层移动到目标设备，并确保数据类型一致
        for layer_idx, layer in enumerate(self.milora_layers):
            if (next(layer.parameters()).device != target_device or
                next(layer.parameters()).dtype != base_dtype):
                self.milora_layers[layer_idx] = layer.to(device=target_device, dtype=base_dtype)

        # 基础模型前向传播（获取隐藏状态）
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        hidden_states = base_outputs.hidden_states
        total_load_balance_loss = torch.zeros(1, device=target_device, dtype=base_dtype, requires_grad=True).sum()

        # 通过每个 MiLoRA 层处理隐藏状态
        milora_outputs = []
        for layer_idx, milora_layer in enumerate(self.milora_layers):
            layer_hidden = hidden_states[layer_idx + 1]  # 跳过 embedding 层

            # 确保层隐藏状态在正确的设备和数据类型上
            if (layer_hidden.device != target_device or
                layer_hidden.dtype != base_dtype):
                layer_hidden = layer_hidden.to(device=target_device, dtype=base_dtype)

            # 确保注意力掩码在正确的设备和数据类型上
            if attention_mask is not None:
                if attention_mask.device != target_device:
                    attention_mask = attention_mask.to(target_device)
                # 注意力掩码通常是整数类型，不需要改变dtype

            # 为每个专家计算输出（这里简化为只使用 q_proj）
            expert_output, load_balance_loss = milora_layer(
                layer_hidden,
                attention_mask,
                expert_name='q_proj',
                training=self.training
            )

            # 确保专家输出在正确的设备和数据类型上
            if (expert_output.device != target_device or
                expert_output.dtype != base_dtype):
                expert_output = expert_output.to(device=target_device, dtype=base_dtype)

            milora_outputs.append(expert_output)

            # 确保负载均衡损失在正确的设备和数据类型上
            if (load_balance_loss.device != target_device or
                load_balance_loss.dtype != base_dtype):
                load_balance_loss = load_balance_loss.to(device=target_device, dtype=base_dtype)

            total_load_balance_loss += load_balance_loss

        return {
            'logits': base_outputs.logits,
            'hidden_states': hidden_states,
            'milora_outputs': milora_outputs,
            'load_balance_loss': total_load_balance_loss * self.load_balance_weight
        }

    def get_model_statistics(self) -> Dict[str, any]:
        """获取模型统计信息"""
        stats = {
            'num_layers': self.num_layers,
            'total_experts': len(self.milora_layers) * 7,
            'layer_statistics': {}
        }

        for layer_idx, layer in enumerate(self.milora_layers):
            layer_stats = layer.get_expert_statistics()
            if layer_stats:
                stats['layer_statistics'][f'layer_{layer_idx}'] = layer_stats

        return stats