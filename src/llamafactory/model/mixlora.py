#!/usr/bin/env python3
"""
MixLoRA Implementation

基于LoRA的参数高效混合专家方法，将LoRA的参数高效性与MoE模型的强大性能结合起来。

核心思想：
- 在冻结的预训练稠密模型基础上，使用多个LoRA模块来构建专家
- 共享FFN层：不改变预训练模型原有的、冻结的FFN层
- LoRA专家：每个"专家"由共享的FFN层 + 独立的LoRA适配器构成
- Top-K路由器：根据输入令牌选择最相关的K个专家
- 负载均衡：使用辅助损失确保专家使用的均衡分布

参考论文：MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
from peft.tuners.lora import LoraLayer


logger = logging.getLogger(__name__)


class MixLoRARouter(nn.Module):
    """
    MixLoRA路由器 - Top-K专家选择

    根据输入特征计算每个专家的匹配分数，通过Top-K操作选择最相关的专家。
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 6,
        top_k: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        初始化MixLoRA路由器

        Args:
            input_dim: 输入特征维度
            num_experts: 专家数量
            top_k: 每次选择的专家数量
            hidden_dim: 路由网络隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        # 路由网络：简单的线性层
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

        # 初始化权重
        self._init_weights()

        logger.info(
            f"MixLoRA Router initialized: {input_dim} -> {num_experts} experts, "
            f"Top-{top_k} routing"
        )

    def _init_weights(self):
        """初始化路由网络权重"""
        # 使用小的初始化确保路由开始时相对均匀
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 - Top-K专家路由

        Args:
            hidden_states: 输入特征 [batch_size, seq_len, input_dim]

        Returns:
            selected_experts: 选中的专家索引 [batch_size, seq_len, top_k]
            expert_weights: 选中专家的权重 [batch_size, seq_len, top_k]
            router_logits: 所有专家的原始logits [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, input_dim = hidden_states.shape

        # 重塑为 [batch_size * seq_len, input_dim] 进行路由计算
        hidden_flat = hidden_states.view(-1, input_dim)

        # 确保gate层在正确的设备和数据类型上
        if self.gate.weight.device != hidden_flat.device or self.gate.weight.dtype != hidden_flat.dtype:
            self.gate = self.gate.to(device=hidden_flat.device, dtype=hidden_flat.dtype)

        # 计算路由logits
        router_logits = self.gate(hidden_flat)  # [batch_size * seq_len, num_experts]

        # Top-K选择
        top_k_logits, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # 两个都是 [batch_size * seq_len, top_k]

        # 对选中的专家进行softmax归一化
        expert_weights = F.softmax(top_k_logits, dim=-1)

        # 重塑回原来的形状
        selected_experts = selected_experts.view(batch_size, seq_len, self.top_k)
        expert_weights = expert_weights.view(batch_size, seq_len, self.top_k)
        router_logits = router_logits.view(batch_size, seq_len, self.num_experts)

        return selected_experts, expert_weights, router_logits

    def compute_load_balancing_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        计算负载均衡损失

        参考Switch Transformer的负载均衡损失：
        L_aux = α * N * Σ (F_i * P_i)

        Args:
            router_logits: 所有专家的原始logits [batch_size, seq_len, num_experts]
            selected_experts: 选中的专家索引 [batch_size, seq_len, top_k]

        Returns:
            load_balancing_loss: 负载均衡损失
        """
        batch_size, seq_len, num_experts = router_logits.shape

        # 计算每个专家被选中的概率 (P_i)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_probs = router_probs.mean(dim=[0, 1])  # [num_experts]

        # 计算每个专家实际被分配的令牌比例 (F_i)
        expert_counts = torch.zeros(num_experts, device=selected_experts.device)
        total_tokens = batch_size * seq_len * self.top_k

        for i in range(num_experts):
            expert_counts[i] = (selected_experts == i).sum().float()

        expert_freqs = expert_counts / total_tokens

        # 负载均衡损失
        load_balancing_loss = self.num_experts * torch.sum(expert_freqs * expert_probs)

        return load_balancing_loss


class MixLoRAExpert(nn.Module):
    """
    MixLoRA专家模块

    每个专家由共享的FFN层 + 独立的LoRA适配器构成。
    """

    def __init__(
        self,
        expert_id: int,
        lora_config: LoraConfig,
        target_modules: List[str],
        base_layer_dict: Dict[str, nn.Module]
    ):
        """
        初始化MixLoRA专家

        Args:
            expert_id: 专家ID
            lora_config: LoRA配置
            target_modules: 目标模块名称列表
            base_layer_dict: 基础层字典 {module_name: base_layer}
        """
        super().__init__()
        self.expert_id = expert_id
        self.lora_config = lora_config
        self.target_modules = target_modules

        # 为每个目标模块创建LoRA适配器
        self.lora_adapters = nn.ModuleDict()

        for module_name, base_layer in base_layer_dict.items():
            if module_name in target_modules:
                # 创建LoRA适配器
                lora_adapter = self._create_lora_adapter(base_layer)
                self.lora_adapters[module_name] = lora_adapter

        logger.info(
            f"MixLoRA Expert {expert_id} initialized with {len(self.lora_adapters)} "
            f"LoRA adapters for modules: {list(self.lora_adapters.keys())}"
        )

    def _create_lora_adapter(self, base_layer: nn.Module) -> nn.Module:
        """
        为给定的基础层创建LoRA适配器

        Args:
            base_layer: 基础线性层

        Returns:
            LoRA适配器模块
        """
        if not isinstance(base_layer, nn.Linear):
            raise ValueError(f"Base layer must be nn.Linear, got {type(base_layer)}")

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # 创建LoRA A和B矩阵，使用基础层的设备和数据类型
        lora_A = nn.Linear(in_features, self.lora_config.r, bias=False,
                          device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        lora_B = nn.Linear(self.lora_config.r, out_features, bias=False,
                          device=base_layer.weight.device, dtype=base_layer.weight.dtype)

        # 初始化LoRA权重
        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(lora_B.weight)

        # 创建LoRA适配器容器
        lora_adapter = nn.ModuleDict({
            'lora_A': lora_A,
            'lora_B': lora_B
        })

        return lora_adapter

    def forward(
        self,
        module_name: str,
        base_output: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        专家前向传播

        Args:
            module_name: 模块名称
            base_output: 基础层的输出 W * x
            input_tensor: 输入张量 x

        Returns:
            专家输出: W * x + B_k * A_k * x
        """
        if module_name not in self.lora_adapters:
            return base_output

        # 获取LoRA适配器
        lora_adapter = self.lora_adapters[module_name]
        lora_A = lora_adapter['lora_A']
        lora_B = lora_adapter['lora_B']

        # 确保LoRA层在正确的设备和数据类型上
        if (lora_A.weight.device != input_tensor.device or
            lora_A.weight.dtype != input_tensor.dtype):
            lora_A = lora_A.to(device=input_tensor.device, dtype=input_tensor.dtype)
            lora_B = lora_B.to(device=input_tensor.device, dtype=input_tensor.dtype)
            # 更新适配器中的引用
            lora_adapter['lora_A'] = lora_A
            lora_adapter['lora_B'] = lora_B

        # 计算LoRA输出: B * A * x
        lora_output = lora_B(lora_A(input_tensor))

        # 应用LoRA缩放
        scaling = self.lora_config.lora_alpha / self.lora_config.r
        lora_output = lora_output * scaling

        # 返回组合输出: W * x + B * A * x
        return base_output + lora_output


class MixLoRALayer(nn.Module):
    """
    MixLoRA层 - 集成路由器和专家的完整MoE层

    这个层替换原来的FFN层，实现MixLoRA的核心功能。
    """

    def __init__(
        self,
        layer_idx: int,
        base_ffn_layers: Dict[str, nn.Module],
        lora_config: LoraConfig,
        num_experts: int = 6,
        top_k: int = 2,
        target_modules: Optional[List[str]] = None
    ):
        """
        初始化MixLoRA层

        Args:
            layer_idx: 层索引
            base_ffn_layers: 基础FFN层字典 {module_name: layer}
            lora_config: LoRA配置
            num_experts: 专家数量
            top_k: Top-K路由
            target_modules: 目标模块列表
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.lora_config = lora_config

        # 默认目标模块
        if target_modules is None:
            target_modules = ['gate_proj', 'up_proj', 'down_proj']
        self.target_modules = target_modules

        # 保存基础FFN层（冻结）
        self.base_ffn_layers = nn.ModuleDict()
        for name, layer in base_ffn_layers.items():
            layer.requires_grad_(False)  # 冻结基础层
            self.base_ffn_layers[name] = layer

        # 确定路由器输入维度
        first_layer = next(iter(base_ffn_layers.values()))
        if hasattr(first_layer, 'in_features'):
            router_input_dim = first_layer.in_features
        else:
            raise ValueError(f"Cannot determine input dimension from base layer: {type(first_layer)}")

        # 创建路由器
        self.router = MixLoRARouter(
            input_dim=router_input_dim,
            num_experts=num_experts,
            top_k=top_k
        )

        # 创建专家
        self.experts = nn.ModuleList([
            MixLoRAExpert(
                expert_id=i,
                lora_config=lora_config,
                target_modules=target_modules,
                base_layer_dict=base_ffn_layers
            )
            for i in range(num_experts)
        ])

        logger.info(
            f"MixLoRA Layer {layer_idx} initialized: {num_experts} experts, "
            f"Top-{top_k} routing, target modules: {target_modules}"
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        MixLoRA层前向传播

        实现计算优化策略：
        1. 先将整个输入序列通过共享FFN层计算
        2. 根据路由权重对结果进行切片和组合

        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_dim]

        Returns:
            output: 输出隐藏状态 [batch_size, seq_len, hidden_dim]
            aux_info: 辅助信息字典，包含路由信息和损失
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. 路由决策
        selected_experts, expert_weights, router_logits = self.router(hidden_states)

        # 2. 共享FFN计算 - 优化策略
        shared_outputs = {}
        for module_name, base_layer in self.base_ffn_layers.items():
            if module_name in self.target_modules:
                # 计算共享的基础输出
                shared_outputs[module_name] = base_layer(hidden_states)

        # 3. 专家计算和聚合
        final_output = torch.zeros_like(hidden_states)

        # 重塑数据进行专家计算
        hidden_flat = hidden_states.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
        selected_flat = selected_experts.view(-1, self.top_k)  # [batch_size * seq_len, top_k]
        weights_flat = expert_weights.view(-1, self.top_k)  # [batch_size * seq_len, top_k]

        # 为每个token计算专家输出
        for token_idx in range(batch_size * seq_len):
            token_hidden = hidden_flat[token_idx:token_idx+1]  # [1, hidden_dim]
            token_experts = selected_flat[token_idx]  # [top_k]
            token_weights = weights_flat[token_idx]  # [top_k]

            token_output = torch.zeros_like(token_hidden)

            # 聚合选中专家的输出
            for k in range(self.top_k):
                expert_id = token_experts[k].item()
                expert_weight = token_weights[k]

                expert = self.experts[expert_id]

                # 计算专家输出（基于共享FFN + LoRA）
                expert_output = token_hidden
                for module_name in self.target_modules:
                    if module_name in shared_outputs:
                        # 获取对应token的共享输出
                        batch_idx = token_idx // seq_len
                        seq_idx = token_idx % seq_len
                        base_output = shared_outputs[module_name][batch_idx:batch_idx+1, seq_idx:seq_idx+1]

                        # 专家处理
                        expert_output = expert(module_name, base_output, expert_output)

                token_output += expert_weight * expert_output

            # 将结果放回最终输出
            batch_idx = token_idx // seq_len
            seq_idx = token_idx % seq_len
            final_output[batch_idx, seq_idx] = token_output.squeeze(0)

        # 4. 计算负载均衡损失
        load_balancing_loss = self.router.compute_load_balancing_loss(
            router_logits, selected_experts
        )

        # 5. 准备辅助信息
        aux_info = {
            'load_balancing_loss': load_balancing_loss,
            'router_logits': router_logits,
            'selected_experts': selected_experts,
            'expert_weights': expert_weights,
            'layer_idx': self.layer_idx
        }

        return final_output, aux_info


class MixLoRAConfig:
    """MixLoRA配置类"""

    def __init__(
        self,
        # LoRA配置
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        # MoE配置
        num_experts: int = 6,
        top_k: int = 2,
        # 损失配置
        aux_loss_coef: float = 0.01,
        # 目标模块配置
        ffn_target_modules: Optional[List[str]] = None,
        attention_target_modules: Optional[List[str]] = None,
        # 其他配置
        apply_mixlora_to_attention: bool = False
    ):
        """
        初始化MixLoRA配置

        Args:
            lora_rank: LoRA秩
            lora_alpha: LoRA alpha参数
            lora_dropout: LoRA dropout概率
            num_experts: 专家数量
            top_k: Top-K路由
            aux_loss_coef: 辅助损失系数
            ffn_target_modules: FFN目标模块
            attention_target_modules: 注意力目标模块
            apply_mixlora_to_attention: 是否在注意力层也应用MixLoRA
        """
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        # 默认目标模块
        if ffn_target_modules is None:
            ffn_target_modules = ['gate_proj', 'up_proj', 'down_proj']
        if attention_target_modules is None:
            attention_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

        self.ffn_target_modules = ffn_target_modules
        self.attention_target_modules = attention_target_modules
        self.apply_mixlora_to_attention = apply_mixlora_to_attention

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'aux_loss_coef': self.aux_loss_coef,
            'ffn_target_modules': self.ffn_target_modules,
            'attention_target_modules': self.attention_target_modules,
            'apply_mixlora_to_attention': self.apply_mixlora_to_attention
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MixLoRAConfig':
        """从字典创建配置"""
        return cls(**config_dict)


def compute_mixlora_total_loss(
    main_loss: torch.Tensor,
    aux_info_list: List[Dict[str, torch.Tensor]],
    aux_loss_coef: float = 0.01
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算MixLoRA总损失

    Args:
        main_loss: 主任务损失
        aux_info_list: 每层的辅助信息列表
        aux_loss_coef: 辅助损失系数

    Returns:
        total_loss: 总损失
        loss_info: 损失信息字典
    """
    # 计算总的负载均衡损失
    total_load_balancing_loss = torch.tensor(0.0, device=main_loss.device)

    for aux_info in aux_info_list:
        if 'load_balancing_loss' in aux_info:
            total_load_balancing_loss += aux_info['load_balancing_loss']

    # 总损失
    total_loss = main_loss + aux_loss_coef * total_load_balancing_loss

    # 损失信息
    loss_info = {
        'main_loss': main_loss,
        'load_balancing_loss': total_load_balancing_loss,
        'aux_loss_coef': aux_loss_coef,
        'total_loss': total_loss,
        'num_mixlora_layers': len(aux_info_list)
    }

    return total_loss, loss_info