#!/usr/bin/env python3
"""
CultureMoE损失函数实现
包括负载均衡损失和文化对比损失
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import re
from collections import defaultdict




def compute_load_balance_loss(gate_probs: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    计算负载均衡损失

    Args:
        gate_probs: [batch_size, seq_len, num_experts] 专家选择概率
        num_experts: 专家数量

    Returns:
        load_balance_loss: 标量损失值
    """
    # 计算每个专家的平均使用率
    expert_usage = gate_probs.mean(dim=[0, 1])  # [num_experts]

    # 理想情况下每个专家的使用率应该是 1/num_experts
    ideal_usage = 1.0 / num_experts

    # 计算方差作为负载均衡损失
    load_balance_loss = torch.var(expert_usage) / (ideal_usage ** 2)

    return load_balance_loss


def compute_culture_contrastive_loss(
    expert_outputs: List[torch.Tensor],
    shared_output: torch.Tensor,
    culture_labels: torch.Tensor
) -> torch.Tensor:
    """
    计算文化对比损失

    新的损失函数公式：
    L = Σ [δ(culture_i, culture_j) * (1 - similarity_ijk) +
           (1 - δ(culture_i, culture_j)) * similarity_ijk +
           (-similarity(shared(i), shared(j)))]

    其中：
    - δ(culture_i, culture_j) = 1 当 culture_i == culture_j，否则为 0
    - 同文化时，希望 similarity_ijk 尽可能大（最小化 1-similarity）
    - 异文化时，希望 similarity_ijk 尽可能小（最小化 similarity）
    - 共享专家输出总是希望相似度大（最小化 -similarity，即最大化 similarity）

    Args:
        expert_outputs: List of [batch_size, seq_len, hidden_size] 每个路由专家的输出
        shared_output: [batch_size, seq_len, hidden_size] 共享专家输出
        culture_labels: [batch_size] 文化标签（整数张量）

    Returns:
        culture_loss: 标量损失值
    """
    if shared_output is None:
        return torch.tensor(0.0, device=expert_outputs[0].device)

    batch_size = expert_outputs[0].shape[0]
    num_experts = len(expert_outputs)

    # 将专家输出堆叠 [num_experts, batch_size, seq_len, hidden_size]
    stacked_expert_outputs = torch.stack(expert_outputs, dim=0)

    # 使用池化来获得句子级别的表示 [num_experts, batch_size, hidden_size]
    expert_sentence_repr = stacked_expert_outputs.mean(dim=2)
    shared_sentence_repr = shared_output.mean(dim=1)  # [batch_size, hidden_size]

    total_loss = 0.0
    num_pairs = 0

    # 计算路由专家的文化对比损失和共享专家损失
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            culture_i = culture_labels[i].item()
            culture_j = culture_labels[j].item()

            # 指示函数：同文化为1，异文化为0
            delta = 1.0 if culture_i == culture_j else 0.0

            # 计算共享专家相似度
            shared_repr_i = shared_sentence_repr[i]  # [hidden_size]
            shared_repr_j = shared_sentence_repr[j]  # [hidden_size]
            shared_similarity = F.cosine_similarity(shared_repr_i, shared_repr_j, dim=0)

            # 对于每个路由专家，计算样本i和样本j的相似度
            for expert_idx in range(num_experts):
                expert_repr_i = expert_sentence_repr[expert_idx, i]  # [hidden_size]
                expert_repr_j = expert_sentence_repr[expert_idx, j]  # [hidden_size]

                # 计算余弦相似度
                similarity_ijk = F.cosine_similarity(expert_repr_i, expert_repr_j, dim=0)

                # 按照新的公式计算损失
                # L = δ(culture_i, culture_j) * (1 - similarity_ijk) +
                #     (1 - δ(culture_i, culture_j)) * similarity_ijk +
                #     (-similarity(shared(i), shared(j)))
                loss_ij = (delta * (1.0 - similarity_ijk) +
                          (1.0 - delta) * similarity_ijk +
                          (-shared_similarity))

                total_loss += loss_ij
                num_pairs += 1

    # 平均损失
    if num_pairs > 0:
        culture_loss = total_loss / num_pairs
    else:
        culture_loss = torch.tensor(0.0, device=expert_outputs[0].device)

    return culture_loss


def compute_total_loss(
    main_loss: torch.Tensor,
    moe_aux_info: List[Dict],
    culture_labels: torch.Tensor,
    lambda_weight: float = 1.0,
    alpha_weight: float = 0.1,
    beta_weight: float = 0.5,
    use_culture_loss: str = "new"
) -> Tuple[torch.Tensor, Dict]:
    """
    计算总损失

    Args:
        main_loss: 主任务损失（交叉熵）
        moe_aux_info: MoE辅助信息列表
        culture_labels: 文化标签张量 [batch_size]
        lambda_weight: 辅助损失权重
        alpha_weight: 负载均衡损失权重
        beta_weight: 文化对比损失权重
        use_culture_loss: 是否使用文化损失

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的详细信息
    """
    device = main_loss.device

    # 初始化辅助损失
    load_balance_loss = torch.tensor(0.0, device=device)
    culture_loss = torch.tensor(0.0, device=device)

    # 计算负载均衡损失
    if len(moe_aux_info) > 0:
        total_load_balance_loss = 0.0
        num_layers = len(moe_aux_info)

        for layer_aux in moe_aux_info:
            gate_probs = layer_aux['gate_probs']
            num_experts = gate_probs.shape[-1]
            layer_load_balance_loss = compute_load_balance_loss(gate_probs, num_experts)
            total_load_balance_loss += layer_load_balance_loss

        load_balance_loss = total_load_balance_loss / num_layers

    # 计算文化对比损失
    if use_culture_loss == "new" and len(moe_aux_info) > 0 and culture_labels is not None:
        total_culture_loss = 0.0
        num_layers = len(moe_aux_info)

        for layer_aux in moe_aux_info:
            expert_outputs = layer_aux['expert_outputs']
            shared_output = layer_aux['shared_output']
            layer_culture_loss = compute_culture_contrastive_loss(
                expert_outputs, shared_output, culture_labels
            )
            total_culture_loss += layer_culture_loss

        culture_loss = total_culture_loss / num_layers

    # 计算总损失
    aux_loss = alpha_weight * load_balance_loss + beta_weight * culture_loss
    total_loss = main_loss + lambda_weight * aux_loss

    # 损失详情
    loss_dict = {
        'main_loss': main_loss.item(),
        'load_balance_loss': load_balance_loss.item(),
        'culture_loss': culture_loss.item(),
        'aux_loss': aux_loss.item(),
        'total_loss': total_loss.item(),
        'lambda': lambda_weight,
        'alpha': alpha_weight,
        'beta': beta_weight
    }

    return total_loss, loss_dict