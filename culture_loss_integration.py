#!/usr/bin/env python3
"""
文化损失集成模块
简化的接口，便于集成到现有训练代码中
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


def compute_culture_loss_enhanced(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    hidden_states: Optional[torch.Tensor] = None,
    expert_outputs: Optional[Dict[int, torch.Tensor]] = None,
    memory_bank: Optional[Dict[str, torch.Tensor]] = None,
    loss_weight: float = 0.01
) -> torch.Tensor:
    """
    增强版文化损失函数 - 简化接口版本
    针对batch_size=1优化，包含多种损失组件

    Args:
        expert_weights: 专家权重 [B, num_experts]
        culture_labels: 文化标签 [B]
        hidden_states: 隐藏状态 [B, L, H] (可选)
        expert_outputs: 专家输出字典 (可选)
        memory_bank: 内存银行 (可选)
        loss_weight: 损失权重

    Returns:
        总的增强文化损失
    """
    if expert_weights is None or culture_labels is None:
        device = culture_labels.device if culture_labels is not None else torch.device('cuda')
        return torch.tensor(0.0, device=device, dtype=torch.float16)

    batch_size = expert_weights.shape[0]
    device = expert_weights.device
    dtype = torch.float16

    # 检查维度匹配
    if expert_weights.shape[0] != culture_labels.shape[0]:
        return torch.tensor(0.0, device=device, dtype=dtype)

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # 1. 专家激活多样性损失
    diversity_loss = compute_diversity_loss(expert_weights, expert_outputs, loss_weight * 0.5)
    total_loss += diversity_loss

    # 2. 文化一致性损失（基于内存银行）
    if memory_bank is not None:
        consistency_loss = compute_consistency_loss(
            expert_weights, culture_labels, memory_bank, loss_weight * 0.3
        )
        total_loss += consistency_loss

    # 3. 基础文化对比损失
    if batch_size >= 2:
        # 标准成对比较
        contrastive_loss = compute_pairwise_loss(expert_weights, culture_labels, loss_weight)
        total_loss += contrastive_loss
    else:
        # batch_size=1时的处理
        if memory_bank is not None:
            # 基于内存银行的对比学习
            memory_contrastive_loss = compute_memory_contrastive_loss(
                expert_weights, culture_labels, memory_bank, loss_weight
            )
            total_loss += memory_contrastive_loss
        else:
            # 回退到专家正则化
            regularization_loss = compute_regularization_loss(expert_weights, loss_weight)
            total_loss += regularization_loss

    return total_loss.to(dtype=dtype)


def compute_diversity_loss(
    expert_weights: torch.Tensor,
    expert_outputs: Optional[Dict[int, torch.Tensor]] = None,
    weight: float = 0.005
) -> torch.Tensor:
    """专家激活多样性损失"""
    device = expert_weights.device
    dtype = torch.float16

    if weight <= 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # 计算专家权重分布的均匀性
    expert_probs = torch.softmax(expert_weights, dim=-1)  # [B, num_experts]
    avg_activation = expert_probs.mean(dim=0)  # [num_experts]

    # 鼓励专家激活的均匀分布
    num_experts = expert_probs.shape[-1]
    uniform_target = torch.ones_like(avg_activation) / num_experts

    # KL散度损失
    kl_loss = F.kl_div(
        torch.log(avg_activation + 1e-8),
        uniform_target,
        reduction='batchmean'
    )

    diversity_loss = kl_loss * weight

    # 如果有专家输出，增加输出多样性约束
    if expert_outputs is not None and len(expert_outputs) > 1:
        output_diversity = compute_output_diversity_simple(expert_outputs)
        diversity_loss += -output_diversity * weight * 0.5  # 负损失鼓励多样性

    return diversity_loss.to(dtype=dtype)


def compute_output_diversity_simple(expert_outputs: Dict[int, torch.Tensor]) -> torch.Tensor:
    """计算专家输出多样性（简化版）"""
    if len(expert_outputs) < 2:
        return torch.tensor(0.0)

    expert_list = list(expert_outputs.keys())
    similarities = []

    # 只计算前两个专家的相似度（减少计算量）
    if len(expert_list) >= 2:
        output1 = expert_outputs[expert_list[0]]  # [B, L, H]
        output2 = expert_outputs[expert_list[1]]  # [B, L, H]

        # 计算输出的余弦相似度
        flat1 = output1.flatten(start_dim=1)  # [B, L*H]
        flat2 = output2.flatten(start_dim=1)  # [B, L*H]

        # 避免零向量
        norm1 = torch.norm(flat1, dim=-1, keepdim=True)
        norm2 = torch.norm(flat2, dim=-1, keepdim=True)

        if (norm1 > 1e-8).all() and (norm2 > 1e-8).all():
            similarity = F.cosine_similarity(flat1, flat2, dim=-1)
            return similarity.mean()

    return torch.tensor(0.0)


def compute_consistency_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    memory_bank: Dict[str, torch.Tensor],
    weight: float = 0.003
) -> torch.Tensor:
    """文化一致性损失（基于内存银行）"""
    device = expert_weights.device
    dtype = torch.float16

    if weight <= 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    consistency_loss = torch.tensor(0.0, device=device, dtype=dtype)
    batch_size = expert_weights.shape[0]

    for i in range(batch_size):
        culture_id = culture_labels[i].item()
        current_weights = expert_weights[i]  # [num_experts]

        weight_key = f"culture_{culture_id}_weights"

        if weight_key in memory_bank:
            # 计算与历史模式的一致性
            historical_weights = memory_bank[weight_key]
            consistency = F.cosine_similarity(
                current_weights.unsqueeze(0),
                historical_weights.unsqueeze(0),
                dim=-1
            )
            consistency_loss += (1.0 - consistency.mean()) * weight

            # 更新内存银行（指数移动平均）
            momentum = 0.9
            memory_bank[weight_key] = (
                momentum * historical_weights +
                (1 - momentum) * current_weights.detach()
            ).to(dtype=dtype)
        else:
            # 初始化
            memory_bank[weight_key] = current_weights.detach().to(dtype=dtype)

    return consistency_loss.to(dtype=dtype)


def compute_memory_contrastive_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    memory_bank: Dict[str, torch.Tensor],
    weight: float = 0.01,
    temperature: float = 0.5
) -> torch.Tensor:
    """基于内存银行的对比学习损失"""
    device = expert_weights.device
    dtype = torch.float16

    if weight <= 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    contrastive_loss = torch.tensor(0.0, device=device, dtype=dtype)
    batch_size = expert_weights.shape[0]

    for i in range(batch_size):
        culture_id = culture_labels[i].item()
        current_weights = expert_weights[i]

        positive_similarities = []
        negative_similarities = []

        # 收集正负样本
        for key, stored_weights in memory_bank.items():
            if key.startswith("culture_") and key.endswith("_weights"):
                stored_culture_id = int(key.split("_")[1])

                similarity = F.cosine_similarity(
                    current_weights.unsqueeze(0),
                    stored_weights.unsqueeze(0),
                    dim=-1
                )

                if stored_culture_id == culture_id:
                    positive_similarities.append(similarity / temperature)
                else:
                    negative_similarities.append(similarity / temperature)

        # 计算对比损失
        if positive_similarities and negative_similarities:
            pos_scores = torch.stack(positive_similarities)
            neg_scores = torch.stack(negative_similarities)

            for pos_score in pos_scores:
                all_scores = torch.cat([pos_score.unsqueeze(0), neg_scores])
                log_prob = pos_score - torch.logsumexp(all_scores, dim=0)
                contrastive_loss += -log_prob * weight

    return contrastive_loss.to(dtype=dtype)


def compute_pairwise_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    weight: float = 0.01
) -> torch.Tensor:
    """成对文化损失（原始实现）"""
    device = expert_weights.device
    dtype = torch.float16
    batch_size = expert_weights.shape[0]

    culture_loss = torch.tensor(0.0, device=device, dtype=dtype)
    count = 0

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            vec1 = expert_weights[i].unsqueeze(0)
            vec2 = expert_weights[j].unsqueeze(0)

            # 检查零向量
            norm1 = torch.norm(vec1)
            norm2 = torch.norm(vec2)
            if norm1 < 1e-8 or norm2 < 1e-8:
                continue

            similarity = F.cosine_similarity(vec1, vec2)
            if torch.isnan(similarity) or torch.isinf(similarity):
                continue

            similarity_scalar = similarity.item() if similarity.numel() == 1 else similarity.mean().item()

            if culture_labels[i] == culture_labels[j]:
                # 相同文化，鼓励相似
                culture_loss += (1.0 - similarity_scalar)
            else:
                # 不同文化，鼓励不同
                culture_loss += similarity_scalar
            count += 1

    if count > 0:
        culture_loss = culture_loss / count * weight

    return culture_loss.to(dtype=dtype)


def compute_regularization_loss(
    expert_weights: torch.Tensor,
    weight: float = 0.01
) -> torch.Tensor:
    """专家权重正则化损失（batch_size=1回退方案）"""
    device = expert_weights.device
    dtype = torch.float16

    # 计算专家权重的熵
    expert_probs = torch.softmax(expert_weights, dim=-1)
    entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=-1)

    # 熵越大越好，损失是负熵
    regularization_loss = -entropy.mean() * weight
    return regularization_loss.to(dtype=dtype)