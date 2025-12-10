#!/usr/bin/env python3
"""
简化版文化损失函数
回到接近原始实现，但保留内存银行机制
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict


def compute_simple_culture_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    memory_bank: Optional[Dict[str, torch.Tensor]] = None,
    loss_weight: float = 0.01
) -> torch.Tensor:
    """
    简化版文化损失函数
    主要保留对比学习，去除过度正则化
    """
    if expert_weights is None or culture_labels is None:
        device = culture_labels.device if culture_labels is not None else torch.device('cuda')
        return torch.tensor(0.0, device=device, dtype=torch.float16, requires_grad=True)

    batch_size = expert_weights.shape[0]
    device = expert_weights.device
    dtype = torch.float16

    # 检查维度匹配
    if expert_weights.shape[0] != culture_labels.shape[0]:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    # 检查输入有效性
    if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    try:
        if batch_size >= 2:
            # 标准成对比较（原始实现）
            return compute_pairwise_culture_loss(expert_weights, culture_labels, loss_weight)
        else:
            # batch_size=1时的处理
            if memory_bank is not None:
                # 轻量级内存银行对比
                return compute_lightweight_memory_loss(
                    expert_weights, culture_labels, memory_bank, loss_weight
                )
            else:
                # 简单的专家正则化
                return compute_simple_regularization(expert_weights, loss_weight)

    except Exception as e:
        print(f"⚠️ Simple culture loss failed: {e}")
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)


def compute_pairwise_culture_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    weight: float = 0.01
) -> torch.Tensor:
    """原始成对文化损失（数值稳定版）"""
    device = expert_weights.device
    dtype = torch.float16
    batch_size = expert_weights.shape[0]

    try:
        culture_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
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

                # 确保similarity是标量
                if similarity.dim() > 0:
                    similarity = similarity.mean()

                if torch.isnan(similarity) or torch.isinf(similarity):
                    continue

                if culture_labels[i] == culture_labels[j]:
                    # 相同文化，鼓励相似
                    loss_increment = (1.0 - similarity) * weight
                else:
                    # 不同文化，鼓励不同
                    loss_increment = similarity * weight

                if not (torch.isnan(loss_increment) or torch.isinf(loss_increment)):
                    culture_loss = culture_loss + loss_increment
                    count += 1

        if count > 0:
            culture_loss = culture_loss / count

        return culture_loss

    except Exception as e:
        print(f"⚠️ Pairwise culture loss failed: {e}")
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)


def compute_lightweight_memory_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    memory_bank: Dict[str, torch.Tensor],
    weight: float = 0.01
) -> torch.Tensor:
    """轻量级内存银行对比损失"""
    device = expert_weights.device
    dtype = torch.float16

    try:
        total_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        batch_size = expert_weights.shape[0]

        for i in range(batch_size):
            culture_id = culture_labels[i].item()
            current_weights = expert_weights[i]

            # 检查当前权重有效性
            if torch.isnan(current_weights).any() or torch.isinf(current_weights).any():
                continue

            weight_key = f"culture_{culture_id}_weights"

            if weight_key in memory_bank:
                # 简单的一致性损失
                historical_weights = memory_bank[weight_key].to(device=device, dtype=dtype)

                if not (torch.isnan(historical_weights).any() or torch.isinf(historical_weights).any()):
                    similarity = F.cosine_similarity(
                        current_weights.unsqueeze(0),
                        historical_weights.unsqueeze(0),
                        dim=-1
                    )

                    if similarity.dim() > 0:
                        similarity = similarity.mean()

                    if not (torch.isnan(similarity) or torch.isinf(similarity)):
                        # 轻微的一致性鼓励
                        loss_increment = (1.0 - similarity) * weight * 0.5
                        if not (torch.isnan(loss_increment) or torch.isinf(loss_increment)):
                            total_loss = total_loss + loss_increment

                # 更新内存银行（更保守的更新）
                momentum = 0.95  # 更高的momentum，更保守的更新
                updated_weights = (
                    momentum * historical_weights +
                    (1 - momentum) * current_weights.detach()
                ).to(dtype=dtype)

                if not (torch.isnan(updated_weights).any() or torch.isinf(updated_weights).any()):
                    memory_bank[weight_key] = updated_weights
            else:
                # 初始化
                memory_bank[weight_key] = current_weights.detach().to(dtype=dtype)

        return total_loss

    except Exception as e:
        print(f"⚠️ Lightweight memory loss failed: {e}")
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)


def compute_simple_regularization(
    expert_weights: torch.Tensor,
    weight: float = 0.01
) -> torch.Tensor:
    """简单的专家正则化（极轻量）"""
    device = expert_weights.device
    dtype = torch.float16

    try:
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # 非常轻微的熵正则化
        expert_probs = torch.softmax(expert_weights, dim=-1)
        entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=-1)

        if entropy.dim() > 0:
            entropy = entropy.mean()

        if torch.isnan(entropy) or torch.isinf(entropy):
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # 非常小的权重，几乎不影响主要训练
        regularization_loss = -entropy * weight * 0.1

        return regularization_loss.to(device=device, dtype=dtype)

    except Exception as e:
        print(f"⚠️ Simple regularization failed: {e}")
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)