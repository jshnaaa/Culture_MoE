#!/usr/bin/env python3
"""
增强版文化损失函数
针对batch_size=1的资源受限环境优化，包含多种损失组件
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


def compute_enhanced_culture_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    hidden_states: Optional[torch.Tensor] = None,
    expert_outputs: Optional[Dict[int, torch.Tensor]] = None,
    loss_weight: float = 0.01,
    diversity_weight: float = 0.005,
    consistency_weight: float = 0.003,
    memory_bank: Optional[Dict[str, torch.Tensor]] = None,
    update_memory: bool = True
) -> torch.Tensor:
    """
    增强版文化损失函数，专为batch_size=1优化

    组件：
    1. 专家激活多样性损失 - 防止专家退化
    2. 文化一致性正则化 - 利用历史信息
    3. 跨文化对比学习 - 基于内存银行的对比学习

    Args:
        expert_weights: 专家权重 [B, num_experts]
        culture_labels: 文化标签 [B]
        hidden_states: 隐藏状态 [B, L, H] (可选，用于表示学习)
        expert_outputs: 专家输出字典 (可选，用于多样性计算)
        loss_weight: 总损失权重
        diversity_weight: 多样性损失权重
        consistency_weight: 一致性损失权重
        memory_bank: 文化表示内存银行
        update_memory: 是否更新内存银行

    Returns:
        enhanced_culture_loss: 增强的文化损失
    """
    if expert_weights is None or culture_labels is None:
        return torch.tensor(0.0, device=culture_labels.device if culture_labels is not None else torch.device('cuda'), dtype=torch.float16)

    batch_size = expert_weights.shape[0]
    device = expert_weights.device
    dtype = torch.float16

    # 检查维度匹配
    if expert_weights.shape[0] != culture_labels.shape[0]:
        return torch.tensor(0.0, device=device, dtype=dtype)

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # 1. 专家激活多样性损失 (Expert Activation Diversity Loss)
    diversity_loss = compute_expert_diversity_loss(
        expert_weights, expert_outputs, diversity_weight
    )
    total_loss += diversity_loss

    # 2. 文化一致性正则化 (Cultural Consistency Regularization)
    if memory_bank is not None:
        consistency_loss = compute_cultural_consistency_loss(
            expert_weights, culture_labels, hidden_states,
            memory_bank, consistency_weight, update_memory
        )
        total_loss += consistency_loss

    # 3. 基础文化对比损失 (适配batch_size=1)
    if batch_size >= 2:
        # 使用原始的成对比较逻辑
        contrastive_loss = compute_pairwise_culture_loss(
            expert_weights, culture_labels, loss_weight
        )
        total_loss += contrastive_loss
    else:
        # batch_size=1时，使用内存银行进行对比学习
        if memory_bank is not None:
            contrastive_loss = compute_memory_based_contrastive_loss(
                expert_weights, culture_labels, memory_bank, loss_weight
            )
            total_loss += contrastive_loss
        else:
            # 回退到专家权重正则化
            regularization_loss = compute_expert_regularization_loss(
                expert_weights, loss_weight
            )
            total_loss += regularization_loss

    return total_loss.to(dtype=dtype)


def compute_expert_diversity_loss(
    expert_weights: torch.Tensor,
    expert_outputs: Optional[Dict[int, torch.Tensor]] = None,
    weight: float = 0.005
) -> torch.Tensor:
    """
    专家激活多样性损失
    目标：防止专家退化，确保不同专家学到不同的表示
    """
    device = expert_weights.device
    dtype = torch.float16

    if weight <= 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # 方法1：基于专家权重的多样性
    # 计算专家权重的方差，鼓励专家权重有差异
    expert_probs = torch.softmax(expert_weights, dim=-1)  # [B, num_experts]

    # 计算每个专家在batch中的平均激活程度
    avg_activation = expert_probs.mean(dim=0)  # [num_experts]

    # 鼓励专家激活的均匀分布（防止某些专家完全不被激活）
    num_experts = expert_probs.shape[-1]
    uniform_target = torch.ones_like(avg_activation) / num_experts

    # KL散度损失：鼓励专家激活分布接近均匀分布
    kl_loss = F.kl_div(
        torch.log(avg_activation + 1e-8),
        uniform_target,
        reduction='batchmean'
    )

    diversity_loss = kl_loss * weight

    # 方法2：如果有专家输出，计算输出多样性
    if expert_outputs is not None and len(expert_outputs) > 1:
        output_diversity = compute_output_diversity(expert_outputs)
        # 鼓励专家输出多样性（负损失）
        diversity_loss += -output_diversity * weight * 0.5

    return diversity_loss.to(dtype=dtype)


def compute_output_diversity(expert_outputs: Dict[int, torch.Tensor]) -> torch.Tensor:
    """
    计算专家输出的多样性
    """
    if len(expert_outputs) < 2:
        return torch.tensor(0.0)

    expert_list = list(expert_outputs.keys())
    similarities = []

    for i in range(len(expert_list)):
        for j in range(i + 1, len(expert_list)):
            output1 = expert_outputs[expert_list[i]]  # [B, L, H]
            output2 = expert_outputs[expert_list[j]]  # [B, L, H]

            # 计算输出的余弦相似度
            flat1 = output1.flatten(start_dim=1)  # [B, L*H]
            flat2 = output2.flatten(start_dim=1)  # [B, L*H]

            # 避免零向量
            norm1 = torch.norm(flat1, dim=-1, keepdim=True)
            norm2 = torch.norm(flat2, dim=-1, keepdim=True)

            if (norm1 > 1e-8).all() and (norm2 > 1e-8).all():
                similarity = F.cosine_similarity(flat1, flat2, dim=-1)
                similarities.append(similarity.mean())

    if similarities:
        # 返回平均相似度，多样性越高相似度越低
        avg_similarity = torch.stack(similarities).mean()
        return avg_similarity
    else:
        return torch.tensor(0.0)


def compute_cultural_consistency_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    memory_bank: Dict[str, torch.Tensor],
    weight: float = 0.003,
    update_memory: bool = True
) -> torch.Tensor:
    """
    文化一致性正则化损失
    利用内存银行维护每种文化的专家激活模式
    """
    device = expert_weights.device
    dtype = torch.float16

    if weight <= 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    consistency_loss = torch.tensor(0.0, device=device, dtype=dtype)
    batch_size = expert_weights.shape[0]

    for i in range(batch_size):
        culture_id = culture_labels[i].item()
        current_weights = expert_weights[i]  # [num_experts]

        # 内存银行的键
        weight_key = f"culture_{culture_id}_weights"

        if weight_key in memory_bank:
            # 获取该文化的历史专家权重模式
            historical_weights = memory_bank[weight_key]  # [num_experts]

            # 计算当前权重与历史模式的一致性
            consistency = F.cosine_similarity(
                current_weights.unsqueeze(0),
                historical_weights.unsqueeze(0),
                dim=-1
            )

            # 鼓励一致性（1 - consistency 作为损失）
            consistency_loss += (1.0 - consistency.mean()) * weight

            if update_memory:
                # 指数移动平均更新内存银行
                momentum = 0.9
                memory_bank[weight_key] = (
                    momentum * historical_weights +
                    (1 - momentum) * current_weights.detach()
                ).to(dtype=dtype)
        else:
            if update_memory:
                # 初始化该文化的权重模式
                memory_bank[weight_key] = current_weights.detach().to(dtype=dtype)

    return consistency_loss.to(dtype=dtype)


def compute_memory_based_contrastive_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    memory_bank: Dict[str, torch.Tensor],
    weight: float = 0.01,
    temperature: float = 0.5
) -> torch.Tensor:
    """
    基于内存银行的对比学习损失
    适用于batch_size=1的情况
    """
    device = expert_weights.device
    dtype = torch.float16

    if weight <= 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    contrastive_loss = torch.tensor(0.0, device=device, dtype=dtype)
    batch_size = expert_weights.shape[0]

    for i in range(batch_size):
        culture_id = culture_labels[i].item()
        current_weights = expert_weights[i]  # [num_experts]

        # 收集正样本（相同文化）和负样本（不同文化）
        positive_similarities = []
        negative_similarities = []

        for key, stored_weights in memory_bank.items():
            if key.startswith("culture_") and key.endswith("_weights"):
                # 提取文化ID
                stored_culture_id = int(key.split("_")[1])

                # 计算相似度
                similarity = F.cosine_similarity(
                    current_weights.unsqueeze(0),
                    stored_weights.unsqueeze(0),
                    dim=-1
                )

                if stored_culture_id == culture_id:
                    positive_similarities.append(similarity / temperature)
                else:
                    negative_similarities.append(similarity / temperature)

        # 计算对比损失（InfoNCE风格）
        if positive_similarities and negative_similarities:
            pos_scores = torch.stack(positive_similarities)
            neg_scores = torch.stack(negative_similarities)

            # 对于每个正样本，计算与所有负样本的对比损失
            for pos_score in pos_scores:
                all_scores = torch.cat([pos_score.unsqueeze(0), neg_scores])
                log_prob = pos_score - torch.logsumexp(all_scores, dim=0)
                contrastive_loss += -log_prob * weight

    return contrastive_loss.to(dtype=dtype)


def compute_pairwise_culture_loss(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    weight: float = 0.01
) -> torch.Tensor:
    """
    原始的成对文化损失（适用于batch_size >= 2）
    """
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
            similarity = similarity.to(dtype=dtype)

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


def compute_expert_regularization_loss(
    expert_weights: torch.Tensor,
    weight: float = 0.01
) -> torch.Tensor:
    """
    专家权重正则化损失（batch_size=1时的回退方案）
    """
    device = expert_weights.device
    dtype = torch.float16

    # 计算专家权重的熵，鼓励专家权重分布均匀
    expert_probs = torch.softmax(expert_weights, dim=-1)  # [B, num_experts]
    entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=-1)  # [B]

    # 熵越大越好，所以损失是负熵
    regularization_loss = -entropy.mean() * weight
    return regularization_loss.to(dtype=dtype)


class CultureMemoryBank:
    """
    文化表示内存银行
    用于在batch_size=1的情况下维护历史文化信息
    """

    def __init__(self, max_size_per_culture: int = 100, device: torch.device = None):
        self.max_size_per_culture = max_size_per_culture
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = {}

    def update(self, culture_id: int, expert_weights: torch.Tensor, hidden_states: Optional[torch.Tensor] = None):
        """更新内存银行"""
        weight_key = f"culture_{culture_id}_weights"

        if weight_key not in self.memory:
            self.memory[weight_key] = expert_weights.detach().to(self.device, dtype=torch.float16)
        else:
            # 指数移动平均
            momentum = 0.9
            self.memory[weight_key] = (
                momentum * self.memory[weight_key] +
                (1 - momentum) * expert_weights.detach()
            ).to(self.device, dtype=torch.float16)

    def get_memory(self) -> Dict[str, torch.Tensor]:
        """获取内存银行"""
        return self.memory

    def clear(self):
        """清空内存银行"""
        self.memory.clear()

    def get_culture_stats(self) -> Dict[str, int]:
        """获取每种文化的统计信息"""
        stats = {}
        for key in self.memory.keys():
            if key.endswith("_weights"):
                culture_id = key.split("_")[1]
                stats[f"culture_{culture_id}"] = 1  # 简化统计
        return stats


# 使用示例和集成函数
def integrate_enhanced_culture_loss(
    model,
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor,
    hidden_states: Optional[torch.Tensor] = None,
    expert_outputs: Optional[Dict[int, torch.Tensor]] = None,
    memory_bank: Optional[CultureMemoryBank] = None,
    loss_config: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """
    集成增强文化损失到现有模型的便捷函数

    Args:
        model: 当前模型实例
        expert_weights: 专家权重
        culture_labels: 文化标签
        hidden_states: 隐藏状态
        expert_outputs: 专家输出
        memory_bank: 内存银行实例
        loss_config: 损失配置字典

    Returns:
        enhanced_culture_loss: 增强的文化损失
    """
    if loss_config is None:
        loss_config = {
            'loss_weight': 0.01,
            'diversity_weight': 0.005,
            'consistency_weight': 0.003
        }

    # 如果没有内存银行，创建一个
    if memory_bank is None:
        memory_bank_dict = {}
    else:
        memory_bank_dict = memory_bank.get_memory()

    enhanced_loss = compute_enhanced_culture_loss(
        expert_weights=expert_weights,
        culture_labels=culture_labels,
        hidden_states=hidden_states,
        expert_outputs=expert_outputs,
        memory_bank=memory_bank_dict,
        **loss_config
    )

    # 更新内存银行
    if memory_bank is not None:
        batch_size = expert_weights.shape[0]
        for i in range(batch_size):
            culture_id = culture_labels[i].item()
            memory_bank.update(culture_id, expert_weights[i])

    return enhanced_loss