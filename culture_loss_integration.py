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
        return torch.tensor(0.0, device=device, dtype=torch.float16, requires_grad=True)

    batch_size = expert_weights.shape[0]
    device = expert_weights.device
    dtype = torch.float16

    # 检查维度匹配
    if expert_weights.shape[0] != culture_labels.shape[0]:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    # 检查输入的有效性
    if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    try:
        total_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # 1. 专家激活多样性损失（大幅降低权重）
        if expert_outputs is not None:  # 只有明确提供expert_outputs时才计算
            diversity_loss = compute_diversity_loss(expert_weights, expert_outputs, loss_weight * 0.1)
            if not (torch.isnan(diversity_loss) or torch.isinf(diversity_loss)):
                total_loss = total_loss + diversity_loss

        # 2. 文化一致性损失（降低权重）
        if memory_bank is not None:
            consistency_loss = compute_consistency_loss(
                expert_weights, culture_labels, memory_bank, loss_weight * 0.2
            )
            if not (torch.isnan(consistency_loss) or torch.isinf(consistency_loss)):
                total_loss = total_loss + consistency_loss

        # 3. 基础文化对比损失
        if batch_size >= 2:
            # 标准成对比较
            contrastive_loss = compute_pairwise_loss(expert_weights, culture_labels, loss_weight)
            if not (torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss)):
                total_loss = total_loss + contrastive_loss
        else:
            # batch_size=1时的处理
            if memory_bank is not None:
                # 基于内存银行的对比学习
                memory_contrastive_loss = compute_memory_contrastive_loss(
                    expert_weights, culture_labels, memory_bank, loss_weight
                )
                if not (torch.isnan(memory_contrastive_loss) or torch.isinf(memory_contrastive_loss)):
                    total_loss = total_loss + memory_contrastive_loss
            else:
                # 回退到专家正则化
                regularization_loss = compute_regularization_loss(expert_weights, loss_weight)
                if not (torch.isnan(regularization_loss) or torch.isinf(regularization_loss)):
                    total_loss = total_loss + regularization_loss

        # 最终检查和限制
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # 限制损失的范围，防止过大
        total_loss = torch.clamp(total_loss, min=-10.0, max=10.0)

        return total_loss.to(device=device, dtype=dtype)

    except Exception as e:
        print(f"⚠️ Enhanced culture loss computation failed: {e}")
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)


def compute_diversity_loss(
    expert_weights: torch.Tensor,
    expert_outputs: Optional[Dict[int, torch.Tensor]] = None,
    weight: float = 0.005
) -> torch.Tensor:
    """专家激活多样性损失"""
    device = expert_weights.device
    dtype = torch.float16

    if weight <= 0:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    try:
        # 计算专家权重分布的均匀性
        expert_probs = torch.softmax(expert_weights, dim=-1)  # [B, num_experts]
        avg_activation = expert_probs.mean(dim=0)  # [num_experts]

        # 数值稳定性检查
        if torch.isnan(avg_activation).any() or torch.isinf(avg_activation).any():
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # 鼓励专家激活的均匀分布
        num_experts = expert_probs.shape[-1]
        uniform_target = torch.ones_like(avg_activation) / num_experts

        # 🔧 修复KL散度计算 - 确保返回标量张量
        log_avg = torch.log(avg_activation + 1e-8)
        kl_loss = F.kl_div(log_avg, uniform_target, reduction='batchmean')

        # 确保kl_loss是标量张量且有梯度
        if kl_loss.dim() > 0:
            kl_loss = kl_loss.mean()

        # 检查数值稳定性
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        diversity_loss = kl_loss * weight

        # 如果有专家输出，增加输出多样性约束
        if expert_outputs is not None and len(expert_outputs) > 1:
            output_diversity = compute_output_diversity_simple(expert_outputs)
            if not (torch.isnan(output_diversity) or torch.isinf(output_diversity)):
                diversity_loss = diversity_loss - output_diversity * weight * 0.5

        # 确保返回标量张量
        if diversity_loss.dim() > 0:
            diversity_loss = diversity_loss.mean()

        return diversity_loss.to(device=device, dtype=dtype)

    except Exception as e:
        print(f"⚠️ Diversity loss computation failed: {e}")
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)


def compute_output_diversity_simple(expert_outputs: Dict[int, torch.Tensor]) -> torch.Tensor:
    """计算专家输出多样性（简化版）"""
    if len(expert_outputs) < 2:
        return torch.tensor(0.0, dtype=torch.float16)

    try:
        expert_list = list(expert_outputs.keys())

        # 只计算前两个专家的相似度（减少计算量）
        if len(expert_list) >= 2:
            output1 = expert_outputs[expert_list[0]]  # [B, L, H]
            output2 = expert_outputs[expert_list[1]]  # [B, L, H]

            # 确保形状一致
            if output1.shape != output2.shape:
                return torch.tensor(0.0, dtype=torch.float16)

            # 计算输出的余弦相似度
            flat1 = output1.flatten(start_dim=1)  # [B, L*H]
            flat2 = output2.flatten(start_dim=1)  # [B, L*H]

            # 避免零向量
            norm1 = torch.norm(flat1, dim=-1, keepdim=True)
            norm2 = torch.norm(flat2, dim=-1, keepdim=True)

            if (norm1 > 1e-8).all() and (norm2 > 1e-8).all():
                similarity = F.cosine_similarity(flat1, flat2, dim=-1)

                # 确保返回标量
                if similarity.dim() > 0:
                    similarity = similarity.mean()

                # 检查数值稳定性
                if torch.isnan(similarity) or torch.isinf(similarity):
                    return torch.tensor(0.0, dtype=torch.float16)

                return similarity.to(dtype=torch.float16)

        return torch.tensor(0.0, dtype=torch.float16)

    except Exception as e:
        print(f"⚠️ Output diversity computation failed: {e}")
        return torch.tensor(0.0, dtype=torch.float16)


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
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    try:
        consistency_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        batch_size = expert_weights.shape[0]

        for i in range(batch_size):
            culture_id = culture_labels[i].item()
            current_weights = expert_weights[i]  # [num_experts]

            # 检查当前权重的有效性
            if torch.isnan(current_weights).any() or torch.isinf(current_weights).any():
                continue

            weight_key = f"culture_{culture_id}_weights"

            if weight_key in memory_bank:
                # 计算与历史模式的一致性
                historical_weights = memory_bank[weight_key]

                # 确保历史权重在正确设备上
                historical_weights = historical_weights.to(device=device, dtype=dtype)

                # 检查历史权重的有效性
                if torch.isnan(historical_weights).any() or torch.isinf(historical_weights).any():
                    # 重新初始化
                    memory_bank[weight_key] = current_weights.detach().to(dtype=dtype)
                    continue

                # 计算余弦相似度
                consistency = F.cosine_similarity(
                    current_weights.unsqueeze(0),
                    historical_weights.unsqueeze(0),
                    dim=-1
                )

                # 确保consistency是标量
                if consistency.dim() > 0:
                    consistency = consistency.mean()

                # 检查数值稳定性
                if torch.isnan(consistency) or torch.isinf(consistency):
                    continue

                # 计算一致性损失
                loss_increment = (1.0 - consistency) * weight
                if not (torch.isnan(loss_increment) or torch.isinf(loss_increment)):
                    consistency_loss = consistency_loss + loss_increment

                # 更新内存银行（指数移动平均）
                momentum = 0.9
                updated_weights = (
                    momentum * historical_weights +
                    (1 - momentum) * current_weights.detach()
                ).to(dtype=dtype)

                # 检查更新后权重的有效性
                if not (torch.isnan(updated_weights).any() or torch.isinf(updated_weights).any()):
                    memory_bank[weight_key] = updated_weights
            else:
                # 初始化
                memory_bank[weight_key] = current_weights.detach().to(dtype=dtype)

        return consistency_loss

    except Exception as e:
        print(f"⚠️ Consistency loss computation failed: {e}")
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)


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
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    try:
        contrastive_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        batch_size = expert_weights.shape[0]

        for i in range(batch_size):
            culture_id = culture_labels[i].item()
            current_weights = expert_weights[i]

            # 检查当前权重的有效性
            if torch.isnan(current_weights).any() or torch.isinf(current_weights).any():
                continue

            positive_similarities = []
            negative_similarities = []

            # 收集正负样本
            for key, stored_weights in memory_bank.items():
                if key.startswith("culture_") and key.endswith("_weights"):
                    try:
                        stored_culture_id = int(key.split("_")[1])

                        # 确保存储权重在正确设备上
                        stored_weights = stored_weights.to(device=device, dtype=dtype)

                        # 检查存储权重的有效性
                        if torch.isnan(stored_weights).any() or torch.isinf(stored_weights).any():
                            continue

                        similarity = F.cosine_similarity(
                            current_weights.unsqueeze(0),
                            stored_weights.unsqueeze(0),
                            dim=-1
                        )

                        # 确保similarity是标量
                        if similarity.dim() > 0:
                            similarity = similarity.mean()

                        # 检查数值稳定性
                        if torch.isnan(similarity) or torch.isinf(similarity):
                            continue

                        similarity_scaled = similarity / temperature

                        if stored_culture_id == culture_id:
                            positive_similarities.append(similarity_scaled)
                        else:
                            negative_similarities.append(similarity_scaled)

                    except (ValueError, IndexError):
                        # 跳过格式不正确的键
                        continue

            # 计算对比损失
            if positive_similarities and negative_similarities:
                try:
                    pos_scores = torch.stack(positive_similarities)
                    neg_scores = torch.stack(negative_similarities)

                    for pos_score in pos_scores:
                        all_scores = torch.cat([pos_score.unsqueeze(0), neg_scores])

                        # 检查scores的有效性
                        if torch.isnan(all_scores).any() or torch.isinf(all_scores).any():
                            continue

                        log_prob = pos_score - torch.logsumexp(all_scores, dim=0)

                        # 检查log_prob的有效性
                        if torch.isnan(log_prob) or torch.isinf(log_prob):
                            continue

                        loss_increment = -log_prob * weight
                        if not (torch.isnan(loss_increment) or torch.isinf(loss_increment)):
                            contrastive_loss = contrastive_loss + loss_increment

                except Exception as e:
                    print(f"⚠️ Contrastive loss stack failed: {e}")
                    continue

        return contrastive_loss

    except Exception as e:
        print(f"⚠️ Memory contrastive loss computation failed: {e}")
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)


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

    try:
        # 检查输入的有效性
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # 计算专家权重的熵
        expert_probs = torch.softmax(expert_weights, dim=-1)

        # 检查softmax结果的有效性
        if torch.isnan(expert_probs).any() or torch.isinf(expert_probs).any():
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=-1)

        # 检查熵计算结果的有效性
        if torch.isnan(entropy).any() or torch.isinf(entropy).any():
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # 确保entropy是标量
        if entropy.dim() > 0:
            entropy = entropy.mean()

        # 熵越大越好，损失是负熵
        regularization_loss = -entropy * weight

        # 最终检查
        if torch.isnan(regularization_loss) or torch.isinf(regularization_loss):
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        return regularization_loss.to(device=device, dtype=dtype)

    except Exception as e:
        print(f"⚠️ Regularization loss computation failed: {e}")
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)