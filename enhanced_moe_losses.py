#!/usr/bin/env python3
"""
增强的MoE损失函数模块
实现完整的损失函数：L = L_h + L_balance
其中 L_balance = αL_aux + βL_o + γL_v

包含4个核心部分：
1. L_h: 主任务损失（交叉熵损失）
2. L_aux: 负载均衡辅助损失
3. L_o: 正交化损失（专家输出差异化约束）
4. L_v: 路由方差损失（路由分数差异化约束）

注意：文化损失L_culture仅用于监控，不加入总损失计算
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any


def compute_enhanced_moe_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    expert_outputs: Dict[int, torch.Tensor],  # 被激活专家的输出
    soft_routing_scores: torch.Tensor,  # 所有专家的soft routing scores [B, num_experts]
    expert_weights: torch.Tensor,  # Top-k专家权重 [B, num_experts]
    activated_experts: list,  # 被激活的专家索引列表
    culture_labels: Optional[torch.Tensor] = None,
    use_culture_loss: bool = False,
    loss_weights: Optional[Dict[str, float]] = None
) -> Dict[str, torch.Tensor]:
    """
    计算增强的MoE损失函数

    Args:
        logits: 模型输出logits [B, L, vocab_size]
        labels: 标签 [B, L]
        expert_outputs: 激活专家的输出 {expert_idx: output_tensor [B, L, H]}
        soft_routing_scores: 所有专家的soft routing分数 [B, num_experts]
        expert_weights: Top-k专家权重 [B, num_experts]
        activated_experts: 被激活的专家索引列表
        culture_labels: 文化标签 [B] (可选)
        use_culture_loss: 是否使用文化损失
        loss_weights: 损失权重字典 {"alpha": 0.01, "beta": 0.1, "gamma": 0.05}

    Returns:
        loss_dict: 包含各种损失的字典
    """
    device = logits.device
    dtype = logits.dtype

    # 默认损失权重
    if loss_weights is None:
        loss_weights = {
            "alpha": 1e-3,  # L_aux权重
            "beta": 1e-3,   # L_o权重
            "gamma": 1e-3   # L_v权重
        }

    loss_dict = {}

    try:
        # 1. L_h: 主任务损失（交叉熵）
        L_h = compute_main_task_loss(logits, labels)
        loss_dict["L_h"] = L_h

        # 2. L_aux: 负载均衡辅助损失
        L_aux = compute_load_balance_loss(soft_routing_scores, expert_weights)
        loss_dict["L_aux"] = L_aux

        # 3. L_o: 正交化损失（专家输出差异化）
        L_o = compute_orthogonalization_loss(expert_outputs, activated_experts)
        loss_dict["L_o"] = L_o

        # 4. L_v: 路由方差损失（路由分数差异化）
        L_v = compute_routing_variance_loss(soft_routing_scores)
        loss_dict["L_v"] = L_v

        # 5. 平衡损失 L_balance = αL_aux + βL_o + γL_v
        L_balance = (
            loss_weights["alpha"] * L_aux +
            loss_weights["beta"] * L_o +
            loss_weights["gamma"] * L_v
        )
        loss_dict["L_balance"] = L_balance

        # 6. 总损失 L = L_h + L_balance (移除文化损失权重)
        if use_culture_loss and culture_labels is not None:
            # 计算文化损失但不加入总损失，仅用于监控
            L_culture = compute_culture_loss_simple(expert_weights, culture_labels)
            loss_dict["L_culture"] = L_culture
        else:
            loss_dict["L_culture"] = torch.tensor(0.0, device=device, dtype=dtype)

        # 纯粹的增强损失：L_total = L_h + L_balance
        L_total = L_h + L_balance
        loss_dict["L_total"] = L_total

        return loss_dict

    except Exception as e:
        print(f"⚠️ Enhanced MoE loss computation failed: {e}")
        # 返回安全的fallback损失
        fallback_loss = compute_main_task_loss(logits, labels)
        return {
            "L_h": fallback_loss,
            "L_aux": torch.tensor(0.0, device=device, dtype=dtype),
            "L_o": torch.tensor(0.0, device=device, dtype=dtype),
            "L_v": torch.tensor(0.0, device=device, dtype=dtype),
            "L_balance": torch.tensor(0.0, device=device, dtype=dtype),
            "L_culture": torch.tensor(0.0, device=device, dtype=dtype),
            "L_total": fallback_loss
        }


def compute_main_task_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算主任务损失 L_h（交叉熵损失）
    """
    try:
        # 标准的语言模型损失计算
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        L_h = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return L_h

    except Exception as e:
        print(f"⚠️ Main task loss computation failed: {e}")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)


def compute_load_balance_loss(
    soft_routing_scores: torch.Tensor,
    expert_weights: torch.Tensor
) -> torch.Tensor:
    """
    计算负载均衡辅助损失 L_aux

    目的：让每个专家整体使用量接近均匀分布，防止专家collapse

    Args:
        soft_routing_scores: [B, num_experts] 所有专家的soft routing概率
        expert_weights: [B, num_experts] Top-k专家权重（稀疏）
    """
    try:
        device = soft_routing_scores.device
        dtype = soft_routing_scores.dtype
        num_experts = soft_routing_scores.shape[-1]

        # 方法：importance * prob_of_selection
        # importance: 每个专家的总概率（p_j = ∑ s_ij）
        importance = soft_routing_scores.sum(0)  # [num_experts]

        # prob_of_selection: 每个专家被选中的比例（top-k时）
        # expert_weights > 0 表示该专家被激活
        prob_of_selection = (expert_weights > 0).float().mean(0)  # [num_experts]

        # L_aux = (importance * prob_of_selection).sum() * num_experts
        # 这会鼓励importance和prob_of_selection都接近均匀
        L_aux = (importance * prob_of_selection).sum() * num_experts

        # 数值稳定性检查
        if torch.isnan(L_aux) or torch.isinf(L_aux):
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        return L_aux

    except Exception as e:
        print(f"⚠️ Load balance loss computation failed: {e}")
        return torch.tensor(0.0, device=soft_routing_scores.device, dtype=soft_routing_scores.dtype, requires_grad=True)


def compute_orthogonalization_loss(
    expert_outputs: Dict[int, torch.Tensor],
    activated_experts: list
) -> torch.Tensor:
    """
    计算正交化损失 L_o（专家输出差异化约束）

    目的：让激活的top-k专家输出更加不同，避免专家同质化

    Args:
        expert_outputs: {expert_idx: output_tensor [B, L, H]}
        activated_experts: 被激活的专家索引列表
    """
    try:
        if len(expert_outputs) < 2:
            # 少于2个专家，无法计算正交化损失
            first_output = next(iter(expert_outputs.values()))
            return torch.tensor(0.0, device=first_output.device, dtype=first_output.dtype, requires_grad=True)

        # 收集所有激活专家的输出
        outputs_list = []
        for expert_idx in activated_experts:
            if expert_idx in expert_outputs:
                output = expert_outputs[expert_idx]  # [B, L, H]
                # 展平为 [B, L*H] 便于计算相似度
                flat_output = output.flatten(start_dim=1)  # [B, L*H]
                outputs_list.append(flat_output)

        if len(outputs_list) < 2:
            first_output = outputs_list[0] if outputs_list else next(iter(expert_outputs.values()))
            return torch.tensor(0.0, device=first_output.device, dtype=first_output.dtype, requires_grad=True)

        # 将所有输出堆叠为 [B, k, L*H]，其中k是激活的专家数
        expert_outputs_tensor = torch.stack(outputs_list, dim=1)  # [B, k, L*H]

        # 计算两两内积：[B, k, k]
        # expert_outputs_tensor @ expert_outputs_tensor.transpose(-1, -2)
        similarity_matrix = torch.bmm(
            expert_outputs_tensor,
            expert_outputs_tensor.transpose(-1, -2)
        )  # [B, k, k]

        # 期望对角线大（专家自身信息），非对角线小（专家差异大）
        # 提取对角线元素
        diagonal = torch.diagonal(similarity_matrix, dim1=-2, dim2=-1)  # [B, k]

        # 创建对角矩阵，然后计算非对角线部分
        diagonal_matrix = torch.diag_embed(diagonal)  # [B, k, k]
        off_diagonal = similarity_matrix - diagonal_matrix  # [B, k, k]

        # 正交化损失 = 非对角线项的平方和
        L_o = (off_diagonal ** 2).sum()

        # 归一化（可选）
        batch_size, k = expert_outputs_tensor.shape[:2]
        L_o = L_o / (batch_size * k * (k - 1))  # 除以非对角线元素总数

        # 数值稳定性检查
        if torch.isnan(L_o) or torch.isinf(L_o):
            first_output = next(iter(expert_outputs.values()))
            return torch.tensor(0.0, device=first_output.device, dtype=first_output.dtype, requires_grad=True)

        return L_o

    except Exception as e:
        print(f"⚠️ Orthogonalization loss computation failed: {e}")
        first_output = next(iter(expert_outputs.values()))
        return torch.tensor(0.0, device=first_output.device, dtype=first_output.dtype, requires_grad=True)


def compute_routing_variance_loss(soft_routing_scores: torch.Tensor) -> torch.Tensor:
    """
    计算路由方差损失 L_v（路由分数差异化约束）

    目的：解决"平均主义路由"问题，鼓励路由器做出更果断的决策

    Args:
        soft_routing_scores: [B, num_experts] 所有专家的soft routing概率
    """
    try:
        device = soft_routing_scores.device
        dtype = soft_routing_scores.dtype

        # 对于每个专家，计算其在整个batch中的路由得分方差
        # soft_routing_scores: [num_tokens, num_experts]

        # 计算每个专家的平均得分
        mean_scores = soft_routing_scores.mean(0, keepdim=True)  # [1, num_experts]

        # 计算方差：((s - mean_s)^2).mean()
        variance = ((soft_routing_scores - mean_scores) ** 2).mean()

        # L_v = -variance（负号表示鼓励方差更大）
        # 方差越大，说明路由决策越果断（有些token得高分，有些得低分）
        L_v = -variance

        # 数值稳定性检查
        if torch.isnan(L_v) or torch.isinf(L_v):
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        return L_v

    except Exception as e:
        print(f"⚠️ Routing variance loss computation failed: {e}")
        return torch.tensor(0.0, device=soft_routing_scores.device, dtype=soft_routing_scores.dtype, requires_grad=True)


def compute_culture_loss_simple(
    expert_weights: torch.Tensor,
    culture_labels: torch.Tensor
) -> torch.Tensor:
    """
    简化的文化损失计算（当启用文化损失时使用）
    """
    try:
        if expert_weights is None or culture_labels is None:
            device = culture_labels.device if culture_labels is not None else torch.device('cuda')
            return torch.tensor(0.0, device=device, dtype=torch.float16, requires_grad=True)

        batch_size = expert_weights.shape[0]
        device = expert_weights.device
        dtype = expert_weights.dtype

        if batch_size < 2:
            # batch_size=1时的简单正则化
            expert_probs = torch.softmax(expert_weights, dim=-1)
            entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=-1)
            return -entropy.mean() * 0.01

        # 成对比较的文化损失
        culture_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        count = 0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                vec1 = expert_weights[i].unsqueeze(0)
                vec2 = expert_weights[j].unsqueeze(0)

                norm1 = torch.norm(vec1)
                norm2 = torch.norm(vec2)
                if norm1 < 1e-8 or norm2 < 1e-8:
                    continue

                similarity = F.cosine_similarity(vec1, vec2)
                if torch.isnan(similarity) or torch.isinf(similarity):
                    continue

                if culture_labels[i] == culture_labels[j]:
                    # 相同文化，鼓励相似
                    culture_loss = culture_loss + (1.0 - similarity.mean())
                else:
                    # 不同文化，鼓励不同
                    culture_loss = culture_loss + similarity.mean()
                count += 1

        if count > 0:
            culture_loss = culture_loss / count

        return culture_loss

    except Exception as e:
        print(f"⚠️ Culture loss computation failed: {e}")
        return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype, requires_grad=True)


# 便捷的集成函数
def integrate_enhanced_moe_loss(
    model_outputs: Any,
    labels: torch.Tensor,
    culture_labels: Optional[torch.Tensor] = None,
    use_culture_loss: bool = False,
    loss_weights: Optional[Dict[str, float]] = None
) -> Dict[str, torch.Tensor]:
    """
    集成增强MoE损失的便捷函数

    Args:
        model_outputs: 模型输出对象，包含logits, expert_weights, expert_outputs等
        labels: 标签张量
        culture_labels: 文化标签（可选）
        use_culture_loss: 是否使用文化损失
        loss_weights: 损失权重配置

    Returns:
        loss_dict: 包含各种损失的字典
    """
    try:
        # 从模型输出中提取必要信息
        logits = model_outputs.logits
        expert_weights = getattr(model_outputs, 'expert_weights', None)
        expert_outputs = getattr(model_outputs, 'expert_outputs', {})
        soft_routing_scores = getattr(model_outputs, 'soft_routing_scores', expert_weights)
        activated_experts = getattr(model_outputs, 'activated_experts', list(expert_outputs.keys()))

        # 如果缺少必要信息，回退到简单损失
        if expert_weights is None or len(expert_outputs) == 0:
            main_loss = compute_main_task_loss(logits, labels)
            return {
                "L_h": main_loss,
                "L_aux": torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
                "L_o": torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
                "L_v": torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
                "L_balance": torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
                "L_culture": torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
                "L_total": main_loss
            }

        # 计算增强损失
        return compute_enhanced_moe_loss(
            logits=logits,
            labels=labels,
            expert_outputs=expert_outputs,
            soft_routing_scores=soft_routing_scores,
            expert_weights=expert_weights,
            activated_experts=activated_experts,
            culture_labels=culture_labels,
            use_culture_loss=use_culture_loss,
            loss_weights=loss_weights
        )

    except Exception as e:
        print(f"⚠️ Enhanced MoE loss integration failed: {e}")
        main_loss = compute_main_task_loss(model_outputs.logits, labels)
        return {
            "L_h": main_loss,
            "L_aux": torch.tensor(0.0, device=model_outputs.logits.device, dtype=model_outputs.logits.dtype),
            "L_o": torch.tensor(0.0, device=model_outputs.logits.device, dtype=model_outputs.logits.dtype),
            "L_v": torch.tensor(0.0, device=model_outputs.logits.device, dtype=model_outputs.logits.dtype),
            "L_balance": torch.tensor(0.0, device=model_outputs.logits.device, dtype=model_outputs.logits.dtype),
            "L_culture": torch.tensor(0.0, device=model_outputs.logits.device, dtype=model_outputs.logits.dtype),
            "L_total": main_loss
        }