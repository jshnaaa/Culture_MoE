# src/llamafactory/train/classification/culture_loss.py
"""
文化专注性损失（Culture Specialization Loss）

目标：最小化专家间在不同文化维度上的信息交叉，
     迫使每个专家学习独立的特征表示。
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CultureSpecializationLoss(nn.Module):
    """
    文化专注性损失

    通过最小化专家间的互信息，使每个专家专注于特定的文化维度
    """

    def __init__(self, num_cultures: int = 6, num_experts: int = 6, lambda_weight: float = 0.1):
        """
        初始化

        Args:
            num_cultures: 文化维度数量（默认 6）
            num_experts: 专家数量（默认 6）
            lambda_weight: 损失权重 λ（默认 0.1）
        """
        super().__init__()
        self.num_cultures = num_cultures
        self.num_experts = num_experts
        self.lambda_weight = lambda_weight

        # 用于累积专家权重的矩阵 [c, n]
        # 使用 buffer 而不是 parameter，因为这不需要梯度
        self.register_buffer('expert_weight_matrix', torch.zeros(num_cultures, num_experts))
        self.register_buffer('culture_counts', torch.zeros(num_cultures))  # 记录每个文化维度的样本数

    def update_expert_weights(self, expert_weights: torch.Tensor, culture_labels: List[List[int]]):
        """
        更新专家权重矩阵

        Args:
            expert_weights: 路由器输出的专家权重 [B, E]
            culture_labels: 文化维度标签列表，例如 [[0], [1, 2], [3]]
        """
        batch_size = expert_weights.size(0)

        for i in range(batch_size):
            # 获取当前样本的文化维度标签
            labels = culture_labels[i]

            if len(labels) == 0:
                continue

            # 获取当前样本的专家权重 [E]
            sample_weights = expert_weights[i]

            # 对于每个文化维度，累加专家权重
            for culture_id in labels:
                if 0 <= culture_id < self.num_cultures:
                    self.expert_weight_matrix[culture_id] += sample_weights
                    self.culture_counts[culture_id] += 1

    def get_normalized_weights(self) -> torch.Tensor:
        """
        获取归一化的专家权重矩阵

        Returns:
            normalized_weights: [C, E]，每行是该文化维度上的平均专家权重
        """
        # 避免除以 0
        counts = self.culture_counts.clamp(min=1.0)

        # 归一化：对于每个文化维度，计算平均权重
        normalized_weights = self.expert_weight_matrix / counts.unsqueeze(1)

        return normalized_weights

    def compute_mutual_information(self, weights_i: torch.Tensor, weights_j: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        计算两个专家权重分布之间的互信息

        I(w_i, w_j) = Σ_w_i Σ_w_j P(w_i, w_j) log(P(w_i, w_j) / (P(w_i) * P(w_j)))

        简化计算：使用 KL 散度的对称版本（JS 散度）作为互信息的近似

        Args:
            weights_i: 专家 i 的权重分布 [C]
            weights_j: 专家 j 的权重分布 [C]
            eps: 数值稳定性参数

        Returns:
            mutual_info: 互信息值（标量）
        """
        # 归一化为概率分布
        p_i = F.softmax(weights_i, dim=0) + eps
        p_j = F.softmax(weights_j, dim=0) + eps

        # 计算联合分布（简化：假设独立，使用外积）
        # 实际上这是一个近似，真实的联合分布需要更多信息
        p_ij = torch.outer(p_i, p_j)  # [C, C]
        p_ij = p_ij / p_ij.sum()  # 归一化

        # 边缘分布
        p_i_marginal = p_ij.sum(dim=1)  # [C]
        p_j_marginal = p_ij.sum(dim=0)  # [C]

        # 计算互信息
        # I(X;Y) = Σ_x Σ_y P(x,y) log(P(x,y) / (P(x)P(y)))
        mutual_info = 0.0
        for k1 in range(self.num_cultures):
            for k2 in range(self.num_cultures):
                if p_ij[k1, k2] > eps:
                    mutual_info += p_ij[k1, k2] * torch.log(
                        p_ij[k1, k2] / (p_i_marginal[k1] * p_j_marginal[k2] + eps)
                    )

        return mutual_info

    def compute_mutual_information_simplified(self, weights_i: torch.Tensor, weights_j: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        简化版互信息计算：使用余弦相似度作为互信息的代理

        目标：最小化专家间的相似度，使它们专注于不同的文化维度

        Args:
            weights_i: 专家 i 的权重分布 [C]
            weights_j: 专家 j 的权重分布 [C]
            eps: 数值稳定性参数

        Returns:
            similarity: 相似度值（标量）
        """
        # 归一化
        weights_i_norm = F.normalize(weights_i.unsqueeze(0), p=2, dim=1)  # [1, C]
        weights_j_norm = F.normalize(weights_j.unsqueeze(0), p=2, dim=1)  # [1, C]

        # 余弦相似度
        similarity = torch.mm(weights_i_norm, weights_j_norm.t()).squeeze()  # 标量

        # 转换为正值（相似度越高，"互信息"越高）
        similarity = (similarity + 1.0) / 2.0  # 映射到 [0, 1]

        return similarity

    def forward(self, expert_weights: torch.Tensor, culture_labels: List[List[int]], use_simplified: bool = True) -> torch.Tensor:
        """
        计算文化专注性损失

        Args:
            expert_weights: 路由器输出的专家权重 [B, E]
            culture_labels: 文化维度标签列表
            use_simplified: 是否使用简化版互信息计算

        Returns:
            loss: 文化专注性损失
        """
        # 1. 更新专家权重矩阵
        self.update_expert_weights(expert_weights, culture_labels)

        # 2. 获取归一化的权重矩阵 [C, E]
        normalized_weights = self.get_normalized_weights()

        # 3. 计算所有专家对之间的互信息
        total_mutual_info = 0.0
        num_pairs = 0

        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):  # i ≠ j，且避免重复计算
                # 获取专家 i 和 j 在所有文化维度上的权重 [C]
                weights_i = normalized_weights[:, i]
                weights_j = normalized_weights[:, j]

                # 计算互信息
                if use_simplified:
                    mutual_info = self.compute_mutual_information_simplified(weights_i, weights_j)
                else:
                    mutual_info = self.compute_mutual_information(weights_i, weights_j)

                total_mutual_info += mutual_info
                num_pairs += 1

        # 4. 平均互信息
        if num_pairs > 0:
            avg_mutual_info = total_mutual_info / num_pairs
        else:
            avg_mutual_info = torch.tensor(0.0, device=expert_weights.device)

        # 5. 文化专注性损失
        L_s = avg_mutual_info

        return L_s

    def reset_statistics(self):
        """重置统计信息（例如在每个 epoch 开始时）"""
        self.expert_weight_matrix.zero_()
        self.culture_counts.zero_()


def compute_culture_aware_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    expert_weights: torch.Tensor,
    culture_labels: List[List[int]],
    culture_loss_module: CultureSpecializationLoss,
    lambda_weight: float = 0.1
) -> tuple:
    """
    计算文化感知损失

    Loss = L_CE + λ * L_s

    Args:
        logits: 模型输出 [B, num_classes]
        labels: 分类标签 [B]
        expert_weights: 路由器输出的专家权重 [B, E]
        culture_labels: 文化维度标签列表
        culture_loss_module: 文化专注性损失模块
        lambda_weight: 损失权重 λ

    Returns:
        total_loss: 总损失
        ce_loss: 交叉熵损失
        culture_loss: 文化专注性损失
    """
    # 1. 交叉熵损失
    ce_loss = F.cross_entropy(logits, labels)

    # 2. 文化专注性损失
    culture_loss = culture_loss_module(expert_weights, culture_labels)

    # 3. 总损失
    total_loss = ce_loss + lambda_weight * culture_loss

    return total_loss, ce_loss, culture_loss


# 使用示例
if __name__ == "__main__":
    # 测试
    batch_size = 4
    num_experts = 6
    num_cultures = 6

    # 模拟数据
    expert_weights = torch.randn(batch_size, num_experts)
    expert_weights = F.softmax(expert_weights, dim=-1)

    culture_labels = [[0], [1, 2], [3], [4, 5]]

    # 创建损失模块
    culture_loss_module = CultureSpecializationLoss(
        num_cultures=num_cultures,
        num_experts=num_experts,
        lambda_weight=0.1
    )

    # 计算损失
    loss = culture_loss_module(expert_weights, culture_labels)

    print(f"Culture Specialization Loss: {loss.item():.4f}")
    print(f"Expert Weight Matrix:\n{culture_loss_module.get_normalized_weights()}")

