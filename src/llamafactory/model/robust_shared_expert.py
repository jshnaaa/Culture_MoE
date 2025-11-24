# src/llamafactory/model/robust_shared_expert.py
"""
稳健共享专家模块
通过蒸馏LoRA模型知识，增强OOD泛化能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging


class RobustSharedExpert(nn.Module):
    """
    稳健共享专家：通过知识蒸馏增强OOD性能

    核心功能：
    1. 从LoRA模型蒸馏跨文化通用知识
    2. 学习文化不变特征
    3. OOD时提供稳定的推理能力
    """

    def __init__(self, hidden_dim: int = 4096, expert_dim: int = 2048,
                 lora_rank: int = 8, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim

        # 主要专家网络（从原shared expert复制结构）
        self.expert_network = nn.Sequential(
            nn.Linear(hidden_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, hidden_dim)
        )

        # 文化不变特征提取器
        self.invariant_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

        # 知识蒸馏投影层
        self.distill_projector = nn.Linear(hidden_dim, hidden_dim)

        # 融合权重（可学习）
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in [self.expert_network, self.invariant_extractor, self.distill_projector]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: torch.Tensor,
                return_distill_logits: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, L, H] 输入隐藏状态
            return_distill_logits: 是否返回蒸馏logits

        Returns:
            输出字典，包含专家输出和蒸馏信息
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. 主要专家推理
        expert_output = self.expert_network(hidden_states)  # [B, L, H]

        # 2. 文化不变特征提取
        invariant_features = self.invariant_extractor(hidden_states)  # [B, L, H]

        # 3. 融合专家输出和不变特征
        fusion_weight = torch.sigmoid(self.fusion_weight)
        fused_output = fusion_weight * expert_output + (1 - fusion_weight) * invariant_features

        outputs = {
            'expert_output': fused_output,
            'raw_expert_output': expert_output,
            'invariant_features': invariant_features,
            'fusion_weight': fusion_weight
        }

        # 4. 如果需要，计算蒸馏logits
        if return_distill_logits:
            distill_logits = self.distill_projector(fused_output)
            outputs['distill_logits'] = distill_logits

        return outputs


class KnowledgeDistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                hard_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算知识蒸馏损失

        Args:
            student_logits: 学生模型(shared expert)的输出
            teacher_logits: 教师模型(LoRA)的输出
            hard_targets: 硬标签(可选)

        Returns:
            损失字典
        """
        # 软蒸馏损失
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)

        losses = {
            'distill_soft_loss': soft_loss,
            'total_distill_loss': self.alpha * soft_loss
        }

        # 硬标签损失（如果提供）
        if hard_targets is not None:
            hard_loss = F.cross_entropy(student_logits, hard_targets)
            losses['distill_hard_loss'] = hard_loss
            losses['total_distill_loss'] = (
                self.alpha * soft_loss + (1 - self.alpha) * hard_loss
            )

        return losses


class CulturalInvarianceLoss(nn.Module):
    """
    文化不变性损失：鼓励shared expert学习跨文化通用特征
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, invariant_features: torch.Tensor,
                culture_ids: torch.Tensor) -> torch.Tensor:
        """
        计算文化不变性损失

        Args:
            invariant_features: [B, L, H] 不变特征
            culture_ids: [B] 文化标签

        Returns:
            不变性损失
        """
        # 对序列维度取平均，得到样本级特征
        sample_features = invariant_features.mean(dim=1)  # [B, H]

        # 计算同文化内的相似性（应该高）和跨文化的相似性（应该适中）
        batch_size = sample_features.size(0)

        intra_culture_sim = 0.0
        inter_culture_sim = 0.0
        intra_count = 0
        inter_count = 0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sim = F.cosine_similarity(
                    sample_features[i:i+1], sample_features[j:j+1], dim=1
                )

                if culture_ids[i] == culture_ids[j]:
                    # 同文化：鼓励相似
                    intra_culture_sim += sim
                    intra_count += 1
                else:
                    # 跨文化：保持适度相似（不要太远，但也不要太近）
                    inter_culture_sim += sim
                    inter_count += 1

        # 计算损失：同文化内应该相似，跨文化应该保持合理距离
        intra_loss = 0.0
        inter_loss = 0.0

        if intra_count > 0:
            avg_intra_sim = intra_culture_sim / intra_count
            intra_loss = F.relu(self.margin - avg_intra_sim)  # 鼓励同文化相似

        if inter_count > 0:
            avg_inter_sim = inter_culture_sim / inter_count
            # 跨文化相似度应该在合理范围内(0.3-0.7)
            inter_loss = F.relu(torch.abs(avg_inter_sim - 0.5) - 0.2)

        total_loss = intra_loss + 0.5 * inter_loss

        return total_loss


def create_robust_shared_expert(hidden_dim: int = 4096, expert_dim: int = 2048,
                               lora_rank: int = 8, dropout: float = 0.1) -> RobustSharedExpert:
    """
    创建稳健共享专家的工厂函数

    Args:
        hidden_dim: 隐藏层维度
        expert_dim: 专家网络维度
        lora_rank: LoRA rank（用于兼容性）
        dropout: Dropout率

    Returns:
        RobustSharedExpert实例
    """
    return RobustSharedExpert(
        hidden_dim=hidden_dim,
        expert_dim=expert_dim,
        lora_rank=lora_rank,
        dropout=dropout
    )