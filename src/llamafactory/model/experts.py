import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class MLPBase(nn.Module):
    """
    简单的MLP模型，作为专家网络的基础部分。
    """

    def __init__(self, experts_input_dim: int, experts_hidden_dim: int, experts_output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(experts_input_dim, experts_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(experts_hidden_dim, experts_output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LoRA(nn.Module):
    """
    低秩适应模块，用于线性层。
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int = 8):
        super().__init__()
        self.rank = rank
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))  # 原始权重矩阵
        self.A = nn.Parameter(torch.randn(input_dim, rank))  # 低秩矩阵A
        self.B = nn.Parameter(torch.randn(rank, output_dim))  # 低秩矩阵B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_rank_weight = torch.matmul(self.A, self.B)  # 低秩近似
        weight_matrix = self.W + low_rank_weight  # 组合原始权重和低秩权重
        return F.linear(x, weight_matrix)


class LoRAExpert(nn.Module):
    """
    结合MLP和LoRA的专家网络。
    ✅ 添加 LayerNorm 以稳定训练
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lora_rank: int = 8, dropout: float = 0.1):
        super().__init__()
        self.base_model = MLPBase(input_dim, hidden_dim, output_dim, dropout)
        self.lora_layer = LoRA(output_dim, output_dim, lora_rank)

        # ✅ 添加 LayerNorm 以稳定专家行为
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ 使用 LayerNorm + FFN + LoRA 的结构
        # y = LayerNorm(x)
        # y = FFN(y)
        # return x + LoRA(y)

        # 先对输入进行 LayerNorm
        normalized = self.layer_norm(x)

        # 通过 MLP 基础模型
        features = self.base_model(normalized)

        # 通过 LoRA 层
        lora_out = self.lora_layer(features)

        # ✅ 残差连接：x + LoRA(LayerNorm(FFN(x)))
        return x + lora_out


class ExpertLayer(nn.Module):
    """
    包含多个LoRA专家的层
    - 默认：输出由专家权重加权求和
    - return_all=True 时：返回所有专家独立输出（不做加权和）
    如果要加权和模式：weighted_output, expert_outputs = experts_layer(features, expert_weights)
    如果你要新的模式（所有专家独立输出）：expert_outputs = experts_layer(features)   # return_all=True 时
    """

    def __init__(self, experts_input_dim: int, experts_hidden_dim: int, experts_output_dim: int,
                 num_experts: int = 6, lora_rank: int = 8, dropout: float = 0.1,
                 return_all: bool = False):
        super().__init__()
        self.experts = nn.ModuleList([
            LoRAExpert(experts_input_dim, experts_hidden_dim, experts_output_dim, lora_rank, dropout)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.return_all = return_all

    def forward(self, features: torch.Tensor, expert_weights: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            features: [B, seq_len, hidden_dim]
            expert_weights: [B, num_experts]  (仅在 return_all=False 时需要)
        Returns:
            if return_all=False:
                weighted_output: [B, seq_len, hidden_dim]
                expert_outputs: List of [B, seq_len, hidden_dim]
            if return_all=True:
                expert_outputs: List of [B, seq_len, hidden_dim]
        """
        # 所有专家独立前向
        expert_outputs = [expert(features) for expert in self.experts]  # list of [B, S, H]

        if self.return_all:
            # 只返回专家独立输出
            return expert_outputs

        else:
            if expert_weights is None:
                raise ValueError("expert_weights must be provided when return_all=False")

            stacked_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, S, H]

            # 广播权重 [B, num_experts, 1, 1]
            weighted_output = torch.sum(
                stacked_outputs * expert_weights[:, :, None, None], dim=1
            )  # [B, S, H]

            return weighted_output, expert_outputs

    def compute_expert_diversity(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算专家间的多样性（输出差异程度）
        """
        # 获取所有专家的输出
        expert_outputs = [expert(features) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=1)  # [B, E, S, H]

        B, E, S, H = stacked_outputs.size()
        flat_outputs = stacked_outputs.view(B * S, E, H)  # [B*S, E, H]

        # 归一化
        normalized_outputs = F.normalize(flat_outputs, p=2, dim=-1)

        # 相似度矩阵 [B*S, E, E]
        similarity_matrix = torch.matmul(normalized_outputs, normalized_outputs.transpose(-2, -1))

        # 去掉对角线
        mask = torch.eye(self.num_experts, device=similarity_matrix.device).bool()
        off_diagonal = similarity_matrix.masked_fill(mask.unsqueeze(0), 0)

        # 计算平均相似度
        avg_similarity = off_diagonal.sum() / (self.num_experts * (self.num_experts - 1) * (B * S))
        diversity_score = 1 - avg_similarity

        return diversity_score



# class ExpertLayer(nn.Module):
#     """
#     包含多个LoRA专家的层，输出由专家权重加权求和得到。
#     """

#     def __init__(self, experts_input_dim: int, experts_hidden_dim: int, experts_output_dim: int, num_experts: int = 6, lora_rank: int = 8, dropout: float = 0.1):
#         super().__init__()
#         self.experts = nn.ModuleList([LoRAExpert(experts_input_dim, experts_hidden_dim, experts_output_dim, lora_rank, dropout) for _ in range(num_experts)])
#         self.num_experts = num_experts

#     def forward(self, features: torch.Tensor, expert_weights: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
#         expert_outputs = [expert(features) for expert in self.experts]
#         stacked_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
#         weighted_output = torch.sum(stacked_outputs * expert_weights.unsqueeze(-1), dim=1)  # 加权求和
#         return weighted_output, expert_outputs
    
#     def compute_expert_diversity(self, features: torch.Tensor) -> torch.Tensor:
#         """
#         计算专家间的多样性（输出差异程度）

#         Args:
#             features: 输入特征 [batch_size, input_dim]

#         Returns:
#             diversity_score: 专家多样性分数
#         """
#         # 获取所有专家的输出
#         expert_outputs = []
#         for expert in self.experts:
#             output = expert(features)
#             expert_outputs.append(output)

#         # 堆叠输出 [batch_size, num_experts, output_dim]
#         stacked_outputs = torch.stack(expert_outputs, dim=1)

#         # 计算专家间的余弦相似度矩阵
#         normalized_outputs = F.normalize(stacked_outputs, p=2, dim=-1)

#         # 计算所有专家对之间的相似度
#         similarity_matrix = torch.matmul(normalized_outputs, normalized_outputs.transpose(-2, -1))

#         # 计算平均相似度（除对角线）
#         mask = torch.eye(self.num_experts, device=similarity_matrix.device).bool()
#         off_diagonal = similarity_matrix.masked_fill(mask.unsqueeze(0), 0)

#         # 多样性 = 1 - 平均相似度
#         avg_similarity = off_diagonal.sum() / (self.num_experts * (self.num_experts - 1) * features.size(0))
#         diversity_score = 1 - avg_similarity

#         return diversity_score
