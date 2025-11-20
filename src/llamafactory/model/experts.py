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

        # 保守的权重初始化，避免数值不稳定
        self._init_weights()

    def _init_weights(self):
        """保守的权重初始化"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，但使用小的gain
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LoRA(nn.Module):
    """
    低秩适应模块，用于线性层。
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int = 8):
        super().__init__()
        self.rank = rank

        # 使用更保守的初始化，避免数值不稳定
        self.W = nn.Parameter(torch.zeros(input_dim, output_dim))  # 原始权重矩阵初始化为0
        self.A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)  # 小的随机初始化
        self.B = nn.Parameter(torch.zeros(rank, output_dim))  # B矩阵初始化为0，这样初始时LoRA贡献为0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.warning("LoRA input contains NaN/Inf")
            return torch.zeros_like(x)

        try:
            # 裁剪参数到合理范围，防止爆炸
            A_clipped = torch.clamp(self.A, min=-10.0, max=10.0)
            B_clipped = torch.clamp(self.B, min=-10.0, max=10.0)
            W_clipped = torch.clamp(self.W, min=-10.0, max=10.0)

            low_rank_weight = torch.matmul(A_clipped, B_clipped)  # 低秩近似

            # 检查低秩权重
            if torch.isnan(low_rank_weight).any() or torch.isinf(low_rank_weight).any():
                logging.warning("Low rank weight contains NaN/Inf, using zeros")
                low_rank_weight = torch.zeros_like(low_rank_weight)

            # 裁剪低秩权重
            low_rank_weight = torch.clamp(low_rank_weight, min=-5.0, max=5.0)

            weight_matrix = W_clipped + low_rank_weight  # 组合原始权重和低秩权重

            # F.linear 期望权重矩阵是 [out_features, in_features]，所以需要转置
            output = F.linear(x, weight_matrix.T)

            # 检查输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.warning("LoRA forward output contains NaN/Inf, using zeros")
                return torch.zeros_like(output)

            # 裁剪输出
            output = torch.clamp(output, min=-20.0, max=20.0)
            return output

        except Exception as e:
            logging.warning(f"LoRA forward failed: {e}, using zeros")
            return torch.zeros(x.shape[0], x.shape[1], self.B.shape[1], device=x.device, dtype=x.dtype)


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

        # ✅ 初始化专家网络权重
        self._init_weights()

    def _init_weights(self):
        """
        ✅ 初始化专家网络权重 - 使用小的初始化防止梯度爆炸
        """
        # MLP 基础模型已经在 MLPBase 中初始化了，不需要重复初始化
        # LoRA 层也已经在 LoRA 类中保守初始化了，不需要重复初始化

        # 只初始化 LayerNorm（如果需要的话）
        if hasattr(self.layer_norm, 'weight'):
            nn.init.ones_(self.layer_norm.weight)
        if hasattr(self.layer_norm, 'bias'):
            nn.init.zeros_(self.layer_norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ 使用 LayerNorm + FFN + LoRA 的结构
        # y = LayerNorm(x)
        # y = FFN(y)
        # return x + LoRA(y)

        # 检查输入数值稳定性
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.warning("LoRAExpert input contains NaN/Inf, using zeros")
            return torch.zeros_like(x)

        try:
            # 先对输入进行 LayerNorm
            normalized = self.layer_norm(x)

            # 检查LayerNorm输出
            if torch.isnan(normalized).any() or torch.isinf(normalized).any():
                logging.warning("LayerNorm output contains NaN/Inf, using input")
                normalized = x

            # 通过 MLP 基础模型
            features = self.base_model(normalized)

            # 检查MLP输出
            if torch.isnan(features).any() or torch.isinf(features).any():
                logging.warning("MLP features contain NaN/Inf, using zeros")
                features = torch.zeros_like(features)

            # 通过 LoRA 层
            lora_out = self.lora_layer(features)

            # 检查LoRA输出
            if torch.isnan(lora_out).any() or torch.isinf(lora_out).any():
                logging.warning("LoRA output contains NaN/Inf, using zeros")
                lora_out = torch.zeros_like(lora_out)

            # 裁剪到合理范围
            lora_out = torch.clamp(lora_out, min=-10.0, max=10.0)

            # ✅ 残差连接：x + LoRA(LayerNorm(FFN(x)))
            output = x + lora_out

            # 检查最终输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.warning("LoRAExpert final output contains NaN/Inf, using input")
                output = x

            return output

        except Exception as e:
            logging.warning(f"LoRAExpert forward failed: {e}, using input")
            return x


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
