import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class MLPBase(nn.Module):
    """
    简单的MLP模型，作为专家网络的基础部分。
    修复了数值稳定性问题。
    """

    def __init__(self, experts_input_dim: int, experts_hidden_dim: int, experts_output_dim: int, dropout: float = 0.1):
        super().__init__()

        # 添加输入层归一化
        self.input_norm = nn.LayerNorm(experts_input_dim)

        self.network = nn.Sequential(
            nn.Linear(experts_input_dim, experts_hidden_dim),
            nn.GELU(),  # 使用GELU替代ReLU，更稳定
            nn.Dropout(dropout),
            nn.Linear(experts_hidden_dim, experts_output_dim)
        )

        # 添加输出层归一化
        self.output_norm = nn.LayerNorm(experts_output_dim)

        self._init_weights()

    def _init_weights(self):
        """使用Xavier初始化确保数值稳定性"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                # 使用Xavier均匀初始化
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # 较小的gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入归一化
        x_norm = self.input_norm(x)

        # 通过网络
        output = self.network(x_norm)

        # 输出归一化
        output_norm = self.output_norm(output)

        # 梯度裁剪防止爆炸
        output_norm = torch.clamp(output_norm, -10.0, 10.0)

        return output_norm


class LoRA(nn.Module):
    """
    低秩适应模块，修复了数值稳定性问题。
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 移除原始权重矩阵W，只保留LoRA适配器
        self.A = nn.Parameter(torch.zeros(input_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, output_dim))

        self._init_weights()

    def _init_weights(self):
        """使用标准LoRA初始化"""
        # A矩阵使用Kaiming均匀初始化
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B矩阵初始化为零
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算LoRA输出: x @ A @ B * scaling
        lora_out = x @ self.A @ self.B * self.scaling

        # 梯度裁剪
        lora_out = torch.clamp(lora_out, -5.0, 5.0)

        return lora_out


class LoRAExpert(nn.Module):
    """
    结合MLP和LoRA的专家网络。
    修复了数值稳定性问题。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lora_rank: int = 8, dropout: float = 0.1):
        super().__init__()

        # 确保输入和输出维度一致（用于残差连接）
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
            nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        else:
            self.projection = None

        # MLP基础模型
        self.base_model = MLPBase(input_dim, hidden_dim, output_dim, dropout)

        # LoRA适配器
        self.lora_layer = LoRA(output_dim, output_dim, lora_rank)

        # 最终层归一化
        self.final_norm = nn.LayerNorm(output_dim)

        # 可学习的缩放因子，初始化为小值
        self.scale_factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检查输入是否包含NaN或Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            # 如果输入有问题，返回零张量
            if self.projection is not None:
                return torch.zeros_like(self.projection(x))
            else:
                return torch.zeros_like(x)

        # 处理维度不匹配
        if self.projection is not None:
            residual = self.projection(x)
        else:
            residual = x

        try:
            # 通过MLP基础模型
            mlp_output = self.base_model(x)

            # 检查MLP输出
            if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
                return self.final_norm(residual)

            # 通过LoRA层
            lora_output = self.lora_layer(mlp_output)

            # 检查LoRA输出
            if torch.isnan(lora_output).any() or torch.isinf(lora_output).any():
                return self.final_norm(residual)

            # 残差连接with scaling
            scale = torch.sigmoid(self.scale_factor)  # 确保scale在(0,1)范围内
            output = residual + scale * lora_output

            # 最终归一化
            output = self.final_norm(output)

            # 最终检查
            if torch.isnan(output).any() or torch.isinf(output).any():
                return self.final_norm(residual)

            return output

        except Exception as e:
            # 如果有任何异常，返回归一化的残差
            return self.final_norm(residual)


class ExpertLayer(nn.Module):
    """
    包含多个LoRA专家的层，修复了数值稳定性问题。
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

        # 专家输出归一化
        self.expert_norm = nn.LayerNorm(experts_output_dim)

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
        # 检查输入
        if torch.isnan(features).any() or torch.isinf(features).any():
            batch_size, seq_len, hidden_dim = features.shape
            zero_output = torch.zeros_like(features)
            expert_outputs = [zero_output for _ in range(self.num_experts)]

            if self.return_all:
                return expert_outputs
            else:
                return zero_output, expert_outputs

        # 所有专家独立前向传播
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            try:
                expert_out = expert(features)
                # 检查专家输出
                if torch.isnan(expert_out).any() or torch.isinf(expert_out).any():
                    expert_out = torch.zeros_like(features)
                expert_outputs.append(expert_out)
            except Exception as e:
                # 如果专家计算失败，使用零输出
                expert_outputs.append(torch.zeros_like(features))

        if self.return_all:
            return expert_outputs
        else:
            if expert_weights is None:
                raise ValueError("expert_weights must be provided when return_all=False")

            # 检查expert_weights
            if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
                # 如果权重有问题，使用均匀权重
                expert_weights = torch.ones_like(expert_weights) / self.num_experts

            try:
                stacked_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, S, H]

                # 归一化权重
                expert_weights = F.softmax(expert_weights, dim=-1)

                # 广播权重并计算加权和
                weighted_output = torch.sum(
                    stacked_outputs * expert_weights[:, :, None, None], dim=1
                )  # [B, S, H]

                # 应用归一化
                weighted_output = self.expert_norm(weighted_output)

                # 最终检查
                if torch.isnan(weighted_output).any() or torch.isinf(weighted_output).any():
                    weighted_output = torch.zeros_like(features)

                return weighted_output, expert_outputs

            except Exception as e:
                # 如果加权计算失败，返回零输出
                return torch.zeros_like(features), expert_outputs

    def compute_expert_diversity(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算专家间的多样性（输出差异程度）
        """
        try:
            # 获取所有专家的输出
            expert_outputs = []
            for expert in self.experts:
                output = expert(features)
                if not (torch.isnan(output).any() or torch.isinf(output).any()):
                    expert_outputs.append(output)

            if len(expert_outputs) < 2:
                return torch.tensor(0.0, device=features.device)

            stacked_outputs = torch.stack(expert_outputs, dim=1)  # [B, E, S, H]

            B, E, S, H = stacked_outputs.size()
            flat_outputs = stacked_outputs.view(B * S, E, H)  # [B*S, E, H]

            # 归一化
            normalized_outputs = F.normalize(flat_outputs, p=2, dim=-1)

            # 相似度矩阵 [B*S, E, E]
            similarity_matrix = torch.matmul(normalized_outputs, normalized_outputs.transpose(-2, -1))

            # 去掉对角线
            mask = torch.eye(E, device=similarity_matrix.device).bool()
            off_diagonal = similarity_matrix.masked_fill(mask.unsqueeze(0), 0)

            # 计算平均相似度
            avg_similarity = off_diagonal.sum() / (E * (E - 1) * (B * S))
            diversity_score = 1 - avg_similarity

            # 确保返回值是有效的
            if torch.isnan(diversity_score) or torch.isinf(diversity_score):
                diversity_score = torch.tensor(0.0, device=features.device)

            return diversity_score

        except Exception as e:
            return torch.tensor(0.0, device=features.device)