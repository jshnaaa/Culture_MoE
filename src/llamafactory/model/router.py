import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
温度参数的作用
温度对softmax分布的影响：
高温度（> 1.0）：软化分布，权重更均匀；多个专家权重相近；更加"民主"的专家选择
低温度（< 1.0）：锐化分布，权重更集中；少数专家获得高权重；更加"专制"的专家选择
"""

"""
ExpertRouter（基础路由器）
特点：
使用固定温度的softmax进行专家权重分配
温度参数是静态的（默认为1.0）
所有输入都使用相同的温度参数
"""

class ExpertRouter(nn.Module):
    """
    路由算法模块
    将Llama共享层的特征映射到6个专家的权重分布
    所有专家都参与计算，通过softmax输出每个专家的权重，最后加权求和
    """

    def __init__(self, router_input_dim: int, num_experts: int = 6, router_hidden_dim: int = 512, dropout: float = 0.1):
        """
        初始化路由模块

        Args:
            router_input_dim: 输入特征维度（来自Llama的hidden_size）
            num_experts: 专家数量，默认为6
            router_hidden_dim: 路由网络的隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        self.input_dim = router_input_dim
        self.num_experts = num_experts
        self.hidden_dim = router_hidden_dim

        # 路由网络：多层感知机
        self.router_network = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, router_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim // 2, num_experts)
        )

        # 初始化权重
        self._init_weights()

        logging.info(f"Full Expert Router initialized: {router_input_dim} -> {num_experts} experts (all experts participate)")

    def _init_weights(self):
        """
        初始化网络权重 - 改进版本
        使用更好的初始化策略确保 Router 稳定性
        """
        for i, module in enumerate(self.router_network):
            if isinstance(module, nn.Linear):
                # ✅ 最后一层（输出层）使用更小的初始化
                if i == len(self.router_network) - 1:
                    # 最后一层：使用小的初始化，使 logits 接近 0
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    # 中间层：使用 Xavier 初始化
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 - 所有专家都参与计算

        Args:
            features: 来自Llama的特征向量 [batch_size, input_dim]
            temperature: softmax温度参数，用于控制分布的平滑程度

        Returns:
            expert_weights: 所有专家的权重分布 [batch_size, num_experts] (和为1)
            router_logits: 路由原始logits [batch_size, num_experts]
        """
        # 通过路由网络计算logits
        router_logits = self.router_network(features)  # [batch_size, num_experts]

        # 应用温度缩放和softmax，确保所有专家权重和为1
        expert_weights = F.softmax(router_logits / temperature, dim=-1)

        return expert_weights, router_logits

    def compute_load_balancing_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失，鼓励专家使用的均匀分布

        Args:
            router_logits: 路由logits [batch_size, num_experts]

        Returns:
            load_balancing_loss: 负载均衡损失
        """
        # 计算每个专家被选择的概率
        expert_probs = F.softmax(router_logits, dim=-1)  # [batch_size, num_experts]

        # 计算每个专家的平均使用率
        expert_usage = expert_probs.mean(dim=0)  # [num_experts]

        # 负载均衡损失：鼓励均匀分布（最小化方差）
        uniform_distribution = torch.ones_like(expert_usage) / self.num_experts
        load_balancing_loss = F.mse_loss(expert_usage, uniform_distribution)

        return load_balancing_loss

    def get_expert_utilization(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        计算专家利用率统计

        Args:
            router_logits: 路由logits [batch_size, num_experts]

        Returns:
            utilization: 每个专家的利用率 [num_experts]
        """
        expert_weights = F.softmax(router_logits, dim=-1)
        utilization = expert_weights.mean(dim=0)
        return utilization

    def entropy_regularization(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        计算熵正则化损失，防止路由过于确定性

        Args:
            expert_weights: 专家权重 [batch_size, num_experts]

        Returns:
            entropy_loss: 熵正则化损失
        """
        # 计算每个样本的熵
        entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-8), dim=-1)

        # 返回负熵（我们希望最大化熵，即最小化负熵）
        entropy_loss = -entropy.mean()

    def negative_entropy_regularization(self, expert_weights: torch.Tensor, lambda_entropy: float = -0.01) -> torch.Tensor:
        """
        ✅ 负熵正则化损失，尖锐化路由

        让 router_probs 不要平均，让每个样本更倾向使用某个专家
        增强专家 specialization

        Args:
            expert_weights: 专家权重 [batch_size, num_experts]
            lambda_entropy: 负熵权重（默认 -0.01，负号表示增加负熵，就是降低熵）

        Returns:
            neg_entropy_loss: 负熵正则化损失
        """
        # 计算每个样本的熵
        entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-8), dim=-1)

        # 返回负熵损失：lambda_entropy * entropy
        # 由于 lambda_entropy 是负数，所以这会最小化熵（让分布更尖锐）
        neg_entropy_loss = lambda_entropy * entropy.mean()

        return neg_entropy_loss

        return entropy_loss


"""
AdaptiveRouter（自适应路由器）
特点：
使用动态温度的softmax进行专家权重分配
温度参数根据输入特征自适应调整
不同输入会有不同的温度，更加灵活
"""

class AdaptiveRouter(ExpertRouter):
    """
    自适应路由器，可以根据输入动态调整路由策略
    """

    def __init__(self, router_input_dim: int, num_experts: int = 6, router_hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__(router_input_dim, num_experts, router_hidden_dim, dropout)

        # 添加一个温度预测网络
        self.temperature_network = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(router_hidden_dim // 4, 1),
            nn.Sigmoid()  # 输出0-1之间的值
        )

    def forward(self, features: torch.Tensor, base_temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        自适应前向传播

        Args:
            features: 输入特征 [batch_size, input_dim]
            base_temperature: 基础温度

        Returns:
            expert_weights: 专家权重分布 [batch_size, num_experts]
            router_logits: 路由原始logits [batch_size, num_experts]
            adaptive_temperature: 自适应温度 [batch_size, 1]
        """
        # 计算路由logits
        router_logits = self.router_network(features)

        # 计算自适应温度（0.1到2.0之间）
        adaptive_temperature = self.temperature_network(features) * 1.9 + 0.1  # [batch_size, 1]

        # 应用自适应温度
        expert_weights = F.softmax(router_logits / adaptive_temperature, dim=-1)

        return expert_weights, router_logits, adaptive_temperature

