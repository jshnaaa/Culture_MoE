# src/llamafactory/model/cultural_components.py
"""
文化感知组件模块
包含文化嵌入、文化感知路由器、文化特定专家等组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math
import logging

from .experts import LoRA


class CulturalEmbeddingLayer(nn.Module):
    """
    文化嵌入层：将大洲标识映射为稠密向量表示
    提供大洲文化上下文感知和细粒度文化控制
    """
    def __init__(self, num_cultures: int = 6, culture_dim: int = 256, hidden_dim: int = 4096):
        super().__init__()

        self.num_cultures = num_cultures
        self.culture_dim = culture_dim
        self.hidden_dim = hidden_dim

        # 文化嵌入表
        self.culture_embeddings = nn.Embedding(num_cultures, culture_dim)

        # 文化特征投影
        self.culture_projection = nn.Linear(culture_dim, hidden_dim)

        # 文化上下文融合 - 使用多头注意力
        self.context_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 文化强度控制门
        self.culture_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重 - 使用小的初始化确保稳定性"""
        # 文化嵌入使用标准初始化
        nn.init.normal_(self.culture_embeddings.weight, mean=0, std=0.02)

        # 投影层使用小的初始化
        nn.init.normal_(self.culture_projection.weight, mean=0, std=0.001)
        if self.culture_projection.bias is not None:
            nn.init.zeros_(self.culture_projection.bias)

        # 门控网络初始化
        for module in self.culture_gate:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, L, H] 输入隐藏状态
            culture_ids: [B] 文化标识

        Returns:
            culturally_aware_states: [B, L, H] 文化感知的隐藏状态
            culture_attention_weights: [B, L, L] 文化注意力权重
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. 获取文化嵌入
        # 确保culture_ids在正确的设备上
        target_device = hidden_states.device
        if culture_ids.device != target_device:
            culture_ids = culture_ids.to(target_device)
        culture_emb = self.culture_embeddings(culture_ids)  # [B, culture_dim]
        culture_features = self.culture_projection(culture_emb)  # [B, H]

        # 2. 扩展文化特征到序列维度
        culture_features = culture_features.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, H]

        # 3. 文化上下文注意力
        # 使用文化特征作为 query，输入状态作为 key 和 value
        culturally_attended, attention_weights = self.context_fusion(
            query=culture_features,
            key=hidden_states,
            value=hidden_states
        )

        # 4. 文化强度门控
        culture_strength = self.culture_gate(culture_features)  # [B, L, 1]

        # 5. 融合原始状态和文化感知状态
        culturally_aware_states = self.layer_norm(
            hidden_states + culture_strength * culturally_attended
        )

        return culturally_aware_states, attention_weights


class CulturalAwareRouter(nn.Module):
    """
    文化感知路由器：结合大洲文化信息进行专家选择
    支持多维度路由决策：内容驱动、大洲文化驱动、亲和性驱动
    """
    def __init__(self, hidden_dim: int = 4096, num_experts: int = 12, num_cultures: int = 6,
                 culture_dim: int = 256, router_hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_cultures = num_cultures
        self.culture_dim = culture_dim

        # 文化嵌入
        self.culture_embeddings = nn.Embedding(num_cultures, culture_dim)

        # 内容路由器（基于输入内容）
        self.content_router = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, router_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim // 2, num_experts)
        )

        # 文化路由器（基于文化身份）
        self.culture_router = nn.Sequential(
            nn.Linear(culture_dim, router_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim // 2, router_hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(router_hidden_dim // 4, num_experts)
        )

        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(num_experts * 2, num_experts * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_experts * 2, num_experts),
            nn.ReLU(),
            nn.Linear(num_experts, num_experts)
        )

        # 文化-专家亲和性矩阵（可学习）
        self.culture_expert_affinity = nn.Parameter(
            torch.randn(num_cultures, num_experts) * 0.1
        )

        # 路由权重参数（可学习的融合权重）
        self.routing_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.2, 0.1]))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 文化嵌入
        nn.init.normal_(self.culture_embeddings.weight, mean=0, std=0.02)

        # 内容路由器 - 使用小的初始化防止梯度爆炸
        for module in self.content_router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)  # 小的gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 文化路由器 - 使用小的初始化
        for module in self.culture_router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)  # 小的gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 融合网络 - 使用小的初始化
        for module in self.fusion_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)  # 小的gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor,
                temperature: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            hidden_states: [B, H] 池化后的隐藏状态
            culture_ids: [B] 文化标识
            temperature: 温度参数

        Returns:
            expert_weights: [B, num_experts] 专家权重
            routing_info: dict 路由信息
        """
        batch_size = hidden_states.shape[0]
        target_device = hidden_states.device

        # 确保culture_ids在正确的设备上
        if culture_ids.device != target_device:
            culture_ids = culture_ids.to(target_device)

        # 1. 内容驱动的路由
        content_logits = self.content_router(hidden_states)  # [B, num_experts]

        # 2. 文化驱动的路由
        culture_emb = self.culture_embeddings(culture_ids)  # [B, culture_dim]
        culture_logits = self.culture_router(culture_emb)  # [B, num_experts]

        # 3. 文化-专家亲和性
        # 确保culture_expert_affinity在正确的设备上
        if self.culture_expert_affinity.device != target_device:
            self.culture_expert_affinity.data = self.culture_expert_affinity.data.to(target_device)
        affinity_logits = self.culture_expert_affinity[culture_ids]  # [B, num_experts]

        # 4. 多维度融合
        combined_features = torch.cat([content_logits, culture_logits], dim=-1)  # [B, num_experts*2]
        fusion_logits = self.fusion_network(combined_features)  # [B, num_experts]

        # 5. 使用可学习的权重进行最终路由决策
        routing_weights = F.softmax(self.routing_weights, dim=0)  # 归一化权重
        final_logits = (
            routing_weights[0] * content_logits +      # 内容驱动
            routing_weights[1] * culture_logits +      # 文化驱动
            routing_weights[2] * affinity_logits +     # 文化亲和性
            routing_weights[3] * fusion_logits         # 深度融合
        )

        # 6. 应用温度和 softmax (数值稳定性优化)
        # 裁剪logits到合理范围，防止溢出
        final_logits = torch.clamp(final_logits, min=-10.0, max=10.0)

        # 使用float32进行softmax计算，避免fp16溢出
        if final_logits.dtype == torch.float16:
            logits_for_softmax = final_logits.float() / temperature
            expert_weights = F.softmax(logits_for_softmax, dim=-1).half()
        else:
            expert_weights = F.softmax(final_logits / temperature, dim=-1)

        # 检查并修复NaN/Inf
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            logging.warning("Router weights contain NaN/Inf, using uniform distribution")
            expert_weights = torch.ones_like(expert_weights) / self.num_experts

        # 7. 收集路由信息
        routing_info = {
            'content_logits': content_logits,
            'culture_logits': culture_logits,
            'affinity_logits': affinity_logits,
            'fusion_logits': fusion_logits,
            'final_logits': final_logits,
            'routing_weights': routing_weights
        }

        return expert_weights, routing_info

    def compute_load_balancing_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失"""
        expert_usage = expert_weights.mean(dim=0)  # [num_experts]

        # 检查数值稳定性
        if torch.isnan(expert_usage).any() or torch.isinf(expert_usage).any():
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)

        uniform_distribution = torch.ones_like(expert_usage) / self.num_experts
        load_balancing_loss = F.mse_loss(expert_usage, uniform_distribution)

        # 确保损失不为0（避免被优化器忽略）
        load_balancing_loss = torch.clamp(load_balancing_loss, min=1e-8, max=1.0)

        return load_balancing_loss

    def entropy_regularization(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """计算熵正则化损失 - 鼓励均匀分布 (数值稳定性优化)"""
        # 检查输入数值稳定性
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            logging.warning("Expert weights contain NaN/Inf, returning zero entropy loss")
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)

        # 检查expert_weights是否已经是概率分布
        weight_sums = expert_weights.sum(dim=-1)
        if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-3):
            logging.warning(f"Expert weights do not sum to 1: {weight_sums[:5].tolist()}")

        # 确保权重为正且和为1
        expert_weights_normalized = torch.clamp(expert_weights, min=1e-8, max=1.0)
        expert_weights_normalized = expert_weights_normalized / expert_weights_normalized.sum(dim=-1, keepdim=True)

        # 🔧 修复81: 使用数值稳定的熵计算，避免log操作
        # 使用方差代替熵计算
        mean_weight = expert_weights_normalized.mean(dim=-1, keepdim=True)
        variance = ((expert_weights_normalized - mean_weight) ** 2).mean(dim=-1)

        # 理想方差（均匀分布的方差）
        num_experts = expert_weights.size(-1)
        ideal_variance = (1.0 / num_experts) * (1.0 - 1.0 / num_experts)

        # 熵正则化损失：鼓励达到理想方差（均匀分布）
        entropy_loss = torch.clamp(ideal_variance - variance, min=0.0).mean()

        # 裁剪损失到合理范围
        entropy_loss = torch.clamp(entropy_loss, min=1e-8, max=1.0)

        return entropy_loss


class CultureSpecificExpert(nn.Module):
    """
    文化特定专家：每个专家专门处理特定文化的知识
    包含文化条件的处理和文化适应机制
    """
    def __init__(self, expert_id: int, primary_culture_ids: List[int], hidden_dim: int = 4096,
                 expert_hidden_dim: int = 4096, lora_rank: int = 32, culture_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.expert_id = expert_id
        self.primary_culture_ids = primary_culture_ids  # 该专家主要负责的文化
        self.hidden_dim = hidden_dim
        self.culture_dim = culture_dim

        # 文化提示向量（可学习的文化知识）
        if len(primary_culture_ids) > 0:
            self.culture_prompt = nn.Parameter(
                torch.randn(len(primary_culture_ids), culture_dim) * 0.01
            )
        else:
            # 通用专家或特殊专家
            self.culture_prompt = nn.Parameter(
                torch.randn(1, culture_dim) * 0.01
            )

        # 文化条件的 MLP
        self.culture_conditioned_mlp = nn.Sequential(
            nn.Linear(hidden_dim + culture_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, hidden_dim)
        )

        # LoRA 层
        self.lora_layer = LoRA(hidden_dim, hidden_dim, lora_rank)

        # 文化适应层
        self.culture_adaptation = nn.Sequential(
            nn.Linear(culture_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Tanh()
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 专家置信度网络
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 文化条件 MLP
        for module in self.culture_conditioned_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # LoRA 层
        nn.init.normal_(self.lora_layer.W, mean=0, std=0.0001)
        nn.init.normal_(self.lora_layer.A, mean=0, std=0.0001)
        nn.init.normal_(self.lora_layer.B, mean=0, std=0.0001)

        # 文化适应层
        for module in self.culture_adaptation:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 置信度网络
        for module in self.confidence_network:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, L, H] 输入隐藏状态
            culture_ids: [B] 文化标识

        Returns:
            expert_output: [B, L, H] 专家输出
            culture_relevance: [B] 文化相关性得分
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        target_device = hidden_states.device

        # 确保culture_ids在正确的设备上
        if culture_ids.device != target_device:
            culture_ids = culture_ids.to(target_device)

        # 1. 计算文化相关性
        culture_relevance = self._compute_culture_relevance(culture_ids)

        # 2. 获取文化提示
        culture_prompt = self._get_culture_prompt(culture_ids)  # [B, culture_dim]

        # 3. 文化适应
        culture_adaptation = self.culture_adaptation(culture_prompt)  # [B, H]
        culture_adaptation = culture_adaptation.unsqueeze(1).expand(-1, seq_len, -1)

        # 4. 文化条件的特征提取
        # 将文化信息与隐藏状态拼接
        culture_prompt_expanded = culture_prompt.unsqueeze(1).expand(-1, seq_len, -1)
        conditioned_input = torch.cat([hidden_states, culture_prompt_expanded], dim=-1)

        # 5. 通过文化条件 MLP
        mlp_output = self.culture_conditioned_mlp(conditioned_input)

        # 6. LoRA 增强
        lora_output = self.lora_layer(mlp_output)

        # 7. 计算专家置信度
        pooled_hidden = hidden_states.mean(dim=1)  # [B, H]
        confidence = self.confidence_network(pooled_hidden).squeeze(-1)  # [B]

        # 8. 文化适应和残差连接
        expert_output = self.layer_norm(
            hidden_states +
            culture_adaptation * lora_output * confidence.unsqueeze(-1).unsqueeze(-1)
        )

        return expert_output, culture_relevance

    def _compute_culture_relevance(self, culture_ids: torch.Tensor) -> torch.Tensor:
        """计算输入文化与该专家的相关性"""
        relevance_scores = []
        for culture_id in culture_ids:
            if len(self.primary_culture_ids) == 0:
                # 冲突处理专家对所有文化都有中等相关性（处理文化冲突）
                relevance_scores.append(0.5)
            elif culture_id.item() in self.primary_culture_ids:
                # 主要负责的文化有高相关性
                relevance_scores.append(1.0)
            else:
                # 其他文化有低相关性
                relevance_scores.append(0.1)
        return torch.tensor(relevance_scores, device=culture_ids.device, dtype=torch.float32)

    def _get_culture_prompt(self, culture_ids: torch.Tensor) -> torch.Tensor:
        """获取文化提示向量"""
        target_device = culture_ids.device
        prompts = []
        for culture_id in culture_ids:
            if len(self.primary_culture_ids) == 0:
                # 冲突处理专家使用平均提示（处理跨文化情况）
                prompt = self.culture_prompt[0]
            elif culture_id.item() in self.primary_culture_ids:
                # 使用对应的文化提示
                idx = self.primary_culture_ids.index(culture_id.item())
                prompt = self.culture_prompt[idx]
            else:
                # 使用平均提示
                prompt = self.culture_prompt.mean(dim=0)

            # 确保prompt在正确的设备上
            if prompt.device != target_device:
                prompt = prompt.to(target_device)
            prompts.append(prompt)

        return torch.stack(prompts)


class CulturalContextAwareness(nn.Module):
    """
    文化上下文感知：理解文本中的大洲文化线索和上下文
    提供大洲文化冲突检测和跨大洲理解能力
    """
    def __init__(self, hidden_dim: int = 4096, num_cultures: int = 6, context_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_cultures = num_cultures
        self.context_dim = context_dim

        # 文化线索检测器
        self.cultural_cue_detector = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim // 2),
            nn.ReLU(),
            nn.Linear(context_dim // 2, num_cultures),
            nn.Sigmoid()  # 多标签：一个文本可能包含多种文化线索
        )

        # 文化冲突检测
        self.conflict_detector = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim // 2),
            nn.ReLU(),
            nn.Linear(context_dim // 2, 1),
            nn.Sigmoid()
        )

        # 跨文化桥梁网络
        self.cross_cultural_bridge = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 文化敏感性检测
        self.sensitivity_detector = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module_list in [self.cultural_cue_detector, self.conflict_detector, self.sensitivity_detector]:
            for module in module_list:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            hidden_states: [B, L, H] 输入序列
            culture_ids: [B] 用户文化标识

        Returns:
            context_aware_states: [B, L, H] 文化上下文感知的状态
            cultural_analysis: dict 文化分析结果
        """
        target_device = hidden_states.device

        # 确保culture_ids在正确的设备上
        if culture_ids.device != target_device:
            culture_ids = culture_ids.to(target_device)

        # 1. 池化获取全局表示
        pooled_states = hidden_states.mean(dim=1)  # [B, H]

        # 2. 检测文化线索
        cultural_cues = self.cultural_cue_detector(pooled_states)  # [B, num_cultures]

        # 3. 检测文化冲突
        conflict_probability = self.conflict_detector(pooled_states)  # [B, 1]

        # 4. 检测文化敏感性
        sensitivity_score = self.sensitivity_detector(pooled_states)  # [B, 1]

        # 5. 跨文化理解
        cross_cultural_states, cross_attention = self.cross_cultural_bridge(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states
        )

        # 6. 文化上下文感知状态
        # 根据敏感性得分调整跨文化理解的强度
        context_aware_states = (
            hidden_states +
            0.1 * sensitivity_score.unsqueeze(1) * cross_cultural_states
        )

        cultural_analysis = {
            'cultural_cues': cultural_cues,
            'conflict_probability': conflict_probability.squeeze(-1),
            'sensitivity_score': sensitivity_score.squeeze(-1),
            'cross_attention_weights': cross_attention
        }

        return context_aware_states, cultural_analysis


def create_culture_assignments(num_experts: int, num_cultures: int = 6) -> List[List[int]]:
    """
    动态创建文化分配方案 - 无通用专家版本

    设计原则：
    1. 共享专家已负责通用知识，路由专家应全部专门化
    2. 每个文化都有专门的专家负责
    3. 保留1个冲突处理专家
    4. 剩余专家通过文化组合或多文化专门化来分配

    Args:
        num_experts: 专家数量
        num_cultures: 文化数量

    Returns:
        culture_assignments: 每个专家负责的文化列表
    """
    if num_experts < 2:
        return [[]]

    culture_assignments = []

    if num_experts >= num_cultures + 1:
        # 专家数量充足：每个文化至少有一个专家 + 冲突专家

        # 1. 为每个文化分配一个专门的专家
        for i in range(num_cultures):
            culture_assignments.append([i])

        # 2. 剩余专家采用文化组合策略，避免通用专家
        remaining_experts = num_experts - num_cultures - 1  # 减去文化专家和冲突专家

        if remaining_experts > 0:
            # 创建文化组合专家：每个专家负责2-3个相关文化
            culture_combinations = [
                [0, 1],     # 亚洲-欧洲（相邻大陆）
                [2, 3],     # 北美-南美（美洲）
                [4, 5],     # 非洲-大洋洲（南半球偏向）
                [0, 2],     # 亚洲-北美（太平洋圈）
                [1, 4],     # 欧洲-非洲（历史联系）
                [0, 3, 5],  # 亚洲-南美-大洋洲（多元文化）
                [1, 2, 4],  # 欧洲-北美-非洲（跨大西洋）
                [0, 4],     # 亚洲-非洲（发展中地区）
                [1, 3],     # 欧洲-南美（拉丁文化）
                [2, 5],     # 北美-大洋洲（英语文化圈）
            ]

            # 分配文化组合专家
            for i in range(min(remaining_experts, len(culture_combinations))):
                culture_assignments.append(culture_combinations[i])

            # 如果还有剩余专家，创建三文化专家
            if remaining_experts > len(culture_combinations):
                extra_experts = remaining_experts - len(culture_combinations)
                for i in range(extra_experts):
                    # 创建三文化组合
                    base_idx = i % num_cultures
                    three_cultures = [
                        base_idx,
                        (base_idx + 2) % num_cultures,
                        (base_idx + 4) % num_cultures
                    ]
                    culture_assignments.append(three_cultures)

        # 3. 最后一个专家作为文化冲突处理专家
        culture_assignments.append([])

    else:
        # 专家数量不足：多个文化共享专家，但避免全文化通用专家
        available_experts = num_experts - 1  # 保留1个冲突专家

        if available_experts >= num_cultures // 2:
            # 每个专家负责2个文化
            culture_pairs = [
                [0, 1], [2, 3], [4, 5],  # 基础配对
                [0, 2], [1, 4], [3, 5],  # 交叉配对
            ]

            for i in range(min(available_experts, len(culture_pairs))):
                culture_assignments.append(culture_pairs[i])

            # 处理剩余文化
            assigned_cultures = set()
            for assignment in culture_assignments:
                assigned_cultures.update(assignment)

            remaining_cultures = [c for c in range(num_cultures) if c not in assigned_cultures]
            if remaining_cultures:
                # 将剩余文化分配给现有专家
                for i, culture in enumerate(remaining_cultures):
                    culture_assignments[i % len(culture_assignments)].append(culture)
        else:
            # 专家数量很少：每个专家负责3个文化
            cultures_per_expert = (num_cultures + available_experts - 1) // available_experts

            for i in range(available_experts):
                start_idx = i * cultures_per_expert
                end_idx = min((i + 1) * cultures_per_expert, num_cultures)
                if start_idx < num_cultures:
                    culture_assignments.append(list(range(start_idx, end_idx)))

        # 文化冲突处理专家
        culture_assignments.append([])

    return culture_assignments