# src/llamafactory/model/dynamic_cultural_components.py
"""
动态文化聚类组件模块
实现基于数据驱动的文化专家动态分配机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math
import logging

from .experts import LoRA


class CultureFeatureExtractor(nn.Module):
    """
    文化特征提取器：从隐藏状态中提取文化特征
    """
    def __init__(self, hidden_dim: int = 4096, culture_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.culture_dim = culture_dim

        # 文化特征提取网络
        self.culture_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, culture_dim),
            nn.Tanh()  # 归一化到[-1,1]
        )

        # 文化强度估计器
        self.culture_strength_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # 文化强度 [0,1]
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.culture_extractor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.culture_strength_estimator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, L, H] 输入隐藏状态

        Returns:
            culture_features: [B, culture_dim] 文化特征
            culture_strength: [B] 文化强度
        """
        # 池化获取全局表示
        pooled = hidden_states.mean(dim=1)  # [B, H]

        # 提取文化特征
        culture_features = self.culture_extractor(pooled)  # [B, culture_dim]

        # 估计文化强度
        culture_strength = self.culture_strength_estimator(pooled).squeeze(-1)  # [B]

        return culture_features, culture_strength


class LearnableCultureClustering(nn.Module):
    """
    可学习的文化聚类模块
    动态学习文化聚类中心，实现专家的动态分配
    """
    def __init__(self, num_experts: int = 12, culture_dim: int = 256, num_cultures: int = 6):
        super().__init__()

        self.num_experts = num_experts
        self.culture_dim = culture_dim
        self.num_cultures = num_cultures

        # 可学习的文化聚类中心 - 每个专家对应一个聚类中心
        self.culture_cluster_centers = nn.Parameter(
            torch.randn(num_experts, culture_dim) * 0.1
        )

        # 聚类温度参数（可学习）
        self.clustering_temperature = nn.Parameter(torch.tensor(1.0))

        # 专家置信度权重（可学习）
        self.expert_confidence_weights = nn.Parameter(
            torch.ones(num_experts) * 0.5
        )

        # 固定文化分配（作为fallback和初始化）
        # 注意：这是一个列表，不能用register_buffer，直接作为属性保存
        self.fixed_culture_assignments = self._create_fixed_assignments()

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 使用固定分配来初始化聚类中心
        with torch.no_grad():
            # 为每个专家创建基于其固定分配的初始聚类中心
            for i, assignment in enumerate(self.fixed_culture_assignments):
                if len(assignment) > 0:
                    # 基于分配的文化创建初始中心
                    center = torch.zeros(self.culture_dim)
                    for culture_id in assignment:
                        # 为每个文化添加一些随机偏移
                        culture_vector = torch.randn(self.culture_dim) * 0.1
                        culture_vector[culture_id % self.culture_dim] = 1.0  # 主要特征
                        center += culture_vector
                    center = center / len(assignment)
                    self.culture_cluster_centers[i] = center
                else:
                    # 冲突处理专家：使用中性初始化
                    self.culture_cluster_centers[i] = torch.randn(self.culture_dim) * 0.05

    def _create_fixed_assignments(self) -> List[List[int]]:
        """创建固定的文化分配作为fallback"""
        assignments = []

        if self.num_experts >= self.num_cultures + 1:
            # 充足专家：每个文化一个专家 + 组合专家
            for i in range(self.num_cultures):
                assignments.append([i])

            # 剩余专家创建组合
            remaining = self.num_experts - self.num_cultures - 1
            combinations = [
                [0, 1], [2, 3], [4, 5], [0, 2], [1, 4], [3, 5]
            ]

            for i in range(min(remaining, len(combinations))):
                assignments.append(combinations[i])

            # 最后一个作为冲突处理专家
            assignments.append([])
        else:
            # 专家不足：多文化共享
            cultures_per_expert = (self.num_cultures + self.num_experts - 2) // (self.num_experts - 1)

            for i in range(self.num_experts - 1):
                start_idx = i * cultures_per_expert
                end_idx = min((i + 1) * cultures_per_expert, self.num_cultures)
                if start_idx < self.num_cultures:
                    assignments.append(list(range(start_idx, end_idx)))

            # 最后一个作为冲突处理专家
            assignments.append([])

        return assignments

    def compute_dynamic_culture_affinity(self, culture_features: torch.Tensor,
                                       current_epoch: int) -> Tuple[torch.Tensor, Dict]:
        """
        计算动态文化亲和性

        Args:
            culture_features: [B, culture_dim] 文化特征
            current_epoch: 当前epoch

        Returns:
            expert_affinities: [B, num_experts] 专家亲和性
            clustering_info: dict 聚类信息
        """
        batch_size = culture_features.shape[0]
        device = culture_features.device

        # 计算每个样本与各聚类中心的相似度
        # 确保聚类中心在正确设备上
        cluster_centers = self.culture_cluster_centers.to(device=culture_features.device, dtype=culture_features.dtype)
        similarities = torch.cosine_similarity(
            culture_features.unsqueeze(1),  # [B, 1, culture_dim]
            cluster_centers.unsqueeze(0),  # [1, num_experts, culture_dim]
            dim=-1
        )  # [B, num_experts]

        # 应用温度和专家置信度
        temperature = torch.clamp(self.clustering_temperature, min=0.1, max=5.0)
        confidence_weights = torch.sigmoid(self.expert_confidence_weights)

        # 确保设备一致性
        temperature = temperature.to(device=similarities.device, dtype=similarities.dtype)
        confidence_weights = confidence_weights.to(device=similarities.device, dtype=similarities.dtype)

        # 计算软分配
        weighted_similarities = similarities * confidence_weights.unsqueeze(0)
        expert_affinities = F.softmax(weighted_similarities / temperature, dim=-1)

        # 收集聚类信息
        clustering_info = {
            'similarities': similarities,
            'temperature': temperature.item(),
            'confidence_weights': confidence_weights,
            'cluster_centers': cluster_centers,  # 使用已经同步设备的聚类中心
            'affinity_entropy': -torch.sum(expert_affinities * torch.log(expert_affinities + 1e-8), dim=-1).mean()
        }

        return expert_affinities, clustering_info

    def compute_fixed_culture_affinity(self, culture_labels: torch.Tensor) -> torch.Tensor:
        """
        计算基于固定分配的文化亲和性

        Args:
            culture_labels: [B] 文化标签

        Returns:
            expert_affinities: [B, num_experts] 专家亲和性
        """
        batch_size = culture_labels.shape[0]
        device = culture_labels.device

        expert_affinities = torch.zeros(batch_size, self.num_experts, device=device)

        for b in range(batch_size):
            culture_id = culture_labels[b].item()

            # 为每个专家计算相关性
            for expert_id, assignment in enumerate(self.fixed_culture_assignments):
                if len(assignment) == 0:
                    # 冲突处理专家：对所有文化都有中等相关性
                    expert_affinities[b, expert_id] = 0.5
                elif culture_id in assignment:
                    # 主要负责的文化：高相关性
                    expert_affinities[b, expert_id] = 1.0
                else:
                    # 其他文化：低相关性
                    expert_affinities[b, expert_id] = 0.1

        # 归一化
        expert_affinities = F.softmax(expert_affinities, dim=-1)

        return expert_affinities

    def get_mixed_affinity(self, culture_features: torch.Tensor, culture_labels: torch.Tensor,
                          current_epoch: int, total_epochs: int = 4) -> Tuple[torch.Tensor, Dict]:
        """
        获取混合的专家亲和性（固定分配 + 动态聚类）

        Args:
            culture_features: [B, culture_dim] 文化特征
            culture_labels: [B] 文化标签
            current_epoch: 当前epoch (1-based)
            total_epochs: 总epoch数

        Returns:
            mixed_affinity: [B, num_experts] 混合亲和性
            affinity_info: dict 亲和性信息
        """
        # 计算动态权重（渐进式）
        dynamic_weight = self.get_dynamic_weight(current_epoch, total_epochs)

        # 计算固定分配亲和性
        fixed_affinity = self.compute_fixed_culture_affinity(culture_labels)

        # 计算动态聚类亲和性
        dynamic_affinity, clustering_info = self.compute_dynamic_culture_affinity(
            culture_features, current_epoch
        )

        # 混合两种亲和性
        mixed_affinity = (
            (1 - dynamic_weight) * fixed_affinity +
            dynamic_weight * dynamic_affinity
        )

        # 收集信息
        affinity_info = {
            'dynamic_weight': dynamic_weight,
            'fixed_affinity': fixed_affinity,
            'dynamic_affinity': dynamic_affinity,
            'clustering_info': clustering_info,
            'epoch': current_epoch
        }

        return mixed_affinity, affinity_info

    @staticmethod
    def get_dynamic_weight(current_epoch: int, total_epochs: int = 4) -> float:
        """
        获取动态聚类权重（渐进式方案）

        Args:
            current_epoch: 当前epoch (1-based)
            total_epochs: 总epoch数

        Returns:
            dynamic_weight: 动态聚类权重 [0, 1]
        """
        if current_epoch <= 1:
            return 0.0  # 第1轮：完全固定分配
        elif current_epoch == 2:
            return 0.3  # 第2轮：30% 动态聚类
        elif current_epoch == 3:
            return 0.7  # 第3轮：70% 动态聚类
        else:  # current_epoch >= 4
            return 1.0  # 第4轮：完全动态聚类

    @staticmethod
    def get_clustering_temperature(current_epoch: int) -> float:
        """
        获取聚类温度（随epoch降低）

        Args:
            current_epoch: 当前epoch (1-based)

        Returns:
            temperature: 聚类温度
        """
        if current_epoch <= 1:
            return 2.0    # 高温度，软聚类
        elif current_epoch == 2:
            return 1.5    # 中等温度
        elif current_epoch == 3:
            return 1.0    # 标准温度
        else:
            return 0.7    # 低温度，硬聚类


class DynamicCultureLoss(nn.Module):
    """
    动态文化聚类相关的损失函数
    """
    def __init__(self):
        super().__init__()

    def clustering_consistency_loss(self, expert_weights: torch.Tensor,
                                  culture_affinities: torch.Tensor) -> torch.Tensor:
        """
        聚类一致性损失：鼓励路由权重与文化亲和性一致

        Args:
            expert_weights: [B, num_experts] 路由器输出的专家权重
            culture_affinities: [B, num_experts] 文化亲和性权重

        Returns:
            consistency_loss: 一致性损失
        """
        # 使用KL散度衡量两个分布的差异
        expert_weights_safe = torch.clamp(expert_weights, min=1e-8, max=1.0)
        culture_affinities_safe = torch.clamp(culture_affinities, min=1e-8, max=1.0)

        # 归一化
        expert_weights_norm = expert_weights_safe / expert_weights_safe.sum(dim=-1, keepdim=True)
        culture_affinities_norm = culture_affinities_safe / culture_affinities_safe.sum(dim=-1, keepdim=True)

        # KL散度
        kl_div = F.kl_div(
            torch.log(expert_weights_norm + 1e-8),
            culture_affinities_norm,
            reduction='batchmean'
        )

        return kl_div

    def clustering_diversity_loss(self, cluster_centers: torch.Tensor) -> torch.Tensor:
        """
        聚类多样性损失：鼓励聚类中心分散，避免聚类塌陷

        Args:
            cluster_centers: [num_experts, culture_dim] 聚类中心

        Returns:
            diversity_loss: 多样性损失
        """
        # 计算聚类中心之间的余弦相似度
        normalized_centers = F.normalize(cluster_centers, p=2, dim=1)
        similarities = torch.mm(normalized_centers, normalized_centers.t())

        # 除去对角线（自相似度）
        mask = ~torch.eye(cluster_centers.size(0), dtype=torch.bool, device=cluster_centers.device)
        off_diagonal_similarities = similarities[mask]

        # 惩罚过高的相似度（大于0.8认为太相似）
        diversity_loss = torch.clamp(off_diagonal_similarities - 0.8, min=0).mean()

        return diversity_loss

    def culture_strength_regularization(self, culture_strength: torch.Tensor) -> torch.Tensor:
        """
        文化强度正则化：鼓励适中的文化强度分布

        Args:
            culture_strength: [B] 文化强度

        Returns:
            strength_loss: 强度正则化损失
        """
        # 鼓励文化强度在0.3-0.8之间
        target_strength = 0.5
        strength_loss = F.mse_loss(culture_strength,
                                 torch.full_like(culture_strength, target_strength))

        return strength_loss


class DynamicCulturalAwareRouter(nn.Module):
    """
    动态文化感知路由器：结合固定分配和动态聚类
    """
    def __init__(self, hidden_dim: int = 4096, num_experts: int = 12, num_cultures: int = 6,
                 culture_dim: int = 256, router_hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_cultures = num_cultures
        self.culture_dim = culture_dim

        # 文化特征提取器
        self.culture_feature_extractor = CultureFeatureExtractor(
            hidden_dim, culture_dim, dropout
        )

        # 可学习文化聚类
        self.culture_clustering = LearnableCultureClustering(
            num_experts, culture_dim, num_cultures
        )

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

        # 融合网络（结合内容和文化信息）
        self.fusion_network = nn.Sequential(
            nn.Linear(num_experts + culture_dim, num_experts * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_experts * 2, num_experts),
            nn.ReLU(),
            nn.Linear(num_experts, num_experts)
        )

        # 路由权重参数（可学习的融合权重）
        self.routing_weights = nn.Parameter(torch.tensor([0.3, 0.4, 0.3]))  # content, culture_affinity, fusion

        # 动态文化损失
        self.culture_loss_module = DynamicCultureLoss()

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 内容路由器
        for module in self.content_router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 融合网络
        for module in self.fusion_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor,
                current_epoch: int, temperature: float = 1.0,
                total_epochs: int = 4) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            hidden_states: [B, H] 池化后的隐藏状态
            culture_ids: [B] 文化标识
            current_epoch: 当前epoch
            temperature: 温度参数
            total_epochs: 总epoch数

        Returns:
            expert_weights: [B, num_experts] 专家权重
            routing_info: dict 路由信息
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # 确保culture_ids在正确的设备上
        if culture_ids.device != device:
            culture_ids = culture_ids.to(device)

        # 1. 提取文化特征
        culture_features, culture_strength = self.culture_feature_extractor(
            hidden_states.unsqueeze(1)  # [B, H] -> [B, 1, H] for feature extractor
        )

        # 2. 内容驱动的路由
        content_logits = self.content_router(hidden_states)  # [B, num_experts]

        # 3. 获取混合文化亲和性
        culture_affinities, affinity_info = self.culture_clustering.get_mixed_affinity(
            culture_features, culture_ids, current_epoch, total_epochs
        )

        # 4. 融合路由（结合内容和文化特征）
        fusion_input = torch.cat([content_logits, culture_features], dim=-1)
        fusion_logits = self.fusion_network(fusion_input)

        # 5. 最终路由决策
        routing_weights = F.softmax(self.routing_weights, dim=0)
        # 确保所有logits数据类型一致
        target_dtype = content_logits.dtype
        target_device = content_logits.device

        culture_logits = torch.log(culture_affinities + 1e-8).to(dtype=target_dtype, device=target_device)
        fusion_logits = fusion_logits.to(dtype=target_dtype, device=target_device)
        routing_weights = routing_weights.to(dtype=target_dtype, device=target_device)

        final_logits = (
            routing_weights[0] * content_logits +
            routing_weights[1] * culture_logits +
            routing_weights[2] * fusion_logits
        )

        # 6. 应用温度和softmax
        final_logits = torch.clamp(final_logits, min=-10.0, max=10.0)
        # 确保温度参数类型匹配
        temperature = torch.tensor(temperature, dtype=final_logits.dtype, device=final_logits.device)

        if final_logits.dtype == torch.float16:
            logits_for_softmax = final_logits.float() / temperature.float()
            expert_weights = F.softmax(logits_for_softmax, dim=-1).half()
        else:
            expert_weights = F.softmax(final_logits / temperature, dim=-1)

        # 检查并修复NaN/Inf
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            logging.warning("Dynamic router weights contain NaN/Inf, using uniform distribution")
            expert_weights = torch.ones_like(expert_weights) / self.num_experts

        # 7. 收集路由信息
        routing_info = {
            'content_logits': content_logits,
            'culture_affinities': culture_affinities,
            'fusion_logits': fusion_logits,
            'final_logits': final_logits,
            'routing_weights': routing_weights,
            'culture_features': culture_features,
            'culture_strength': culture_strength,
            'affinity_info': affinity_info
        }

        return expert_weights, routing_info

    def compute_dynamic_culture_losses(self, routing_info: Dict) -> Dict[str, torch.Tensor]:
        """
        计算动态文化相关的损失

        Args:
            routing_info: 路由信息

        Returns:
            losses: 各种损失的字典
        """
        losses = {}

        # 1. 聚类一致性损失
        expert_weights = routing_info.get('expert_weights')  # 需要从外部传入
        culture_affinities = routing_info['culture_affinities']

        if expert_weights is not None:
            consistency_loss = self.culture_loss_module.clustering_consistency_loss(
                expert_weights, culture_affinities
            )
            losses['clustering_consistency_loss'] = consistency_loss

        # 2. 聚类多样性损失
        cluster_centers = self.culture_clustering.culture_cluster_centers
        diversity_loss = self.culture_loss_module.clustering_diversity_loss(cluster_centers)
        losses['clustering_diversity_loss'] = diversity_loss

        # 3. 文化强度正则化
        culture_strength = routing_info['culture_strength']
        strength_loss = self.culture_loss_module.culture_strength_regularization(culture_strength)
        losses['culture_strength_loss'] = strength_loss

        return losses

    def compute_load_balancing_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失"""
        expert_usage = expert_weights.mean(dim=0)

        if torch.isnan(expert_usage).any() or torch.isinf(expert_usage).any():
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)

        uniform_distribution = torch.ones_like(expert_usage) / self.num_experts
        load_balancing_loss = F.mse_loss(expert_usage, uniform_distribution)

        return torch.clamp(load_balancing_loss, min=1e-8, max=1.0)

    def entropy_regularization(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """计算熵正则化损失"""
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            logging.warning("Expert weights contain NaN/Inf, returning zero entropy loss")
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)

        # 确保权重为正且和为1
        expert_weights_normalized = torch.clamp(expert_weights, min=1e-8, max=1.0)
        expert_weights_normalized = expert_weights_normalized / expert_weights_normalized.sum(dim=-1, keepdim=True)

        # 计算熵
        log_weights = torch.log(expert_weights_normalized + 1e-8)
        entropy = -torch.sum(expert_weights_normalized * log_weights, dim=-1)

        # 最大熵
        max_entropy = torch.log(torch.tensor(expert_weights.size(-1), dtype=entropy.dtype, device=entropy.device))

        # 熵正则化损失
        entropy_loss = (max_entropy - entropy).mean()

        return torch.clamp(entropy_loss, min=1e-8, max=max_entropy.item())