# src/llamafactory/model/dynamic_enhanced_culturemoe.py
"""
动态增强的文化感知 CultureMoE 模型
集成了动态文化聚类和自适应专家分配机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
import math

from .CultureMoE import LlamaSharedRouterExpertsModel
from .cultural_components import (
    CulturalEmbeddingLayer,
    CultureSpecificExpert,
    CulturalContextAwareness
)
from .dynamic_cultural_components import DynamicCulturalAwareRouter
from .moe_args import ModelArgs


class DynamicEnhancedCultureMoE(LlamaSharedRouterExpertsModel):
    """
    动态增强的文化感知 MoE 模型

    新增功能：
    1. 动态文化聚类：数据驱动的专家分配
    2. 渐进式学习：从固定分配到动态聚类的平滑过渡
    3. 混合路由策略：结合内容和动态文化亲和性
    4. 自适应温度调节：随训练进程调整聚类硬度
    """

    def __init__(self, llama_model, config, args: ModelArgs, culture_loss_lambda=-1,
                 moe_fusion=0.4, num_cultures=6, culture_dim=256, use_gate=True):
        # 调用父类初始化
        super().__init__(llama_model, config, args, culture_loss_lambda, moe_fusion)

        self.num_cultures = num_cultures
        self.culture_dim = culture_dim
        self.use_gate = use_gate
        hidden_dim = self.config.hidden_size

        logging.info(f"Initializing Dynamic Enhanced CultureMoE with {args.num_experts} experts and {num_cultures} cultures")

        # ✅ 1. 文化嵌入层（保持原有）
        self.cultural_embedding = CulturalEmbeddingLayer(
            num_cultures=num_cultures,
            culture_dim=culture_dim,
            hidden_dim=hidden_dim
        )

        # ✅ 2. 替换为动态文化感知路由器
        self.router = DynamicCulturalAwareRouter(
            hidden_dim=hidden_dim,
            num_experts=args.num_experts,
            num_cultures=num_cultures,
            culture_dim=culture_dim,
            router_hidden_dim=args.router_hidden_dim,
            dropout=args.dropout
        )

        # ✅ 3. 文化特定专家（保持原有架构，但专家分配将动态调整）
        self.cultural_experts = nn.ModuleList([
            CultureSpecificExpert(
                expert_id=i,
                primary_culture_ids=[],  # 初始为空，将通过动态聚类分配
                hidden_dim=hidden_dim,
                expert_hidden_dim=args.experts_hidden_dim,
                lora_rank=args.lora_rank,
                culture_dim=culture_dim,
                dropout=args.dropout
            ) for i in range(args.num_experts)
        ])

        # 移除原有的 experts_layer（如果存在）
        if hasattr(self, 'experts_layer'):
            delattr(self, 'experts_layer')

        # ✅ 4. 文化上下文感知（保持原有）
        self.cultural_context = CulturalContextAwareness(
            hidden_dim=hidden_dim,
            num_cultures=num_cultures,
            context_dim=512,
            dropout=args.dropout
        )

        # ✅ 5. 文化感知的门控机制（可选）
        if self.use_gate:
            self.cultural_gate = nn.Sequential(
                nn.Linear(hidden_dim + culture_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            logging.info("Cultural gate mechanism enabled")
        else:
            self.cultural_gate = None
            logging.info("Cultural gate mechanism disabled")

        # 移除原有的 gate_linear 和 gate_sigmoid（如果存在）
        if hasattr(self, 'gate_linear'):
            delattr(self, 'gate_linear')
        if hasattr(self, 'gate_sigmoid'):
            delattr(self, 'gate_sigmoid')

        # ✅ 6. 动态文化损失增强
        self.culture_loss_alpha_enhanced = nn.Parameter(torch.tensor(2.0))
        self.culture_loss_beta_enhanced = nn.Parameter(torch.tensor(1.0))

        # 动态损失权重
        self.dynamic_loss_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.02]))  # consistency, diversity, strength

        # 初始化新增组件
        self._init_enhanced_components()

        logging.info("Dynamic Enhanced CultureMoE initialization completed")

    def _init_enhanced_components(self):
        """初始化增强组件"""
        if self.use_gate and self.cultural_gate is not None:
            for module in self.cultural_gate:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=0.001)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, -1.0)

    def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None,
                attention_mask_mask=None, labels=None, culture_labels=None, culture_ids=None,
                culture_ids_multi=None, use_culture_loss=False, culture_loss_lambda=0.5,
                culture_loss_alpha=2.0, culture_loss_beta=1.0, use_shared_experts=True,
                router_temperature=2.0, load_balance_weight=0.01, entropy_weight=0.1,
                current_epoch=1, total_epochs=4, **kwargs):
        """
        动态增强的前向传播

        Args:
            current_epoch: 当前epoch (1-based)
            total_epochs: 总epoch数
            其他参数: 与原模型相同
        """
        # DataParallel兼容性处理
        is_dataparallel = hasattr(self, 'module')

        device = input_ids.device
        dtype = self.shared[0].weight.dtype

        # 处理culture_ids
        if culture_ids is None:
            if culture_labels is not None:
                if culture_labels.device != device:
                    culture_labels = culture_labels.to(device)
                culture_ids = culture_labels
            else:
                culture_ids = torch.zeros(input_ids.shape[0], dtype=torch.long, device=device)

        if culture_ids is not None and culture_ids.device != device:
            culture_ids = culture_ids.to(device)

        # ✅ Step 1: LLaMA forward for h_all
        outputs_all = self.llama_model.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_all = outputs_all.last_hidden_state  # [B, L, H]
        if hidden_all.device != device:
            hidden_all = hidden_all.to(device)
        if hidden_all.dtype != dtype:
            hidden_all = hidden_all.to(dtype)

        h_all = hidden_all.clone()

        # ✅ Step 2: LLaMA forward for h_no (如果使用shared专家)
        if use_shared_experts and input_ids_mask is not None:
            outputs_no = self.llama_model.model(
                input_ids_mask,
                attention_mask=attention_mask_mask,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_no = outputs_no.last_hidden_state
            if hidden_no.device != device:
                hidden_no = hidden_no.to(device)
            if hidden_no.dtype != dtype:
                hidden_no = hidden_no.to(dtype)

            h_no = hidden_no.clone()
        else:
            h_no = h_all.clone()

        # ✅ Step 3: 文化嵌入增强
        if next(self.cultural_embedding.parameters()).device != device:
            self.cultural_embedding = self.cultural_embedding.to(device)
        culturally_embedded_states, culture_attention = self.cultural_embedding(
            h_all, culture_ids
        )

        # ✅ Step 4: 文化上下文感知
        if next(self.cultural_context.parameters()).device != device:
            self.cultural_context = self.cultural_context.to(device)
        context_aware_states, cultural_analysis = self.cultural_context(
            culturally_embedded_states, culture_ids
        )

        # ✅ Step 5: Shared 层处理
        if use_shared_experts:
            h_no = h_no.to(device=device, dtype=dtype)
            shared_out = self.shared(h_no)
        else:
            shared_out = torch.zeros_like(context_aware_states)

        # ✅ Step 6: 动态文化感知路由
        if use_shared_experts:
            pooled = shared_out.mean(dim=1)  # [B, H]
        else:
            pooled = context_aware_states.mean(dim=1)  # [B, H]

        # 确保路由器在正确的设备上
        if next(self.router.parameters()).device != device:
            self.router = self.router.to(device)

        # 使用动态路由器
        expert_weights, routing_info = self.router(
            pooled, culture_ids, current_epoch, router_temperature, total_epochs
        )

        # ✅ Step 7: 文化特定专家处理
        expert_outputs = []
        culture_relevances = []

        # 从动态聚类获取当前的专家-文化分配
        culture_affinities = routing_info['culture_affinities']  # [B, num_experts]

        for i, expert in enumerate(self.cultural_experts):
            # 确保专家在正确的设备上
            if next(expert.parameters()).device != device:
                self.cultural_experts[i] = expert.to(device)

            # 动态更新专家的文化相关性（基于当前batch的聚类结果）
            batch_relevance = culture_affinities[:, i]  # [B]

            output, _ = expert(context_aware_states, culture_ids)
            expert_outputs.append(output)
            culture_relevances.append(batch_relevance)

        # ✅ Step 8: MoE 预热权重
        moe_warmup_weight = self.get_moe_warmup_weight()

        # ✅ Step 9: 专家输出加权融合
        weighted_expert_outs = []
        for i in range(len(expert_outputs)):
            expert_out = expert_outputs[i]
            if torch.isnan(expert_out).any() or torch.isinf(expert_out).any():
                logging.warning(f"⚠️  Expert {i} output contains NaN/Inf, replacing with zeros")
                expert_out = torch.zeros_like(expert_out)

            weight = expert_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            if torch.isnan(weight).any() or torch.isinf(weight).any():
                logging.warning(f"⚠️  Expert {i} weight contains NaN/Inf, replacing with uniform weight")
                weight = torch.ones_like(weight) / len(expert_outputs)

            weighted_out = expert_out * weight
            weighted_expert_outs.append(weighted_out)

        expert_sum = torch.stack(weighted_expert_outs, dim=0).sum(dim=0)

        if torch.isnan(expert_sum).any() or torch.isinf(expert_sum).any():
            logging.error("⚠️  Expert sum contains NaN/Inf, replacing with zeros")
            expert_sum = torch.zeros_like(expert_sum)

        # ✅ Step 10: 处理序列长度不匹配
        if shared_out.size(1) != expert_sum.size(1):
            min_len = min(shared_out.size(1), expert_sum.size(1))
            shared_out = shared_out[:, :min_len, :]
            expert_sum = expert_sum[:, :min_len, :]

        # ✅ Step 11: 文化感知门控机制
        if self.use_gate and self.cultural_gate is not None:
            if next(self.cultural_gate.parameters()).device != device:
                self.cultural_gate = self.cultural_gate.to(device)

            # 获取文化特征
            culture_features = routing_info['culture_features']  # [B, culture_dim]
            culture_features_expanded = culture_features.unsqueeze(1).expand(-1, shared_out.size(1), -1)

            gate_input = torch.cat([shared_out, culture_features_expanded], dim=-1)
            cultural_gate = self.cultural_gate(gate_input)

            moe_fusion_alpha = self.moe_fusion_alpha.to(device=cultural_gate.device, dtype=cultural_gate.dtype)
            moe_gated = cultural_gate * moe_fusion_alpha * expert_sum
            enhanced_hidden = shared_out + moe_warmup_weight * moe_gated
        else:
            moe_fusion_alpha = self.moe_fusion_alpha.to(device=expert_sum.device, dtype=expert_sum.dtype)
            moe_direct = moe_fusion_alpha * expert_sum
            enhanced_hidden = shared_out + moe_warmup_weight * moe_direct

        # ✅ Step 12: 检查数值稳定性
        if torch.isnan(enhanced_hidden).any() or torch.isinf(enhanced_hidden).any():
            logging.error("⚠️  Enhanced hidden states contain NaN/Inf, attempting repair")
            enhanced_hidden = torch.nan_to_num(enhanced_hidden, nan=0.0, posinf=10.0, neginf=-10.0)
            enhanced_hidden = torch.clamp(enhanced_hidden, min=-50.0, max=50.0)

        # 生成logits
        lm_head_dtype = self.llama_model.lm_head.weight.dtype
        if enhanced_hidden.dtype != lm_head_dtype:
            enhanced_hidden = enhanced_hidden.to(lm_head_dtype)

        logits = self.llama_model.lm_head(enhanced_hidden)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.error("⚠️  Logits contain NaN/Inf, attempting repair")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        logits = torch.clamp(logits, min=-100, max=100)

        # ✅ Step 13: 构建输出
        outputs = {
            'logits': logits,
            'expert_weights': expert_weights
        }

        # 添加动态路由信息
        if not is_dataparallel:
            outputs['cultural_analysis'] = cultural_analysis
            outputs['culture_attention'] = culture_attention
            outputs['routing_info'] = routing_info
            outputs['culture_relevances'] = torch.stack(culture_relevances, dim=1)
            outputs['dynamic_weight'] = routing_info['affinity_info']['dynamic_weight']
            outputs['clustering_info'] = routing_info['affinity_info']['clustering_info']
        else:
            outputs['culture_relevances'] = torch.stack(culture_relevances, dim=1)

        # ✅ Step 14: 计算损失
        if labels is not None:
            if labels.dim() == 1:
                raise ValueError(
                    f"Expected labels to be 2D [batch_size, seq_len] for generative training, "
                    f"but got 1D [batch_size]. Labels shape: {labels.shape}, Logits shape: {logits.shape}"
                )

            # 处理序列长度不匹配
            if labels.size(1) != logits.size(1):
                min_len = min(labels.size(1), logits.size(1))
                labels = labels[:, :min_len]
                logits = logits[:, :min_len, :]

            # 生成损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if shift_labels.device != shift_logits.device:
                shift_labels = shift_labels.to(shift_logits.device)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            generation_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs['generation_loss'] = generation_loss

            # 文化损失
            if use_culture_loss and culture_labels is not None:
                if culture_labels.device != device:
                    culture_labels = culture_labels.to(device)

                if is_dataparallel:
                    # DataParallel模式下使用基础文化损失
                    culture_loss = self.compute_culture_loss(
                        expert_weights=expert_weights,
                        culture_labels=culture_labels,
                        margin=culture_loss_alpha,
                        lambda_diff=culture_loss_beta
                    )
                else:
                    # 非DataParallel模式下使用动态增强的文化损失
                    culture_loss = self.compute_dynamic_enhanced_culture_loss(
                        expert_weights=expert_weights,
                        culture_labels=culture_labels,
                        routing_info=routing_info,
                        cultural_analysis=cultural_analysis,
                        culture_relevances=torch.stack(culture_relevances, dim=1),
                        culture_labels_multi=culture_ids_multi,
                        margin=culture_loss_alpha,
                        lambda_diff=culture_loss_beta
                    )

                if culture_loss.device != generation_loss.device:
                    culture_loss = culture_loss.to(generation_loss.device)

                outputs['culture_loss'] = culture_loss

                # 计算组件损失
                with torch.no_grad():
                    spec_loss, div_loss = self.compute_culture_loss_components(
                        expert_weights, culture_labels
                    )
                    outputs['specialization_loss'] = spec_loss
                    outputs['diversity_loss'] = div_loss

                # 动态聚类相关损失
                if not is_dataparallel:
                    routing_info['expert_weights'] = expert_weights  # 添加expert_weights到routing_info
                    dynamic_losses = self.router.compute_dynamic_culture_losses(routing_info)

                    # 添加到输出
                    for loss_name, loss_value in dynamic_losses.items():
                        outputs[loss_name] = loss_value

                # 传统的防塌陷损失
                load_balance_loss = self.router.compute_load_balancing_loss(expert_weights)
                entropy_loss = self.router.entropy_regularization(expert_weights)

                if load_balance_loss is None:
                    load_balance_loss = torch.tensor(0.0, device=generation_loss.device)
                if entropy_loss is None:
                    entropy_loss = torch.tensor(0.0, device=generation_loss.device)

                outputs['load_balance_loss'] = load_balance_loss
                outputs['entropy_loss'] = entropy_loss

                # 使用模型的可学习culture_loss_lambda
                lambda_value = self.culture_loss_lambda

                # 损失数值稳定性检查
                has_nan_loss = False
                if torch.isnan(generation_loss) or torch.isinf(generation_loss):
                    logging.error(f"⚠️  Generation loss is NaN/Inf: {generation_loss}")
                    has_nan_loss = True

                if torch.isnan(culture_loss) or torch.isinf(culture_loss):
                    logging.warning(f"⚠️  Culture loss is NaN/Inf: {culture_loss}")
                    has_nan_loss = True

                if torch.isnan(load_balance_loss) or torch.isinf(load_balance_loss):
                    logging.warning(f"⚠️  Load balance loss is NaN/Inf: {load_balance_loss}")
                    has_nan_loss = True

                if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
                    logging.warning(f"⚠️  Entropy loss is NaN/Inf: {entropy_loss}")
                    has_nan_loss = True

                if has_nan_loss:
                    logging.warning("⚠️  Detected NaN/Inf in loss components, returning NaN loss for batch skipping")
                    total_loss = torch.tensor(float('nan'), device=generation_loss.device, dtype=generation_loss.dtype)
                    outputs['loss'] = total_loss
                    return outputs

                # 安全的总损失计算
                culture_component = torch.clamp(lambda_value * culture_loss, min=-10.0, max=10.0)
                load_component = torch.clamp(load_balance_weight * load_balance_loss, min=-1.0, max=1.0)
                entropy_component = torch.clamp(entropy_weight * entropy_loss, min=-1.0, max=1.0)

                # 动态损失组件
                dynamic_component = torch.tensor(0.0, device=generation_loss.device)
                if not is_dataparallel and 'clustering_consistency_loss' in outputs:
                    consistency_loss = outputs['clustering_consistency_loss']
                    diversity_loss_dyn = outputs['clustering_diversity_loss']
                    strength_loss = outputs['culture_strength_loss']

                    # 应用动态损失权重
                    dynamic_weights = torch.sigmoid(self.dynamic_loss_weights)
                    dynamic_component = (
                        dynamic_weights[0] * consistency_loss +
                        dynamic_weights[1] * diversity_loss_dyn +
                        dynamic_weights[2] * strength_loss
                    )
                    dynamic_component = torch.clamp(dynamic_component, min=-1.0, max=1.0)

                total_loss = generation_loss + culture_component + load_component + entropy_component + dynamic_component

                # 最终损失检查
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logging.warning("⚠️  Total loss is NaN or Inf after component combination, returning NaN for batch skipping")
                    total_loss = torch.tensor(float('nan'), device=generation_loss.device, dtype=generation_loss.dtype)
                    outputs['loss'] = total_loss
                    return outputs
                elif total_loss < 0:
                    logging.warning(f"⚠️  Total loss is negative ({total_loss:.6f}), adjusting weights")
                    total_loss = (
                        generation_loss +
                        0.5 * culture_component +
                        0.001 * load_component +
                        0.01 * entropy_component +
                        0.1 * dynamic_component
                    )
                    if total_loss < 0:
                        logging.warning("⚠️  Total loss still negative, using generation loss only")
                        total_loss = generation_loss

                if not is_dataparallel:
                    outputs['culture_loss_lambda'] = lambda_value.item()
                    outputs['dynamic_loss_weights'] = dynamic_weights
                else:
                    outputs['culture_loss_lambda'] = lambda_value

            else:
                # 无文化损失的情况
                outputs['culture_loss'] = torch.tensor(0.0, device=generation_loss.device)
                outputs['specialization_loss'] = torch.tensor(0.0, device=generation_loss.device)
                outputs['diversity_loss'] = torch.tensor(0.0, device=generation_loss.device)

                load_balance_loss = self.router.compute_load_balancing_loss(expert_weights)
                entropy_loss = self.router.entropy_regularization(expert_weights)

                if load_balance_loss is None:
                    load_balance_loss = torch.tensor(0.0, device=generation_loss.device)
                if entropy_loss is None:
                    entropy_loss = torch.tensor(0.0, device=generation_loss.device)

                outputs['load_balance_loss'] = load_balance_loss
                outputs['entropy_loss'] = entropy_loss

                # 损失检查
                has_nan_loss = (torch.isnan(generation_loss) or torch.isinf(generation_loss) or
                              torch.isnan(load_balance_loss) or torch.isinf(load_balance_loss) or
                              torch.isnan(entropy_loss) or torch.isinf(entropy_loss))

                if has_nan_loss:
                    total_loss = torch.tensor(float('nan'), device=generation_loss.device, dtype=generation_loss.dtype)
                    outputs['loss'] = total_loss
                    return outputs

                load_component = torch.clamp(load_balance_weight * load_balance_loss, min=-1.0, max=1.0)
                entropy_component = torch.clamp(entropy_weight * entropy_loss, min=-1.0, max=1.0)

                total_loss = generation_loss + load_component + entropy_component

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    total_loss = torch.tensor(float('nan'), device=generation_loss.device, dtype=generation_loss.dtype)
                    outputs['loss'] = total_loss
                    return outputs
                elif total_loss < 0:
                    total_loss = generation_loss

            outputs['loss'] = total_loss

        return outputs

    def compute_dynamic_enhanced_culture_loss(self, expert_weights, culture_labels, routing_info,
                                            cultural_analysis, culture_relevances, culture_labels_multi=None,
                                            margin=0.5, lambda_diff=1.0, eps=1e-8):
        """
        动态增强的文化损失，结合动态聚类信息

        Args:
            expert_weights: [B, num_experts] 专家权重
            culture_labels: [B] 主要文化标签
            routing_info: dict 路由信息
            cultural_analysis: dict 文化分析结果
            culture_relevances: [B, num_experts] 专家文化相关性
            culture_labels_multi: List[List[int]] 多标签文化信息（可选）
            margin: 不同文化间的最小距离
            lambda_diff: 不同文化排斥力权重
            eps: 数值稳定性常数

        Returns:
            enhanced_culture_loss: 动态增强的文化损失
        """
        device = expert_weights.device
        dtype = expert_weights.dtype

        # 1. 基础对比损失
        base_culture_loss = self.compute_culture_loss(
            expert_weights, culture_labels, margin, lambda_diff, eps
        )

        # 2. 动态文化亲和性损失
        culture_affinities = routing_info['culture_affinities']  # [B, num_experts]

        # 鼓励专家权重与动态文化亲和性一致
        affinity_consistency_loss = F.mse_loss(expert_weights, culture_affinities)

        # 3. 文化特征质量损失
        culture_features = routing_info['culture_features']  # [B, culture_dim]
        culture_strength = routing_info['culture_strength']  # [B]

        # 鼓励文化特征的多样性和质量
        culture_feature_diversity = 0.0
        if culture_features.shape[0] > 1:
            # 计算文化特征之间的相似度
            normalized_features = F.normalize(culture_features, p=2, dim=1)
            similarities = torch.mm(normalized_features, normalized_features.t())
            # 除去对角线
            mask = ~torch.eye(culture_features.shape[0], dtype=torch.bool, device=device)
            off_diagonal_sims = similarities[mask]
            # 惩罚过高的相似度
            culture_feature_diversity = torch.clamp(off_diagonal_sims - 0.6, min=0).mean()

        # 4. 文化强度适中性损失
        target_strength = 0.5
        strength_loss = F.mse_loss(culture_strength,
                                 torch.full_like(culture_strength, target_strength))

        # 5. 聚类质量损失（从clustering_info获取）
        clustering_info = routing_info['affinity_info']['clustering_info']
        cluster_quality_loss = 0.0

        if 'affinity_entropy' in clustering_info:
            # 鼓励适中的亲和性熵（既不过于确定，也不过于随机）
            affinity_entropy = clustering_info['affinity_entropy']
            target_entropy = math.log(expert_weights.shape[1]) * 0.7  # 70%的最大熵
            entropy_deviation = torch.abs(affinity_entropy - target_entropy)
            cluster_quality_loss = entropy_deviation

        # 6. 动态权重调节
        dynamic_weight = routing_info['affinity_info']['dynamic_weight']

        # 在动态聚类权重较高时，更重视动态相关的损失
        dynamic_factor = dynamic_weight * 0.5 + 0.5  # 0.5到1.0之间

        # 7. 综合增强损失
        enhanced_culture_loss = (
            base_culture_loss +
            dynamic_factor * (
                0.2 * affinity_consistency_loss +
                0.1 * culture_feature_diversity +
                0.05 * strength_loss +
                0.03 * cluster_quality_loss
            )
        )

        return enhanced_culture_loss

    def get_culture_expert_info(self):
        """获取文化专家信息（动态版本）"""
        info = []

        # 获取当前的聚类中心
        cluster_centers = self.router.culture_clustering.culture_cluster_centers

        for i, expert in enumerate(self.cultural_experts):
            # 分析聚类中心来推断专家的文化倾向
            center = cluster_centers[i]

            # 计算与各个文化的相似度（简化版本）
            culture_similarities = []
            for culture_id in range(self.num_cultures):
                # 这里使用简化的相似度计算
                # 在实际应用中，可能需要更复杂的分析
                sim = torch.cosine_similarity(
                    center.unsqueeze(0),
                    torch.randn(1, self.culture_dim, device=center.device) * 0.1,
                    dim=1
                ).item()
                culture_similarities.append(sim)

            # 找到最相关的文化
            top_cultures = sorted(range(len(culture_similarities)),
                                key=lambda x: culture_similarities[x], reverse=True)[:3]

            info.append({
                'expert_id': i,
                'primary_cultures': top_cultures,
                'role': f"Dynamic Cultures {top_cultures}",
                'param_count': sum(p.numel() for p in expert.parameters()),
                'cluster_center_norm': torch.norm(center).item(),
                'culture_similarities': culture_similarities
            })

        return info

    def print_trainable_parameters(self):
        """打印可训练参数信息（动态版本）"""
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        # 统计所有参数
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()

        # 统计各组件参数
        cultural_expert_params = 0
        router_params = 0
        cultural_embedding_params = 0
        cultural_context_params = 0
        cultural_gate_params = 0
        shared_params = 0
        base_model_params = 0
        dynamic_clustering_params = 0

        for name, param in self.named_parameters():
            if 'cultural_experts' in name:
                cultural_expert_params += param.numel()
            elif 'router' in name:
                router_params += param.numel()
                if 'culture_clustering' in name or 'culture_feature_extractor' in name:
                    dynamic_clustering_params += param.numel()
            elif 'cultural_embedding' in name:
                cultural_embedding_params += param.numel()
            elif 'cultural_context' in name:
                cultural_context_params += param.numel()
            elif 'cultural_gate' in name:
                cultural_gate_params += param.numel()
            elif 'shared' in name:
                shared_params += param.numel()
            else:
                base_model_params += param.numel()

        trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0

        print("=" * 60)
        print("📊 Dynamic Enhanced CultureMoE Parameter Statistics")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable percentage: {trainable_percentage:.4f}%")
        print("")
        print("Component breakdown:")
        print(f"  Base model (LLaMA): {base_model_params:,}")
        print(f"  Cultural experts: {cultural_expert_params:,}")
        print(f"  Cultural router: {router_params:,}")
        print(f"    └─ Dynamic clustering: {dynamic_clustering_params:,}")
        print(f"  Cultural embedding: {cultural_embedding_params:,}")
        print(f"  Cultural context: {cultural_context_params:,}")
        print(f"  Cultural gate: {cultural_gate_params:,}")
        print(f"  Shared layers: {shared_params:,}")
        print("=" * 60)