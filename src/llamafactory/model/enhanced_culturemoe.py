# src/llamafactory/model/enhanced_culturemoe.py
"""
增强的文化感知 CultureMoE 模型
集成了文化嵌入、文化感知路由器、文化特定专家等组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .CultureMoE import LlamaSharedRouterExpertsModel
from .cultural_components import (
    CulturalEmbeddingLayer,
    CulturalAwareRouter,
    CultureSpecificExpert,
    CulturalContextAwareness,
    create_culture_assignments
)
from .moe_args import ModelArgs


class EnhancedCultureMoE(LlamaSharedRouterExpertsModel):
    """
    增强的文化感知 MoE 模型

    新增功能：
    1. 文化嵌入层：显式文化表示
    2. 文化感知路由器：多维度路由决策
    3. 文化特定专家：专门的文化知识处理
    4. 文化上下文感知：文化线索检测和跨文化理解
    """

    def __init__(self, llama_model, config, args: ModelArgs, culture_loss_lambda=-1,
                 moe_fusion=0.4, num_cultures=6, culture_dim=256, use_gate=True):
        # 调用父类初始化，但不使用其 router 和 experts_layer
        super().__init__(llama_model, config, args, culture_loss_lambda, moe_fusion)

        self.num_cultures = num_cultures
        self.culture_dim = culture_dim
        self.use_gate = use_gate
        hidden_dim = self.config.hidden_size

        logging.info(f"Initializing Enhanced CultureMoE with {args.num_experts} experts and {num_cultures} cultures, gate={use_gate}")

        # ✅ 1. 文化嵌入层
        self.cultural_embedding = CulturalEmbeddingLayer(
            num_cultures=num_cultures,
            culture_dim=culture_dim,
            hidden_dim=hidden_dim
        )

        # ✅ 2. 替换原有路由器为文化感知路由器
        self.router = CulturalAwareRouter(
            hidden_dim=hidden_dim,
            num_experts=args.num_experts,
            num_cultures=num_cultures,
            culture_dim=culture_dim,
            router_hidden_dim=args.router_hidden_dim,
            dropout=args.dropout
        )

        # ✅ 3. 创建文化分配方案
        self.culture_assignments = create_culture_assignments(args.num_experts, num_cultures)

        # ✅ 4. 替换原有专家为文化特定专家
        self.cultural_experts = nn.ModuleList([
            CultureSpecificExpert(
                expert_id=i,
                primary_culture_ids=self.culture_assignments[i],
                hidden_dim=hidden_dim,
                expert_hidden_dim=args.experts_hidden_dim,
                lora_rank=args.lora_rank,
                culture_dim=culture_dim,
                dropout=args.dropout
            ) for i in range(args.num_experts)
        ])

        # 移除原有的 experts_layer
        delattr(self, 'experts_layer')

        # ✅ 5. 文化上下文感知
        self.cultural_context = CulturalContextAwareness(
            hidden_dim=hidden_dim,
            num_cultures=num_cultures,
            context_dim=512,
            dropout=args.dropout
        )

        # ✅ 6. 文化感知的门控机制（可选，替换原有gate_linear）
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
            logging.info("Cultural gate mechanism disabled for ablation study")

        # 移除原有的 gate_linear 和 gate_sigmoid
        delattr(self, 'gate_linear')
        delattr(self, 'gate_sigmoid')

        # ✅ 7. 文化损失增强
        self.culture_loss_alpha_enhanced = nn.Parameter(torch.tensor(2.0))
        self.culture_loss_beta_enhanced = nn.Parameter(torch.tensor(1.0))

        # 初始化新增组件
        self._init_enhanced_components()

        logging.info("Enhanced CultureMoE initialization completed")
        self._log_culture_assignments()

    def _init_enhanced_components(self):
        """初始化增强组件"""
        # 文化感知门控初始化（仅在启用时）
        if self.use_gate and self.cultural_gate is not None:
            for module in self.cultural_gate:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=0.001)
                    if module.bias is not None:
                        # 门控的bias初始化为负值，确保初期影响较小
                        nn.init.constant_(module.bias, -1.0)

    def _log_culture_assignments(self):
        """记录文化分配方案"""
        logging.info("Culture assignments for experts:")
        for i, assignment in enumerate(self.culture_assignments):
            if len(assignment) == 0:
                logging.info(f"  Expert {i}: Conflict resolution specialist")
            elif len(assignment) == self.num_cultures:
                logging.info(f"  Expert {i}: Cross-cultural generalist")
            else:
                logging.info(f"  Expert {i}: Cultures {assignment}")

    def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None,
                attention_mask_mask=None, labels=None, culture_labels=None, culture_ids=None,
                culture_ids_multi=None, use_culture_loss=False, culture_loss_lambda=0.5,
                culture_loss_alpha=2.0, culture_loss_beta=1.0, use_shared_experts=True,
                router_temperature=2.0, load_balance_weight=0.01, entropy_weight=0.1, **kwargs):
        """
        增强的前向传播，包含文化感知组件

        Args:
            input_ids: [B, L] 输入token IDs
            attention_mask: [B, L] 注意力掩码
            input_ids_mask: [B, L] mask输入（用于shared专家）
            attention_mask_mask: [B, L] mask注意力掩码
            labels: [B, L] 生成标签
            culture_labels: [B] 文化标签（用于文化损失）
            culture_ids: [B] 文化ID（用于文化感知组件）
            culture_ids_multi: List[List[int]] 多标签文化ID（可选）
            use_culture_loss: 是否使用文化损失
            其他参数: 与原模型相同

        Returns:
            outputs: dict 包含所有输出和分析结果
        """
        # DataParallel兼容性处理
        is_dataparallel = hasattr(self, 'module')

        device = input_ids.device
        dtype = self.shared[0].weight.dtype

        # 如果没有提供culture_ids，使用culture_labels
        if culture_ids is None:
            if culture_labels is not None:
                # 确保culture_labels在正确的设备上
                if culture_labels.device != device:
                    culture_labels = culture_labels.to(device)
                culture_ids = culture_labels
            else:
                culture_ids = torch.zeros(input_ids.shape[0], dtype=torch.long, device=device)

        # 确保culture_ids在正确的设备上
        if culture_ids is not None and culture_ids.device != device:
            culture_ids = culture_ids.to(device)

        # ✅ Step 1: LLaMA forward for h_all (instruction + input)
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

        # ✅ Step 2: LLaMA forward for h_no (instruction_mask + input) - 只有使用shared专家时
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
        # 确保文化嵌入组件在正确的设备上
        if next(self.cultural_embedding.parameters()).device != device:
            self.cultural_embedding = self.cultural_embedding.to(device)
        culturally_embedded_states, culture_attention = self.cultural_embedding(
            h_all, culture_ids
        )

        # ✅ Step 4: 文化上下文感知
        # 确保文化上下文组件在正确的设备上
        if next(self.cultural_context.parameters()).device != device:
            self.cultural_context = self.cultural_context.to(device)
        context_aware_states, cultural_analysis = self.cultural_context(
            culturally_embedded_states, culture_ids
        )

        # ✅ Step 5: Shared 层处理
        if use_shared_experts:
            h_no = h_no.to(device=device, dtype=dtype)
            shared_out = self.shared(h_no)  # [B, L, H]
        else:
            shared_out = torch.zeros_like(context_aware_states)

        # ✅ Step 6: 文化感知路由
        if use_shared_experts:
            pooled = shared_out.mean(dim=1)  # [B, H]
        else:
            pooled = context_aware_states.mean(dim=1)  # [B, H]

        # 确保路由器在正确的设备上
        if next(self.router.parameters()).device != device:
            self.router = self.router.to(device)
        expert_weights, routing_info = self.router(pooled, culture_ids, router_temperature)

        # 为DataParallel兼容性提取final_logits
        final_logits = routing_info['final_logits']

        # ✅ Step 7: 文化特定专家处理
        expert_outputs = []
        culture_relevances = []
        for i, expert in enumerate(self.cultural_experts):
            # 确保每个专家在正确的设备上
            if next(expert.parameters()).device != device:
                self.cultural_experts[i] = expert.to(device)
            output, relevance = expert(context_aware_states, culture_ids)
            expert_outputs.append(output)
            culture_relevances.append(relevance)

        # ✅ Step 8: MoE 预热权重
        moe_warmup_weight = self.get_moe_warmup_weight()

        # ✅ Step 9: 专家输出加权融合
        weighted_expert_outs = []
        for i in range(len(expert_outputs)):
            # 检查专家输出是否有异常
            expert_out = expert_outputs[i]
            if torch.isnan(expert_out).any() or torch.isinf(expert_out).any():
                logging.warning(f"⚠️  Expert {i} output contains NaN/Inf, replacing with zeros")
                expert_out = torch.zeros_like(expert_out)

            # 检查专家权重是否有异常
            weight = expert_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            if torch.isnan(weight).any() or torch.isinf(weight).any():
                logging.warning(f"⚠️  Expert {i} weight contains NaN/Inf, replacing with uniform weight")
                weight = torch.ones_like(weight) / len(expert_outputs)

            weighted_out = expert_out * weight
            weighted_expert_outs.append(weighted_out)

        expert_sum = torch.stack(weighted_expert_outs, dim=0).sum(dim=0)  # [B, L, H]

        # 检查融合后的专家输出
        if torch.isnan(expert_sum).any() or torch.isinf(expert_sum).any():
            logging.error("⚠️  Expert sum contains NaN/Inf, replacing with zeros")
            expert_sum = torch.zeros_like(expert_sum)

        # ✅ Step 10: 处理序列长度不匹配
        if shared_out.size(1) != expert_sum.size(1):
            min_len = min(shared_out.size(1), expert_sum.size(1))
            shared_out = shared_out[:, :min_len, :]
            expert_sum = expert_sum[:, :min_len, :]

        # ✅ Step 11: 文化感知门控机制（可选）
        if self.use_gate and self.cultural_gate is not None:
            # 确保文化门控在正确的设备上
            if next(self.cultural_gate.parameters()).device != device:
                self.cultural_gate = self.cultural_gate.to(device)
            # 获取文化嵌入
            culture_emb = self.cultural_embedding.culture_embeddings(culture_ids)  # [B, culture_dim]
            culture_emb_expanded = culture_emb.unsqueeze(1).expand(-1, shared_out.size(1), -1)  # [B, L, culture_dim]

            # 拼接shared输出和文化嵌入
            gate_input = torch.cat([shared_out, culture_emb_expanded], dim=-1)  # [B, L, H + culture_dim]
            cultural_gate = self.cultural_gate(gate_input)  # [B, L, H]

            # 应用文化感知门控
            moe_fusion_alpha = self.moe_fusion_alpha.to(device=cultural_gate.device, dtype=cultural_gate.dtype)
            moe_gated = cultural_gate * moe_fusion_alpha * expert_sum  # [B, L, H]
            enhanced_hidden = shared_out + moe_warmup_weight * moe_gated  # [B, L, H]
        else:
            # 不使用门控机制：直接融合专家输出
            moe_fusion_alpha = self.moe_fusion_alpha.to(device=expert_sum.device, dtype=expert_sum.dtype)
            moe_direct = moe_fusion_alpha * expert_sum  # [B, L, H]
            enhanced_hidden = shared_out + moe_warmup_weight * moe_direct  # [B, L, H]

        # ✅ Step 12: 检查enhanced_hidden的数值稳定性
        if torch.isnan(enhanced_hidden).any() or torch.isinf(enhanced_hidden).any():
            logging.error("⚠️  Enhanced hidden states contain NaN/Inf, attempting repair")
            enhanced_hidden = torch.nan_to_num(enhanced_hidden, nan=0.0, posinf=10.0, neginf=-10.0)
            enhanced_hidden = torch.clamp(enhanced_hidden, min=-50.0, max=50.0)

        # 生成logits
        lm_head_dtype = self.llama_model.lm_head.weight.dtype
        if enhanced_hidden.dtype != lm_head_dtype:
            enhanced_hidden = enhanced_hidden.to(lm_head_dtype)

        logits = self.llama_model.lm_head(enhanced_hidden)  # [B, L, vocab_size]

        # ✅ Step 13: 检查和修复异常值
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.error("⚠️  Logits contain NaN/Inf, attempting repair")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        logits = torch.clamp(logits, min=-100, max=100)

        # ✅ Step 14: 构建输出
        outputs = {
            'logits': logits,
            'expert_weights': expert_weights
        }

        # DataParallel兼容性：只在非DataParallel模式下添加复杂类型
        if not is_dataparallel:
            outputs['cultural_analysis'] = cultural_analysis
            outputs['culture_attention'] = culture_attention
            outputs['routing_info'] = routing_info
            outputs['culture_relevances'] = torch.stack(culture_relevances, dim=1)
            outputs['culture_assignments'] = self.culture_assignments
        else:
            # DataParallel模式下只返回基本tensor
            outputs['culture_relevances'] = torch.stack(culture_relevances, dim=1)

        # ✅ Step 15: 计算损失
        if labels is not None:
            # 检查labels维度
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
                # 确保culture_labels在正确的设备上
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
                    # 非DataParallel模式下使用增强的文化损失
                    culture_loss = self.compute_enhanced_culture_loss(
                        expert_weights=expert_weights,
                        culture_labels=culture_labels,
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

                # 防塌陷损失
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

                # 损失数值稳定性检查（在计算总损失前）
                # 检查各个组件损失是否有异常
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

                # 检查lambda_value是否异常
                if torch.isnan(lambda_value) or torch.isinf(lambda_value):
                    logging.warning(f"⚠️  Lambda value is NaN/Inf: {lambda_value}")
                    has_nan_loss = True

                # 如果检测到任何NaN/Inf，返回NaN损失让训练循环跳过这个batch
                if has_nan_loss:
                    logging.warning("⚠️  Detected NaN/Inf in loss components, returning NaN loss for batch skipping")
                    total_loss = torch.tensor(float('nan'), device=generation_loss.device, dtype=generation_loss.dtype)
                    outputs['loss'] = total_loss
                    return outputs

                # 安全的总损失计算（使用torch.clamp防止溢出）
                culture_component = torch.clamp(lambda_value * culture_loss, min=-10.0, max=10.0)
                load_component = torch.clamp(load_balance_weight * load_balance_loss, min=-1.0, max=1.0)
                entropy_component = torch.clamp(entropy_weight * entropy_loss, min=-1.0, max=1.0)

                total_loss = generation_loss + culture_component + load_component + entropy_component

                # 最终损失检查和修正
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logging.warning("⚠️  Total loss is NaN or Inf after component combination, returning NaN for batch skipping")
                    total_loss = torch.tensor(float('nan'), device=generation_loss.device, dtype=generation_loss.dtype)
                    outputs['loss'] = total_loss
                    return outputs
                elif total_loss < 0:
                    logging.warning(f"⚠️  Total loss is negative ({total_loss:.6f}), adjusting weights")
                    # 如果总损失为负，减少正则化项的权重
                    total_loss = (
                        generation_loss +
                        0.5 * culture_component +    # 减少文化损失权重
                        0.001 * load_component +     # 减少负载均衡权重
                        0.01 * entropy_component     # 减少熵权重
                    )
                    if total_loss < 0:
                        # 如果仍为负，只使用生成损失
                        logging.warning("⚠️  Total loss still negative, using generation loss only")
                        total_loss = generation_loss

                # DataParallel兼容性：标量值处理
                if not is_dataparallel:
                    outputs['culture_loss_lambda'] = lambda_value.item()
                else:
                    outputs['culture_loss_lambda'] = lambda_value
            else:
                outputs['culture_loss'] = torch.tensor(0.0, device=generation_loss.device)
                outputs['specialization_loss'] = torch.tensor(0.0, device=generation_loss.device)
                outputs['diversity_loss'] = torch.tensor(0.0, device=generation_loss.device)

                # 防塌陷损失
                load_balance_loss = self.router.compute_load_balancing_loss(expert_weights)
                entropy_loss = self.router.entropy_regularization(expert_weights)

                if load_balance_loss is None:
                    load_balance_loss = torch.tensor(0.0, device=generation_loss.device)
                if entropy_loss is None:
                    entropy_loss = torch.tensor(0.0, device=generation_loss.device)

                outputs['load_balance_loss'] = load_balance_loss
                outputs['entropy_loss'] = entropy_loss

                # 损失数值稳定性检查（无文化损失情况）
                has_nan_loss_no_culture = False
                if torch.isnan(generation_loss) or torch.isinf(generation_loss):
                    logging.error(f"⚠️  Generation loss is NaN/Inf: {generation_loss}")
                    has_nan_loss_no_culture = True

                if torch.isnan(load_balance_loss) or torch.isinf(load_balance_loss):
                    logging.warning(f"⚠️  Load balance loss is NaN/Inf: {load_balance_loss}")
                    has_nan_loss_no_culture = True

                if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
                    logging.warning(f"⚠️  Entropy loss is NaN/Inf: {entropy_loss}")
                    has_nan_loss_no_culture = True

                # 如果检测到任何NaN/Inf，返回NaN损失让训练循环跳过这个batch
                if has_nan_loss_no_culture:
                    logging.warning("⚠️  Detected NaN/Inf in loss components (no culture), returning NaN loss for batch skipping")
                    total_loss = torch.tensor(float('nan'), device=generation_loss.device, dtype=generation_loss.dtype)
                    outputs['loss'] = total_loss
                    return outputs

                # 安全的总损失计算
                load_component = torch.clamp(load_balance_weight * load_balance_loss, min=-1.0, max=1.0)
                entropy_component = torch.clamp(entropy_weight * entropy_loss, min=-1.0, max=1.0)

                total_loss = generation_loss + load_component + entropy_component

                # 最终损失检查和修正
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logging.warning("⚠️  Total loss is NaN or Inf after component combination (no culture), returning NaN for batch skipping")
                    total_loss = torch.tensor(float('nan'), device=generation_loss.device, dtype=generation_loss.dtype)
                    outputs['loss'] = total_loss
                    return outputs
                elif total_loss < 0:
                    logging.warning(f"⚠️  Total loss is negative ({total_loss:.6f}), adjusting weights")
                    # 如果总损失为负，减少正则化项的权重
                    total_loss = (
                        generation_loss +
                        0.001 * load_component +     # 减少负载均衡权重
                        0.01 * entropy_component     # 减少熵权重
                    )
                    if total_loss < 0:
                        # 如果仍为负，只使用生成损失
                        logging.warning("⚠️  Total loss still negative, using generation loss only")
                        total_loss = generation_loss

            outputs['loss'] = total_loss

        return outputs

    def compute_enhanced_culture_loss(self, expert_weights, culture_labels, cultural_analysis,
                                    culture_relevances, culture_labels_multi=None, margin=0.5, lambda_diff=1.0, eps=1e-8):
        """
        增强的文化损失，结合多个文化感知组件的信息

        Args:
            expert_weights: [B, num_experts] 专家权重
            culture_labels: [B] 主要文化标签
            cultural_analysis: dict 文化分析结果
            culture_relevances: [B, num_experts] 专家文化相关性
            culture_labels_multi: List[List[int]] 多标签文化信息（可选）
            margin: 不同文化间的最小距离
            lambda_diff: 不同文化排斥力权重
            eps: 数值稳定性常数

        Returns:
            enhanced_culture_loss: 增强的文化损失
        """
        device = expert_weights.device
        dtype = expert_weights.dtype

        # 1. 基础对比损失（继承自原模型）
        base_culture_loss = self.compute_culture_loss(
            expert_weights, culture_labels, margin, lambda_diff, eps
        )

        # 2. 文化相关性损失（支持多标签）
        # 鼓励高文化相关性的专家获得更高权重
        relevance_loss = 0.0
        batch_size = expert_weights.shape[0]

        for b in range(batch_size):
            expert_weight = expert_weights[b]  # [num_experts]
            relevance = culture_relevances[b]  # [num_experts]

            # 使用多标签信息（如果可用）
            if culture_labels_multi is not None and b < len(culture_labels_multi):
                culture_ids = culture_labels_multi[b]  # List[int]
                # 对于多标签情况，计算所有相关文化的平均相关性
                total_relevance = 0.0
                for culture_id in culture_ids:
                    # 这里需要重新计算每个文化的相关性
                    # 简化处理：使用现有的relevance向量
                    total_relevance += torch.sum(expert_weight * relevance)
                weighted_relevance = total_relevance / len(culture_ids)
            else:
                # 单标签情况
                weighted_relevance = torch.sum(expert_weight * relevance)

            relevance_loss += (1.0 - weighted_relevance) ** 2

        relevance_loss = relevance_loss / batch_size

        # 3. 文化冲突惩罚
        # 当检测到文化冲突时，增加损失以鼓励更谨慎的处理
        conflict_penalty = cultural_analysis['conflict_probability'].mean()

        # 4. 文化敏感性损失（转换为正向损失）
        # 当内容具有文化敏感性时，如果专家使用不当，增加损失
        sensitivity_score = cultural_analysis['sensitivity_score'].mean()
        # 转换为损失：敏感性高但相关性低时损失大
        sensitivity_loss = torch.clamp(sensitivity_score * (1.0 - relevance_loss), min=0.0)

        # 5. 综合增强损失（全部为正值）
        enhanced_culture_loss = (
            base_culture_loss +
            0.1 * relevance_loss +
            0.05 * conflict_penalty +
            0.02 * sensitivity_loss  # 改为正向损失
        )

        return enhanced_culture_loss

    def get_culture_expert_info(self):
        """获取文化专家信息"""
        info = []
        for i, expert in enumerate(self.cultural_experts):
            assignment = self.culture_assignments[i]
            if len(assignment) == 0:
                role = "Conflict Resolution"
            elif len(assignment) == self.num_cultures:
                role = "Cross-Cultural Generalist"
            else:
                role = f"Cultures {assignment}"

            info.append({
                'expert_id': i,
                'primary_cultures': assignment,
                'role': role,
                'param_count': sum(p.numel() for p in expert.parameters())
            })
        return info

    def print_trainable_parameters(self):
        """打印可训练参数信息"""
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

        for name, param in self.named_parameters():
            if 'cultural_experts' in name:
                cultural_expert_params += param.numel()
            elif 'router' in name:
                router_params += param.numel()
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
        print("📊 Enhanced CultureMoE Parameter Statistics")
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
        print(f"  Cultural embedding: {cultural_embedding_params:,}")
        print(f"  Cultural context: {cultural_context_params:,}")
        print(f"  Cultural gate: {cultural_gate_params:,}")
        print(f"  Shared layers: {shared_params:,}")
        print("=" * 60)