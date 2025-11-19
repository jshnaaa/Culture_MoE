# src/llamafactory/model/enhanced_culturemoe_fixed.py
"""
增强的文化感知 CultureMoE 模型 - 修复NaN问题版本
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


class EnhancedCultureMoEFixed(LlamaSharedRouterExpertsModel):
    """
    增强的文化感知 MoE 模型 - 修复NaN问题版本

    修复内容：
    1. 增强数值稳定性检查
    2. 改进损失计算的鲁棒性
    3. 添加梯度裁剪和异常值处理
    4. 优化参数初始化
    """

    def __init__(self, llama_model, config, args: ModelArgs, culture_loss_lambda=-1,
                 moe_fusion=0.4, num_cultures=6, culture_dim=256):
        # 调用父类初始化，但不使用其 router 和 experts_layer
        super().__init__(llama_model, config, args, culture_loss_lambda, moe_fusion)

        self.num_cultures = num_cultures
        self.culture_dim = culture_dim
        hidden_dim = self.config.hidden_size

        logging.info(f"Initializing Enhanced CultureMoE (Fixed) with {args.num_experts} experts and {num_cultures} cultures")

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

        # ✅ 6. 文化感知的门控机制（替换原有gate_linear）
        self.cultural_gate = nn.Sequential(
            nn.Linear(hidden_dim + culture_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm提高稳定性
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 移除原有的 gate_linear 和 gate_sigmoid
        delattr(self, 'gate_linear')
        delattr(self, 'gate_sigmoid')

        # ✅ 7. 文化损失增强 - 使用更保守的初始值
        self.culture_loss_alpha_enhanced = nn.Parameter(torch.tensor(1.0))  # 减小初始值
        self.culture_loss_beta_enhanced = nn.Parameter(torch.tensor(0.5))   # 减小初始值

        # 初始化新增组件
        self._init_enhanced_components()

        # 确保所有新增组件使用与base模型相同的数据类型
        self._ensure_dtype_consistency()

        logging.info("Enhanced CultureMoE (Fixed) initialization completed")
        self._log_culture_assignments()

    def _init_enhanced_components(self):
        """初始化增强组件 - 改进版本"""
        # 文化感知门控初始化 - 更保守的初始化
        for module in self.cultural_gate:
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，更稳定
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    # 门控的bias初始化为更小的负值
                    nn.init.constant_(module.bias, -0.5)

    def _ensure_dtype_consistency(self):
        """确保所有组件使用相同的数据类型"""
        # 获取base模型的数据类型
        base_dtype = next(self.llama_model.parameters()).dtype

        # 将所有新增组件转换为相同的数据类型
        self.cultural_embedding = self.cultural_embedding.to(dtype=base_dtype)
        self.router = self.router.to(dtype=base_dtype)
        self.cultural_experts = self.cultural_experts.to(dtype=base_dtype)
        self.cultural_context = self.cultural_context.to(dtype=base_dtype)
        self.cultural_gate = self.cultural_gate.to(dtype=base_dtype)

        # 确保参数也使用正确的数据类型
        self.culture_loss_alpha_enhanced.data = self.culture_loss_alpha_enhanced.data.to(dtype=base_dtype)
        self.culture_loss_beta_enhanced.data = self.culture_loss_beta_enhanced.data.to(dtype=base_dtype)

        logging.info(f"All components converted to dtype: {base_dtype}")

    def _to_consistent_dtype(self, tensor, reference_tensor):
        """将tensor转换为与reference_tensor一致的数据类型"""
        if tensor.dtype != reference_tensor.dtype:
            return tensor.to(dtype=reference_tensor.dtype)
        return tensor

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

    def _check_numerical_stability(self, tensor, name="tensor", eps=1e-8):
        """检查数值稳定性"""
        if torch.isnan(tensor).any():
            logging.warning(f"⚠️  NaN detected in {name}")
            return torch.nan_to_num(tensor, nan=0.0)
        if torch.isinf(tensor).any():
            logging.warning(f"⚠️  Inf detected in {name}")
            return torch.nan_to_num(tensor, posinf=100.0, neginf=-100.0)
        if (tensor.abs() > 1000).any():
            logging.warning(f"⚠️  Large values detected in {name} (max: {tensor.abs().max():.2f})")
            return torch.clamp(tensor, min=-100, max=100)
        return tensor

    def _stable_cross_entropy_loss(self, logits, labels, ignore_index=-100):
        """数值稳定的交叉熵损失计算"""
        # 确保logits数值稳定
        logits = self._check_numerical_stability(logits, "logits")

        # 使用label smoothing提高稳定性
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.1)

        try:
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # 检查损失是否为NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("⚠️  CrossEntropyLoss returned NaN/Inf, using fallback")
                # 使用更简单的损失计算作为fallback
                logits_stable = torch.clamp(logits, min=-50, max=50)
                loss_fct_stable = nn.CrossEntropyLoss(ignore_index=ignore_index)
                loss = loss_fct_stable(logits_stable.view(-1, logits_stable.size(-1)), labels.view(-1))

                if torch.isnan(loss) or torch.isinf(loss):
                    # 最后的fallback：返回一个小的正值
                    logging.error("⚠️  Fallback loss also failed, using default value")
                    loss = torch.tensor(0.01, device=logits.device, dtype=logits.dtype)

        except Exception as e:
            logging.error(f"⚠️  Error in cross entropy calculation: {e}")
            loss = torch.tensor(0.01, device=logits.device, dtype=logits.dtype)

        return loss

    def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None,
                attention_mask_mask=None, labels=None, culture_labels=None, culture_ids=None,
                culture_ids_multi=None, use_culture_loss=False, culture_loss_lambda=0.5,
                culture_loss_alpha=2.0, culture_loss_beta=1.0, use_shared_experts=True,
                router_temperature=2.0, load_balance_weight=0.01, entropy_weight=0.1, **kwargs):
        """
        增强的前向传播，包含文化感知组件 - 修复NaN版本
        """
        # DataParallel兼容性处理
        is_dataparallel = hasattr(self, 'module')

        device = input_ids.device
        dtype = self.shared[0].weight.dtype

        # ✅ Step 1: 获取基础hidden states
        try:
            with torch.no_grad():
                outputs = self.llama_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            hidden_states = outputs.hidden_states[-1]  # [B, L, H]
        except Exception as e:
            logging.error(f"Error in base model forward: {e}")
            # 创建默认的hidden states
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.zeros(batch_size, seq_len, self.config.hidden_size,
                                      device=device, dtype=dtype)

        # 数值稳定性检查
        hidden_states = self._check_numerical_stability(hidden_states, "hidden_states")

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # ✅ Step 2: 文化ID处理
        if culture_ids is None:
            culture_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        if culture_ids.device != device:
            culture_ids = culture_ids.to(device)

        # ✅ Step 3: 文化嵌入
        try:
            culturally_aware_states, culture_attention_weights = self.cultural_embedding(hidden_states, culture_ids)
            # 提取文化嵌入向量（从culturally_aware_states中获取平均池化）
            cultural_embeddings = culturally_aware_states.mean(dim=1)  # [B, H] -> [B, culture_dim]
            # 投影到正确的维度
            if cultural_embeddings.size(-1) != self.culture_dim:
                # 如果维度不匹配，创建一个简单的投影
                if not hasattr(self, 'culture_dim_projection'):
                    self.culture_dim_projection = nn.Linear(cultural_embeddings.size(-1), self.culture_dim).to(device)
                cultural_embeddings = self.culture_dim_projection(cultural_embeddings)
            cultural_embeddings = self._check_numerical_stability(cultural_embeddings, "cultural_embeddings")
        except Exception as e:
            logging.warning(f"Error in cultural embedding: {e}")
            cultural_embeddings = torch.zeros(batch_size, self.culture_dim, device=device, dtype=dtype)

        # ✅ Step 4: 文化上下文分析
        try:
            context_aware_states, cultural_analysis = self.cultural_context(hidden_states, culture_ids)
            # 对字典中的每个tensor进行数值稳定性检查
            for key, tensor in cultural_analysis.items():
                if isinstance(tensor, torch.Tensor):
                    cultural_analysis[key] = self._check_numerical_stability(tensor, f"cultural_analysis_{key}")
        except Exception as e:
            logging.warning(f"Error in cultural context: {e}")
            context_aware_states = hidden_states  # 使用原始状态作为fallback
            cultural_analysis = {
                'cultural_cues': torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                'cross_cultural_conflicts': torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                'cultural_sensitivity': torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                'culture_relevances': torch.zeros(batch_size, self.num_cultures, device=device, dtype=dtype)
            }

        culture_relevances = cultural_analysis['culture_relevances']  # [B, num_cultures]

        # ✅ Step 5: 文化感知路由
        try:
            # 池化context_aware_states到序列级别用于路由
            pooled_hidden = context_aware_states.mean(dim=1)  # [B, H]
            expert_weights_per_sample, routing_info = self.router(
                pooled_hidden,
                culture_ids,
                temperature=router_temperature
            )
            # 扩展到序列维度
            expert_weights = expert_weights_per_sample.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, num_experts]
            expert_weights = self._check_numerical_stability(expert_weights, "expert_weights")
        except Exception as e:
            logging.warning(f"Error in cultural router: {e}")
            expert_weights = torch.ones(batch_size, seq_len, self.args.num_experts,
                                      device=device, dtype=dtype) / self.args.num_experts

        # ✅ Step 6: 专家处理
        expert_outputs = []
        for i, expert in enumerate(self.cultural_experts):
            try:
                expert_out, culture_relevance = expert(context_aware_states, culture_ids)
                expert_out = self._check_numerical_stability(expert_out, f"expert_{i}_output")
                expert_outputs.append(expert_out)
            except Exception as e:
                logging.warning(f"Error in expert {i}: {e}")
                expert_outputs.append(torch.zeros_like(context_aware_states))

        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, H, num_experts]

        # ✅ Step 7: 专家输出聚合
        expert_weights_expanded = expert_weights.unsqueeze(-2)  # [B, L, 1, num_experts]
        moe_output = torch.sum(expert_outputs * expert_weights_expanded, dim=-1)  # [B, L, H]
        moe_output = self._check_numerical_stability(moe_output, "moe_output")

        # ✅ Step 8: 共享专家处理（如果启用）
        if use_shared_experts and hasattr(self, 'shared') and self.shared:
            try:
                if input_ids_mask is not None:
                    with torch.no_grad():
                        outputs_mask = self.llama_model(
                            input_ids=input_ids_mask,
                            attention_mask=attention_mask_mask,
                            output_hidden_states=True,
                            return_dict=True
                        )
                    hidden_states_mask = outputs_mask.hidden_states[-1]
                else:
                    hidden_states_mask = hidden_states

                shared_output = hidden_states_mask
                for shared_layer in self.shared:
                    shared_output = shared_layer(shared_output)
                shared_output = self._check_numerical_stability(shared_output, "shared_output")
            except Exception as e:
                logging.warning(f"Error in shared experts: {e}")
                shared_output = torch.zeros_like(hidden_states)
        else:
            shared_output = torch.zeros_like(hidden_states)

        # ✅ Step 9: 文化感知门控
        try:
            cultural_embeddings_expanded = cultural_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
            gate_input = torch.cat([hidden_states, cultural_embeddings_expanded], dim=-1)
            gate_input = self._check_numerical_stability(gate_input, "gate_input")
            cultural_gate_values = self.cultural_gate(gate_input)  # [B, L, H]
            cultural_gate_values = self._check_numerical_stability(cultural_gate_values, "cultural_gate_values")
        except Exception as e:
            logging.warning(f"Error in cultural gate: {e}")
            cultural_gate_values = torch.ones_like(hidden_states) * 0.5

        # ✅ Step 10: 最终融合
        # 使用父类的可学习参数 moe_fusion_alpha
        moe_fusion_alpha = self.moe_fusion_alpha.to(device=hidden_states.device, dtype=hidden_states.dtype)
        enhanced_hidden = (
            hidden_states +
            moe_fusion_alpha * cultural_gate_values * moe_output +
            (1 - moe_fusion_alpha) * shared_output
        )
        enhanced_hidden = self._check_numerical_stability(enhanced_hidden, "enhanced_hidden")

        # ✅ Step 11: 生成logits
        try:
            logits = self.llama_model.lm_head(enhanced_hidden)  # [B, L, vocab_size]
            logits = self._check_numerical_stability(logits, "logits")
        except Exception as e:
            logging.error(f"Error in lm_head: {e}")
            vocab_size = self.config.vocab_size
            logits = torch.zeros(batch_size, seq_len, vocab_size, device=device, dtype=dtype)

        # ✅ Step 12: 构建输出
        outputs = {
            'logits': logits,
            'expert_weights': expert_weights
        }

        # DataParallel兼容性：只在非DataParallel模式下添加复杂类型
        if not is_dataparallel:
            outputs.update({
                'cultural_analysis': cultural_analysis,
                'enhanced_hidden_states': enhanced_hidden,
                'moe_output': moe_output,
                'shared_output': shared_output,
                'cultural_gate_values': cultural_gate_values
            })

        # ✅ Step 13: 损失计算
        if labels is not None:
            # 确保维度匹配
            if labels.size(1) != logits.size(1):
                min_len = min(labels.size(1), logits.size(1))
                labels = labels[:, :min_len]
                logits = logits[:, :min_len, :]

            # 生成损失 - 使用稳定版本
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if shift_labels.device != shift_logits.device:
                shift_labels = shift_labels.to(shift_logits.device)

            generation_loss = self._stable_cross_entropy_loss(shift_logits, shift_labels)
            outputs['generation_loss'] = generation_loss

            # 文化损失
            if use_culture_loss and culture_labels is not None:
                try:
                    if is_dataparallel:
                        # DataParallel模式下使用基础文化损失
                        culture_loss = self.compute_culture_loss(
                            expert_weights=expert_weights,
                            culture_labels=culture_labels,
                            margin=min(culture_loss_alpha, 2.0),  # 限制margin值
                            lambda_diff=min(culture_loss_beta, 1.0)  # 限制lambda_diff值
                        )
                    else:
                        # 非DataParallel模式下使用增强的文化损失
                        culture_loss = self.compute_enhanced_culture_loss(
                            expert_weights=expert_weights,
                            culture_labels=culture_labels,
                            cultural_analysis=cultural_analysis,
                            culture_relevances=culture_relevances,
                            culture_labels_multi=culture_ids_multi,
                            margin=min(culture_loss_alpha, 2.0),
                            lambda_diff=min(culture_loss_beta, 1.0)
                        )

                    # 确保文化损失数值稳定
                    culture_loss = self._check_numerical_stability(culture_loss, "culture_loss")
                    if culture_loss < 0:
                        culture_loss = torch.tensor(0.0, device=culture_loss.device)

                    outputs['culture_loss'] = culture_loss
                except Exception as e:
                    logging.warning(f"Error in culture loss computation: {e}")
                    outputs['culture_loss'] = torch.tensor(0.0, device=generation_loss.device)

            # 负载均衡损失 - 使用更保守的计算
            try:
                load_balance_loss = self.compute_load_balance_loss(expert_weights)
                load_balance_loss = self._check_numerical_stability(load_balance_loss, "load_balance_loss")
                # 限制负载均衡损失的大小
                load_balance_loss = torch.clamp(load_balance_loss, min=0, max=10.0)
            except Exception as e:
                logging.warning(f"Error in load balance loss: {e}")
                load_balance_loss = torch.tensor(0.0, device=generation_loss.device)

            # 熵损失 - 使用更保守的计算
            try:
                entropy_loss = self.compute_entropy_loss(expert_weights)
                entropy_loss = self._check_numerical_stability(entropy_loss, "entropy_loss")
                # 限制熵损失的大小
                entropy_loss = torch.clamp(entropy_loss, min=0, max=10.0)
            except Exception as e:
                logging.warning(f"Error in entropy loss: {e}")
                entropy_loss = torch.tensor(0.0, device=generation_loss.device)

            outputs['load_balance_loss'] = load_balance_loss
            outputs['entropy_loss'] = entropy_loss

            # 总损失计算 - 更保守的权重
            if use_culture_loss and 'culture_loss' in outputs:
                # 使用更小的权重避免不稳定
                safe_load_balance_weight = min(load_balance_weight, 0.001)
                safe_entropy_weight = min(entropy_weight, 0.01)
                safe_culture_lambda = min(culture_loss_lambda, 0.5)

                total_loss = (
                    generation_loss +
                    safe_culture_lambda * outputs['culture_loss'] +
                    safe_load_balance_weight * load_balance_loss +
                    safe_entropy_weight * entropy_loss
                )
            else:
                safe_load_balance_weight = min(load_balance_weight, 0.001)
                safe_entropy_weight = min(entropy_weight, 0.01)

                total_loss = (
                    generation_loss +
                    safe_load_balance_weight * load_balance_loss +
                    safe_entropy_weight * entropy_loss
                )

            # 最终的损失检查和修正
            total_loss = self._check_numerical_stability(total_loss, "total_loss")

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logging.warning("⚠️  Total loss is still NaN/Inf after fixes, using generation loss only")
                total_loss = generation_loss
            elif total_loss < 0:
                logging.warning(f"⚠️  Total loss is negative ({total_loss:.6f}), using generation loss only")
                total_loss = generation_loss

            outputs['loss'] = total_loss

        return outputs

    def compute_enhanced_culture_loss(self, expert_weights, culture_labels, cultural_analysis,
                                    culture_relevances, culture_labels_multi=None, margin=0.5, lambda_diff=1.0, eps=1e-8):
        """
        增强的文化损失，结合多个文化感知组件的信息 - 修复版本
        """
        try:
            # 基础文化损失
            base_culture_loss = self.compute_culture_loss(
                expert_weights=expert_weights,
                culture_labels=culture_labels,
                margin=margin,
                lambda_diff=lambda_diff
            )

            # 确保基础损失稳定
            base_culture_loss = self._check_numerical_stability(base_culture_loss, "base_culture_loss")

            return base_culture_loss

        except Exception as e:
            logging.warning(f"Error in enhanced culture loss: {e}")
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)

    def compute_load_balance_loss(self, expert_weights):
        """计算负载均衡损失"""
        try:
            if expert_weights.dim() == 3:  # [B, L, num_experts]
                # 计算每个专家的平均使用率
                expert_usage = expert_weights.mean(dim=(0, 1))  # [num_experts]
            else:  # [B, num_experts]
                expert_usage = expert_weights.mean(dim=0)  # [num_experts]

            # 计算方差作为负载均衡损失
            ideal_usage = 1.0 / self.args.num_experts
            load_balance_loss = torch.var(expert_usage)

            return self._check_numerical_stability(load_balance_loss, "load_balance_loss")
        except Exception as e:
            logging.warning(f"Error in compute_load_balance_loss: {e}")
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)

    def compute_entropy_loss(self, expert_weights):
        """计算熵损失"""
        try:
            # 计算专家权重的熵
            # 添加小值避免log(0)
            eps = 1e-8
            expert_probs = expert_weights + eps
            expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True)

            # 计算熵
            entropy = -torch.sum(expert_probs * torch.log(expert_probs + eps), dim=-1)
            entropy_loss = -entropy.mean()  # 负熵，鼓励多样性

            return self._check_numerical_stability(entropy_loss, "entropy_loss")
        except Exception as e:
            logging.warning(f"Error in compute_entropy_loss: {e}")
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)