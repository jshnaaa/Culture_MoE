# src/llamafactory/model/enhanced_culturemoe_with_distill.py
"""
带知识蒸馏的增强CultureMoE模型
集成稳健共享专家和知识蒸馏功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import math

from .enhanced_culturemoe import EnhancedCultureMoE
from .robust_shared_expert import RobustSharedExpert, KnowledgeDistillationLoss, CulturalInvarianceLoss
from .teacher_model_loader import TeacherModelLoader
from .moe_args import ModelArgs


class EnhancedCultureMoEWithDistill(EnhancedCultureMoE):
    """
    带知识蒸馏的增强CultureMoE模型

    新增功能：
    1. 稳健共享专家（通过蒸馏增强）
    2. 知识蒸馏损失
    3. 文化不变性学习
    4. 自适应蒸馏权重
    """

    def __init__(self, llama_model, config, args: ModelArgs, culture_loss_lambda=-1,
                 moe_fusion=0.4, num_cultures=6, culture_dim=256, use_gate=True,
                 teacher_model_path: Optional[str] = None,
                 teacher_lora_path: Optional[str] = None,
                 distill_temperature: float = 4.0,
                 distill_alpha: float = 0.7):

        # 调用父类初始化
        super().__init__(llama_model, config, args, culture_loss_lambda,
                        moe_fusion, num_cultures, culture_dim, use_gate)

        self.distill_temperature = distill_temperature
        self.distill_alpha = distill_alpha

        # 用稳健共享专家替换原来的shared expert
        if hasattr(self, 'shared_expert'):
            # 保存原shared expert的参数用于初始化
            original_shared = self.shared_expert

            # 创建新的稳健共享专家
            self.robust_shared_expert = RobustSharedExpert(
                hidden_dim=self.config.hidden_size,
                expert_dim=args.shared_hidden_dim,
                lora_rank=args.lora_rank,
                dropout=args.dropout
            )

            # 初始化稳健共享专家的主网络为原shared expert的权重
            self._initialize_robust_shared_expert(original_shared)

            # 移除原shared expert
            del self.shared_expert

        # 知识蒸馏损失函数
        self.distill_loss_fn = KnowledgeDistillationLoss(
            temperature=distill_temperature,
            alpha=distill_alpha
        )

        # 文化不变性损失函数
        self.invariant_loss_fn = CulturalInvarianceLoss(margin=1.0)

        # Teacher模型加载器（延迟初始化）
        self.teacher_loader = None
        self.teacher_model_path = teacher_model_path
        self.teacher_lora_path = teacher_lora_path

        # 蒸馏权重调度器
        self.distill_weight_scheduler = DistillWeightScheduler()

        logging.info("Enhanced CultureMoE with Knowledge Distillation initialized")

    def _initialize_robust_shared_expert(self, original_shared: nn.Module):
        """
        用原shared expert的权重初始化稳健共享专家的主网络

        Args:
            original_shared: 原来的shared expert
        """
        try:
            # 获取原shared expert的state dict
            original_state = original_shared.state_dict()

            # 初始化robust shared expert的主网络
            robust_state = self.robust_shared_expert.expert_network.state_dict()

            # 复制匹配的权重
            for name, param in original_state.items():
                if name in robust_state and param.shape == robust_state[name].shape:
                    robust_state[name].copy_(param)
                    logging.debug(f"Copied weight: {name}")

            logging.info("Robust shared expert initialized with original shared expert weights")

        except Exception as e:
            logging.warning(f"Failed to initialize robust shared expert: {e}")
            logging.info("Using random initialization for robust shared expert")

    def setup_teacher_model(self):
        """设置Teacher模型"""
        if self.teacher_model_path and self.teacher_loader is None:
            try:
                self.teacher_loader = TeacherModelLoader(
                    base_model_path=self.teacher_model_path,
                    lora_weights_path=self.teacher_lora_path,
                    device=next(self.parameters()).device
                )
                self.teacher_loader.load_teacher_model()
                logging.info("Teacher model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load teacher model: {e}")
                self.teacher_loader = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                input_ids_mask: Optional[torch.Tensor] = None,
                attention_mask_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                culture_labels: Optional[torch.Tensor] = None,
                culture_ids: Optional[torch.Tensor] = None,
                culture_ids_multi: Optional[List] = None,
                use_culture_loss: bool = True,
                culture_loss_lambda: float = 0.5,
                culture_loss_alpha: float = 0.5,
                culture_loss_beta: float = 1.0,
                use_shared_experts: bool = True,
                router_temperature: float = 1.0,
                load_balance_weight: float = 0.01,
                entropy_weight: float = 0.1,
                current_epoch: int = 1,
                total_epochs: int = 6,
                enable_distillation: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播（增强版，包含知识蒸馏）

        Args:
            enable_distillation: 是否启用知识蒸馏

        Returns:
            包含所有损失的输出字典
        """
        # 1. 基础MoE前向传播
        outputs = self._forward_moe_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids_mask=input_ids_mask,
            attention_mask_mask=attention_mask_mask,
            labels=labels,
            culture_labels=culture_labels,
            culture_ids=culture_ids,
            culture_ids_multi=culture_ids_multi,
            use_culture_loss=use_culture_loss,
            culture_loss_lambda=culture_loss_lambda,
            culture_loss_alpha=culture_loss_alpha,
            culture_loss_beta=culture_loss_beta,
            use_shared_experts=use_shared_experts,
            router_temperature=router_temperature,
            load_balance_weight=load_balance_weight,
            entropy_weight=entropy_weight
        )

        # 2. 知识蒸馏（如果启用且有teacher模型）
        if enable_distillation and self.teacher_loader is not None and self.training:
            distill_outputs = self._compute_distillation_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                student_logits=outputs['logits'],
                culture_ids=culture_ids,
                current_epoch=current_epoch,
                total_epochs=total_epochs
            )
            outputs.update(distill_outputs)

        return outputs

    def _forward_moe_base(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        基础MoE前向传播（调用父类方法但使用稳健共享专家）
        """
        # 临时替换shared expert调用
        original_forward = super().forward

        # 修改shared expert的调用
        def robust_shared_forward(hidden_states):
            robust_outputs = self.robust_shared_expert(
                hidden_states, return_distill_logits=True
            )
            return robust_outputs['expert_output']

        # 暂时替换shared expert
        if hasattr(self, 'shared_expert_forward'):
            original_shared_forward = self.shared_expert_forward
        else:
            original_shared_forward = None

        self.shared_expert_forward = robust_shared_forward

        try:
            # 调用父类的forward方法
            outputs = original_forward(**kwargs)
        finally:
            # 恢复原来的方法
            if original_shared_forward is not None:
                self.shared_expert_forward = original_shared_forward
            elif hasattr(self, 'shared_expert_forward'):
                delattr(self, 'shared_expert_forward')

        return outputs

    def _compute_distillation_loss(self, input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor,
                                  student_logits: torch.Tensor,
                                  culture_ids: Optional[torch.Tensor],
                                  current_epoch: int,
                                  total_epochs: int) -> Dict[str, torch.Tensor]:
        """
        计算知识蒸馏损失

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力mask
            student_logits: 学生模型(MoE)的logits
            culture_ids: 文化标签
            current_epoch: 当前epoch
            total_epochs: 总epoch数

        Returns:
            蒸馏损失字典
        """
        distill_outputs = {}

        try:
            # 1. 获取teacher logits
            teacher_logits = self.teacher_loader.get_teacher_logits(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 2. 获取robust shared expert的logits
            # 需要重新计算shared expert的输出用于蒸馏
            with torch.no_grad():
                llama_outputs = self.llama_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = llama_outputs.hidden_states[-1]

            robust_outputs = self.robust_shared_expert(
                hidden_states, return_distill_logits=True
            )
            shared_logits = robust_outputs['distill_logits']

            # 3. 计算蒸馏权重（随epoch变化）
            distill_weight = self.distill_weight_scheduler.get_weight(
                current_epoch, total_epochs
            )

            # 4. 计算知识蒸馏损失
            distill_losses = self.distill_loss_fn(
                student_logits=shared_logits,
                teacher_logits=teacher_logits
            )

            # 5. 计算文化不变性损失
            if culture_ids is not None:
                invariant_loss = self.invariant_loss_fn(
                    invariant_features=robust_outputs['invariant_features'],
                    culture_ids=culture_ids
                )
                distill_outputs['cultural_invariant_loss'] = invariant_loss
            else:
                distill_outputs['cultural_invariant_loss'] = torch.tensor(0.0)

            # 6. 组合蒸馏损失
            total_distill_loss = (
                distill_weight * distill_losses['total_distill_loss'] +
                0.1 * distill_outputs['cultural_invariant_loss']
            )

            distill_outputs.update({
                'distill_soft_loss': distill_losses['distill_soft_loss'],
                'total_distill_loss': total_distill_loss,
                'distill_weight': distill_weight,
                'robust_fusion_weight': robust_outputs['fusion_weight']
            })

        except Exception as e:
            logging.warning(f"Failed to compute distillation loss: {e}")
            # 返回零损失以避免训练中断
            distill_outputs.update({
                'distill_soft_loss': torch.tensor(0.0),
                'total_distill_loss': torch.tensor(0.0),
                'cultural_invariant_loss': torch.tensor(0.0),
                'distill_weight': 0.0,
                'robust_fusion_weight': 0.5
            })

        return distill_outputs

    def get_total_loss(self, outputs: Dict[str, torch.Tensor],
                      distill_weight: float = 0.3,
                      invariant_weight: float = 0.1) -> torch.Tensor:
        """
        计算总损失（包含蒸馏损失）

        Args:
            outputs: 模型输出
            distill_weight: 蒸馏损失权重
            invariant_weight: 不变性损失权重

        Returns:
            总损失
        """
        # 基础损失
        total_loss = outputs.get('loss', torch.tensor(0.0))

        # 添加蒸馏损失
        if 'total_distill_loss' in outputs:
            total_loss = total_loss + distill_weight * outputs['total_distill_loss']

        # 添加不变性损失
        if 'cultural_invariant_loss' in outputs:
            total_loss = total_loss + invariant_weight * outputs['cultural_invariant_loss']

        return total_loss

    def cleanup_teacher(self):
        """清理teacher模型资源"""
        if self.teacher_loader is not None:
            self.teacher_loader.cleanup()
            self.teacher_loader = None


class DistillWeightScheduler:
    """
    蒸馏权重调度器：动态调整蒸馏损失权重
    """

    def __init__(self, initial_weight: float = 0.5, final_weight: float = 0.2):
        self.initial_weight = initial_weight
        self.final_weight = final_weight

    def get_weight(self, current_epoch: int, total_epochs: int) -> float:
        """
        获取当前epoch的蒸馏权重

        Args:
            current_epoch: 当前epoch (1-based)
            total_epochs: 总epoch数

        Returns:
            蒸馏权重
        """
        if total_epochs <= 1:
            return self.initial_weight

        # 线性衰减
        progress = (current_epoch - 1) / (total_epochs - 1)
        weight = self.initial_weight + progress * (self.final_weight - self.initial_weight)

        return max(weight, 0.0)


def create_enhanced_culturemoe_with_distill(
    llama_model, config, args: ModelArgs,
    culture_loss_lambda: float = -1,
    moe_fusion: float = 0.4,
    num_cultures: int = 6,
    culture_dim: int = 256,
    use_gate: bool = True,
    teacher_model_path: Optional[str] = None,
    teacher_lora_path: Optional[str] = None,
    distill_temperature: float = 4.0,
    distill_alpha: float = 0.7) -> EnhancedCultureMoEWithDistill:
    """
    创建带知识蒸馏的增强CultureMoE模型

    Args:
        llama_model: 基础LLaMA模型
        config: 模型配置
        args: MoE参数
        teacher_model_path: Teacher模型路径
        teacher_lora_path: Teacher LoRA权重路径
        distill_temperature: 蒸馏温度
        distill_alpha: 蒸馏权重

    Returns:
        EnhancedCultureMoEWithDistill实例
    """
    return EnhancedCultureMoEWithDistill(
        llama_model=llama_model,
        config=config,
        args=args,
        culture_loss_lambda=culture_loss_lambda,
        moe_fusion=moe_fusion,
        num_cultures=num_cultures,
        culture_dim=culture_dim,
        use_gate=use_gate,
        teacher_model_path=teacher_model_path,
        teacher_lora_path=teacher_lora_path,
        distill_temperature=distill_temperature,
        distill_alpha=distill_alpha
    )