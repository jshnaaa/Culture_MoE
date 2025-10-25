# src/llamafactory/train/classification/trainer.py
from typing import Dict, Tuple

from transformers.trainer import *

from .culture_loss import CultureSpecializationLoss, compute_culture_aware_loss


class ClassificationTrainer(Trainer):
    """
    用于三分类任务的自定义 Trainer
    适配 CultureMoE 模型的输出格式
    支持文化专注性损失
    """

    def __init__(self, *args, use_culture_loss: bool = True, lambda_weight: float = 0.1, **kwargs):
        """
        初始化 Trainer

        Args:
            use_culture_loss: 是否使用文化专注性损失
            lambda_weight: 文化损失权重 λ
        """
        super().__init__(*args, **kwargs)

        self.use_culture_loss = use_culture_loss
        self.lambda_weight = lambda_weight

        # 创建文化专注性损失模块
        if self.use_culture_loss:
            self.culture_loss_module = CultureSpecializationLoss(
                num_cultures=6,
                num_experts=6,
                lambda_weight=lambda_weight
            )
            # 将模块移到正确的设备
            if torch.cuda.is_available():
                self.culture_loss_module = self.culture_loss_module.cuda()

    def compute_loss(
            self,
            model: PreTrainedModel,
            inputs: Dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Optional[int] = None  # 忽略
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        重写 loss 计算逻辑，支持文化专注性损失

        Args:
            model: CultureMoE 模型
            inputs: 包含 input_ids, attention_mask, labels, culture_labels
            return_outputs: 是否返回模型输出

        Returns:
            loss 或 (loss, outputs)
        """
        # ✅ 安全地提取标签
        if "labels" in inputs:
            labels = inputs.pop("labels")  # [B]，整数标签 0/1/2
        else:
            # 如果没有 labels，返回 None（评估时可能发生）
            labels = None

        # ✅ 提取文化维度标签
        culture_labels = inputs.pop("culture_labels", None)  # List[List[int]]

        # 前向传播 - CultureMoE 返回 logits_avg [B, num_classes]
        # ✅ 支持双路输入
        logits = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            input_ids_mask=inputs.get("input_ids_mask"),
            attention_mask_mask=inputs.get("attention_mask_mask")
        )

        # 如果没有标签，只返回 logits（用于推理）
        if labels is None:
            outputs = {"logits": logits}
            return (None, outputs) if return_outputs else None

        # 确保 labels 是 Long 类型
        labels = labels.long()

        # ✅ 获取专家权重（从模型中）
        # 需要修改模型以返回专家权重
        expert_weights = None
        if hasattr(model, 'get_expert_weights'):
            expert_weights = model.get_expert_weights()
        elif hasattr(model, 'module') and hasattr(model.module, 'get_expert_weights'):
            # DDP 模式
            expert_weights = model.module.get_expert_weights()

        # 计算损失
        if self.use_culture_loss and expert_weights is not None and culture_labels is not None:
            # 使用文化感知损失
            total_loss, ce_loss, culture_loss = compute_culture_aware_loss(
                logits=logits,
                labels=labels,
                expert_weights=expert_weights,
                culture_labels=culture_labels,
                culture_loss_module=self.culture_loss_module,
                lambda_weight=self.lambda_weight
            )

            # 记录损失组件（用于日志）
            outputs = {
                "logits": logits,
                "ce_loss": ce_loss.detach(),
                "culture_loss": culture_loss.detach(),
                "total_loss": total_loss.detach()
            }
            loss = total_loss
        else:
            # 只使用交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = {"logits": logits}

        # 返回结果
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        重写预测步骤，用于评估

        Returns:
            (loss, predictions, labels)
        """
        # ✅ 安全地检查和提取标签
        has_labels = "labels" in inputs

        if has_labels:
            labels = inputs.get("labels")
            # 确保 labels 在正确的设备上
            if isinstance(labels, torch.Tensor):
                model_device = next(model.parameters()).device  # 从模型参数获取设备
                labels = labels.to(model_device)
        else:
            labels = None

        # 前向传播
        with torch.no_grad():
            if has_labels:
                # ✅ 创建 inputs 的副本，避免修改原始数据
                inputs_copy = {k: v for k, v in inputs.items()}
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs_copy, return_outputs=True)
                logits = outputs["logits"]
            else:
                loss = None
                # ✅ 直接调用模型，不通过 compute_loss
                logits = model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask")
                )

        # 获取预测类别
        if logits is not None:
            preds = torch.argmax(logits, dim=-1)  # [B]
        else:
            preds = None

        if prediction_loss_only:
            return (loss, None, None)

        #
        if preds is not None:
            preds = preds.detach()
        if labels is not None:
            labels = labels.detach()
        # if loss is not None:
        #     loss = loss.detach().cpu()

        return (loss, preds, labels)