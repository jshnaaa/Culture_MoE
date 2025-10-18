# src/llamafactory/train/classification/trainer.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple
from transformers import Trainer
from transformers.trainer import *
from transformers.modeling_utils import PreTrainedModel
import numpy as np


class ClassificationTrainer(Trainer):
    """
    用于三分类任务的自定义 Trainer
    适配 CultureMoE 模型的输出格式
    """

    def compute_loss(
            self,
            model: PreTrainedModel,
            inputs: Dict[str, Any],
            return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        重写 loss 计算逻辑

        Args:
            model: CultureMoE 模型
            inputs: 包含 input_ids, attention_mask, labels
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

        # 前向传播 - CultureMoE 返回 logits_avg [B, num_classes]
        logits = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask")
        )

        # 如果没有标签，只返回 logits（用于推理）
        if labels is None:
            outputs = {"logits": logits}
            return (None, outputs) if return_outputs else None

        # 确保 labels 是 Long 类型
        labels = labels.long()

        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        # 返回结果
        outputs = {"logits": logits}
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
                labels = labels.to(model.device)
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

        # 转移到 CPU
        if preds is not None:
            preds = preds.detach().cpu()
        if labels is not None:
            labels = labels.detach().cpu()
        if loss is not None:
            loss = loss.detach().cpu()

        return (loss, preds, labels)