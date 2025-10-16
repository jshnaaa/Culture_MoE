# src/llamafactory/train/classification/trainer.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple
from transformers import Trainer
from transformers.trainer import *
from transformers.modeling_utils import PreTrainedModel


class ClassificationTrainer(Trainer):
    """
    用于三分类任务的自定义 Trainer
    适配 CultureMoE 模型的输出格式
    所有操作在 GPU 上进行
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
        # 提取标签（已经在 GPU 上）
        labels = inputs.pop("labels")  # [B]，整数标签 0/1/2

        # 前向传播 - CultureMoE 返回 logits_avg [B, num_classes]
        logits = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask")
        )

        # 确保 labels 是 Long 类型（保持在 GPU 上）
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
        所有张量保持在 GPU 上

        Returns:
            (loss, predictions, labels)
        """
        has_labels = "labels" in inputs
        labels = inputs.pop("labels") if has_labels else None

        # 前向传播（在 GPU 上）
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits = outputs["logits"]
            else:
                loss = None
                logits = model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask")
                )

        # 获取预测类别（保持在 GPU 上）
        if logits is not None:
            preds = torch.argmax(logits, dim=-1)  # [B]
        else:
            preds = None

        if prediction_loss_only:
            return (loss, None, None)

        # ✅ 不转移到 CPU，保持在 GPU 上
        # Trainer 会自动处理设备转移
        return (loss, preds, labels)