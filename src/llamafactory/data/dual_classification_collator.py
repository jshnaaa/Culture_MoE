# src/llamafactory/data/dual_classification_collator.py
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DualClassificationDataCollator:
    """
    双路输入的分类任务 DataCollator
    处理两组输入：(instruction + input) 和 (instruction_mask + input)
    """
    tokenizer: PreTrainedTokenizer
    padding: bool = True
    max_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        将一批样本整理成模型输入格式

        Args:
            features: 包含 input_ids, attention_mask, input_ids_mask, attention_mask_mask, labels, culture_labels 的样本列表

        Returns:
            批次数据字典
        """
        # 1. 提取 labels（分类标签）
        labels = [feature["labels"] for feature in features]

        # 2. 提取 culture_labels（文化维度标签）
        # culture_labels 是列表的列表，例如 [[0], [1, 2], [3]]
        culture_labels = [feature.get("culture_labels", []) for feature in features]

        # 3. 提取第一组输入（instruction + input）
        batch_full = {
            "input_ids": [feature["input_ids"] for feature in features],
            "attention_mask": [feature["attention_mask"] for feature in features]
        }

        # 4. 提取第二组输入（instruction_mask + input）
        batch_mask = {
            "input_ids": [feature["input_ids_mask"] for feature in features],
            "attention_mask": [feature["attention_mask_mask"] for feature in features]
        }

        # 5. Padding 第一组
        batch_full_padded = self.tokenizer.pad(
            batch_full,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 6. Padding 第二组
        batch_mask_padded = self.tokenizer.pad(
            batch_mask,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 7. 组合结果
        result = {
            # 第一组：instruction + input (用于 h_all)
            "input_ids": batch_full_padded["input_ids"],
            "attention_mask": batch_full_padded["attention_mask"],

            # 第二组：instruction_mask + input (用于 h_no)
            "input_ids_mask": batch_mask_padded["input_ids"],
            "attention_mask_mask": batch_mask_padded["attention_mask"],

            # 分类标签（output 字段）
            "labels": torch.tensor(labels, dtype=torch.long),

            # 文化维度标签（label 字段）
            # 保持为列表格式，因为每个样本可能有不同数量的标签
            "culture_labels": culture_labels
        }

        return result

