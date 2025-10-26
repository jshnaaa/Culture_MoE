# src/llamafactory/train/classification/metrics.py
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def compute_classification_metrics(eval_pred, num_classes=2) -> Dict[str, float]:
    """
    计算分类任务的评估指标

    Args:
        eval_pred: (predictions, labels) 元组
        num_classes: 分类数量（2 或 3）

    Returns:
        包含各种指标的字典
    """
    predictions, labels = eval_pred

    # predictions 可能是 logits 或已经是类别索引
    if len(predictions.shape) > 1:
        preds = np.argmax(predictions, axis=-1)
    else:
        preds = predictions

    # 计算各种指标
    accuracy = accuracy_score(labels, preds)

    # ✅ 根据分类数量动态设置标签
    if num_classes == 2:
        class_labels = [0, 1]
        target_names = ["no (0)", "yes (1)"]
        label_names = ["no", "yes"]
    else:  # num_classes == 3
        class_labels = [0, 1, 2]
        target_names = ["no (0)", "neutral (1)", "yes (2)"]
        label_names = ["no", "neutral", "yes"]

    # 计算每个类别的 precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=class_labels, zero_division=0
    )

    # 计算宏平均和加权平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    # 混淆矩阵
    cm = confusion_matrix(labels, preds, labels=class_labels)

    # 打印详细报告
    print("\n" + "=" * 50)
    print("Classification Report:")
    print("=" * 50)
    print(classification_report(
        labels, preds,
        target_names=target_names,
        digits=4,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print("                Predicted")
    if num_classes == 2:
        print("              no  yes")
        for i, row in enumerate(cm):
            label_name = label_names[i]
            print(f"Actual {label_name:7s} {row[0]:3d}  {row[1]:3d}")
    else:
        print("              no  neutral  yes")
        for i, row in enumerate(cm):
            label_name = label_names[i]
            print(f"Actual {label_name:7s} {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")
    print("=" * 50 + "\n")

    # ✅ 返回指标字典（动态适配分类数）
    result = {
        "accuracy": accuracy,
        # 平均指标
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

    # 添加每个类别的指标
    if num_classes == 2:
        result.update({
            "precision_no": precision[0],
            "precision_yes": precision[1],
            "recall_no": recall[0],
            "recall_yes": recall[1],
            "f1_no": f1[0],
            "f1_yes": f1[1],
        })
    else:  # num_classes == 3
        result.update({
            "precision_no": precision[0],
            "precision_neutral": precision[1],
            "precision_yes": precision[2],
            "recall_no": recall[0],
            "recall_neutral": recall[1],
            "recall_yes": recall[2],
            "f1_no": f1[0],
            "f1_neutral": f1[1],
            "f1_yes": f1[2],
        })

    return result
