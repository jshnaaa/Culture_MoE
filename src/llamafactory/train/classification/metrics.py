# src/llamafactory/train/classification/metrics.py
import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """
    计算分类任务的评估指标

    Args:
        eval_pred: (predictions, labels) 元组

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

    # 计算每个类别的 precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2]
    )

    # 计算宏平均和加权平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )

    # 混淆矩阵
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    # 打印详细报告
    print("\n" + "=" * 50)
    print("Classification Report:")
    print("=" * 50)
    print(classification_report(
        labels, preds,
        target_names=["no (0)", "neutral (1)", "yes (2)"],
        digits=4
    ))

    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              no  neutral  yes")
    for i, row in enumerate(cm):
        label_name = ["no", "neutral", "yes"][i]
        print(f"Actual {label_name:7s} {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")
    print("=" * 50 + "\n")

    # 返回指标字典
    return {
        "accuracy": accuracy,
        # 每个类别的指标
        "precision_no": precision[0],
        "precision_neutral": precision[1],
        "precision_yes": precision[2],
        "recall_no": recall[0],
        "recall_neutral": recall[1],
        "recall_yes": recall[2],
        "f1_no": f1[0],
        "f1_neutral": f1[1],
        "f1_yes": f1[2],
        # 平均指标
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }