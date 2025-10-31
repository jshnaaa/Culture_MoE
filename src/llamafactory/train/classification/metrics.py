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
    计算分类任务的评估指标（支持 2/3/4/5 分类）

    Args:
        eval_pred: (predictions, labels) 元组
        num_classes: 分类数量（2/3/4/5）

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

    # ✅ 检查数据集中实际的类别数
    actual_num_classes = len(np.unique(labels))
    if actual_num_classes != num_classes:
        print(f"\n⚠️  Warning: Expected {num_classes} classes, but found {actual_num_classes} unique classes in the dataset")
        print(f"   Unique labels in data: {sorted(np.unique(labels).tolist())}")
        print(f"   Missing classes: {sorted(set(range(num_classes)) - set(np.unique(labels).tolist()))}")

    # ✅ 根据分类数量动态设置标签
    class_labels = list(range(num_classes))

    if num_classes == 2:
        target_names = ["no (0)", "yes (1)"]
        label_names = ["no", "yes"]
    elif num_classes == 3:
        target_names = ["no (0)", "neutral (1)", "yes (2)"]
        label_names = ["no", "neutral", "yes"]
    elif num_classes == 4:
        target_names = ["strongly_disagree (0)", "disagree (1)", "agree (2)", "strongly_agree (3)"]
        label_names = ["strongly_disagree", "disagree", "agree", "strongly_agree"]
    elif num_classes == 5:
        target_names = ["strongly_disagree (0)", "disagree (1)", "neutral (2)", "agree (3)", "strongly_agree (4)"]
        label_names = ["strongly_disagree", "disagree", "neutral", "agree", "strongly_agree"]
    else:
        target_names = [f"class_{i} ({i})" for i in range(num_classes)]
        label_names = [f"class_{i}" for i in range(num_classes)]

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
        labels=class_labels,  # ✅ 显式指定所有类别标签
        target_names=target_names,
        digits=4,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print("                Predicted")

    # ✅ 动态打印混淆矩阵
    if num_classes == 2:
        print("              no  yes")
        for i, row in enumerate(cm):
            label_name = label_names[i]
            print(f"Actual {label_name:7s} {row[0]:3d}  {row[1]:3d}")
    elif num_classes == 3:
        print("              no  neutral  yes")
        for i, row in enumerate(cm):
            label_name = label_names[i]
            print(f"Actual {label_name:7s} {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")
    elif num_classes == 4:
        print("              SD   D    A    SA")  # SD=Strongly Disagree, D=Disagree, A=Agree, SA=Strongly Agree
        for i, row in enumerate(cm):
            label_name = label_names[i][:4]  # 缩短标签名
            print(f"Actual {label_name:4s} {row[0]:3d}  {row[1]:3d}  {row[2]:3d}  {row[3]:3d}")
    elif num_classes == 5:
        print("              SD   D    N    A    SA")  # N=Neutral
        for i, row in enumerate(cm):
            label_name = label_names[i][:4]
            print(f"Actual {label_name:4s} {row[0]:3d}  {row[1]:3d}  {row[2]:3d}  {row[3]:3d}  {row[4]:3d}")
    else:
        # 通用打印（适用于任意分类数）
        header = "              " + "  ".join([f"C{i}" for i in range(num_classes)])
        print(header)
        for i, row in enumerate(cm):
            row_str = "  ".join([f"{val:3d}" for val in row])
            print(f"Actual C{i}   {row_str}")

    print("=" * 50 + "\n")

    # ✅ 返回指标字典（动态适配分类数）
    result = {
        "accuracy": accuracy,
        # 平均指标（使用 macro 平均，对所有类别一视同仁）
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        # 详细的平均指标
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

    # 添加每个类别的指标（通用方式）
    for i in range(num_classes):
        if i < len(precision):  # 确保索引有效
            result[f"precision_class_{i}"] = precision[i]
            result[f"recall_class_{i}"] = recall[i]
            result[f"f1_class_{i}"] = f1[i]

    return result
