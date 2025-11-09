#!/usr/bin/env python3
"""
从 generated_answers.json 计算评估指标

使用方法：
    python eval_from_generated_answers.py --input generated_answers.json
    python eval_from_generated_answers.py --input /path/to/output_dir/generated_answers.json
"""

import argparse
import json
import os

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def detect_task_type(data):
    """检测任务类型"""
    # 检查前几个样本
    sample_outputs = [d["true"] for d in data[:10] if d["true"]]

    # 检查是否是 yes/no/neutral
    if all(s.lower() in ["yes", "no", "neutral"] for s in sample_outputs if s):
        return "text"

    # 检查是否是 TRUE/FALSE
    if all(s.upper() in ["TRUE", "FALSE"] for s in sample_outputs if s):
        return "bool"

    # 检查是否是数字
    try:
        [int(s) for s in sample_outputs if s]
        return "number"
    except:
        pass

    return "other"


def evaluate_text_classification(data):
    """评估文本分类任务（yes/no/neutral）"""
    label_to_id = {"yes": 0, "no": 1, "neutral": 2}
    id_to_label = {0: "yes", 1: "no", 2: "neutral"}

    pred_labels = []
    true_labels = []

    for item in data:
        pred = item["predicted"].strip().lower()
        true = item["true"].strip().lower()

        # 提取标签
        pred_label = "neutral"  # 默认
        for label_str in ["yes", "no", "neutral"]:
            if label_str in pred:
                pred_label = label_str
                break

        true_label = true if true in label_to_id else "neutral"

        pred_labels.append(label_to_id[pred_label])
        true_labels.append(label_to_id[true_label])

    # 计算指标
    accuracy = sum(p == t for p, t in zip(pred_labels, true_labels)) / len(pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )

    # 每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0
    )

    print("\n" + "="*80)
    print("📊 Classification Metrics (yes/no/neutral)")
    print("="*80)
    print(f"Accuracy:  {accuracy:.4f} ({sum(p == t for p, t in zip(pred_labels, true_labels))}/{len(pred_labels)})")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall:    {recall:.4f} (weighted)")
    print(f"F1 Score:  {f1:.4f} (weighted)")
    print("\nPer-class metrics:")
    print("-" * 80)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 80)
    for label_id, label_str in id_to_label.items():
        print(f"{label_str:<10} {precision_per_class[label_id]:<12.4f} {recall_per_class[label_id]:<12.4f} {f1_per_class[label_id]:<12.4f} {support[label_id]:<10}")
    print("="*80)

    # 混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    print("-" * 80)
    print(f"{'':>10} {'yes':<10} {'no':<10} {'neutral':<10}")
    print("-" * 80)
    for i, label in enumerate(["yes", "no", "neutral"]):
        print(f"{label:>10} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}")
    print("="*80)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_number_classification(data):
    """评估数字分类任务"""
    pred_labels = []
    true_labels = []

    for item in data:
        pred = item["predicted"].strip()
        true = item["true"].strip()

        # 提取数字
        try:
            pred_num = int(pred) if pred else -1
        except:
            pred_num = -1

        try:
            true_num = int(true) if true else -1
        except:
            true_num = -1

        if true_num != -1:  # 只统计有效样本
            pred_labels.append(pred_num)
            true_labels.append(true_num)

    # 计算准确率
    correct = sum(p == t for p, t in zip(pred_labels, true_labels))
    accuracy = correct / len(pred_labels) if len(pred_labels) > 0 else 0.0

    # 获取所有类别
    unique_labels = sorted(set(true_labels))

    print("\n" + "="*80)
    print("📊 Classification Metrics (Number)")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(pred_labels)})")
    print(f"Classes:  {unique_labels}")

    # 如果类别数量合理，计算更多指标
    if len(unique_labels) <= 20:
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0, labels=unique_labels
        )
        print(f"Precision: {precision:.4f} (weighted)")
        print(f"Recall:    {recall:.4f} (weighted)")
        print(f"F1 Score:  {f1:.4f} (weighted)")

        # 每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, zero_division=0, labels=unique_labels
        )

        print("\nPer-class metrics:")
        print("-" * 80)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 80)
        for i, label in enumerate(unique_labels):
            print(f"{label:<10} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f} {support_per_class[i]:<10}")

        # 混淆矩阵（如果类别不太多）
        if len(unique_labels) <= 10:
            cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
            print("\nConfusion Matrix:")
            print("-" * 80)
            header = f"{'':>10} " + " ".join(f"{l:<10}" for l in unique_labels)
            print(header)
            print("-" * 80)
            for i, label in enumerate(unique_labels):
                row = f"{label:>10} " + " ".join(f"{cm[i][j]:<10}" for j in range(len(unique_labels)))
                print(row)

    print("="*80)

    return {
        "accuracy": accuracy
    }


def evaluate_bool_classification(data):
    """评估布尔分类任务（TRUE/FALSE）"""
    label_to_id = {"true": 0, "false": 1}
    id_to_label = {0: "TRUE", 1: "FALSE"}

    pred_labels = []
    true_labels = []

    for item in data:
        pred = item["predicted"].strip().upper()
        true = item["true"].strip().upper()

        # 提取标签
        pred_label = "FALSE"  # 默认
        if "TRUE" in pred:
            pred_label = "TRUE"
        elif "FALSE" in pred:
            pred_label = "FALSE"

        true_label = true if true in ["TRUE", "FALSE"] else "FALSE"

        pred_labels.append(label_to_id[pred_label.lower()])
        true_labels.append(label_to_id[true_label.lower()])

    # 计算指标
    accuracy = sum(p == t for p, t in zip(pred_labels, true_labels)) / len(pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )

    print("\n" + "="*80)
    print("📊 Classification Metrics (TRUE/FALSE)")
    print("="*80)
    print(f"Accuracy:  {accuracy:.4f} ({sum(p == t for p, t in zip(pred_labels, true_labels))}/{len(pred_labels)})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*80)

    # 混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    print("-" * 80)
    print(f"{'':>10} {'TRUE':<10} {'FALSE':<10}")
    print("-" * 80)
    for i, label in enumerate(["TRUE", "FALSE"]):
        print(f"{label:>10} {cm[i][0]:<10} {cm[i][1]:<10}")
    print("="*80)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_exact_match(data):
    """评估精确匹配"""
    correct = sum(d["predicted"].strip() == d["true"].strip() for d in data)
    accuracy = correct / len(data) if len(data) > 0 else 0.0

    print("\n" + "="*80)
    print("📊 Exact Match Metrics")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(data)})")
    print("="*80)

    return {
        "accuracy": accuracy
    }


def main():
    parser = argparse.ArgumentParser(description="从 generated_answers.json 计算评估指标")
    parser.add_argument("--input", type=str, default="generated_answers.json",
                        help="generated_answers.json 文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="保存评估结果的 JSON 文件路径（可选）")

    args = parser.parse_args()

    # 读取数据
    if not os.path.exists(args.input):
        print(f"❌ Error: File not found: {args.input}")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n📂 Loaded {len(data)} samples from: {args.input}")

    # 检测任务类型
    task_type = detect_task_type(data)
    print(f"🔍 Detected task type: {task_type}")

    # 根据任务类型评估
    if task_type == "text":
        metrics = evaluate_text_classification(data)
    elif task_type == "bool":
        metrics = evaluate_bool_classification(data)
    elif task_type == "number":
        metrics = evaluate_number_classification(data)
    else:
        metrics = evaluate_exact_match(data)

    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Metrics saved to: {args.output}")

    print("")


if __name__ == "__main__":
    main()

