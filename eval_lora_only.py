#!/usr/bin/env python3
"""
评估 LoRA Only 模型（合并后的完整模型）

使用方法：
    python eval_lora_only.py \
        --model_path /path/to/merged_model \
        --test_file /path/to/test_data.json \
        --output_dir /path/to/output

示例：
    python eval_lora_only.py \
        --model_path /root/autodl-tmp/CultureMoE/Culture_Alignment/llama_merge_2 \
        --test_file /root/autodl-fs/wvs_2_merge.json \
        --output_dir /root/autodl-tmp/CultureMoE/Culture_Alignment/lora_only_test_results/llama_2class_20241102_1234
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data
from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator


def load_merged_model(model_path: str, device: str = "cuda"):
    """
    加载合并后的 LoRA 模型

    Args:
        model_path: 合并后的模型路径
        device: 设备

    Returns:
        model: 模型
        tokenizer: Tokenizer
    """
    print("\n" + "="*80)
    print("Loading Merged LoRA Model")
    print("="*80)
    print(f"Model path: {model_path}")
    print("="*80)
    print("")

    # 1. 加载 Tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载模型
    print("\n2. Loading merged model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # 设置为评估模式
    model.eval()

    print("   ✅ Model loaded")
    print(f"      Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*80)
    print("✅ Model Loaded Successfully!")
    print("="*80)
    print("")

    return model, tokenizer


def evaluate_model(model, tokenizer, test_dataset, device, batch_size=8, num_classes=2):
    """
    在测试集上评估模型

    Args:
        model: 模型
        tokenizer: Tokenizer
        test_dataset: 测试数据集
        device: 设备
        batch_size: 批次大小
        num_classes: 分类数量

    Returns:
        dict: 评估结果
    """
    print("\n" + "="*80)
    print("Evaluating Model on Test Set")
    print("="*80)
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("="*80)
    print("")

    # 创建 DataLoader
    data_collator = DualClassificationDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        padding=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2
    )

    # 评估
    model.eval()
    all_preds = []
    all_labels = []
    all_culture_preds = []
    all_culture_labels = []

    print("Running evaluation...")

    # ✅ 获取模型的实际设备
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 移动到模型所在的设备
            input_ids = batch['input_ids'].to(model_device)
            attention_mask = batch['attention_mask'].to(model_device)

            # 处理标签
            if isinstance(batch['labels'], torch.Tensor):
                labels = batch['labels']
                culture_labels = batch['culture_labels']
            else:
                labels = torch.tensor(batch['labels'])
                culture_labels = torch.tensor(batch['culture_labels'])

            # 前向传播 - 使用生成模式获取 logits
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 获取 logits（最后一层的输出）
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            # ✅ 对于分类任务，我们需要将 vocab logits 映射到类别
            # 这里假设使用特定 token 的 logits 作为类别分数
            # 例如：使用前 num_classes 个 token 的 logits
            class_logits = logits[:, :num_classes]  # [B, num_classes]

            preds = torch.argmax(class_logits, dim=-1)

            # 对于 LoRA Only 模型，没有单独的 culture 分类头
            # 使用相同的预测作为 culture 预测
            culture_preds = preds

            # ✅ 统一处理函数
            def to_flat_list(data):
                """将张量/数组/列表转换为一维列表"""
                if isinstance(data, torch.Tensor):
                    data = data.cpu().numpy()
                elif isinstance(data, list):
                    try:
                        data = np.array(data)
                    except (ValueError, TypeError):
                        flat = []
                        for item in data:
                            if isinstance(item, (list, tuple)):
                                flat.extend(item)
                            else:
                                flat.append(item)
                        return flat

                if isinstance(data, np.ndarray):
                    if data.ndim > 1:
                        data = data.flatten()
                    return data.tolist()

                return data

            # 收集结果
            all_preds.extend(to_flat_list(preds))
            all_labels.extend(to_flat_list(labels))
            all_culture_preds.extend(to_flat_list(culture_preds))
            all_culture_labels.extend(to_flat_list(culture_labels))

    # ✅ 调试：打印长度
    print(f"\nDebug - Collected samples:")
    print(f"  all_preds: {len(all_preds)}")
    print(f"  all_labels: {len(all_labels)}")
    print(f"  all_culture_preds: {len(all_culture_preds)}")
    print(f"  all_culture_labels: {len(all_culture_labels)}")

    # ✅ 确保所有列表长度一致
    min_len = min(len(all_preds), len(all_labels), len(all_culture_preds), len(all_culture_labels))
    if not (len(all_preds) == len(all_labels) == len(all_culture_preds) == len(all_culture_labels)):
        print(f"\n⚠️  Warning: Inconsistent lengths detected! Truncating to {min_len} samples")
        all_preds = all_preds[:min_len]
        all_labels = all_labels[:min_len]
        all_culture_preds = all_culture_preds[:min_len]
        all_culture_labels = all_culture_labels[:min_len]

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_culture_preds = np.array(all_culture_preds)
    all_culture_labels = np.array(all_culture_labels)

    # 计算指标
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)

    # 主任务指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print("\n📊 Main Task (Classification):")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1:        {f1:.4f}")

    # 文化任务指标
    culture_accuracy = accuracy_score(all_culture_labels, all_culture_preds)
    culture_precision, culture_recall, culture_f1, _ = precision_recall_fscore_support(
        all_culture_labels, all_culture_preds, average='macro', zero_division=0
    )

    print("\n📊 Culture Task:")
    print(f"   Accuracy:  {culture_accuracy:.4f}")
    print(f"   Precision: {culture_precision:.4f}")
    print(f"   Recall:    {culture_recall:.4f}")
    print(f"   F1:        {culture_f1:.4f}")

    # 详细分类报告
    print("\n📋 Detailed Classification Report (Main Task):")
    print(classification_report(all_labels, all_preds, zero_division=0))

    print("\n📋 Detailed Classification Report (Culture Task):")
    print(classification_report(all_culture_labels, all_culture_preds, zero_division=0))

    # 返回结果
    results = {
        "main_task": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist()
        },
        "culture_task": {
            "accuracy": float(culture_accuracy),
            "precision": float(culture_precision),
            "recall": float(culture_recall),
            "f1": float(culture_f1),
            "predictions": all_culture_preds.tolist(),
            "labels": all_culture_labels.tolist()
        },
        "num_samples": len(all_labels),
        "num_classes": num_classes
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="评估 LoRA Only 模型（合并后的完整模型）")
    parser.add_argument("--model_path", type=str, required=True,
                        help="合并后的模型路径")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="评估结果输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="评估批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="分类数量")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LoRA Only Model Evaluation")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model path: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    print("")

    # 1. 加载模型
    model, tokenizer = load_merged_model(
        model_path=args.model_path,
        device=args.device
    )

    # 2. 加载测试数据
    print("\n" + "="*80)
    print("Loading Test Dataset")
    print("="*80)
    print(f"Test file: {args.test_file}")
    print("="*80)
    print("")

    # 检查测试文件是否存在
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")

    # 加载测试数据（不分割，全部作为测试集）
    datasets = load_and_process_dual_classification_data(
        data_path=args.test_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=0.0  # 不分割，全部作为测试集
    )
    test_dataset = datasets['train']  # val_split=0 时，所有数据在 train 中

    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

    # 3. 评估模型
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=args.device,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )

    # 4. 保存评估结果
    print("\n" + "="*80)
    print("Saving Evaluation Results")
    print("="*80)

    # 保存详细结果
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Detailed results saved to: {results_file}")

    # 保存摘要
    summary = {
        "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_path": args.model_path,
        "test_file": args.test_file,
        "num_samples": results['num_samples'],
        "num_classes": results['num_classes'],
        "main_task": {
            "accuracy": results['main_task']['accuracy'],
            "precision": results['main_task']['precision'],
            "recall": results['main_task']['recall'],
            "f1": results['main_task']['f1']
        },
        "culture_task": {
            "accuracy": results['culture_task']['accuracy'],
            "precision": results['culture_task']['precision'],
            "recall": results['culture_task']['recall'],
            "f1": results['culture_task']['f1']
        }
    }

    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Summary saved to: {summary_file}")

    print("\n" + "="*80)
    print("✅ Evaluation Completed Successfully!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print("")
    print("📊 Summary:")
    print(f"   Main Task Accuracy:    {results['main_task']['accuracy']:.4f}")
    print(f"   Main Task F1:          {results['main_task']['f1']:.4f}")
    print(f"   Culture Task Accuracy: {results['culture_task']['accuracy']:.4f}")
    print(f"   Culture Task F1:       {results['culture_task']['f1']:.4f}")
    print("="*80)
    print("")


if __name__ == "__main__":
    main()

