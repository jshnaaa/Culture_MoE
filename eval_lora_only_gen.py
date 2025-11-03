#!/usr/bin/env python3
"""
评估 LoRA Only 模型（生成式版本）

使用方法：
    python eval_lora_only_gen.py \
        --model_path /path/to/merged_model \
        --test_file /path/to/test_data.json \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
import re
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_model(model_path: str, device: str = "cuda"):
    """加载模型"""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    print("✅ Model loaded\n")
    return model, tokenizer


def generate_answer(model, tokenizer, instruction: str, input_text: str, max_new_tokens: int = 10):
    """生成答案"""
    device = next(model.parameters()).device

    # 构建输入
    full_input = f"{instruction}\n{input_text}"

    # Tokenize
    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪婪解码
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取生成的部分（去掉输入）
    answer = answer[len(full_input):].strip()

    return answer


def extract_label(answer: str):
    """从答案中提取标签（数字）"""
    # 尝试匹配数字
    match = re.search(r'\b(\d+)\b', answer)
    if match:
        return int(match.group(1))

    # 如果没有找到，返回 -1
    return -1


def evaluate_model(model, tokenizer, test_data, device):
    """评估模型"""
    print("Running evaluation...")

    all_preds = []
    all_labels = []
    all_culture_preds = []
    all_culture_labels = []

    for item in tqdm(test_data, desc="Evaluating"):
        instruction = item['instruction']
        input_text = item['input']
        label = int(item['output'])  # 真实标签
        culture_label = int(item.get('label', label))  # 文化标签

        # 生成答案
        answer = generate_answer(model, tokenizer, instruction, input_text)

        # 提取预测标签
        pred = extract_label(answer)

        all_preds.append(pred)
        all_labels.append(label)
        all_culture_preds.append(pred)  # 生成式模型没有单独的 culture 预测
        all_culture_labels.append(culture_label)

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
        "num_samples": len(all_labels)
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="评估 LoRA Only 模型（生成式版本）")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="评估结果输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LoRA Only Model Evaluation (Generative Version)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model path: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    print("")

    # 加载模型
    model, tokenizer = load_model(args.model_path, args.device)

    # 加载测试数据
    print(f"Loading test data from: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ Loaded {len(test_data)} test samples\n")

    # 评估模型
    results = evaluate_model(model, tokenizer, test_data, args.device)

    # 保存评估结果
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

