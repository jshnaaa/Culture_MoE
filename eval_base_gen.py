#!/usr/bin/env python3
"""
评估 Base 模型（生成式版本）

使用方法：
    python eval_base_gen.py \
        --model_path /path/to/base_model \
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


def generate_answer(model, tokenizer, instruction: str, input_text: str, num_classes: int = 5, max_new_tokens: int = 3):
    """
    生成答案 - 强制只生成数字（选项编号）

    Args:
        model: 模型
        tokenizer: tokenizer
        instruction: 指令
        input_text: 输入文本
        num_classes: 类别数量
        max_new_tokens: 最大生成 token 数（默认3，足够生成一个数字）

    Returns:
        raw_answer: 原始生成的答案
        predicted_label: 提取的标签（整数）
    """
    device = next(model.parameters()).device

    # 构建输入 - 明确要求只输出数字
    full_input = f"{instruction}\n{input_text}\n\nPlease answer with ONLY ONE NUMBER (0 to {num_classes-1}).\nYour answer:"

    # Tokenize
    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate - 严格限制生成长度
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,  # 贪婪解码
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            num_beams=1  # 禁用 beam search
        )

    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取生成的部分（去掉输入的 prompt）
    raw_answer = full_output[len(full_input):].strip()

    # 只保留第一个字符（应该是数字）
    if raw_answer:
        raw_answer = raw_answer.split()[0]  # 取第一个词
        if len(raw_answer) > 1:
            raw_answer = raw_answer[0]  # 只取第一个字符

    # 提取标签
    predicted_label = extract_label(raw_answer, num_classes)

    return raw_answer, predicted_label


def extract_label(answer: str, num_classes: int = 5):
    """
    从答案中提取标签

    多层提取策略：
    1. 尝试直接匹配数字
    2. 尝试匹配第一个数字
    3. 如果失败，返回默认值（中间类别）
    """
    if not answer:
        return num_classes // 2  # 默认返回中间类别

    # 方法1：尝试直接转换为整数
    try:
        label = int(answer)
        if 0 <= label < num_classes:
            return label
    except ValueError:
        pass

    # 方法2：尝试匹配第一个数字
    match = re.search(r'(\d+)', answer)
    if match:
        label = int(match.group(1))
        if 0 <= label < num_classes:
            return label

    # 方法3：默认返回中间类别
    print(f"⚠️  Warning: Failed to extract label from '{answer}', using default {num_classes // 2}")
    return num_classes // 2


def evaluate_model(model, tokenizer, test_data, device, num_classes: int = 5, save_answers: bool = True, output_dir: str = None):
    """
    评估模型

    Args:
        model: 模型
        tokenizer: tokenizer
        test_data: 测试数据
        device: 设备
        num_classes: 类别数量
        save_answers: 是否保存详细答案
        output_dir: 输出目录
    """
    print("Running evaluation...")

    all_preds = []
    all_labels = []
    all_culture_preds = []
    all_culture_labels = []
    all_answers = []  # 保存详细答案

    failed_count = 0  # 统计提取失败的数量

    for item in tqdm(test_data, desc="Evaluating"):
        instruction = item['instruction']
        input_text = item['input']
        label = int(item['output'])
        culture_label = int(item.get('label', label))

        # 生成答案
        raw_answer, pred = generate_answer(model, tokenizer, instruction, input_text, num_classes)

        # 统计失败
        if pred == num_classes // 2 and raw_answer and not raw_answer.isdigit():
            failed_count += 1

        all_preds.append(pred)
        all_labels.append(label)
        all_culture_preds.append(pred)
        all_culture_labels.append(culture_label)

        # 保存详细答案
        if save_answers:
            all_answers.append({
                "instruction": instruction,
                "input": input_text,
                "true_label": label,
                "culture_label": culture_label,
                "predicted_label": pred,
                "raw_answer": raw_answer,
                "correct": (pred == label),
                "culture_correct": (pred == culture_label)
            })

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_culture_preds = np.array(all_culture_preds)
    all_culture_labels = np.array(all_culture_labels)

    # 打印提取失败统计
    if failed_count > 0:
        print(f"\n⚠️  Warning: {failed_count}/{len(test_data)} samples failed to extract valid labels")
        print(f"   Failed rate: {100 * failed_count / len(test_data):.2f}%")

    # 保存详细答案
    if save_answers and output_dir:
        answers_file = os.path.join(output_dir, "generated_answers.json")
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(all_answers, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {len(all_answers)} detailed answers to: {answers_file}")

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
        "num_samples": len(all_labels),
        "failed_extractions": failed_count,
        "failed_rate": float(failed_count / len(test_data)) if len(test_data) > 0 else 0.0
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="评估 Base 模型（生成式版本）")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="评估结果输出目录")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="类别数量")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--save_answers", action="store_true", default=True,
                        help="是否保存详细答案")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("Base Model Evaluation (Generative Version)")
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
    results = evaluate_model(
        model,
        tokenizer,
        test_data,
        args.device,
        num_classes=args.num_classes,
        save_answers=args.save_answers,
        output_dir=args.output_dir
    )

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

