#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 Base 模型 + LoRA 权重还原完整模型并评估（生成式版本）

用法：
    python eval_lora_only_from_components.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --test_file /path/to/test.json \
        --output_dir /path/to/output
"""

import os
import json
import argparse
import re
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model_from_components(base_model_path: str, lora_weights_path: str, device: str = "cuda"):
    """
    从 base 模型和 LoRA 权重还原完整模型

    Args:
        base_model_path: base 模型路径
        lora_weights_path: LoRA 权重路径
        device: 设备

    Returns:
        model: 合并后的完整模型
        tokenizer: tokenizer
    """
    print("Loading model from components...")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA weights: {lora_weights_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 base 模型
    print("Step 1: Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载 LoRA 权重
    print("Step 2: Loading LoRA weights...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        is_trainable=False
    )

    # 合并 LoRA 权重到 base 模型（仅在内存中）
    print("Step 3: Merging LoRA weights into base model...")
    model = peft_model.merge_and_unload()

    # 清理内存
    del base_model
    del peft_model
    torch.cuda.empty_cache()

    print("✅ Model loaded and merged successfully (in-memory only)")

    return model, tokenizer


def generate_answer(model, tokenizer, instruction: str, input_text: str, num_classes: int = 10, max_new_tokens: int = 10):
    """
    生成答案

    Args:
        model: 模型
        tokenizer: tokenizer
        instruction: 指令
        input_text: 输入文本
        num_classes: 类别数量
        max_new_tokens: 最大生成 token 数

    Returns:
        raw_answer: 原始生成的答案
        predicted_label: 提取的标签（整数）
    """
    device = next(model.parameters()).device

    # 构建 prompt
    if num_classes <= 10:
        full_input = f"{instruction}\n{input_text}\n\nPlease answer with ONLY ONE NUMBER (1 to {num_classes}).\nYour answer:"
    else:
        full_input = f"{instruction}\n{input_text}\n\nYour answer:"

    # Tokenize
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            num_beams=1
        )

    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取生成的部分
    raw_answer = full_output[len(full_input):].strip()

    # 提取第一个词
    if raw_answer:
        raw_answer = raw_answer.split()[0]
        raw_answer = raw_answer.strip('.,!?;:')

    # 提取标签
    predicted_label = extract_label(raw_answer, num_classes)

    return raw_answer, predicted_label


def extract_label(answer: str, num_classes: int = 10):
    """
    从答案中提取标签

    Args:
        answer: 模型回答
        num_classes: 类别数量

    Returns:
        label: 标签（0-indexed）
    """
    if not answer:
        return num_classes // 2

    # 尝试直接转换为整数
    try:
        label = int(answer)
        # 1-indexed -> 0-indexed
        if 1 <= label <= num_classes:
            return label - 1
        elif 0 <= label < num_classes:
            return label
        elif label > num_classes:
            return num_classes - 1
    except ValueError:
        pass

    # 尝试匹配第一个数字
    match = re.search(r'(\d+)', answer)
    if match:
        label = int(match.group(1))
        if 1 <= label <= num_classes:
            return label - 1
        elif 0 <= label < num_classes:
            return label
        elif label > num_classes:
            return num_classes - 1

    # 默认返回中间类别
    return num_classes // 2


def evaluate_model(model, tokenizer, test_data, num_classes: int = 10, output_dir: str = None):
    """
    评估模型

    Args:
        model: 模型
        tokenizer: tokenizer
        test_data: 测试数据
        num_classes: 类别数量
        output_dir: 输出目录

    Returns:
        results: 评估结果
    """
    print("\nRunning evaluation...")
    print(f"Number of classes: {num_classes}")

    all_preds = []
    all_labels = []
    all_answers = []

    failed_count = 0

    for item in tqdm(test_data, desc="Evaluating"):
        instruction = item['instruction']
        input_text = item['input']

        # 处理标签（1-indexed -> 0-indexed）
        label = int(item['output'])
        if label >= 1:
            label = label - 1

        # 生成答案
        raw_answer, pred = generate_answer(model, tokenizer, instruction, input_text, num_classes)

        # 统计失败
        if pred == num_classes // 2 and raw_answer and not raw_answer.isdigit():
            failed_count += 1

        all_preds.append(pred)
        all_labels.append(label)

        # 保存详细答案
        all_answers.append({
            "instruction": instruction,
            "input": input_text,
            "true_label": label,
            "predicted_label": pred,
            "raw_answer": raw_answer,
            "correct": (pred == label)
        })

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 保存详细答案
    if output_dir:
        answers_file = os.path.join(output_dir, "generated_answers.json")
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(all_answers, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {len(all_answers)} detailed answers to: {answers_file}")

    # 打印提取失败统计
    if failed_count > 0:
        print(f"\n⚠️  Warning: {failed_count}/{len(test_data)} samples failed to extract valid labels")
        print(f"   Failed rate: {100 * failed_count / len(test_data):.2f}%")

    # 计算指标
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print(f"\n📊 Classification Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1:        {f1:.4f}")
    print("="*80)

    # 构建结果字典
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "num_samples": len(all_labels),
        "failed_extractions": failed_count,
        "failed_rate": float(failed_count / len(test_data)) if len(test_data) > 0 else 0.0
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="从 Base 模型 + LoRA 权重评估（生成式版本）")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Base 模型路径")
    parser.add_argument("--lora_weights_path", type=str, required=True,
                        help="LoRA 权重路径")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="评估结果输出目录")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="类别数量（默认 10）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LoRA Model Evaluation (From Components)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.lora_weights_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Num classes: {args.num_classes}")
    print("="*80)
    print("")

    # 加载模型
    model, tokenizer = load_model_from_components(
        args.base_model_path,
        args.lora_weights_path,
        args.device
    )

    # 加载测试数据
    print(f"\nLoading test data from: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ Loaded {len(test_data)} test samples")

    # 评估模型
    results = evaluate_model(
        model,
        tokenizer,
        test_data,
        num_classes=args.num_classes,
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
        "base_model_path": args.base_model_path,
        "lora_weights_path": args.lora_weights_path,
        "test_file": args.test_file,
        "num_samples": results['num_samples'],
        "num_classes": args.num_classes,
        "accuracy": results['accuracy'],
        "precision": results['precision'],
        "recall": results['recall'],
        "f1": results['f1'],
        "failed_extractions": results['failed_extractions'],
        "failed_rate": results['failed_rate']
    }

    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Summary saved to: {summary_file}")

    print("\n" + "="*80)
    print("✅ Evaluation completed successfully!")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   Samples: {results['num_samples']}")
    print("")


if __name__ == "__main__":
    main()

