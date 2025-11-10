#!/usr/bin/env python3
"""
在 CultureLLM 数据集上评估 Base 模型（LLaMA 或 Qwen）

使用方法：
    python ft_base.py \
        --base_model_path /path/to/base_model \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output

数据格式：
    {
        "text": "### Question: ... ### Answer: 10",
        "text_mask": "...",
        "label": "1"
    }

说明：
    - 只使用 text 字段进行评估
    - text_mask 和 label 字段被忽略
    - 模型生成答案，通过正则表达式提取数字
    - 计算准确率
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CultureLLMDataset:
    """
    CultureLLM 数据集

    数据格式：
    {
        "text": "### Question: ... ### Answer: 10",
        "text_mask": "...",
        "label": "1"
    }
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: 数据文件路径
        """
        print(f"Loading data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item.get('label', '')

        return {
            'text': text,
            'label': label
        }


def extract_answer_from_text(text: str) -> str:
    """
    从生成的文本中提取答案

    使用正则表达式查找 "Answer: " 后面的数字

    Args:
        text: 生成的文本

    Returns:
        提取的答案（数字字符串），如果没有找到则返回空字符串
    """
    # 查找 "Answer: " 后面的数字
    match = re.search(r'Answer:\s*(\d+)', text)
    if match:
        return match.group(1)

    return ""


def generate_answer(model, tokenizer, text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    """
    使用模型生成答案

    Args:
        model: 模型
        tokenizer: tokenizer
        text: 输入文本
        device: 设备
        max_new_tokens: 最大生成 token 数

    Returns:
        生成的文本
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None
        )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text


def evaluate_base_model(model, tokenizer, dataset, device, output_dir):
    """
    在数据集上评估 Base 模型

    Args:
        model: 模型
        tokenizer: tokenizer
        dataset: 数据集
        device: 设备
        output_dir: 输出目录

    Returns:
        dict: 包含评估指标的字典
    """
    model.eval()

    correct = 0
    total = 0
    generated_data = []

    print("\nGenerating answers on dataset...")

    for idx in tqdm(range(len(dataset)), desc="Generating"):
        sample = dataset[idx]
        text = sample['text']
        true_label = sample['label']

        # 生成答案
        generated_text = generate_answer(model, tokenizer, text, device)

        # 提取答案
        predicted_answer = extract_answer_from_text(generated_text)

        # 比对答案
        if predicted_answer == true_label:
            correct += 1
        total += 1

        # 保存生成的数据
        generated_data.append({
            'text': text,
            'true_label': true_label,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': predicted_answer == true_label
        })

    accuracy = correct / total if total > 0 else 0

    # 保存生成的答案
    with open(os.path.join(output_dir, 'generated_answers.json'), 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'generated_data': generated_data
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base model on CultureLLM dataset")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (JSON format)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Evaluating Base Model on CultureLLM Dataset")
    print("="*80)
    print(f"Base model: {args.base_model_path}")
    print(f"Training data: {args.train_file}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded")

    # 加载数据
    print("\nLoading dataset...")
    dataset = CultureLLMDataset(args.train_file)
    print("✅ Dataset loaded")

    # 加载模型
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Base model loaded")

    # 评估
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80 + "\n")

    eval_metrics = evaluate_base_model(model, tokenizer, dataset, args.device, args.output_dir)

    # 保存评估结果
    results = {
        'accuracy': eval_metrics['accuracy'],
        'correct': eval_metrics['correct'],
        'total': eval_metrics['total'],
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存配置
    config = {
        'base_model': args.base_model_path,
        'max_length': args.max_length
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 打印结果
    print("\n" + "="*80)
    print("📊 Evaluation Results")
    print("="*80)
    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Correct: {eval_metrics['correct']}/{eval_metrics['total']}")
    print("="*80)

    print(f"\n✅ Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nFiles generated:")
    print(f"  - eval_results.json (评估结果)")
    print(f"  - generated_answers.json (生成的答案)")
    print(f"  - config.json (配置)")
    print("="*80)


if __name__ == "__main__":
    main()

