#!/usr/bin/env python3
"""
在新格式 CultureLLM 数据集上评估 Base 模型（LLaMA 或 Qwen）

使用方法：
    python ft_base_gen.py \
        --base_model_path /path/to/base_model \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output

数据格式（新格式）：
    {
        "instruction": "### Question: ... ### Answer: ",
        "instruction_mask": "### Question: ... [MASK] ### Answer: ",
        "input": "",
        "output": "1",
        "label": "0"
    }

说明：
    - 只使用 instruction + input 字段进行评估
    - instruction_mask 和 label 字段被忽略
    - 模型生成答案，通过正则表达式提取数字
    - 计算准确率
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CultureLLMNewFormatDataset:
    """
    CultureLLM 新格式数据集

    数据格式：
    {
        "instruction": "### Question: ... ### Answer: ",
        "instruction_mask": "### Question: ... [MASK] ### Answer: ",
        "input": "",
        "output": "1",
        "label": "0"
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

        # 新格式：instruction + input
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        label = item.get('label', '')

        # 构建完整的文本
        if input_text:
            full_text = f"{instruction}{input_text}"
        else:
            full_text = instruction

        return {
            'text': full_text,
            'output': output_text,
            'label': label
        }


def extract_answer_from_text(text: str) -> str:
    """
    从生成的文本中提取答案

    使用正则表达式查找数字

    Args:
        text: 生成的文本

    Returns:
        提取的答案（数字字符串），如果没有找到则返回空字符串
    """
    # 查找数字（1-10 或 1-4）
    match = re.search(r'\d+', text)
    if match:
        return match.group(0)

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
    # ✅ 检查输入文本是否为空
    if not text or text.strip() == "":
        return ""

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ✅ 检查 input_ids 是否为空
    if inputs['input_ids'].shape[1] == 0:
        return ""

    with torch.no_grad():
        try:
            # 尝试使用最简单的配置
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"⚠️  Warning: Generation failed with error: {e}")
            print("   Trying alternative configuration...")
            # 如果失败，尝试更简单的配置
            try:
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    max_new_tokens=max_new_tokens
                )
            except Exception as e2:
                print(f"⚠️  Warning: Alternative generation also failed: {e2}")
                return ""

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
        true_output = sample['output']
        label = sample['label']

        # 生成答案
        generated_text = generate_answer(model, tokenizer, text, device)

        # 提取答案
        predicted_answer = extract_answer_from_text(generated_text)

        # 比对答案
        if predicted_answer == true_output:
            correct += 1
        total += 1

        # 保存生成的数据
        generated_data.append({
            'text': text,
            'true_output': true_output,
            'label': label,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': predicted_answer == true_output
        })

    accuracy = correct / total if total > 0 else 0

    # 保存生成的答案
    with open(os.path.join(output_dir, 'generated_answers.json'), 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # 打印前五条生成的答案
    print("\n📋 前五条生成的答案:")
    print("-" * 100)
    for idx in range(min(5, len(generated_data))):
        item = generated_data[idx]
        print(f"\n样本 {idx + 1}:")
        print(f"  Question: {item['text'][:80]}...")
        print(f"  True Output: {item['true_output']}")
        print(f"  Generated Text: {item['generated_text']}")
        print(f"  Predicted Answer: {item['predicted_answer']}")
        print(f"  Correct: {'✅' if item['correct'] else '❌'}")
    print("\n" + "-" * 100)

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'generated_data': generated_data
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base model on CultureLLM dataset (new format)")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (new format JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Evaluating Base Model on CultureLLM Dataset (New Format)")
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
    dataset = CultureLLMNewFormatDataset(args.train_file)
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
        'max_length': args.max_length,
        'data_format': 'new_format (instruction + instruction_mask + input + output + label)'
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

