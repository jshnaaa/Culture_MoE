#!/usr/bin/env python3
"""
将 CultureLLM 数据格式转换为 Alpaca 格式（适合 LoRA SFT）

原始格式（CultureLLM）：
{
    "text": "### Question: ... ### Answer: 2",
    "text_mask": "...",
    "label": "1"
}

转换后格式（Alpaca）：
{
    "instruction": "Give me the answer from 1 to 4: ...",
    "input": "",
    "output": "2"
}

使用方法：
    python convert_culturellm_to_alpaca.py \
        --input_file /path/to/cultureLLM_merge_gen.json \
        --output_file /path/to/cultureLLM_alpaca.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


def extract_question_and_answer(text: str):
    """
    从 CultureLLM 格式的 text 中提取 Question 和 Answer

    Args:
        text: 原始文本，格式为 "### Question: ... ### Answer: 2"

    Returns:
        tuple: (question, answer) 或 (None, None) 如果提取失败
    """
    # 提取 Question 部分
    question_match = re.search(r'### Question:\s*(.*?)\n\s*### Answer:', text, re.DOTALL)
    if not question_match:
        return None, None

    question = question_match.group(1).strip()

    # 提取 Answer 部分
    answer_match = re.search(r'### Answer:\s*(\d+)', text)
    if not answer_match:
        return None, None

    answer = answer_match.group(1).strip()

    return question, answer


def convert_to_alpaca(input_file: str, output_file: str):
    """
    将 CultureLLM 格式转换为 Alpaca 格式

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"Loading data from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # 转换数据
    converted_data = []
    failed_count = 0

    for idx, item in enumerate(data):
        text = item.get('text', '')

        # 提取 Question 和 Answer
        question, answer = extract_question_and_answer(text)

        if question is None or answer is None:
            failed_count += 1
            if idx < 5:  # 只显示前 5 个失败的样本
                print(f"  ⚠️  Failed to extract from sample {idx}: {text[:100]}")
            continue

        # 转换为 Alpaca 格式
        alpaca_item = {
            "instruction": question,
            "input": "",
            "output": answer
        }

        converted_data.append(alpaca_item)

    print(f"\n✅ Conversion completed:")
    print(f"   Original samples: {len(data)}")
    print(f"   Converted samples: {len(converted_data)}")
    print(f"   Failed samples: {failed_count}")

    # 保存转换后的数据
    print(f"\nSaving to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(converted_data)} samples")

    # 显示示例
    print(f"\n📋 Example conversions:")
    print(f"\nOriginal format:")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))
    print(f"\nConverted format:")
    print(json.dumps(converted_data[0], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Convert CultureLLM format to Alpaca format")

    parser.add_argument("--input_file", type=str, required=True,
                        help="Input file path (CultureLLM format)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path (Alpaca format)")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Converting CultureLLM format to Alpaca format")
    print("="*80 + "\n")

    # 检查输入文件
    if not Path(args.input_file).exists():
        print(f"❌ Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 转换
    convert_to_alpaca(args.input_file, args.output_file)

    print("\n" + "="*80)
    print("✅ Conversion completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

