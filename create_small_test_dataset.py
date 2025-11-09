#!/usr/bin/env python3
"""
创建小规模测试数据集

用法：
    python create_small_test_dataset.py \
        --input_file /path/to/large_dataset.json \
        --output_file /path/to/small_dataset.json \
        --num_samples 100 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def create_small_dataset(input_file: str, output_file: str, num_samples: int = 100, seed: int = 42):
    """
    从大数据集中随机抽取小规模数据集

    Args:
        input_file: 输入数据文件路径
        output_file: 输出数据文件路径
        num_samples: 抽取的样本数量
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)

    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # 确保 num_samples 不超过数据集大小
    num_samples = min(num_samples, len(data))

    # 随机抽取样本
    sampled_data = random.sample(data, num_samples)

    print(f"Sampled: {len(sampled_data)} samples")

    # 分析标签分布
    labels = {}
    for item in sampled_data:
        label = str(item.get('output', 'unknown'))
        labels[label] = labels.get(label, 0) + 1

    print(f"\nLabel distribution:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count} ({100*count/len(sampled_data):.1f}%)")

    # 保存到文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Small dataset saved to: {output_file}")
    print(f"   Size: {len(sampled_data)} samples")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="创建小规模测试数据集")
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入数据文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出数据文件路径")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="抽取的样本数量（默认 100）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")

    args = parser.parse_args()

    create_small_dataset(
        args.input_file,
        args.output_file,
        args.num_samples,
        args.seed
    )


if __name__ == "__main__":
    main()

