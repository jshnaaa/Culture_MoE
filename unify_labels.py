#!/usr/bin/env python3
"""
统一三个数据集的标签格式为数字

CultureLLM: 1-10 → 保持不变
NormAD: yes/no/neutral → 11/12/13
CulturalBench: TRUE/FALSE → 14/15

这样所有标签都是数字，可以合并训练
"""

import argparse
import json


def unify_culturellm(data):
    """CultureLLM: 1-10 保持不变"""
    for item in data:
        # 确保是整数
        item['output'] = int(item['output'])
    return data


def unify_normad(data):
    """NormAD: yes/no/neutral → 11/12/13"""
    label_map = {
        'yes': 11,
        'no': 12,
        'neutral': 13
    }

    for item in data:
        output = str(item['output']).lower().strip()
        if output in label_map:
            item['output'] = label_map[output]
        else:
            print(f"Warning: Unknown label '{output}' in NormAD, skipping")
            item['output'] = 13  # 默认为 neutral

    return data


def unify_culturalbench(data):
    """CulturalBench: TRUE/FALSE → 14/15"""
    label_map = {
        'true': 14,
        'false': 15
    }

    for item in data:
        output = str(item['output']).upper().strip()
        if output in label_map:
            item['output'] = label_map[output]
        else:
            print(f"Warning: Unknown label '{output}' in CulturalBench, skipping")
            item['output'] = 15  # 默认为 FALSE

    return data


def main():
    parser = argparse.ArgumentParser(description="统一三个数据集的标签格式")
    parser.add_argument("--culturellm", type=str, required=True,
                        help="CultureLLM 数据集路径")
    parser.add_argument("--normad", type=str, required=True,
                        help="NormAD 数据集路径")
    parser.add_argument("--culturalbench", type=str, required=True,
                        help="CulturalBench 数据集路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径")

    args = parser.parse_args()

    print("="*80)
    print("Unifying labels for three datasets")
    print("="*80)

    # 加载数据集
    print("\nLoading datasets...")
    with open(args.culturellm, 'r', encoding='utf-8') as f:
        culturellm_data = json.load(f)
    print(f"  CultureLLM: {len(culturellm_data)} samples")

    with open(args.normad, 'r', encoding='utf-8') as f:
        normad_data = json.load(f)
    print(f"  NormAD: {len(normad_data)} samples")

    with open(args.culturalbench, 'r', encoding='utf-8') as f:
        culturalbench_data = json.load(f)
    print(f"  CulturalBench: {len(culturalbench_data)} samples")

    # 统一标签
    print("\nUnifying labels...")
    culturellm_data = unify_culturellm(culturellm_data)
    print(f"  CultureLLM: 1-10 (unchanged)")

    normad_data = unify_normad(normad_data)
    print(f"  NormAD: yes/no/neutral → 11/12/13")

    culturalbench_data = unify_culturalbench(culturalbench_data)
    print(f"  CulturalBench: TRUE/FALSE → 14/15")

    # 合并数据集
    print("\nMerging datasets...")
    merged_data = culturellm_data + normad_data + culturalbench_data
    print(f"  Total: {len(merged_data)} samples")

    # 保存
    print(f"\nSaving to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print("\n✅ Done!")
    print("="*80)
    print("\nLabel mapping:")
    print("  CultureLLM:     1-10  (original)")
    print("  NormAD:         11 (yes), 12 (no), 13 (neutral)")
    print("  CulturalBench:  14 (TRUE), 15 (FALSE)")
    print("="*80)


if __name__ == "__main__":
    main()

