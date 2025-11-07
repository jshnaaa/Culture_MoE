#!/usr/bin/env python3
"""
统一三个数据集的标签格式并合并

标签映射：
- CultureLLM:    1-10  (保持不变)
- NormAD:        11 (yes), 12 (no), 13 (neutral)
- CulturalBench: 14 (TRUE), 15 (FALSE)

使用方法：
    python prepare_unified_dataset.py \
        --culturellm /path/to/cultureLLM_merge_gen.json \
        --normad /path/to/normad_merge_gen.json \
        --culturalbench /path/to/CulturalBench_merge_gen.json \
        --output /path/to/unified_dataset.json
"""

import argparse
import json
from collections import Counter


def process_culturellm(data):
    """CultureLLM: 1-10 保持不变"""
    processed = []
    label_counts = Counter()

    for item in data:
        try:
            # 确保是整数
            label = int(item['output'])
            if 1 <= label <= 10:
                item['output'] = str(label)  # 转为字符串以保持一致性
                item['dataset_source'] = 'CultureLLM'
                processed.append(item)
                label_counts[label] += 1
            else:
                print(f"Warning: CultureLLM label out of range: {label}")
        except (ValueError, KeyError) as e:
            print(f"Warning: Invalid CultureLLM sample: {e}")

    return processed, label_counts


def process_normad(data):
    """NormAD: yes/no/neutral → 11/12/13"""
    label_map = {
        'yes': 11,
        'no': 12,
        'neutral': 13
    }

    processed = []
    label_counts = Counter()
    original_counts = Counter()

    for item in data:
        try:
            output = str(item['output']).lower().strip()
            original_counts[output] += 1

            if output in label_map:
                item['output'] = str(label_map[output])
                item['dataset_source'] = 'NormAD'
                item['original_label'] = output  # 保存原始标签用于调试
                processed.append(item)
                label_counts[label_map[output]] += 1
            else:
                print(f"Warning: Unknown NormAD label '{output}', skipping")
        except (ValueError, KeyError) as e:
            print(f"Warning: Invalid NormAD sample: {e}")

    return processed, label_counts, original_counts


def process_culturalbench(data):
    """CulturalBench: TRUE/FALSE → 14/15"""
    label_map = {
        'true': 14,
        'false': 15
    }

    processed = []
    label_counts = Counter()
    original_counts = Counter()

    for item in data:
        try:
            output = str(item['output']).upper().strip()
            original_counts[output] += 1

            if output in label_map:
                item['output'] = str(label_map[output])
                item['dataset_source'] = 'CulturalBench'
                item['original_label'] = output  # 保存原始标签用于调试
                processed.append(item)
                label_counts[label_map[output]] += 1
            else:
                print(f"Warning: Unknown CulturalBench label '{output}', skipping")
        except (ValueError, KeyError) as e:
            print(f"Warning: Invalid CulturalBench sample: {e}")

    return processed, label_counts, original_counts


def main():
    parser = argparse.ArgumentParser(description="统一三个数据集的标签格式并合并")
    parser.add_argument("--culturellm", type=str, required=True,
                        help="CultureLLM 数据集路径")
    parser.add_argument("--normad", type=str, required=True,
                        help="NormAD 数据集路径")
    parser.add_argument("--culturalbench", type=str, required=True,
                        help="CulturalBench 数据集路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径")
    parser.add_argument("--shuffle", action="store_true",
                        help="是否打乱数据集顺序")

    args = parser.parse_args()

    print("="*80)
    print("Preparing Unified Dataset")
    print("="*80)
    print("")

    # 加载数据集
    print("Step 1: Loading datasets...")
    print("-"*80)

    with open(args.culturellm, 'r', encoding='utf-8') as f:
        culturellm_data = json.load(f)
    print(f"  ✅ CultureLLM: {len(culturellm_data)} samples")

    with open(args.normad, 'r', encoding='utf-8') as f:
        normad_data = json.load(f)
    print(f"  ✅ NormAD: {len(normad_data)} samples")

    with open(args.culturalbench, 'r', encoding='utf-8') as f:
        culturalbench_data = json.load(f)
    print(f"  ✅ CulturalBench: {len(culturalbench_data)} samples")

    print(f"\n  Total input samples: {len(culturellm_data) + len(normad_data) + len(culturalbench_data)}")
    print("")

    # 处理数据集
    print("Step 2: Processing and unifying labels...")
    print("-"*80)

    culturellm_processed, culturellm_counts = process_culturellm(culturellm_data)
    print(f"  ✅ CultureLLM: {len(culturellm_processed)} samples processed")
    print(f"     Label distribution: {dict(culturellm_counts)}")

    normad_processed, normad_counts, normad_original = process_normad(normad_data)
    print(f"  ✅ NormAD: {len(normad_processed)} samples processed")
    print(f"     Original labels: {dict(normad_original)}")
    print(f"     Mapped labels: {dict(normad_counts)}")

    culturalbench_processed, culturalbench_counts, culturalbench_original = process_culturalbench(culturalbench_data)
    print(f"  ✅ CulturalBench: {len(culturalbench_processed)} samples processed")
    print(f"     Original labels: {dict(culturalbench_original)}")
    print(f"     Mapped labels: {dict(culturalbench_counts)}")

    print("")

    # 合并数据集
    print("Step 3: Merging datasets...")
    print("-"*80)

    merged_data = culturellm_processed + normad_processed + culturalbench_processed
    print(f"  Total merged samples: {len(merged_data)}")

    # 打乱数据（可选）
    if args.shuffle:
        import random
        random.seed(42)
        random.shuffle(merged_data)
        print(f"  ✅ Data shuffled (seed=42)")

    print("")

    # 保存
    print("Step 4: Saving unified dataset...")
    print("-"*80)
    print(f"  Output file: {args.output}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved {len(merged_data)} samples")
    print("")

    # 统计信息
    print("="*80)
    print("Summary")
    print("="*80)
    print("")
    print("Dataset composition:")
    print(f"  CultureLLM:    {len(culturellm_processed):>6} samples ({len(culturellm_processed)/len(merged_data)*100:.1f}%)")
    print(f"  NormAD:        {len(normad_processed):>6} samples ({len(normad_processed)/len(merged_data)*100:.1f}%)")
    print(f"  CulturalBench: {len(culturalbench_processed):>6} samples ({len(culturalbench_processed)/len(merged_data)*100:.1f}%)")
    print(f"  Total:         {len(merged_data):>6} samples")
    print("")

    print("Label mapping:")
    print("  CultureLLM:    1-10   (original)")
    print("  NormAD:        11 (yes), 12 (no), 13 (neutral)")
    print("  CulturalBench: 14 (TRUE), 15 (FALSE)")
    print("")

    print("Unified label range: 1-15")
    print("")

    print("="*80)
    print("✅ Done!")
    print("="*80)
    print("")
    print("Next step:")
    print(f"  sh run_train_lora_only_gen_unified.sh llama")
    print("")


if __name__ == "__main__":
    main()

