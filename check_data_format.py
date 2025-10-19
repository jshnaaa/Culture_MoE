#!/usr/bin/env python3
"""
检查数据集的类别分布
"""
import json
import sys
from collections import Counter

def check_dataset(file_path):
    """检查数据集的类别分布"""
    print(f"\n{'='*60}")
    print(f"Checking: {file_path}")
    print(f"{'='*60}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 统计标签
        labels = [item['output'] for item in data]
        label_counts = Counter(labels)

        print(f"\nTotal samples: {len(data)}")
        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = count / len(data) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")

        # 判断是二分类还是三分类
        unique_labels = sorted(label_counts.keys())
        print(f"\nUnique labels: {unique_labels}")

        if len(unique_labels) == 2:
            print("✅ This is a BINARY classification dataset")
            if unique_labels == [0, 1]:
                print("   Labels: 0=no, 1=yes")
            elif unique_labels == [0, 2]:
                print("   Labels: 0=no, 2=yes")
        elif len(unique_labels) == 3:
            print("✅ This is a THREE-class classification dataset")
            if unique_labels == [0, 1, 2]:
                print("   Labels: 0=no, 1=neutral, 2=yes")
        else:
            print(f"⚠️  Unexpected number of classes: {len(unique_labels)}")

        # 检查字段
        print(f"\nData fields:")
        if data:
            fields = list(data[0].keys())
            print(f"  {fields}")

            # 检查是否有 instruction_mask
            if 'instruction_mask' in fields:
                print("  ✅ Has 'instruction_mask' (dual input)")
            else:
                print("  ❌ No 'instruction_mask' (single input)")

        print(f"{'='*60}\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_data_format.py <data_file1> [data_file2] ...")
        print("\nExample:")
        print("  python check_data_format.py /root/autodl-fs/normad_ed_merge.json")
        print("  python check_data_format.py /root/autodl-fs/CulturalBench_Hard_merge.json")
        sys.exit(1)

    for file_path in sys.argv[1:]:
        check_dataset(file_path)

