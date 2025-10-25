#!/usr/bin/env python3
"""
测试文化维度标签的数据加载
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from transformers import AutoTokenizer
from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data
from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator


def create_test_data():
    """创建测试数据"""
    test_data = [
        {
            "instruction": "Question about culture 0",
            "instruction_mask": "Masked question",
            "input": "Test input 1",
            "output": 0,
            "label": "0"  # 单个文化维度
        },
        {
            "instruction": "Question about culture 1 and 2",
            "instruction_mask": "Masked question",
            "input": "Test input 2",
            "output": 1,
            "label": "1,2"  # 多个文化维度
        },
        {
            "instruction": "Question about culture 3",
            "instruction_mask": "Masked question",
            "input": "Test input 3",
            "output": 2,
            "label": "3"
        },
        {
            "instruction": "Question about culture 4 and 5",
            "instruction_mask": "Masked question",
            "input": "Test input 4",
            "output": 0,
            "label": "4,5"
        },
    ]

    # 保存测试数据
    test_file = "test_culture_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Created test data: {test_file}")
    print(f"   Total samples: {len(test_data)}")

    return test_file


def test_data_loading():
    """测试数据加载"""
    print("="*60)
    print("Testing Culture Label Data Loading")
    print("="*60)

    # 1. 创建测试数据
    test_file = create_test_data()

    # 2. 加载 tokenizer
    print("\n1. Loading tokenizer...")
    model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"

    # 如果模型路径不存在，使用一个简单的 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("   ✅ Tokenizer loaded from model path")
    except:
        print("   ⚠️  Model path not found, using gpt2 tokenizer for testing")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载和处理数据
    print("\n2. Loading and processing data...")
    data = load_and_process_dual_classification_data(
        data_path=test_file,
        tokenizer=tokenizer,
        max_length=512,
        val_split=0,
        num_proc=1
    )

    dataset = data['train']
    print(f"   ✅ Dataset loaded: {len(dataset)} samples")

    # 4. 检查数据格式
    print("\n3. Checking data format...")
    print(f"   Dataset columns: {dataset.column_names}")

    # 5. 查看样本
    print("\n4. Sample data:")
    for i in range(min(4, len(dataset))):
        sample = dataset[i]
        print(f"\n   Sample {i}:")
        print(f"     input_ids shape: {len(sample['input_ids'])}")
        print(f"     input_ids_mask shape: {len(sample['input_ids_mask'])}")
        print(f"     labels: {sample['labels']}")
        print(f"     culture_labels: {sample['culture_labels']}")

    # 6. 测试 DataCollator
    print("\n5. Testing DataCollator...")
    collator = DualClassificationDataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=512
    )

    # 创建一个批次
    batch_samples = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = collator(batch_samples)

    print(f"   Batch keys: {batch.keys()}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   input_ids_mask shape: {batch['input_ids_mask'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   labels: {batch['labels']}")
    print(f"   culture_labels: {batch['culture_labels']}")

    # 7. 验证 culture_labels 格式
    print("\n6. Validating culture_labels format...")
    for i, culture_label in enumerate(batch['culture_labels']):
        print(f"   Sample {i}: {culture_label} (type: {type(culture_label)})")
        if isinstance(culture_label, list):
            print(f"     ✅ Correct format (list)")
            for label_id in culture_label:
                if isinstance(label_id, int) and 0 <= label_id <= 5:
                    print(f"       ✅ Valid dimension: {label_id}")
                else:
                    print(f"       ❌ Invalid dimension: {label_id}")
        else:
            print(f"     ❌ Wrong format (expected list, got {type(culture_label)})")

    print("\n" + "="*60)
    print("✅ Test completed!")
    print("="*60)

    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n🗑️  Cleaned up test file: {test_file}")


if __name__ == "__main__":
    test_data_loading()

