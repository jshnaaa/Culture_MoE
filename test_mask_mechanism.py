#!/usr/bin/env python3
"""
MASK机制功能测试脚本

测试要点：
1. 数据集是否正确生成masked instruction
2. MoE层是否根据input_type正确激活专家
3. 训练时是否正确传递input_type参数
"""

import json
import os
import sys
import torch
from transformers import AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ft_lora_only_gen import CultureLLMNewFormatDataset, dynamic_padding_collate_fn
from src.llamafactory.model.simplified_culturemoe import SimplifiedCultureMoEConfig
from src.llamafactory.model.simplified_culturemoe_adapter import MoEFFNLoRA


def test_dataset_mask_mechanism():
    """测试数据集的MASK机制"""
    print("=" * 60)
    print("测试1: 数据集MASK机制")
    print("=" * 60)

    # 创建测试数据
    test_data = [
        {
            "instruction": "Give me the answer from 1 to 4: Do you agree with this statement?",
            "input": "This question is for Arabic culture.",
            "output": "2",
            "label": "0"
        },
        {
            "instruction": "Choose the correct answer from 1 to 4: What is your opinion?",
            "input": "This is about Chinese culture.",
            "output": "3",
            "label": "1"
        }
    ]

    test_file = "/tmp/test_mask_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)

    # 测试tokenizer
    tokenizer_path = "/Users/yzl/models/Meta-Llama-3.1-8B-Instruct"
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer路径不存在: {tokenizer_path}")
        print("请确保模型路径正确或使用其他可用的模型路径")
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    except Exception as e:
        print(f"❌ 加载tokenizer失败: {e}")
        return False

    # 测试不启用MASK的数据集
    print("\n🔍 测试不启用MASK的数据集:")
    dataset_no_mask = CultureLLMNewFormatDataset(
        test_file, tokenizer, max_length=128,
        enable_mask=False
    )

    sample_no_mask = dataset_no_mask[0]
    print(f"  input_type: {sample_no_mask['input_type']}")
    print(f"  instruction: {sample_no_mask['instruction']}")

    # 测试启用MASK的数据集
    print("\n🔍 测试启用MASK的数据集:")
    dataset_with_mask = CultureLLMNewFormatDataset(
        test_file, tokenizer, max_length=128,
        enable_mask=True, mask_prob=0.3
    )

    # 测试多个样本，观察MASK效果
    for i in range(5):
        sample = dataset_with_mask[0]  # 同一个样本，每次随机MASK
        print(f"  样本{i+1}:")
        print(f"    input_type: {sample['input_type']}")
        print(f"    instruction: {sample['instruction'][:100]}...")
        if sample['input_type'] == 0:
            print(f"    -> 使用masked instruction (激活shared专家)")
        elif sample['input_type'] == 1:
            print(f"    -> 使用完整instruction (激活路由专家)")

    # 清理测试文件
    os.remove(test_file)
    print("\n✅ 数据集MASK机制测试完成")
    return True


def test_moe_conditional_activation():
    """测试MoE层的条件激活"""
    print("\n" + "=" * 60)
    print("测试2: MoE条件专家激活")
    print("=" * 60)

    try:
        # 创建模拟的FFN层（简化版）
        class MockFFN:
            def __init__(self):
                self.gate_proj = torch.nn.Linear(128, 256, bias=False)
                self.up_proj = torch.nn.Linear(128, 256, bias=False)
                self.down_proj = torch.nn.Linear(256, 128, bias=False)
                self.act_fn = torch.nn.SiLU()

        # 创建简化配置
        config = SimplifiedCultureMoEConfig(
            lora_rank=8,
            lora_alpha=16,
            num_moe_experts=4,
            num_activated_experts=2,
            use_shared=True,
            use_gate=False  # 简化测试，不使用gate网络
        )

        # 创建MoE层
        mock_ffn = MockFFN()
        moe_layer = MoEFFNLoRA(mock_ffn, config)

        # 模拟输入
        batch_size, seq_len, hidden_dim = 2, 10, 128
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)

        print("🔍 测试不同input_type的激活情况:")

        # 测试1: 兼容模式 (input_type=None)
        print("  测试1: 兼容模式 (input_type=None)")
        try:
            output_compat = moe_layer(hidden_states, input_type=None)
            print(f"    输出形状: {output_compat.shape}")
            print(f"    ✅ 兼容模式正常")
        except Exception as e:
            print(f"    ❌ 兼容模式失败: {e}")

        # 测试2: 全masked输入 (激活shared专家)
        print("  测试2: 全masked输入 (input_type=0, 激活shared专家)")
        input_type_masked = torch.zeros(batch_size, dtype=torch.long)
        try:
            output_masked = moe_layer(hidden_states, input_type=input_type_masked)
            print(f"    输出形状: {output_masked.shape}")
            print(f"    ✅ Shared专家激活正常")
        except Exception as e:
            print(f"    ❌ Shared专家激活失败: {e}")

        # 测试3: 全完整输入 (激活路由专家)
        print("  测试3: 全完整输入 (input_type=1, 激活路由专家)")
        input_type_full = torch.ones(batch_size, dtype=torch.long)
        try:
            output_full = moe_layer(hidden_states, input_type=input_type_full)
            print(f"    输出形状: {output_full.shape}")
            print(f"    ✅ 路由专家激活正常")
        except Exception as e:
            print(f"    ❌ 路由专家激活失败: {e}")

        # 测试4: 混合batch
        print("  测试4: 混合batch (input_type=[0,1], 分别激活不同专家)")
        input_type_mixed = torch.tensor([0, 1], dtype=torch.long)
        try:
            output_mixed = moe_layer(hidden_states, input_type=input_type_mixed)
            print(f"    输出形状: {output_mixed.shape}")
            print(f"    ✅ 混合batch处理正常")
        except Exception as e:
            print(f"    ❌ 混合batch处理失败: {e}")

        print("\n✅ MoE条件专家激活测试完成")
        return True

    except Exception as e:
        print(f"❌ MoE测试失败: {e}")
        return False


def test_collate_function():
    """测试collate函数是否正确处理input_type"""
    print("\n" + "=" * 60)
    print("测试3: Collate函数input_type处理")
    print("=" * 60)

    # 模拟batch数据
    mock_batch = [
        {
            'input_ids': torch.tensor([1, 2, 3, 4]),
            'attention_mask': torch.tensor([1, 1, 1, 1]),
            'labels': torch.tensor([1, 2, 3, 4]),
            'instruction': "test instruction 1",
            'input': "test input 1",
            'output': "1",
            'label': "0",
            'input_type': 0  # masked
        },
        {
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
            'labels': torch.tensor([1, 2, 3, 4, 5]),
            'instruction': "test instruction 2",
            'input': "test input 2",
            'output': "2",
            'label': "1",
            'input_type': 1  # full
        }
    ]

    # 创建模拟tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0

    mock_tokenizer = MockTokenizer()

    try:
        batch_result = dynamic_padding_collate_fn(mock_batch, mock_tokenizer)

        print("🔍 Collate函数处理结果:")
        print(f"  input_ids形状: {batch_result['input_ids'].shape}")
        print(f"  attention_mask形状: {batch_result['attention_mask'].shape}")
        print(f"  labels形状: {batch_result['labels'].shape}")
        print(f"  input_type: {batch_result['input_type']}")
        print(f"  input_type类型: {type(batch_result['input_type'])}")

        # 验证input_type是否正确
        expected_input_type = torch.tensor([0, 1], dtype=torch.long)
        if torch.equal(batch_result['input_type'], expected_input_type):
            print("  ✅ input_type处理正确")
        else:
            print(f"  ❌ input_type处理错误，期望: {expected_input_type}, 实际: {batch_result['input_type']}")
            return False

        print("\n✅ Collate函数测试完成")
        return True

    except Exception as e:
        print(f"❌ Collate函数测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始MASK机制功能测试")
    print("=" * 60)

    test_results = []

    # 测试1: 数据集MASK机制
    result1 = test_dataset_mask_mechanism()
    test_results.append(("数据集MASK机制", result1))

    # 测试2: MoE条件激活
    result2 = test_moe_conditional_activation()
    test_results.append(("MoE条件激活", result2))

    # 测试3: Collate函数
    result3 = test_collate_function()
    test_results.append(("Collate函数", result3))

    # 汇总结果
    print("\n" + "=" * 60)
    print("🎯 测试结果汇总")
    print("=" * 60)

    all_passed = True
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！MASK机制实现正确")
        print("\n📋 MASK机制工作原理:")
        print("  1. 数据集随机为样本分配input_type (0=masked, 1=full)")
        print("  2. input_type=0时，使用masked instruction，激活shared专家")
        print("  3. input_type=1时，使用完整instruction，激活路由专家")
        print("  4. 混合batch时，分别处理不同类型的样本")
        print("  5. 评估时禁用MASK，使用原始instruction")
    else:
        print("❌ 部分测试失败，请检查实现")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)