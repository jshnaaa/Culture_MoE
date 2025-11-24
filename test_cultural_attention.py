#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文化感知注意力机制的脚本
"""

import sys
import os
import torch

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from llamafactory.model.lora_enhanced_culturemoe import LoRACultureMoEConfig
    from llamafactory.model.cultural_aware_attention import CulturallyAwareAttention
    print("✅ Successfully imported cultural aware attention modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def test_cultural_attention_config():
    """测试文化感知注意力配置"""
    print("\n🧪 Testing Cultural Attention Configuration...")

    # 创建配置
    config = LoRACultureMoEConfig(
        num_experts=8,
        top_k=2,
        num_cultures=6,
        culture_dim=256,
        lora_rank=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
        enable_cultural_attention=True
    )

    print(f"   - Number of cultures: {config.num_cultures}")
    print(f"   - Culture dimension: {config.culture_dim}")
    print(f"   - Cultural attention enabled: {config.enable_cultural_attention}")
    print(f"   - LoRA rank: {config.lora_rank}")
    print(f"   - LoRA alpha: {config.lora_alpha}")

    return config

def test_cultural_attention_creation():
    """测试文化感知注意力组件创建"""
    print("\n🧪 Testing Cultural Attention Component Creation...")

    # 模拟Llama配置
    class MockLlamaConfig:
        def __init__(self):
            self.hidden_size = 4096
            self.num_attention_heads = 32
            self.max_position_embeddings = 2048
            self.rope_theta = 10000

    mock_config = MockLlamaConfig()
    lora_config = LoRACultureMoEConfig(enable_cultural_attention=True)

    try:
        # 创建文化感知注意力
        cultural_attention = CulturallyAwareAttention(
            config=mock_config,
            layer_idx=0,
            lora_config=lora_config
        )
        print("   ✅ Successfully created CulturallyAwareAttention")

        # 检查组件
        print(f"   - Culture embeddings shape: {cultural_attention.culture_embeddings.weight.shape}")
        print(f"   - Hidden size: {cultural_attention.hidden_size}")
        print(f"   - Number of heads: {cultural_attention.num_heads}")

        return cultural_attention

    except Exception as e:
        print(f"   ❌ Error creating cultural attention: {e}")
        return None

def test_forward_pass():
    """测试前向传播"""
    print("\n🧪 Testing Forward Pass...")

    # 创建模拟输入
    batch_size = 2
    seq_len = 10
    hidden_size = 4096

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    culture_ids = torch.randint(0, 6, (batch_size,))
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    print(f"   - Input shape: {hidden_states.shape}")
    print(f"   - Culture IDs: {culture_ids}")

    # 创建注意力模块
    class MockLlamaConfig:
        def __init__(self):
            self.hidden_size = 4096
            self.num_attention_heads = 32
            self.max_position_embeddings = 2048
            self.rope_theta = 10000

    mock_config = MockLlamaConfig()
    lora_config = LoRACultureMoEConfig(enable_cultural_attention=True)

    try:
        cultural_attention = CulturallyAwareAttention(
            config=mock_config,
            layer_idx=0,
            lora_config=lora_config
        )

        # 前向传播
        output, attn_weights, past_kv = cultural_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            culture_ids=culture_ids,
            output_attentions=True
        )

        print(f"   ✅ Forward pass successful")
        print(f"   - Output shape: {output.shape}")
        if attn_weights is not None:
            print(f"   - Attention weights shape: {attn_weights.shape}")

        return True

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Testing Cultural Aware Attention Implementation")
    print("=" * 60)

    # 测试配置
    config = test_cultural_attention_config()

    # 测试组件创建
    attention = test_cultural_attention_creation()

    # 测试前向传播
    if attention is not None:
        success = test_forward_pass()

        if success:
            print("\n🎉 All tests passed! Cultural aware attention is working correctly.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
    else:
        print("\n❌ Could not create attention module. Please check the implementation.")

    print("=" * 60)

if __name__ == "__main__":
    main()