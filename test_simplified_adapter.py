#!/usr/bin/env python3
"""
测试简化版CultureMoE适配器的PeftModel兼容性修复
"""

import torch
import torch.nn as nn
from src.llamafactory.model.simplified_culturemoe import SimplifiedCultureMoEConfig
from src.llamafactory.model.simplified_culturemoe_adapter import SimplifiedCultureMoEAdapter, MoEFFNLoRA

# 创建一个简单的模拟模型来测试
class MockFFN(nn.Module):
    def __init__(self, hidden_dim=4096, intermediate_dim=11008):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MockFFN()

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer() for _ in range(32)])

class MockPeftModel(nn.Module):
    """模拟PeftModel结构"""
    def __init__(self):
        super().__init__()
        self.base_model = MockBaseModel()

class MockBaseModel(nn.Module):
    """模拟PeftModel.base_model结构"""
    def __init__(self):
        super().__init__()
        self.model = MockModel()

def test_adapter():
    print("🧪 测试简化版CultureMoE适配器的PeftModel兼容性...")

    # 创建配置
    config = SimplifiedCultureMoEConfig(
        lora_rank=16,
        lora_alpha=32,
        num_moe_experts=4,
        num_activated_experts=2,
        use_shared=True,
        use_gate=True,
        use_culture_loss="new"
    )

    # 测试1: 普通模型
    print("\n🔍 测试1: 普通模型结构")
    try:
        mock_model = MockModel()
        adapter = SimplifiedCultureMoEAdapter(mock_model, config)
        print("✅ 普通模型测试成功")
    except Exception as e:
        print(f"❌ 普通模型测试失败: {e}")

    # 测试2: PeftModel包装的模型
    print("\n🔍 测试2: PeftModel包装的模型结构")
    try:
        mock_peft_model = MockPeftModel()
        adapter = SimplifiedCultureMoEAdapter(mock_peft_model, config)
        print("✅ PeftModel测试成功")
    except Exception as e:
        print(f"❌ PeftModel测试失败: {e}")
        print(f"   错误详情: {str(e)}")

    # 测试3: 验证MoE层类型
    print("\n🔍 测试3: 验证MoE层替换")
    try:
        mock_model = MockModel()
        adapter = SimplifiedCultureMoEAdapter(mock_model, config)

        # 检查最后8层是否被正确替换
        layers = mock_model.layers
        target_layers = list(range(24, 32))  # 最后8层

        moe_count = 0
        for layer_idx in target_layers:
            if isinstance(layers[layer_idx].mlp, MoEFFNLoRA):
                moe_count += 1

        print(f"✅ 成功替换 {moe_count}/8 层为MoE")

        # 测试配置信息
        print(f"   - MoE专家数: {config.num_moe_experts}")
        print(f"   - 激活专家数: {config.num_activated_experts}")
        print(f"   - 使用共享专家: {config.use_shared}")
        print(f"   - 使用Gate网络: {config.use_gate}")
        print(f"   - LoRA rank: {config.lora_rank}")

    except Exception as e:
        print(f"❌ MoE层验证失败: {e}")

    # 测试4: 验证错误处理机制
    print("\n🔍 测试4: 验证错误处理机制")
    try:
        mock_model = MockModel()
        adapter = SimplifiedCultureMoEAdapter(mock_model, config)

        # 测试get_expert_weights_for_culture_loss的错误处理
        expert_weights = adapter.get_expert_weights_for_culture_loss()
        print(f"✅ Expert weights获取: {expert_weights is not None}")

        # 测试get_accumulated_z_loss的错误处理
        z_loss = adapter.get_accumulated_z_loss()
        print(f"✅ Z-loss计算: {z_loss.item():.6f}")

    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")

if __name__ == "__main__":
    test_adapter()