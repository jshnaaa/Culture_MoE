#!/usr/bin/env python3
"""
本地测试Only MoE模型基本功能
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.only_moe_model import OnlyMoEModel, OnlyMoEConfig


def create_mock_model():
    """创建一个简单的mock模型用于测试"""
    class MockMLP(torch.nn.Module):
        def __init__(self, hidden_size=4096, intermediate_size=11008):
            super().__init__()
            self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
            self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
            self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float16)
            self.act_fn = torch.nn.SiLU()

        def forward(self, x):
            gate_out = self.act_fn(self.gate_proj(x))
            up_out = self.up_proj(x)
            return self.down_proj(gate_out * up_out)

    class MockLayer(torch.nn.Module):
        def __init__(self, hidden_size=4096):
            super().__init__()
            self.mlp = MockMLP(hidden_size)

        def forward(self, x):
            return self.mlp(x)

    class MockModel(torch.nn.Module):
        def __init__(self, hidden_size=4096, num_layers=28):
            super().__init__()
            self.layers = torch.nn.ModuleList([MockLayer(hidden_size) for _ in range(num_layers)])

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
            # 简单的mock实现
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.randn(batch_size, seq_len, 4096, dtype=torch.float16, device=input_ids.device)

            all_hidden_states = [hidden_states]
            for layer in self.layers:
                hidden_states = layer(hidden_states)
                all_hidden_states.append(hidden_states)

            class MockOutput:
                def __init__(self):
                    self.hidden_states = all_hidden_states
                    self.logits = torch.randn(batch_size, seq_len, 32000, dtype=torch.float16, device=input_ids.device)

            return MockOutput()

    class MockConfig:
        def __init__(self):
            self.hidden_size = 4096
            self.vocab_size = 32000

    model = MockModel()
    model.config = MockConfig()
    return model


def test_only_moe_local():
    """本地测试Only MoE"""
    print("Creating mock base model...")

    # 创建mock模型
    base_model = create_mock_model()

    # 移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)

    print(f"Using device: {device}")

    # 创建OnlyMoE配置
    config = OnlyMoEConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        num_moe_experts=2,
        top_k=2,
        moe_layers=[26, 27],  # 最后两层
        aux_loss_coef=0.01,
        dropout=0.1
    )

    # 创建OnlyMoE模型
    print("Creating OnlyMoE model...")
    try:
        model = OnlyMoEModel(base_model, config)
        model.print_trainable_parameters()
        print("✅ OnlyMoE model created successfully")
    except Exception as e:
        print(f"❌ OnlyMoE model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试前向传播
    print("Testing forward pass...")
    try:
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :seq_len//2] = -100  # mask前半部分

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        loss = outputs.loss
        moe_aux_loss = outputs.moe_aux_loss

        print(f"✅ Forward pass successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  MoE aux loss: {moe_aux_loss.item():.6f}")

        # 测试反向传播
        print("Testing backward pass...")
        loss.backward()
        print("✅ Backward pass successful")

        # 检查梯度
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        print(f"Parameters with gradients: {grad_count}")

        # 测试保存模型
        print("Testing model save...")
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_model(temp_dir)
            print("✅ Model save successful")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✅ All local tests passed!")
    return True


if __name__ == "__main__":
    success = test_only_moe_local()
    exit(0 if success else 1)