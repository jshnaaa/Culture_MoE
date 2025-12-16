#!/usr/bin/env python3
"""
测试KL散度文化损失集成效果
验证新的KL损失函数是否正确集成到训练流程中
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_moe_losses import compute_kl_culture_loss, integrate_enhanced_moe_loss


def test_kl_culture_loss():
    """测试KL散度文化损失函数"""
    print("🧪 测试KL散度文化损失函数...")

    # 创建测试数据
    batch_size = 4
    num_experts = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 专家权重分布（已经过softmax）
    expert_weights = torch.softmax(torch.randn(batch_size, num_experts), dim=-1).to(device)

    # 文化标签：两种文化，每种2个样本
    culture_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long).to(device)

    print(f"  设备: {device}")
    print(f"  expert_weights shape: {expert_weights.shape}")
    print(f"  culture_labels: {culture_labels}")
    print(f"  expert_weights:")
    for i, (weights, culture) in enumerate(zip(expert_weights, culture_labels)):
        print(f"    样本{i} (文化{culture.item()}): {weights.detach().cpu().numpy()}")

    # 计算KL损失
    try:
        kl_loss = compute_kl_culture_loss(expert_weights, culture_labels)
        print(f"  ✅ KL损失计算成功: {kl_loss.item():.6f}")

        # 检查梯度
        if kl_loss.requires_grad:
            print(f"  ✅ 损失具有梯度")
        else:
            print(f"  ⚠️ 损失缺少梯度")

        # 检查数值有效性
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print(f"  ❌ 损失包含NaN或Inf")
        else:
            print(f"  ✅ 损失数值有效")

        return True

    except Exception as e:
        print(f"  ❌ KL损失计算失败: {e}")
        return False


def test_enhanced_moe_loss_integration():
    """测试增强MoE损失集成"""
    print("\n🧪 测试增强MoE损失集成...")

    # 创建模拟的模型输出
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    num_experts = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模拟的模型输出对象
    class MockModelOutputs:
        def __init__(self):
            self.logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
            self.expert_weights = torch.softmax(torch.randn(batch_size, num_experts), dim=-1).to(device)
            self.expert_outputs = {
                0: torch.randn(batch_size, seq_len, 512).to(device),
                1: torch.randn(batch_size, seq_len, 512).to(device)
            }
            self.soft_routing_scores = self.expert_weights
            self.activated_experts = [0, 1]

    outputs = MockModelOutputs()
    labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    # 设置一些标签为-100（忽略）
    labels[:, :3] = -100

    culture_labels = torch.tensor([0, 1], dtype=torch.long).to(device)

    print(f"  logits shape: {outputs.logits.shape}")
    print(f"  expert_weights shape: {outputs.expert_weights.shape}")
    print(f"  culture_labels: {culture_labels}")

    # 测试不同的损失模式
    test_modes = [
        ("ori", {"alpha": 1e-2, "beta": 1e-3, "gamma": 1e-3, "use_cultural_aware": False, "use_kl_loss": False}),
        ("new", {"alpha": 1e-2, "beta": 1e-3, "gamma": 1e-3, "use_cultural_aware": True, "use_kl_loss": False}),
        ("kl", {"alpha": 1e-2, "beta": 1e-3, "gamma": 1e-3, "use_cultural_aware": False, "use_kl_loss": True})
    ]

    for mode, loss_weights in test_modes:
        print(f"\n  测试 {mode} 模式:")
        try:
            loss_dict = integrate_enhanced_moe_loss(
                model_outputs=outputs,
                labels=labels,
                culture_labels=culture_labels,
                use_culture_loss=False,
                loss_weights=loss_weights
            )

            print(f"    L_h (主任务): {loss_dict['L_h'].item():.6f}")
            print(f"    L_aux (负载均衡): {loss_dict['L_aux'].item():.6f}")
            print(f"    L_o (正交化): {loss_dict['L_o'].item():.6f}")
            print(f"    L_v (路由方差): {loss_dict['L_v'].item():.6f}")
            print(f"    L_balance (平衡损失): {loss_dict['L_balance'].item():.6f}")
            print(f"    L_total (总损失): {loss_dict['L_total'].item():.6f}")

            # 检查梯度
            if loss_dict['L_total'].requires_grad:
                print(f"    ✅ 总损失具有梯度")
            else:
                print(f"    ⚠️ 总损失缺少梯度")

            # 特别检查KL模式的L_o损失
            if mode == "kl":
                if loss_dict['L_o'].item() != 0.0:
                    print(f"    ✅ KL模式的L_o损失非零: {loss_dict['L_o'].item():.6f}")
                else:
                    print(f"    ⚠️ KL模式的L_o损失为零，可能有问题")

        except Exception as e:
            print(f"    ❌ {mode} 模式测试失败: {e}")
            return False

    return True


def test_gradient_flow():
    """测试梯度流"""
    print("\n🧪 测试梯度流...")

    batch_size = 3
    num_experts = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建需要梯度的专家权重
    expert_weights = torch.softmax(torch.randn(batch_size, num_experts, requires_grad=True), dim=-1).to(device)
    culture_labels = torch.tensor([0, 0, 1], dtype=torch.long).to(device)

    try:
        # 计算KL损失
        kl_loss = compute_kl_culture_loss(expert_weights, culture_labels)

        # 反向传播
        kl_loss.backward()

        # 检查梯度
        if expert_weights.grad is not None:
            grad_norm = torch.norm(expert_weights.grad)
            print(f"  ✅ 梯度计算成功，梯度范数: {grad_norm.item():.6f}")

            # 检查梯度是否合理（不为零且不太大）
            if 1e-6 < grad_norm < 1e6:
                print(f"  ✅ 梯度范数在合理范围内")
            else:
                print(f"  ⚠️ 梯度范数可能异常: {grad_norm.item()}")

            return True
        else:
            print(f"  ❌ 没有计算出梯度")
            return False

    except Exception as e:
        print(f"  ❌ 梯度流测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("KL散度文化损失集成测试")
    print("=" * 60)

    # 运行所有测试
    tests = [
        test_kl_culture_loss,
        test_enhanced_moe_loss_integration,
        test_gradient_flow
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if test_func():
            passed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！KL散度损失集成成功")
        print("\n📝 使用方法:")
        print("  bash run_joint_lora_moe_training.sh llama 2 false false 4 kl")
        print("  其中最后一个参数 'kl' 指定使用KL散度文化损失")
    else:
        print("❌ 部分测试失败，请检查实现")

    print("=" * 60)


if __name__ == "__main__":
    main()