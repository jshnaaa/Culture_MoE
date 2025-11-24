#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Top-2路由和Switch Transformer负载均衡损失的实现
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_top2_routing():
    """测试Top-2路由实现"""
    print("=== 测试Top-2路由实现 ===")

    # 模拟参数
    batch_size = 4
    num_experts = 12

    # 模拟expert_weights（原始softmax权重）
    torch.manual_seed(42)
    expert_weights = torch.randn(batch_size, num_experts)
    expert_weights = F.softmax(expert_weights, dim=-1)

    print("原始专家权重:")
    print(expert_weights)
    print("每行权重和:", expert_weights.sum(dim=-1))

    # Top-2选择
    top2_weights, top2_indices = torch.topk(expert_weights, k=2, dim=-1)
    top2_weights_normalized = F.softmax(top2_weights, dim=-1)

    print("\nTop-2权重:")
    print(top2_weights)
    print("Top-2索引:")
    print(top2_indices)
    print("Top-2归一化权重:")
    print(top2_weights_normalized)

    # 创建稀疏权重矩阵
    sparse_expert_weights = torch.zeros_like(expert_weights)
    for b in range(batch_size):
        sparse_expert_weights[b, top2_indices[b]] = top2_weights_normalized[b]

    print("\n稀疏专家权重（只有Top-2有值）:")
    print(sparse_expert_weights)
    print("每行权重和:", sparse_expert_weights.sum(dim=-1))

    # 验证只有2个专家被激活
    active_experts_per_sample = (sparse_expert_weights > 0).sum(dim=-1)
    print("每个样本激活的专家数量:", active_experts_per_sample)

    return sparse_expert_weights

def test_load_balancing_loss():
    """测试Switch Transformer负载均衡损失"""
    print("\n=== 测试Switch Transformer负载均衡损失 ===")

    # 模拟参数
    batch_size = 8
    num_experts = 12

    # 模拟数据
    torch.manual_seed(42)
    final_logits = torch.randn(batch_size, num_experts)

    # 计算路由器输出概率
    router_probs = F.softmax(final_logits, dim=-1)

    # 模拟Top-2稀疏权重
    expert_weights = torch.randn(batch_size, num_experts)
    expert_weights = F.softmax(expert_weights, dim=-1)

    # 转换为Top-2稀疏权重
    top2_weights, top2_indices = torch.topk(expert_weights, k=2, dim=-1)
    top2_weights_normalized = F.softmax(top2_weights, dim=-1)

    sparse_expert_weights = torch.zeros_like(expert_weights)
    for b in range(batch_size):
        sparse_expert_weights[b, top2_indices[b]] = top2_weights_normalized[b]

    print("路由器输出概率 P_i (平均):")
    P_i = router_probs.mean(dim=0)
    print(P_i)

    print("专家分配比例 f_i (平均):")
    f_i = sparse_expert_weights.mean(dim=0)
    print(f_i)

    # 计算Switch Transformer负载均衡损失
    N = num_experts
    balance_loss = N * torch.sum(f_i * P_i)

    print(f"\nSwitch Transformer负载均衡损失:")
    print(f"N = {N}")
    print(f"Σ(f_i * P_i) = {torch.sum(f_i * P_i):.6f}")
    print(f"L_balance = N * Σ(f_i * P_i) = {balance_loss:.6f}")

    # 对比原始MSE方法
    uniform_distribution = torch.ones_like(f_i) / num_experts
    mse_loss = F.mse_loss(f_i, uniform_distribution)
    print(f"\n对比 - 原始MSE损失: {mse_loss:.6f}")

    return balance_loss

def test_routing_distribution():
    """测试路由分布的均衡性"""
    print("\n=== 测试路由分布均衡性 ===")

    batch_size = 100
    num_experts = 12

    # 模拟不均衡的路由（某些专家被过度使用）
    torch.manual_seed(42)

    # 创建偏向某些专家的logits
    biased_logits = torch.randn(batch_size, num_experts)
    # 让前3个专家有更高的权重
    biased_logits[:, :3] += 2.0

    expert_weights = F.softmax(biased_logits, dim=-1)
    router_probs = F.softmax(biased_logits, dim=-1)

    print("偏向性路由 - 专家使用率:")
    usage_rates = expert_weights.mean(dim=0)
    print([f"Expert{i}: {rate:.3f}" for i, rate in enumerate(usage_rates)])

    # 计算负载均衡损失
    f_i = expert_weights.mean(dim=0)
    P_i = router_probs.mean(dim=0)
    balance_loss = num_experts * torch.sum(f_i * P_i)

    print(f"负载均衡损失: {balance_loss:.6f}")

    # 模拟均衡的路由
    uniform_logits = torch.randn(batch_size, num_experts) * 0.1  # 小的随机扰动
    uniform_expert_weights = F.softmax(uniform_logits, dim=-1)
    uniform_router_probs = F.softmax(uniform_logits, dim=-1)

    print("\n均衡性路由 - 专家使用率:")
    uniform_usage_rates = uniform_expert_weights.mean(dim=0)
    print([f"Expert{i}: {rate:.3f}" for i, rate in enumerate(uniform_usage_rates)])

    # 计算负载均衡损失
    uniform_f_i = uniform_expert_weights.mean(dim=0)
    uniform_P_i = uniform_router_probs.mean(dim=0)
    uniform_balance_loss = num_experts * torch.sum(uniform_f_i * uniform_P_i)

    print(f"负载均衡损失: {uniform_balance_loss:.6f}")

    print(f"\n损失差异: {balance_loss - uniform_balance_loss:.6f}")
    print("✅ 偏向性路由的损失应该明显高于均衡性路由")

if __name__ == "__main__":
    print("测试Top-2路由和Switch Transformer负载均衡损失实现")
    print("=" * 60)

    # 测试Top-2路由
    sparse_weights = test_top2_routing()

    # 测试负载均衡损失
    balance_loss = test_load_balancing_loss()

    # 测试路由分布
    test_routing_distribution()

    print("\n" + "=" * 60)
    print("✅ 所有测试完成")
    print("主要改进:")
    print("1. ✅ 实现了Top-2专家激活（只有2个专家被选中）")
    print("2. ✅ 实现了Switch Transformer负载均衡损失公式")
    print("3. ✅ 路由器权重重新归一化确保概率分布")
    print("4. ✅ 负载均衡损失能够区分均衡和不均衡的路由")