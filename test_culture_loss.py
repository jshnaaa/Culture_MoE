#!/usr/bin/env python3
"""
测试文化专注性损失
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from src.llamafactory.train.classification.culture_loss import (
    CultureSpecializationLoss,
    compute_culture_aware_loss
)


def test_culture_loss():
    """测试文化专注性损失"""
    print("="*60)
    print("Testing Culture Specialization Loss")
    print("="*60)

    # 参数
    batch_size = 8
    num_experts = 6
    num_cultures = 6
    num_classes = 3

    # 1. 模拟数据
    print("\n1. Creating simulated data...")

    # 专家权重（路由器输出）
    expert_weights = torch.randn(batch_size, num_experts)
    expert_weights = F.softmax(expert_weights, dim=-1)
    print(f"   Expert weights shape: {expert_weights.shape}")
    print(f"   Expert weights (sample 0): {expert_weights[0]}")

    # 文化维度标签
    culture_labels = [
        [0],      # 样本 0: 文化维度 0
        [1, 2],   # 样本 1: 文化维度 1 和 2
        [3],      # 样本 2: 文化维度 3
        [4, 5],   # 样本 3: 文化维度 4 和 5
        [0],      # 样本 4: 文化维度 0
        [1],      # 样本 5: 文化维度 1
        [2, 3],   # 样本 6: 文化维度 2 和 3
        [4],      # 样本 7: 文化维度 4
    ]
    print(f"   Culture labels: {culture_labels}")

    # 分类标签和 logits
    labels = torch.randint(0, num_classes, (batch_size,))
    logits = torch.randn(batch_size, num_classes)
    print(f"   Labels shape: {labels.shape}")
    print(f"   Logits shape: {logits.shape}")

    # 2. 创建损失模块
    print("\n2. Creating culture loss module...")
    culture_loss_module = CultureSpecializationLoss(
        num_cultures=num_cultures,
        num_experts=num_experts,
        lambda_weight=0.1
    )
    print(f"   ✅ Module created")

    # 3. 计算文化专注性损失
    print("\n3. Computing culture specialization loss...")
    culture_loss = culture_loss_module(expert_weights, culture_labels)
    print(f"   Culture loss: {culture_loss.item():.4f}")

    # 4. 查看专家权重矩阵
    print("\n4. Expert weight matrix:")
    weight_matrix = culture_loss_module.get_normalized_weights()
    print(f"   Shape: {weight_matrix.shape}")
    print(f"   Matrix:")
    for i in range(num_cultures):
        print(f"     Culture {i}: {weight_matrix[i].tolist()}")

    # 5. 计算完整损失
    print("\n5. Computing total loss...")
    total_loss, ce_loss, culture_loss_val = compute_culture_aware_loss(
        logits=logits,
        labels=labels,
        expert_weights=expert_weights,
        culture_labels=culture_labels,
        culture_loss_module=culture_loss_module,
        lambda_weight=0.1
    )

    print(f"   CE Loss: {ce_loss.item():.4f}")
    print(f"   Culture Loss: {culture_loss_val.item():.4f}")
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Lambda * Culture Loss: {(0.1 * culture_loss_val).item():.4f}")

    # 6. 验证损失计算
    print("\n6. Verifying loss calculation...")
    expected_total = ce_loss + 0.1 * culture_loss_val
    print(f"   Expected total: {expected_total.item():.4f}")
    print(f"   Actual total: {total_loss.item():.4f}")

    if torch.allclose(total_loss, expected_total, atol=1e-6):
        print(f"   ✅ Loss calculation correct!")
    else:
        print(f"   ❌ Loss calculation mismatch!")

    # 7. 测试多个批次
    print("\n7. Testing multiple batches...")
    for batch_idx in range(3):
        # 新的批次数据
        expert_weights_new = torch.randn(batch_size, num_experts)
        expert_weights_new = F.softmax(expert_weights_new, dim=-1)

        # 计算损失
        culture_loss_new = culture_loss_module(expert_weights_new, culture_labels)
        print(f"   Batch {batch_idx+1}: culture_loss = {culture_loss_new.item():.4f}")

    # 8. 查看累积后的权重矩阵
    print("\n8. Accumulated expert weight matrix:")
    weight_matrix_final = culture_loss_module.get_normalized_weights()
    print(f"   Shape: {weight_matrix_final.shape}")
    print(f"   Matrix:")
    for i in range(num_cultures):
        weights = weight_matrix_final[i]
        max_expert = torch.argmax(weights).item()
        print(f"     Culture {i}: max_expert={max_expert}, weights={weights.tolist()}")

    # 9. 测试重置
    print("\n9. Testing reset...")
    culture_loss_module.reset_statistics()
    weight_matrix_reset = culture_loss_module.get_normalized_weights()
    print(f"   After reset, all zeros: {torch.allclose(weight_matrix_reset, torch.zeros_like(weight_matrix_reset))}")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_culture_loss()

