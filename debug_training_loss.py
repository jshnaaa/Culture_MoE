#!/usr/bin/env python3
"""
调试训练损失异常高的问题
"""

import json
import torch
import torch.nn.functional as F

# 模拟数据
batch_size = 4
num_classes = 2
num_experts = 6

# 1. 模拟正常的 logits
logits_normal = torch.randn(batch_size, num_classes)
labels = torch.randint(0, num_classes, (batch_size,))

# 2. 计算正常的交叉熵损失
ce_loss_normal = F.cross_entropy(logits_normal, labels)
print("=" * 60)
print("正常情况：")
print("=" * 60)
print(f"Logits: {logits_normal}")
print(f"Labels: {labels}")
print(f"CE Loss: {ce_loss_normal.item():.4f}")
print(f"Expected: 0.5-1.5")
print()

# 3. 模拟异常的 logits（值过大）
logits_abnormal = torch.randn(batch_size, num_classes) * 10  # 放大10倍
ce_loss_abnormal = F.cross_entropy(logits_abnormal, labels)
print("=" * 60)
print("异常情况（logits 过大）：")
print("=" * 60)
print(f"Logits: {logits_abnormal}")
print(f"CE Loss: {ce_loss_abnormal.item():.4f}")
print(f"Problem: Loss is too high!")
print()

# 4. 检查文化损失
expert_weights = torch.softmax(torch.randn(batch_size, num_experts), dim=-1)
print("=" * 60)
print("专家权重：")
print("=" * 60)
print(f"Expert weights: {expert_weights}")
print(f"Sum (should be ~1.0): {expert_weights.sum(dim=-1)}")
print()

# 5. 模拟文化损失计算
# 假设文化损失是专家权重的熵
entropy = -(expert_weights * torch.log(expert_weights + 1e-10)).sum(dim=-1).mean()
print("=" * 60)
print("文化损失（熵）：")
print("=" * 60)
print(f"Culture loss (entropy): {entropy.item():.4f}")
print(f"Expected: 0.5-2.0")
print()

# 6. 总损失
lambda_weight = 0.01
total_loss = ce_loss_normal + lambda_weight * entropy
print("=" * 60)
print("总损失：")
print("=" * 60)
print(f"CE Loss: {ce_loss_normal.item():.4f}")
print(f"Culture Loss: {entropy.item():.4f}")
print(f"Lambda: {lambda_weight}")
print(f"Total Loss: {total_loss.item():.4f}")
print(f"Expected: 0.5-2.0")
print()

# 7. 检查你的实际损失
print("=" * 60)
print("你的训练日志：")
print("=" * 60)
print("loss: 4.1-4.5  ← 异常高！")
print("grad_norm: 137-300  ← 梯度爆炸！")
print()

print("=" * 60)
print("可能的原因：")
print("=" * 60)
print("1. ✅ Logits 值过大（分类头初始化不当）")
print("2. ✅ 文化损失计算错误（可能返回了很大的值）")
print("3. ✅ 标签错误（可能不是 0/1，而是其他值）")
print("4. ✅ 数据预处理错误")
print()

print("=" * 60)
print("建议的调试步骤：")
print("=" * 60)
print("1. 在 trainer.py 的 compute_loss 中添加打印：")
print("   print(f'CE Loss: {ce_loss.item():.4f}')")
print("   print(f'Culture Loss: {culture_loss.item():.4f}')")
print("   print(f'Total Loss: {total_loss.item():.4f}')")
print()
print("2. 检查 logits 的范围：")
print("   print(f'Logits min/max: {logits.min().item():.2f}/{logits.max().item():.2f}')")
print()
print("3. 检查标签：")
print("   print(f'Labels: {labels}')")
print("   print(f'Labels unique: {labels.unique()}')")
print()
print("4. 暂时禁用文化损失：")
print("   --use_culture_loss False")
print()

