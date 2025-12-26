#!/usr/bin/env python3
"""
修复culture loss过低问题的建议方案
"""

print("🔧 Culture Loss修复方案")
print("=" * 60)

print("\n📊 问题分析:")
print("1. MASK机制：70%样本激活路由专家，30%激活shared专家")
print("2. Batch size=2：平均只有1.4个样本有expert_weights")
print("3. Culture loss需要≥2个样本才能计算成对相似性")
print("4. 结果：大多数batch的culture loss=0")

print("\n💡 解决方案（按优先级）:")

print("\n方案1：增加batch size（推荐）")
print("- 将batch_size从2增加到4或8")
print("- 确保每个batch有足够样本计算culture loss")
print("- 相应调整gradient_accumulation_steps保持有效batch size")

print("\n方案2：调整MASK概率")
print("- 将激活路由专家的概率从70%提高到85-90%")
print("- 代码位置：ft_lora_only_gen.py:138")
print("- 修改：use_mask = random.random() < 0.1  # 降低到10%")

print("\n方案3：增加culture loss权重")
print("- 将culture_loss_weight从0.01增加到0.05或0.1")
print("- 让culture loss对总损失有更大影响")

print("\n方案4：修改culture loss计算逻辑")
print("- 允许单样本情况下的culture loss计算")
print("- 使用历史expert_weights进行比较")

print("\n🔧 具体修改建议:")

print("\n1. 修改训练脚本batch size:")
print("   run_simplified_culturemoe.sh:")
print("   BATCH_SIZE=4  # 从2改为4")
print("   GRADIENT_ACCUMULATION=8  # 从16改为8，保持有效batch size=32")

print("\n2. 修改MASK概率:")
print("   ft_lora_only_gen.py:138:")
print("   use_mask = random.random() < 0.1  # 从0.3改为0.1")

print("\n3. 增加culture loss权重:")
print("   run_simplified_culturemoe.sh或训练参数:")
print("   --culture_loss_weight 0.05  # 从0.01改为0.05")

print("\n📈 预期效果:")
print("- Batch size=4时，平均3.6个样本激活路由专家")
print("- 几乎每个batch都能计算culture loss")
print("- Culture loss从0.003提升到0.01-0.05范围")
print("- 模型学习更好的文化感知专家路由")

print("\n⚠️ 注意事项:")
print("- 增加batch size会增加显存使用")
print("- 需要监控显存是否充足")
print("- 可以先尝试方案2（调整MASK概率）")