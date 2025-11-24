# Phase 1: 知识蒸馏增强CultureMoE - 使用说明

## 概述

Phase 1 实现了通过知识蒸馏来改善CultureMoE模型OOD（分布外）性能的方案。核心思想是：

1. **稳健共享专家**：用一个增强的共享专家替换原来的简单共享专家
2. **知识蒸馏**：从LoRA微调模型（teacher）向稳健共享专家（student）蒸馏知识
3. **文化不变性学习**：学习跨文化通用特征，提高OOD泛化能力
4. **自适应蒸馏权重**：训练过程中动态调整蒸馏损失权重

## 新增文件

### 1. 核心模型组件
- `src/llamafactory/model/robust_shared_expert.py` - 稳健共享专家实现
- `src/llamafactory/model/teacher_model_loader.py` - Teacher模型加载器
- `src/llamafactory/model/enhanced_culturemoe_with_distill.py` - 带蒸馏的增强模型

### 2. 训练脚本
- `ft_enhanced_culturemoe_with_distill.py` - Phase 1 训练脚本
- `run_ft_enhanced_culturemoe_with_distill.sh` - 便捷运行脚本

### 3. 说明文档
- `PHASE1_DISTILLATION_README.md` - 本文档

## 快速开始

### 基本用法

```bash
# 使用默认参数训练
bash run_ft_enhanced_culturemoe_with_distill.sh llama 1

# 使用2个GPU训练
bash run_ft_enhanced_culturemoe_with_distill.sh llama 2
```

### 完整参数示例

```bash
bash run_ft_enhanced_culturemoe_with_distill.sh llama 1 True 12 0.4 0.5 0.5 1.0 True 3.0 0.01 0.1 1 True False 4.0 0.7
```

### 参数说明

| 参数位置 | 参数名 | 默认值 | 说明 |
|---------|--------|--------|------|
| 1 | model_type | - | 模型类型 (llama) |
| 2 | num_gpus | - | GPU数量 (1或2) |
| 3 | freeze_base_model | True | 是否冻结基础模型 |
| 4 | num_experts | 12 | 专家数量 |
| 5 | moe_fusion | 0.4 | MoE融合系数 |
| 6 | culture_loss_lambda | 0.5 | 文化损失权重 |
| 7 | culture_loss_alpha | 0.5 | 文化损失margin |
| 8 | culture_loss_beta | 1.0 | 文化损失lambda_diff |
| 9 | use_culture_loss | True | 是否使用文化损失 |
| 10 | router_temperature | 3.0 | 路由器温度 |
| 11 | load_balance_weight | 0.01 | 负载均衡权重 |
| 12 | entropy_weight | 0.1 | 熵正则化权重 |
| 13 | eval_interval | 1 | 评估间隔 |
| 14 | use_shared_experts | True | 是否使用共享专家 |
| 15 | use_gate | False | 是否使用门控机制 |
| 16 | distill_temperature | 4.0 | 蒸馏温度 |
| 17 | distill_alpha | 0.7 | 蒸馏权重 |

## 关键特性

### 1. 稳健共享专家 (RobustSharedExpert)

- **主要专家网络**：从原shared expert复制结构和权重
- **文化不变特征提取器**：学习跨文化通用特征
- **知识蒸馏投影层**：用于与teacher模型对齐
- **融合权重**：可学习的权重平衡专家输出和不变特征

### 2. 知识蒸馏机制

- **Teacher模型**：使用LoRA微调模型作为教师
- **Student模型**：稳健共享专家作为学生
- **蒸馏损失**：KL散度损失，温度可调
- **权重调度**：训练过程中动态调整蒸馏权重

### 3. 文化不变性学习

- **不变性损失**：鼓励同文化内相似，跨文化适度相似
- **余弦相似度**：基于余弦相似度计算文化间距离
- **自适应边界**：动态调整文化间的合理距离

## 输出文件

训练完成后，输出目录包含：

```
output/ft_enhanced_culturemoe_with_distill_YYYY-MM-DD_HH-MM-SS/
├── best_distill_moe/                    # 最佳模型
│   └── distill_moe_weights.pth         # 蒸馏MoE权重
├── distill_training.log                 # 训练日志
├── distill_config.json                  # 训练配置
└── distill_epoch_results.json          # 每轮训练结果
```

## 训练监控

### 关键指标

1. **蒸馏损失** (distill_loss)：student和teacher的知识对齐程度
2. **不变性损失** (invariant_loss)：文化不变特征学习效果
3. **融合权重** (fusion_weight)：专家输出和不变特征的平衡
4. **生成损失** (generation_loss)：基础语言建模能力
5. **文化损失** (culture_loss)：文化感知能力

### 日志示例

```
=== Distillation Epoch 1 Summary ===
Total Loss: 2.1234
  ├─ Generation Loss: 1.8500
  ├─ Culture Loss: 0.1200
  ├─ Distillation Loss: 0.1234
  └─ Invariant Loss: 0.0300
```

## 预期效果

### Phase 1 目标

1. **OOD性能提升**：在分布外测试集上性能接近或超过LoRA基线
2. **ID性能保持**：在分布内测试集上保持现有优势
3. **知识传递**：成功从LoRA模型蒸馏跨文化通用知识
4. **稳定训练**：无NaN/Inf梯度，训练过程稳定

### 评估方法

1. **对比基线**：与原始LoRA模型和原始MoE模型对比
2. **OOD测试**：在未见过的文化/领域数据上测试
3. **消融研究**：关闭蒸馏功能，验证蒸馏效果
4. **专家分析**：分析稳健共享专家的激活模式

## 下一步计划

如果Phase 1效果良好：

- **Phase 2**：实现自适应文化感知路由器
- **Phase 3**：集成元学习机制
- **Phase 4**：多模态文化理解扩展

## 故障排除

### 常见问题

1. **Teacher模型加载失败**
   - 检查teacher_model_path和teacher_lora_path是否正确
   - 确保路径存在且可访问

2. **蒸馏损失异常**
   - 调整distill_temperature参数（建议范围：2.0-8.0）
   - 检查teacher和student的输出维度是否匹配

3. **内存不足**
   - 减小batch_size
   - 使用单GPU训练
   - 启用梯度累积

4. **训练不稳定**
   - 降低学习率
   - 增加warmup步数
   - 检查梯度裁剪设置

### 调试模式

在训练脚本中添加详细调试信息：

```python
# 在train_epoch函数中添加
if self.global_step <= 10:
    logging.info(f"Debug step {self.global_step}:")
    logging.info(f"  Distill weight: {outputs.get('distill_weight', 0):.4f}")
    logging.info(f"  Fusion weight: {outputs.get('robust_fusion_weight', 0):.4f}")
```

## 联系与支持

如有问题或需要帮助，请：

1. 检查训练日志中的详细错误信息
2. 验证所有依赖文件是否存在
3. 确认GPU内存和系统资源充足
4. 参考本文档的故障排除部分