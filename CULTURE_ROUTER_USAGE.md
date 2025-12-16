# 文化感知路由机制使用指南

## 概述

已在`run_joint_lora_moe_training.sh`脚本中新增`USE_CULTURE_ROUTER`参数（第7个参数），用于控制是否启用基于文化感知和冲突检测的路由机制。

## 参数说明

### 新的参数顺序
```bash
bash run_joint_lora_moe_training.sh [backbone] [data_id] [use_shared] [use_gate] [num_moe_experts] [use_culture_loss] [use_culture_router] [use_lora] [num_gpus] [lora_rank] [lora_alpha]
```

### 参数详解

| 位置 | 参数名 | 默认值 | 说明 |
|------|--------|--------|------|
| 1 | `backbone` | `llama` | 模型骨干（llama/qwen） |
| 2 | `data_id` | `2` | 数据集ID |
| 3 | `use_shared` | `false` | 是否使用共享专家 |
| 4 | `use_gate` | `false` | 是否使用MoE内部Gate |
| 5 | `num_moe_experts` | `4` | MoE专家数量 |
| 6 | `use_culture_loss` | `ori` | 文化损失模式（ori/new/kl/false） |
| 7 | **`use_culture_router`** | **`false`** | **是否启用文化感知路由** |
| 8 | `use_lora` | `true` | 是否启用LoRA |
| 9 | `num_gpus` | `2` | GPU数量 |
| 10 | `lora_rank` | `16` | LoRA rank |
| 11 | `lora_alpha` | `32` | LoRA alpha |

## 使用方法

### 1. 标准路由（默认）
```bash
# 使用标准路由机制
bash run_joint_lora_moe_training.sh llama 2 false false 4 ori false
```

### 2. 启用文化感知路由
```bash
# 启用文化感知冲突检测路由
bash run_joint_lora_moe_training.sh llama 2 false false 4 ori true
```

### 3. 文化感知路由 + KL散度损失
```bash
# 同时启用文化感知路由和KL散度文化损失
bash run_joint_lora_moe_training.sh llama 2 false false 4 kl true
```

### 4. 完整参数示例
```bash
# 完整指定所有参数
bash run_joint_lora_moe_training.sh llama 2 false false 4 new true true 2 16 32
#                                   ↑     ↑ ↑     ↑     ↑ ↑   ↑    ↑    ↑ ↑  ↑
#                                   模型  数据 共享 Gate  专家 损失 路由 LoRA GPU rank alpha
```

## 路由机制对比

### 标准路由 (`use_culture_router=false`)
- 使用传统的MoE路由机制
- 损失函数：`L = L_h + L_balance`
- 其中：`L_balance = αL_aux + βL_o + γL_v`

### 文化感知路由 (`use_culture_router=true`)
- 启用基于文化感知和冲突检测的路由机制
- **扩展损失函数**：`L = L_h + L_balance + L_cultural + L_conflict + L_consistency + L_specialization`
- 包含文化路由、冲突检测、冲突解决等组件

## 损失函数切换规则

当`USE_CULTURE_ROUTER=true`时，**无论`USE_CULTURE_LOSS`设置为什么值**（只要不是`false`），都会使用扩展后的损失函数：

| USE_CULTURE_ROUTER | USE_CULTURE_LOSS | 实际使用的损失函数 |
|-------------------|------------------|-----------------|
| `false` | `ori/new/kl` | 标准损失 + 对应的文化损失 |
| `false` | `false` | 仅标准损失 |
| **`true`** | **`ori`** | **扩展损失函数** |
| **`true`** | **`new`** | **扩展损失函数** |
| **`true`** | **`kl`** | **扩展损失函数** |
| **`true`** | **`false`** | **扩展损失函数**（但不含文化损失组件） |

## 配置输出示例

启用文化感知路由时，脚本会显示：

```
配置信息:
  模型: llama (Meta-Llama-3.1-8B-Instruct)
  数据: CulturalBench (CulturalBench_merge_gen.json)
  总层数: 32
  训练模式: 联合训练 (预训练LoRA + MoE专家LoRA + Router)
  共享专家: false
  MoE内部Gate: false
  MoE专家数: 4
  文化损失模式: ori (ori=原始L_o, new=文化感知L_o, kl=KL散度L_o, false=仅L_aux)
  文化感知路由: true (true=启用文化感知冲突检测路由, false=标准路由)  ← 新增
  启用预训练LoRA: true
  LoRA配置: rank=16, alpha=32
  GPU: 2卡
```

## 实验建议

### 渐进式实验
1. **基线实验**：
   ```bash
   bash run_joint_lora_moe_training.sh llama 2 false false 4 false false
   ```

2. **标准文化损失**：
   ```bash
   bash run_joint_lora_moe_training.sh llama 2 false false 4 ori false
   ```

3. **文化感知路由**：
   ```bash
   bash run_joint_lora_moe_training.sh llama 2 false false 4 ori true
   ```

### 对比实验
```bash
# 对比不同文化损失 + 文化感知路由的效果
bash run_joint_lora_moe_training.sh llama 2 false false 4 ori true   # 原始损失
bash run_joint_lora_moe_training.sh llama 2 false false 4 new true   # 对比损失
bash run_joint_lora_moe_training.sh llama 2 false false 4 kl true    # KL散度损失
```

## 注意事项

1. **参数位置变化**：由于新增了`use_culture_router`参数，原来的参数位置都后移了一位
2. **向后兼容**：如果不指定第7个参数，默认为`false`，保持原有行为
3. **训练复杂度**：启用文化感知路由会显著增加训练复杂度和计算开销
4. **数据需求**：文化感知路由需要高质量的文化标注数据才能发挥最佳效果

## 配置文件

训练配置会保存在`config.json`中，包含新的`use_culture_router`参数：

```json
{
    "training_config": {
        "use_culture_loss": "ori",
        "use_culture_router": true,
        ...
    }
}
```

这样可以方便后续分析和复现实验结果。