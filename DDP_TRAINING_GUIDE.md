# LoRA增强FFN集成CultureMoE DDP训练指南

## 概述

本指南介绍如何使用DDP（DistributedDataParallel）进行LoRA增强的FFN集成CultureMoE模型的多卡训练和评估。

## 文件说明

### 核心脚本文件

1. **train_lora_culturemoe_ffn_integrated_ddp.py** - DDP多卡训练脚本
2. **eval_lora_culturemoe_ffn_integrated_ddp.py** - DDP多卡评估脚本
3. **run_lora_culturemoe_ddp.sh** - 一键启动脚本
4. **train_lora_culturemoe_ffn_integrated.py** - 单卡训练脚本（已修复）
5. **eval_lora_culturemoe_ffn_integrated.py** - 单卡评估脚本（已修复）

### 修复的问题

1. ✅ **变量命名错误修复**：修复了模型中`hidden_states_mask`变量命名不一致的问题
2. ✅ **参数传递错误修复**：修复了LoRACultureMoELlamaModel中参数传递的错误
3. ✅ **LoRA参数移除**：移除了命令行中的`--lora_rank`和`--lora_alpha`参数，现在使用固定值
4. ✅ **DDP支持添加**：添加了完整的DDP多卡训练和评估支持

## 使用方法

### 方式1：使用一键启动脚本（推荐）

```bash
# 基本用法（双卡训练）
./run_lora_culturemoe_ddp.sh \\
    --data_path ./data/your_training_data.json \\
    --test_data_path ./data/your_test_data.json

# 自定义配置
./run_lora_culturemoe_ddp.sh \\
    --base_model meta-llama/Llama-2-7b-hf \\
    --data_path ./data/culture_train.json \\
    --test_data_path ./data/culture_test.json \\
    --output_dir ./outputs/my_experiment \\
    --num_epochs 10 \\
    --batch_size 8 \\
    --learning_rate 1e-4 \\
    --num_experts 16 \\
    --num_gpus 4 \\
    --seed 42

# 单卡训练
./run_lora_culturemoe_ddp.sh \\
    --data_path ./data/culture_train.json \\
    --num_gpus 1

# 查看帮助
./run_lora_culturemoe_ddp.sh --help
```

### 方式2：直接使用Python脚本

#### 训练

```bash
# DDP双卡训练
python train_lora_culturemoe_ffn_integrated_ddp.py \\
    --base_model meta-llama/Llama-2-7b-hf \\
    --data_path ./data/culture_train.json \\
    --output_dir ./outputs/lora_culturemoe \\
    --num_epochs 8 \\
    --batch_size 4 \\
    --learning_rate 5e-4 \\
    --num_experts 8 \\
    --num_gpus 2 \\
    --seed 42

# 单卡训练
python train_lora_culturemoe_ffn_integrated_ddp.py \\
    --data_path ./data/culture_train.json \\
    --num_gpus 1
```

#### 评估

```bash
# DDP双卡评估
python eval_lora_culturemoe_ffn_integrated_ddp.py \\
    --base_model meta-llama/Llama-2-7b-hf \\
    --lora_weights ./outputs/lora_culturemoe/final_lora_weights.pt \\
    --test_data ./data/culture_test.json \\
    --output_dir ./evaluation_results \\
    --batch_size 8 \\
    --num_experts 8 \\
    --num_gpus 2

# 单卡评估
python eval_lora_culturemoe_ffn_integrated_ddp.py \\
    --lora_weights ./outputs/lora_culturemoe/final_lora_weights.pt \\
    --test_data ./data/culture_test.json \\
    --num_gpus 1
```

## 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_model` | `meta-llama/Llama-2-7b-hf` | 基础模型路径 |
| `--data_path` | 必需 | 训练数据路径 |
| `--output_dir` | `./outputs/lora_culturemoe` | 输出目录 |
| `--num_epochs` | 8 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--learning_rate` | 5e-4 | 学习率 |
| `--num_experts` | 8 | 专家数量 |
| `--num_gpus` | 2 | GPU数量（1=单卡，2+=DDP） |
| `--progressive_training` | False | 是否使用渐进式训练 |
| `--seed` | 42 | 随机种子 |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_model` | `meta-llama/Llama-2-7b-hf` | 基础模型路径 |
| `--lora_weights` | 必需 | LoRA权重路径 |
| `--test_data` | 必需 | 测试数据路径 |
| `--output_dir` | `./evaluation_results` | 输出目录 |
| `--batch_size` | 8 | 批次大小 |
| `--num_experts` | 8 | 专家数量 |
| `--num_gpus` | 2 | GPU数量 |

### 固定的LoRA参数

以下LoRA参数现在是固定的，不再作为命令行参数：

- `lora_rank`: 16
- `lora_alpha`: 32.0
- `lora_dropout`: 0.1
- `attention_lora_targets`: ["q_proj", "k_proj", "v_proj", "o_proj"]
- `expert_lora_targets`: ["gate_proj", "up_proj", "down_proj"]

## 数据格式

训练和测试数据应为JSON格式，每条数据包含以下字段：

```json
{
  "instruction": "In Asian culture, respect for elders is fundamental to social harmony. What value does this represent?",
  "instruction_mask": "Respect for elders is fundamental to social harmony. What value does this represent?",
  "input": "",
  "output": "0",
  "label": "0"
}
```

字段说明：
- `instruction`: 原始指令（包含文化信息）
- `instruction_mask`: 移除文化特定信息的指令
- `input`: 输入字段（通常为空）
- `output`: 期望的输出答案
- `label`: 文化标签（0-5，对应6个大洲）

## 环境要求

### 硬件要求

- **GPU**: 至少1张NVIDIA GPU，推荐2张或更多用于DDP训练
- **内存**: 推荐至少32GB系统内存
- **存储**: 足够空间存储模型权重和训练数据

### 软件要求

```bash
# Python依赖
torch>=2.0.0
transformers>=4.30.0
numpy
tqdm
scikit-learn
matplotlib
seaborn
```

### CUDA设置

确保正确设置CUDA环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1  # 根据实际GPU情况调整
export TOKENIZERS_PARALLELISM=false
```

## 性能优化建议

### 内存优化

1. **批次大小调整**：根据GPU内存调整`batch_size`
   - 单卡16GB: batch_size=2-4
   - 双卡16GB: batch_size=4-8
   - 更大GPU: 可适当增加

2. **梯度累积**：对于大批次训练，可以使用梯度累积

### 训练加速

1. **多卡训练**：使用`--num_gpus 2`或更多进行DDP训练
2. **混合精度**：模型内部已支持混合精度训练
3. **数据加载**：确保数据加载不成为瓶颈

## 监控和调试

### 日志监控

训练过程中会输出以下信息：
- 损失值变化（总损失、语言模型损失、负载均衡损失等）
- 专家利用率统计
- 验证集性能

### 常见问题

1. **CUDA内存不足**
   - 减少`batch_size`
   - 检查是否有其他进程占用GPU

2. **DDP初始化失败**
   - 检查端口是否被占用
   - 确保所有GPU可用

3. **数据加载错误**
   - 检查数据格式是否正确
   - 确保数据路径存在

## 输出文件

### 训练输出

```
outputs/
├── final_lora_weights.pt          # 最终LoRA权重
├── training_logs/                  # 训练日志
└── checkpoints/                    # 检查点（如果启用）
```

### 评估输出

```
evaluation_results/
├── evaluation_report.txt           # 文本报告
├── detailed_results.json          # 详细结果
├── confusion_matrix.png           # 混淆矩阵
├── expert_utilization.png         # 专家利用率图表
└── cultural_specialization.png    # 文化专业化图表
```

## 示例使用流程

```bash
# 1. 准备数据
cp your_train_data.json ./data/culture_train.json
cp your_test_data.json ./data/culture_test.json

# 2. 运行训练和评估
./run_lora_culturemoe_ddp.sh \\
    --data_path ./data/culture_train.json \\
    --test_data_path ./data/culture_test.json \\
    --num_gpus 2 \\
    --num_epochs 10 \\
    --batch_size 8

# 3. 查看结果
cat ./outputs/lora_culturemoe_ddp/evaluation_results/evaluation_report.txt
```

## 注意事项

1. **数据路径**：确保数据路径正确，脚本会检查文件是否存在
2. **GPU数量**：脚本会自动检查可用GPU数量并调整
3. **权限**：确保脚本有执行权限：`chmod +x run_lora_culturemoe_ddp.sh`
4. **端口冲突**：如果同时运行多个DDP任务，可能需要修改端口设置
5. **模型路径**：确保基础模型路径正确，支持本地路径和HuggingFace模型ID