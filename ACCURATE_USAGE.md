# 准确的使用说明

## ⚠️ 重要澄清

### save_model=false 时实际保存的文件

之前的说明不够准确。实际情况是：

**save_model=false 时会保存**：
1. ✅ `eval_results.json` - 评估结果（~1 KB）
2. ✅ `trainer_state.json` - 训练状态（~5 KB）
3. ✅ `training_args.bin` - 训练参数（~10 KB）
4. ✅ `runs/` - TensorBoard 日志（~30-50 MB）

**save_model=false 时不会保存**：
1. ❌ 模型权重文件（`pytorch_model.bin`、`adapter_model.bin` 等）
2. ❌ Checkpoint 目录（`checkpoint-xxx/`）
3. ❌ Tokenizer 文件
4. ❌ 模型配置文件
5. ❌ 优化器状态

## 磁盘空间对比

| 配置 | 评估结果 | 训练日志 | 模型权重 | Checkpoint | 总计 |
|------|---------|---------|---------|-----------|------|
| `save_model=false` | ✅ 1 KB | ✅ ~50 MB | ❌ | ❌ | **~50 MB** |
| `save_model=true` | ✅ 1 KB | ✅ ~50 MB | ✅ ~50 MB | ✅ ~800 MB | **~900 MB** |

**节省空间**：~850 MB（每个训练任务）

## 详细文件列表

### save_model=false 的输出目录

```
output/lora_only_llama_3class_20251028_1234/
├── eval_results.json          # ✅ 评估结果
├── trainer_state.json          # ✅ 训练状态（loss 历史等）
├── training_args.bin           # ✅ 训练参数
└── runs/                       # ✅ TensorBoard 日志
    └── events.out.tfevents.xxx
```

**用途**：
- 查看模型性能（accuracy、F1 等）
- 查看训练曲线
- 分析训练过程
- **无法用于推理**（没有模型权重）

### save_model=true 的输出目录

```
output/lora_only_llama_3class_20251028_1234/
├── eval_results.json          # ✅ 评估结果
├── trainer_state.json          # ✅ 训练状态
├── training_args.bin           # ✅ 训练参数
├── runs/                       # ✅ TensorBoard 日志
├── checkpoint-500/             # ✅ 训练检查点
│   ├── model.safetensors
│   ├── optimizer.pt
│   └── ...
├── checkpoint-1000/            # ✅ 训练检查点
├── checkpoint-1500/            # ✅ 训练检查点
├── adapter_model.bin           # ✅ LoRA 权重
├── adapter_config.json         # ✅ LoRA 配置
├── config.json                 # ✅ 模型配置
├── tokenizer.json              # ✅ Tokenizer
└── ...
```

**用途**：
- 所有 save_model=false 的功能
- **可以用于推理**（有模型权重）
- 可以恢复训练（有 checkpoint）

## 使用场景

### 场景 1：快速实验（save_model=false）

**适用于**：
- 测试不同超参数
- 对比不同配置
- 只关心性能指标
- 不需要使用模型

**命令**：
```bash
sh run_train_lora_only.sh llama 2 false
sh run_train_lora_only.sh llama 3 false
sh run_train_lora_only.sh llama 4 false
```

**优势**：
- ✅ 快速迭代
- ✅ 节省磁盘空间（每个任务节省 ~850 MB）
- ✅ 保留评估结果和训练曲线

**限制**：
- ❌ 无法用于推理
- ❌ 无法恢复训练

### 场景 2：保存最佳模型（save_model=true）

**适用于**：
- 训练最终模型
- 需要使用模型进行推理
- 需要恢复训练
- 需要部署模型

**命令**：
```bash
sh run_train_lora_only.sh llama 3 true
```

**优势**：
- ✅ 可以用于推理
- ✅ 可以恢复训练
- ✅ 可以部署使用

**限制**：
- ❌ 占用较多磁盘空间

## 推荐工作流程

### 步骤 1：快速实验（不保存模型）

```bash
# 测试多个配置
sh run_train_lora_only.sh llama 2 false
sh run_train_lora_only.sh llama 3 false
sh run_train_lora_only.sh llama 4 false
sh run_train_lora_only.sh llama 5 false

# 对比评估结果
cat output/lora_only_llama_*class_*/eval_results.json
```

**磁盘占用**：4 × 50 MB = 200 MB

### 步骤 2：选择最佳配置

```bash
# 假设发现 3 分类效果最好
# 查看详细结果
cat output/lora_only_llama_3class_*/eval_results.json
```

### 步骤 3：保存最佳模型

```bash
# 重新训练并保存模型
sh run_train_lora_only.sh llama 3 true
```

**磁盘占用**：900 MB

### 总磁盘占用

- 实验阶段：200 MB（4 个配置）
- 最终模型：900 MB（1 个模型）
- **总计**：1.1 GB

**如果全部保存模型**：4 × 900 MB = 3.6 GB

**节省空间**：2.5 GB

## 查看保存的文件

### 查看评估结果

```bash
# 查看 JSON 文件
cat output/lora_only_llama_3class_*/eval_results.json

# 格式化输出
cat output/lora_only_llama_3class_*/eval_results.json | jq

# 提取特定指标
cat output/lora_only_llama_3class_*/eval_results.json | jq '.eval_accuracy'
```

### 查看训练曲线

```bash
# 启动 TensorBoard
tensorboard --logdir output/lora_only_llama_3class_*/runs

# 在浏览器中打开 http://localhost:6006
```

### 查看训练状态

```bash
# 查看训练状态
cat output/lora_only_llama_3class_*/trainer_state.json | jq

# 查看最终 loss
cat output/lora_only_llama_3class_*/trainer_state.json | jq '.log_history[-1]'
```

## 常见问题

### Q1: save_model=false 时真的只保存 eval_results.json 吗？

**A**: 不是。还会保存：
- `trainer_state.json` - 训练状态
- `training_args.bin` - 训练参数
- `runs/` - TensorBoard 日志

但**不会保存模型权重和 checkpoint**。

### Q2: 为什么还要保存训练日志？

**A**: 训练日志很重要：
- 可以查看训练曲线（loss、accuracy）
- 可以分析训练过程（是否过拟合）
- 可以调试问题
- 占用空间很小（~50 MB）

### Q3: 如何完全不保存任何日志？

**A**: 不推荐，但如果真的需要：

修改 `TrainingArguments`：
```python
training_args = TrainingArguments(
    # ...
    report_to=[],  # 禁用 TensorBoard
    logging_strategy="no",  # 禁用日志
)
```

然后手动删除 `trainer_state.json` 和 `training_args.bin`。

但这样做会失去所有训练信息，只为节省 ~50 MB，**非常不值得**。

### Q4: save_model=false 时能恢复训练吗？

**A**: 不能。因为没有保存：
- 模型权重
- 优化器状态
- Checkpoint

### Q5: save_model=false 时能用模型推理吗？

**A**: 不能。因为没有保存模型权重。

## 总结

### 准确的说法

**save_model=false 时**：
- ✅ 不保存模型权重（节省 ~850 MB）
- ✅ 不保存 checkpoint
- ⚠️  仍然保存训练日志和状态（~50 MB）
- ✅ 保存评估结果

**适用场景**：
- 快速实验和对比
- 只关心性能指标
- 不需要使用模型

**不适用场景**：
- 需要使用模型推理
- 需要恢复训练
- 需要部署模型

### 修正后的 README

之前说"只保存评估结果"不够准确，应该说：

> **save_model=false 时**：不保存模型权重和 checkpoint，但会保存训练日志、状态和评估结果（总计约 50 MB）

这样更准确！

