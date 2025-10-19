# Checkpoint LoRA 权重保存完整指南

## 🎯 问题描述

默认情况下，Hugging Face Trainer 只在训练结束时保存 PEFT/LoRA 权重，中间 checkpoint 不包含 LoRA 权重。这导致：
- ❌ 无法使用中间 checkpoint 进行评估
- ❌ 训练中断后无法从 checkpoint 恢复 LoRA 权重
- ❌ 无法比较不同 checkpoint 的性能

## ✅ 解决方案

我们创建了自定义回调 `SaveFullModelCallback`，确保每个 checkpoint 都包含：
1. ✅ LoRA 权重（`adapter_config.json`, `adapter_model.bin`）
2. ✅ MoE 组件权重（`pytorch_model.bin`）
3. ✅ Tokenizer 配置
4. ✅ 训练状态

## 📝 实现步骤

### 步骤 1：创建回调类

已创建文件：`src/llamafactory/train/classification/callbacks.py`

包含两个回调类：
- `SavePeftModelCallback`: 基础版本，只保存 LoRA 权重
- `SaveFullModelCallback`: 完整版本，保存 LoRA + Tokenizer

### 步骤 2：修改训练脚本

已修改文件：`src/llamafactory/train/classification/workflow.py`

添加了：
```python
from .callbacks import SavePeftModelCallback, SaveFullModelCallback

# 在创建 Trainer 时
callbacks = []
if args.use_llama_lora and not args.freeze_llama:
    callbacks.append(SaveFullModelCallback())

trainer = ClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_classification_metrics,
    callbacks=callbacks,  # ✅ 添加回调
)
```

### 步骤 3：重新训练

使用修改后的代码重新训练：

```bash
bash run_train_ddp_dual.sh
```

训练时会看到：
```
Creating trainer...
✅ Added SaveFullModelCallback to save LoRA weights in every checkpoint

Starting training...
...
✅ [Checkpoint-500] Saved LoRA weights
✅ [Checkpoint-500] Saved tokenizer
...
✅ [Checkpoint-1000] Saved LoRA weights
✅ [Checkpoint-1000] Saved tokenizer
```

## 🔍 验证

### 方法 1：使用验证脚本

```bash
# 训练完成后验证
bash verify_checkpoint_save.sh /root/autodl-fs/output/classi_dual_20251019_XXXXXX
```

预期输出：
```
============================================================
Checkpoint LoRA Weights Verification
============================================================

Checking output directory: /root/autodl-fs/output/classi_dual_20251019_XXXXXX

📁 Final Model:
  ✅ adapter_config.json found
  ✅ adapter_model found

📁 Checkpoints:
  ✅ checkpoint-500 - Has LoRA weights
  ✅ checkpoint-1000 - Has LoRA weights
  ✅ checkpoint-1500 - Has LoRA weights

============================================================
Summary:
============================================================
Total checkpoints: 3
Checkpoints with LoRA: 3

✅ SUCCESS! All checkpoints contain LoRA weights!
```

### 方法 2：手动检查

```bash
# 检查单个 checkpoint
bash check_checkpoint.sh /root/autodl-fs/output/classi_dual_20251019_XXXXXX/checkpoint-1000
```

### 方法 3：使用 checkpoint 评估

```bash
# 现在可以使用任何 checkpoint 进行评估
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_XXXXXX/checkpoint-1000"
bash run_eval_dual_binary.sh
```

## 📊 Checkpoint 目录结构

训练完成后，每个 checkpoint 应该包含：

```
checkpoint-1000/
├── adapter_config.json          # ✅ LoRA 配置
├── adapter_model.bin            # ✅ LoRA 权重
├── pytorch_model.bin            # ✅ MoE 权重
├── trainer_state.json           # ✅ 训练状态
├── training_args.bin            # ✅ 训练参数
├── tokenizer_config.json        # ✅ Tokenizer 配置
├── tokenizer.json               # ✅ Tokenizer
└── special_tokens_map.json      # ✅ 特殊 token 映射
```

## 🎯 使用场景

### 场景 1：评估不同 checkpoint

```bash
# 评估 checkpoint-500
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_XXXXXX/checkpoint-500"
bash run_eval_dual_binary.sh

# 评估 checkpoint-1000
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_XXXXXX/checkpoint-1000"
bash run_eval_dual_binary.sh

# 评估最终模型
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_XXXXXX"
bash run_eval_dual_binary.sh
```

### 场景 2：从 checkpoint 恢复训练

```bash
# 如果训练中断，可以从任何 checkpoint 恢复
python examples/train_classification.py \
    --resume_from_checkpoint /root/autodl-fs/output/classi_dual_20251019_XXXXXX/checkpoint-1000 \
    ...其他参数...
```

### 场景 3：选择最佳 checkpoint

```bash
# 评估所有 checkpoint，选择性能最好的
for ckpt in /root/autodl-fs/output/classi_dual_20251019_XXXXXX/checkpoint-*; do
    echo "Evaluating $ckpt"
    MODEL_PATH="$ckpt" bash run_eval_dual_binary.sh
done
```

## ⚠️ 注意事项

### 1. 显存占用

保存 LoRA 权重会增加磁盘占用：
- 每个 checkpoint: ~100-200MB（LoRA） + ~500MB（MoE） = ~600-700MB
- 如果有 10 个 checkpoint: ~6-7GB

**解决方案**：
```python
# 在 TrainingArguments 中限制保存的 checkpoint 数量
save_total_limit=3  # 只保留最近 3 个 checkpoint
```

### 2. 训练速度

保存 LoRA 权重会略微增加保存时间（~1-2 秒/checkpoint），但不影响训练速度。

### 3. 兼容性

确保使用相同版本的 `peft` 库加载 checkpoint：
```bash
pip install peft==0.7.0  # 或你训练时使用的版本
```

## 🔧 故障排除

### 问题 1：Checkpoint 中没有 LoRA 权重

**检查**：
```bash
bash verify_checkpoint_save.sh /root/autodl-fs/output/classi_dual_20251019_XXXXXX
```

**可能原因**：
1. 训练时没有使用 LoRA（`--use_llama_lora False`）
2. 回调没有正确添加
3. 训练在保存 checkpoint 前中断

**解决方案**：
- 确保训练参数包含 `--use_llama_lora True`
- 确保 `--freeze_llama False`
- 重新训练

### 问题 2：评估时找不到 LoRA 权重

**错误信息**：
```
⚠️  Warning: Could not load LoRA weights: Can't find 'adapter_config.json'
```

**解决方案**：
```bash
# 检查 checkpoint 内容
bash check_checkpoint.sh /path/to/checkpoint

# 如果 checkpoint 不完整，使用最终模型
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_XXXXXX"
```

### 问题 3：训练时没有看到保存 LoRA 的日志

**检查**：
```bash
# 查看训练日志
tail -f /root/autodl-fs/output/classi_dual_20251019_XXXXXX/logs/events.out.tfevents.*
```

**可能原因**：
- 回调没有正确导入
- 训练参数设置错误

**解决方案**：
- 确认 `callbacks.py` 文件存在
- 确认 `workflow.py` 正确导入并使用回调

## 📚 相关文件

- `src/llamafactory/train/classification/callbacks.py` - 回调实现
- `src/llamafactory/train/classification/workflow.py` - 训练流程
- `check_checkpoint.sh` - 检查单个 checkpoint
- `verify_checkpoint_save.sh` - 验证所有 checkpoint
- `run_eval_dual_binary.sh` - 评估脚本

## 🎉 总结

现在你的训练流程会：
1. ✅ 在每个 checkpoint 保存 LoRA 权重
2. ✅ 在每个 checkpoint 保存 Tokenizer
3. ✅ 在最终模型保存完整权重
4. ✅ 支持从任何 checkpoint 恢复训练
5. ✅ 支持评估任何 checkpoint

**下次训练时，所有 checkpoint 都会包含 LoRA 权重！** 🚀

