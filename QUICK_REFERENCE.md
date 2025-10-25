# 🚀 快速参考：Checkpoint LoRA 权重

## ✅ 问题已解决

每个 checkpoint 现在都包含 LoRA 权重！

## 📝 快速命令

### 1. 训练（自动保存 LoRA）

```bash
bash run_train_ddp_lora_dual.sh
```

### 2. 验证 checkpoint

```bash
# 验证所有 checkpoint
bash verify_checkpoint_save.sh /root/autodl-fs/output/classi_dual_20251019_XXXXXX

# 检查单个 checkpoint
bash check_checkpoint.sh /root/autodl-fs/output/classi_dual_20251019_XXXXXX/checkpoint-1000
```

### 3. 评估 checkpoint

```bash
# 方法 1：修改脚本
vim run_eval_dual_binary.sh
# 修改 MODEL_PATH="/path/to/checkpoint-1000"

bash run_eval_dual_binary.sh

# 方法 2：直接指定
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_XXXXXX/checkpoint-1000" \
bash run_eval_dual_binary.sh
```

### 4. 比较所有 checkpoint

```bash
# 评估所有 checkpoint
for ckpt in /root/autodl-fs/output/classi_dual_20251019_XXXXXX/checkpoint-*; do
    echo "Evaluating $(basename $ckpt)"
    MODEL_PATH="$ckpt" bash run_eval_dual_binary.sh
done
```

## 🔍 检查清单

### ✅ 训练前

- [ ] 确认使用最新代码（包含 `callbacks.py`）
- [ ] 确认训练参数：`--use_llama_lora True`
- [ ] 确认训练参数：`--freeze_llama False`

### ✅ 训练中

查看日志，应该看到：
```
✅ Added SaveFullModelCallback to save LoRA weights in every checkpoint
...
✅ [Checkpoint-500] Saved LoRA weights
✅ [Checkpoint-500] Saved tokenizer
```

### ✅ 训练后

```bash
# 验证 checkpoint
bash verify_checkpoint_save.sh <output_dir>

# 应该看到
✅ SUCCESS! All checkpoints contain LoRA weights!
```

## 📊 Checkpoint 内容

每个 checkpoint 应包含：

```
checkpoint-1000/
├── adapter_config.json        ✅ LoRA 配置
├── adapter_model.bin          ✅ LoRA 权重
├── pytorch_model.bin          ✅ MoE 权重
├── tokenizer_config.json      ✅ Tokenizer
└── trainer_state.json         ✅ 训练状态
```

## ⚠️ 故障排除

### 问题：Checkpoint 没有 LoRA 权重

**检查**：
```bash
bash check_checkpoint.sh /path/to/checkpoint
```

**解决**：
1. 确认使用了最新代码
2. 确认训练参数正确
3. 重新训练

### 问题：评估时找不到 LoRA

**错误**：
```
⚠️  Warning: Could not load LoRA weights
```

**解决**：
```bash
# 使用最终模型
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_XXXXXX"

# 或检查 checkpoint
bash check_checkpoint.sh /path/to/checkpoint
```

## 📚 详细文档

- `SOLUTION_SUMMARY.md` - 完整解决方案
- `CHECKPOINT_LORA_GUIDE.md` - 详细使用指南
- `src/llamafactory/train/classification/callbacks.py` - 回调实现

## 🎯 核心改动

### 1. 新增文件

```
src/llamafactory/train/classification/callbacks.py  ← 新增
```

### 2. 修改文件

```python
# src/llamafactory/train/classification/workflow.py
from .callbacks import SaveFullModelCallback  # ← 新增

callbacks = []
if args.use_llama_lora:
    callbacks.append(SaveFullModelCallback())  # ← 新增

trainer = ClassificationTrainer(
    ...
    callbacks=callbacks,  # ← 新增
)
```

## ✨ 效果

### 修改前 ❌

```
checkpoint-1000/
├── pytorch_model.bin          # 只有 MoE 权重
└── trainer_state.json
```

评估时：
```
⚠️  Warning: Could not load LoRA weights
   Using base model only
```

### 修改后 ✅

```
checkpoint-1000/
├── adapter_config.json        # ✅ LoRA 配置
├── adapter_model.bin          # ✅ LoRA 权重
├── pytorch_model.bin          # ✅ MoE 权重
└── ...
```

评估时：
```
✅ LoRA weights loaded and merged
✅ MoE weights loaded
```

---

**问题已彻底解决！下次训练时，所有 checkpoint 都会自动包含 LoRA 权重！** 🎉

