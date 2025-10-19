# 如何让 checkpoint 包含 LoRA 权重

## 问题
默认情况下，Hugging Face Trainer 只在最终模型中保存 LoRA 权重，中间 checkpoint 不包含。

## 解决方案

### 方法 1：修改 Trainer 的保存逻辑

在 `src/llamafactory/train/classification/workflow.py` 中添加自定义保存回调：

```python
from transformers import TrainerCallback

class SavePeftModelCallback(TrainerCallback):
    """保存 PEFT 模型的回调"""

    def on_save(self, args, state, control, **kwargs):
        """在保存 checkpoint 时也保存 LoRA 权重"""
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}"
        )

        # 保存 LoRA 权重
        if hasattr(kwargs.get("model"), "llama_model"):
            llama_model = kwargs["model"].llama_model
            if hasattr(llama_model, "save_pretrained"):
                peft_model_path = checkpoint_folder
                llama_model.save_pretrained(peft_model_path)
                print(f"✅ Saved LoRA weights to {peft_model_path}")

        return control

# 在创建 Trainer 时添加回调
trainer = ClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_classification_metrics,
    callbacks=[SavePeftModelCallback()],  # ✅ 添加这行
)
```

### 方法 2：使用最终模型

最简单的方法是使用训练完成后的最终模型，它包含所有权重：

```bash
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_131545"
```

### 方法 3：手动合并 checkpoint

如果必须使用某个 checkpoint，可以手动合并：

```python
# 1. 从最终模型加载 LoRA 权重
final_model_path = "/root/autodl-fs/output/classi_dual_20251019_131545"

# 2. 从 checkpoint 加载 MoE 权重
checkpoint_path = "/root/autodl-fs/output/classi_dual_20251019_131545/checkpoint-3000"

# 3. 在评估脚本中分别加载
# - LoRA 权重从 final_model_path
# - MoE 权重从 checkpoint_path
```

## 推荐做法

**对于评估**：使用最终模型
```bash
MODEL_PATH="/root/autodl-fs/output/classi_dual_20251019_131545"
```

**对于训练**：如果需要从 checkpoint 恢复训练，Trainer 会自动处理

## 检查 checkpoint

使用提供的脚本检查：
```bash
bash check_checkpoint.sh /root/autodl-fs/output/classi_dual_20251019_131545/checkpoint-3000

