## 📝 LoRA Only 模型保存选项

### 问题背景

训练 LoRA only 模型时，模型文件可能很大（几个 GB），如果只需要评估结果而不需要保存模型，可以节省大量磁盘空间。

---

## 🔧 解决方案

添加 `--save_model` 参数，控制是否保存模型：
- **默认**：不保存模型，只保存评估结果
- **添加 `--save_model`**：保存模型和评估结果

---

## 📊 支持的脚本

| 脚本 | 支持 `--save_model` | 默认行为 |
|------|-------------------|---------|
| `train_and_eval_lora_only.py` | ✅ | 不保存模型 |
| `train_and_eval_lora_only_flexible.py` | ✅ | 不保存模型 |
| `run_train_lora_only_flexible.sh` | ✅ | 不保存模型 |

---

## 🚀 使用方法

### 方法 1：不保存模型（默认）

#### Python 脚本
```bash
python train_and_eval_lora_only_flexible.py \
    --model_path /path/to/model \
    --train_file /path/to/data.json \
    --output_dir /path/to/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-6 \
    --lora_rank 8
    # 不添加 --save_model，默认不保存模型
```

#### Shell 脚本
```bash
# 修改 run_train_lora_only_flexible.sh
SAVE_MODEL=false  # 不保存模型（默认）

bash run_train_lora_only_flexible.sh
```

**输出**：
```
8. Skipping model saving (--save_model not set)

9. Final evaluation on validation set...

Final Evaluation Results
============================================================
   Accuracy:   0.6500
   Precision:  0.6800
   Recall:     0.6200
   F1:         0.6485
============================================================

✅ Training and evaluation completed!
   Eval results saved to: /path/to/output/eval_results.json
```

**保存的文件**：
- ✅ `/path/to/output/eval_results.json`（评估结果）
- ❌ 不保存模型文件

---

### 方法 2：保存模型

#### Python 脚本
```bash
python train_and_eval_lora_only_flexible.py \
    --model_path /path/to/model \
    --train_file /path/to/data.json \
    --output_dir /path/to/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-6 \
    --lora_rank 8 \
    --save_model  # ✅ 添加此参数以保存模型
```

#### Shell 脚本
```bash
# 修改 run_train_lora_only_flexible.sh
SAVE_MODEL=true  # 保存模型

bash run_train_lora_only_flexible.sh
```

**输出**：
```
8. Saving model to /path/to/output...
   ✅ Model saved

9. Final evaluation on validation set...

Final Evaluation Results
============================================================
   Accuracy:   0.6500
   Precision:  0.6800
   Recall:     0.6200
   F1:         0.6485
============================================================

✅ Training and evaluation completed!
   Model saved to: /path/to/output
   Eval results saved to: /path/to/output/eval_results.json
```

**保存的文件**：
- ✅ `/path/to/output/eval_results.json`（评估结果）
- ✅ `/path/to/output/adapter_model.bin`（LoRA 权重）
- ✅ `/path/to/output/adapter_config.json`（LoRA 配置）
- ✅ `/path/to/output/pytorch_model.bin`（分类头权重）
- ✅ `/path/to/output/config.json`（模型配置）
- ✅ `/path/to/output/tokenizer*`（Tokenizer 文件）

---

## 📊 磁盘空间对比

### 不保存模型（默认）

| 文件 | 大小 |
|------|------|
| `eval_results.json` | ~10 KB |
| **总计** | **~10 KB** |

### 保存模型

| 文件 | 大小 |
|------|------|
| `adapter_model.bin` | ~50 MB |
| `pytorch_model.bin` | ~2 GB |
| `tokenizer*` | ~5 MB |
| `eval_results.json` | ~10 KB |
| **总计** | **~2 GB** |

**节省空间**：~2 GB（约 99.5%）

---

## 🔍 代码实现

### Python 脚本

```python
def train_lora_only_flexible(args: LoRATrainingArguments):
    """训练 LoRA only 模型"""

    # ... 训练代码 ...

    # 10. 保存模型（可选）
    if args.save_model:
        print(f"\n8. Saving model to {args.output_dir}...")
        trainer.save_model()
        trainer.save_state()

        # 保存 LoRA 权重
        model.llama_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print("   ✅ Model saved")
    else:
        print(f"\n8. Skipping model saving (--save_model not set)")

    # 11. 最终评估
    print(f"\n9. Final evaluation on validation set...")
    metrics = trainer.evaluate()

    # ... 打印结果 ...

    # 保存指标（总是保存）
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Training and evaluation completed!")
    if args.save_model:
        print(f"   Model saved to: {args.output_dir}")
    print(f"   Eval results saved to: {os.path.join(args.output_dir, 'eval_results.json')}")

    return metrics
```

### Shell 脚本

```bash
#!/bin/bash

# 配置参数
SAVE_MODEL=false  # 是否保存模型（true/false）

# 构建命令
CMD="python train_and_eval_lora_only_flexible.py \
    --model_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_rank $LORA_RANK"

# 添加可选参数
if [ "$SAVE_MODEL" = "true" ]; then
    CMD="$CMD --save_model"
fi

# 运行训练
eval $CMD

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    if [ "$SAVE_MODEL" = "true" ]; then
        echo "Model saved to: $OUTPUT_DIR"
    fi
    echo "Eval results saved to: $OUTPUT_DIR/eval_results.json"
fi
```

---

## 💡 使用建议

### 何时不保存模型

✅ **推荐场景**：
1. **快速实验**：只需要评估结果，不需要部署模型
2. **超参数搜索**：测试多组超参数，只保留最佳结果
3. **磁盘空间有限**：节省存储空间
4. **只关心性能**：不需要后续使用模型

**示例**：
```bash
# 快速测试不同的学习率
for lr in 1e-5 2e-5 5e-5; do
    python train_and_eval_lora_only_flexible.py \
        --model_path /path/to/model \
        --train_file /path/to/data.json \
        --output_dir /path/to/output_lr_${lr} \
        --learning_rate $lr
        # 不保存模型，只保存评估结果
done
```

---

### 何时保存模型

✅ **推荐场景**：
1. **最终训练**：需要部署或进一步使用模型
2. **模型对比**：需要加载模型进行详细分析
3. **继续训练**：需要从检查点恢复训练
4. **模型分享**：需要分享给他人使用

**示例**：
```bash
# 最终训练，保存模型
python train_and_eval_lora_only_flexible.py \
    --model_path /path/to/model \
    --train_file /path/to/data.json \
    --output_dir /path/to/final_model \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --save_model  # ✅ 保存模型
```

---

## 📝 评估结果格式

### `eval_results.json`

```json
{
  "eval_loss": 0.5234,
  "eval_accuracy": 0.6500,
  "eval_precision": 0.6800,
  "eval_recall": 0.6200,
  "eval_f1": 0.6485,
  "eval_runtime": 123.45,
  "eval_samples_per_second": 32.1,
  "eval_steps_per_second": 4.0,
  "epoch": 3.0
}
```

**说明**：
- 包含所有评估指标
- 可以用于后续分析和对比
- 文件很小（~10 KB），不占用空间

---

## 🎯 快速参考

### 不保存模型（默认）

```bash
# Python
python train_and_eval_lora_only_flexible.py \
    --model_path /path/to/model \
    --train_file /path/to/data.json \
    --output_dir /path/to/output

# Shell（修改 SAVE_MODEL=false）
bash run_train_lora_only_flexible.sh
```

### 保存模型

```bash
# Python
python train_and_eval_lora_only_flexible.py \
    --model_path /path/to/model \
    --train_file /path/to/data.json \
    --output_dir /path/to/output \
    --save_model  # ✅ 添加此参数

# Shell（修改 SAVE_MODEL=true）
bash run_train_lora_only_flexible.sh
```

---

## 📊 总结

| 选项 | 保存模型 | 保存评估结果 | 磁盘占用 | 适用场景 |
|------|---------|------------|---------|---------|
| **默认**（不保存） | ❌ | ✅ | ~10 KB | 快速实验、超参数搜索 |
| **`--save_model`** | ✅ | ✅ | ~2 GB | 最终训练、模型部署 |

**推荐**：
- 🚀 **快速实验**：不保存模型（默认）
- 💾 **最终训练**：保存模型（添加 `--save_model`）

---

**现在可以根据需要灵活选择是否保存模型了！** 🎉

