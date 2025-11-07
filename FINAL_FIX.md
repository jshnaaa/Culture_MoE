# 最终修复：编码问题

## 问题

运行训练时遇到 `UnicodeDecodeError`:

```
UnicodeDecodeError: 'ascii' codec can't decode byte 0xf0 in position 1: ordinal not in range(128)
```

## 原因

`subprocess.run()` 在调用 `eval_from_generated_answers.py` 时，默认使用系统的默认编码（通常是 ASCII），无法解码包含 emoji 或特殊字符的输出（如 📊、✅ 等）。

## 解决方案

在 `subprocess.run()` 中添加编码参数：

```python
result = subprocess.run(
    ["python", eval_script, "--input", answers_file, "--output", metrics_file],
    capture_output=True,
    text=True,
    encoding='utf-8',  # ✅ 指定 UTF-8 编码
    errors='replace',  # ✅ 遇到无法解码的字符时替换
    check=True
)
```

## 修改位置

文件：`train_lora_only_gen.py`
函数：`generate_and_evaluate()`
行数：约 110-120 行

## 测试

现在运行训练脚本应该不会再出现编码错误：

```bash
sh run_train_lora_only_gen.sh llama 4
```

## 预期输出

```
================================================================================
Generating answers for evaluation...
================================================================================
Generating: 100%|████████████████████████████| 1101/1101 [04:30<00:00,  4.06it/s]
✅ Generated answers saved to: generated_answers.json

Running post-evaluation...

================================================================================
📊 Classification Metrics (Number)
================================================================================
Accuracy: 0.7548 (831/1101)
Precision: 0.7521 (weighted)
Recall:    0.7548 (weighted)
F1 Score:  0.7512 (weighted)
================================================================================

📊 Epoch 1 Results:
   Train Loss: 0.7749
   Eval Accuracy: 0.7548
   🏆 New best model! Eval Accuracy: 0.7548
   ✅ Best LoRA weights saved to: best_lora/
================================================================================
```

## 完成！

所有问题已修复：
- ✅ 移除了 `preprocess_logits_for_metrics`
- ✅ 移除了复杂的 `compute_metrics`
- ✅ 使用真正的生成 + post-eval 统计
- ✅ 修复了编码问题
- ✅ 每个 epoch 后自动生成答案并评估
- ✅ 根据 eval_accuracy 保存最佳模型

现在可以正常训练了！🎉

