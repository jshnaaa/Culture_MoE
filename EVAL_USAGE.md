# 评估脚本使用说明

## 问题说明

训练时 `compute_metrics()` 显示 `Accuracy: 0.0000`，但 `generated_answers.json` 中的预测结果是正确的。这说明问题在于评估时的对齐逻辑，而不是模型本身。

## 解决方案

使用后处理脚本 `eval_from_generated_answers.py` 直接从 `generated_answers.json` 计算准确率。

## 使用方法

### 1. 基本用法

```bash
# 在输出目录中运行
cd /path/to/output_dir
python /Users/yzl/ownCode/Culture_Moe/eval_from_generated_answers.py --input generated_answers.json
```

### 2. 指定完整路径

```bash
python /Users/yzl/ownCode/Culture_Moe/eval_from_generated_answers.py \
    --input /path/to/output_dir/generated_answers.json
```

### 3. 保存评估结果

```bash
python /Users/yzl/ownCode/Culture_Moe/eval_from_generated_answers.py \
    --input generated_answers.json \
    --output eval_metrics.json
```

## 输出示例

### 数字分类任务（CultureLLM）

```
📂 Loaded 1101 samples from: generated_answers.json
🔍 Detected task type: number

================================================================================
📊 Classification Metrics (Number)
================================================================================
Accuracy: 0.7548 (831/1101)
Classes:  [1, 2, 3, 4]
Precision: 0.7521 (weighted)
Recall:    0.7548 (weighted)
F1 Score:  0.7512 (weighted)

Per-class metrics:
--------------------------------------------------------------------------------
Class      Precision    Recall       F1           Support
--------------------------------------------------------------------------------
1          0.7234       0.8012       0.7604       245
2          0.7891       0.7456       0.7667       312
3          0.7123       0.7234       0.7178       289
4          0.7845       0.7512       0.7675       255

Confusion Matrix:
--------------------------------------------------------------------------------
           1          2          3          4
--------------------------------------------------------------------------------
         1 196        28         15         6
         2 31         232        35         14
         3 18         42         209        20
         4 9          15         22         209
================================================================================
```

### 文本分类任务（NormAD - yes/no/neutral）

```
📂 Loaded 500 samples from: generated_answers.json
🔍 Detected task type: text

================================================================================
📊 Classification Metrics (yes/no/neutral)
================================================================================
Accuracy:  0.8520 (426/500)
Precision: 0.8498 (weighted)
Recall:    0.8520 (weighted)
F1 Score:  0.8505 (weighted)

Per-class metrics:
--------------------------------------------------------------------------------
Class      Precision    Recall       F1           Support
--------------------------------------------------------------------------------
yes        0.8234       0.8512       0.8371       167
no         0.8756       0.8423       0.8586       178
neutral    0.8512       0.8645       0.8578       155

Confusion Matrix:
--------------------------------------------------------------------------------
           yes        no         neutral
--------------------------------------------------------------------------------
       yes 142        15         10
        no 18         150        10
   neutral 12         8          135
================================================================================
```

## 集成到训练流程

### 方法 1：训练后手动运行

```bash
# 训练
sh run_train_lora_only_gen.sh llama 4

# 评估
python eval_from_generated_answers.py \
    --input outputs/llama_cultureLLM_lora_only_gen/generated_answers.json
```

### 方法 2：修改训练脚本自动运行

在 `run_train_lora_only_gen.sh` 末尾添加：

```bash
# 训练完成后自动评估
if [ -f "${OUTPUT_DIR}/generated_answers.json" ]; then
    echo "Running post-evaluation..."
    python eval_from_generated_answers.py \
        --input "${OUTPUT_DIR}/generated_answers.json" \
        --output "${OUTPUT_DIR}/eval_metrics.json"
fi
```

## 注意事项

1. **文件路径**：确保 `generated_answers.json` 存在
2. **任务类型**：脚本会自动检测任务类型（text/number/other）
3. **输出格式**：
   - 终端：彩色格式化输出
   - JSON 文件：机器可读的指标

## 为什么这个方法有效？

1. **绕过对齐问题**：直接使用已解码的文本，避免 token 对齐错误
2. **简单可靠**：字符串比较比 token ID 对齐更直观
3. **易于调试**：可以直接查看 `generated_answers.json` 中的预测结果

## 下一步

如果需要修复 `compute_metrics()` 中的对齐问题，可以参考这个脚本的逻辑。但在大多数情况下，使用后处理脚本已经足够。

