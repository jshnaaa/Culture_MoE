# 📊 评估指南：使用合并后的模型

## 🎯 概述

现在 `run_eval_dual_binary.sh` 已经更新为使用合并后的模型进行评估。

## 🚀 完整工作流程

### 步骤 1：训练模型

```bash
bash run_train_ddp_lora_dual.sh
```

**输出**：
```
/root/autodl-fs/output/classi_dual_20251019_131545/
├── adapter_config.json      # LoRA 配置
├── adapter_model.bin         # LoRA 权重
├── pytorch_model.bin         # MoE 权重
└── ...
```

### 步骤 2：合并 LoRA 权重

```bash
# 编辑合并脚本，设置正确的路径
vim run_merge_lora.sh

# 运行合并
bash run_merge_lora.sh
```

**输出**：
```
/root/autodl-fs/merged_models/llama_lora_merged_20251019/
├── config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── tokenizer_config.json
└── ...
```

### 步骤 3：评估合并后的模型

```bash
# 编辑评估脚本，设置正确的路径
vim run_eval_dual_binary.sh

# 修改以下变量：
# MERGED_MODEL_PATH="/root/autodl-fs/merged_models/llama_lora_merged_20251019"
# MOE_CHECKPOINT_PATH="/root/autodl-fs/output/classi_dual_20251019_131545"

# 运行评估
bash run_eval_dual_binary.sh
```

## 📝 配置说明

### run_eval_dual_binary.sh 配置

```bash
# ✅ 方式 1：使用已经合并好的模型（推荐）
MERGED_MODEL_PATH="/root/autodl-fs/merged_models/llama_lora_merged_20251019"

# ✅ 方式 2：如果还没有合并，可以在评估前自动合并
# 取消注释以下代码块：
# BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
# LORA_CHECKPOINT_PATH="/root/autodl-fs/output/classi_dual_20251019_131545"
# MERGED_MODEL_PATH="/root/autodl-fs/merged_models/llama_lora_merged_$(date +%Y%m%d_%H%M%S)"
#
# echo "============================================================"
# echo "Step 1: Merging LoRA weights..."
# echo "============================================================"
# python merge_lora_and_save.py \
#     --base_model_path $BASE_MODEL_PATH \
#     --lora_checkpoint_path $LORA_CHECKPOINT_PATH \
#     --output_path $MERGED_MODEL_PATH \
#     --device cuda:0

# MoE 权重路径
MOE_CHECKPOINT_PATH="/root/autodl-fs/output/classi_dual_20251019_131545"

# 测试数据
TEST_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"

# 映射策略
MAPPING_STRATEGY="merge_neutral_to_no"  # 或 "merge_neutral_to_yes"
```

## 🎯 两种使用方式

### 方式 1：分步执行（推荐）

```bash
# 1. 先合并
bash run_merge_lora.sh

# 2. 再评估
bash run_eval_dual_binary.sh
```

**优点**：
- ✅ 清晰明确
- ✅ 可以重复使用合并后的模型
- ✅ 便于调试

### 方式 2：一键执行

编辑 `run_eval_dual_binary.sh`，取消注释合并代码块：

```bash
# 取消注释这部分
BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
LORA_CHECKPOINT_PATH="/root/autodl-fs/output/classi_dual_20251019_131545"
MERGED_MODEL_PATH="/root/autodl-fs/merged_models/llama_lora_merged_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "Step 1: Merging LoRA weights..."
echo "============================================================"
python merge_lora_and_save.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_checkpoint_path $LORA_CHECKPOINT_PATH \
    --output_path $MERGED_MODEL_PATH \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "❌ Merge failed! Please check the error messages above."
    exit 1
fi
echo ""
```

然后运行：

```bash
bash run_eval_dual_binary.sh
```

**优点**：
- ✅ 一键完成
- ✅ 自动合并 + 评估

**缺点**：
- ⚠️ 每次都会重新合并（耗时）
- ⚠️ 占用更多磁盘空间

## 📊 预期输出

### 合并阶段

```
============================================================
Merging LoRA Weights
============================================================
Base model: /root/autodl-tmp/.../Meta-Llama-3.1-8B-Instruct
LoRA checkpoint: /root/autodl-fs/output/classi_dual_20251019_131545
Output: /root/autodl-fs/merged_models/llama_lora_merged_20251019
============================================================

1. Loading base model from: /root/autodl-tmp/.../Meta-Llama-3.1-8B-Instruct
   ✅ Base model loaded on cuda:0

2. Loading tokenizer...
   ✅ Tokenizer loaded from checkpoint

3. Loading LoRA weights from: /root/autodl-fs/output/classi_dual_20251019_131545
   ✅ LoRA weights loaded

4. Merging LoRA weights into base model...
   ✅ LoRA weights merged

5. Saving merged model to: /root/autodl-fs/merged_models/llama_lora_merged_20251019
   ✅ Merged model saved

✅ Merge completed successfully!
```

### 评估阶段

```
============================================================
Evaluating Merged Model
============================================================
Merged model: /root/autodl-fs/merged_models/llama_lora_merged_20251019
MoE checkpoint: /root/autodl-fs/output/classi_dual_20251019_131545
Test file: /root/autodl-fs/CulturalBench_Hard_merge.json
Mapping strategy: merge_neutral_to_no
============================================================

Loading merged model from /root/autodl-fs/merged_models/llama_lora_merged_20251019...
✅ Tokenizer loaded
Loading merged LLaMA model...
Using device: cuda:0
✅ Merged LLaMA model loaded (LoRA weights already merged)
Loading MoE weights from /root/autodl-fs/output/.../pytorch_model.bin...
✅ MoE weights loaded
✅ Model loaded on: cuda:0
   Model dtype: torch.float16

Loading test data from /root/autodl-fs/CulturalBench_Hard_merge.json...
Mapping strategy: merge_neutral_to_no
Test dataset size: 4908

Evaluating on 4908 samples...
100%|████████████████████████████████| 614/614 [05:23<00:00,  1.90it/s]

============================================================
Evaluation Results (Binary Classification)
============================================================

📊 Overall Metrics:
   Accuracy:   0.8234
   Precision:  0.8156
   Recall:     0.7923
   F1:         0.8038

📊 Classification Report:
              precision    recall  f1-score   support

      no (0)     0.8312    0.8545    0.8427      2454
     yes (1)     0.8156    0.7923    0.8038      2454

    accuracy                         0.8234      4908
   macro avg     0.8234    0.8234    0.8232      4908
weighted avg     0.8234    0.8234    0.8232      4908

📊 Confusion Matrix:
           Predicted
           no   yes
Actual no  2097  357
Actual yes  510 1944

💾 Saving results to /root/autodl-fs/eval_results_merged_20251019_XXXXXX.json...
✅ Results saved!

============================================================

============================================================
✅ Evaluation completed successfully!
============================================================
Mapping strategy: merge_neutral_to_no
Results saved to: /root/autodl-fs/eval_results_merged_20251019_XXXXXX.json
============================================================
```

## 🔧 故障排除

### 问题 1：找不到合并后的模型

**错误**：
```
❌ Evaluation failed!
Possible issues:
  1. Merged model not found at: /root/autodl-fs/merged_models/llama_lora_merged_20251019
     → Run: bash run_merge_lora.sh first
```

**解决方案**：
```bash
# 先运行合并
bash run_merge_lora.sh

# 或检查路径是否正确
ls -la /root/autodl-fs/merged_models/
```

### 问题 2：找不到 MoE checkpoint

**错误**：
```
⚠️  Warning: pytorch_model.bin not found
```

**解决方案**：
```bash
# 检查 checkpoint 路径
ls -la /root/autodl-fs/output/classi_dual_20251019_131545/

# 确保包含 pytorch_model.bin
bash check_checkpoint.sh /root/autodl-fs/output/classi_dual_20251019_131545
```

### 问题 3：显存不足

**解决方案**：
```bash
# 编辑 run_eval_dual_binary.sh
# 修改 batch_size
--batch_size 4  # 从 8 改为 4 或 2
```

## 📚 相关文档

- `FINAL_SOLUTION.md` - 完整解决方案
- `LORA_MERGE_WORKFLOW.md` - 详细工作流程
- `merge_lora_and_save.py` - 合并脚本
- `examples/eval_classification_dual_merged.py` - 评估脚本

## 🎉 总结

### 修改内容

✅ **run_eval_dual_binary.sh** 已更新为：
- 使用 `examples/eval_classification_dual_merged.py`
- 需要先合并 LoRA 权重
- 支持两种使用方式（分步/一键）

### 使用流程

```
训练 → 合并 → 评估
  ↓      ↓      ↓
LoRA  完整   快速
权重  模型   推理
```

### 快速命令

```bash
# 训练
bash run_train_ddp_lora_dual.sh

# 合并
bash run_merge_lora.sh

# 评估
bash run_eval_dual_binary.sh
```

---

**现在你可以使用合并后的模型进行快速评估了！** 🚀

