# 🚀 快速开始：LoRA 合并与评估

## 📝 三步完成评估

### 步骤 1：训练模型

```bash
bash run_train_ddp_lora_dual.sh
```

### 步骤 2：合并 LoRA 权重

```bash
# 编辑脚本设置路径
vim run_merge_lora.sh

# 运行合并
bash run_merge_lora.sh
```

### 步骤 3：评估模型

```bash
# 编辑脚本设置路径
vim run_eval_dual_binary.sh

# 运行评估
bash run_eval_dual_binary.sh
```

## 🎯 关键路径配置

### run_merge_lora.sh

```bash
BASE_MODEL_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
LORA_CHECKPOINT_PATH="/root/autodl-fs/output/classi_dual_20251019_131545"
OUTPUT_PATH="/root/autodl-fs/merged_models/llama_lora_merged_$(date +%Y%m%d_%H%M%S)"
```

### run_eval_dual_binary.sh

```bash
MERGED_MODEL_PATH="/root/autodl-fs/merged_models/llama_lora_merged_20251019"
MOE_CHECKPOINT_PATH="/root/autodl-fs/output/classi_dual_20251019_131545"
TEST_FILE="/root/autodl-fs/CulturalBench_Hard_merge.json"
```

## ✅ 检查清单

### 训练前
- [ ] 确认训练数据路径正确
- [ ] 确认基础模型路径正确
- [ ] 确认输出目录有足够空间

### 合并前
- [ ] 确认训练已完成
- [ ] 确认 checkpoint 包含 LoRA 权重
  ```bash
  bash check_checkpoint.sh /path/to/checkpoint
  ```
- [ ] 确认有足够磁盘空间（~16GB）

### 评估前
- [ ] 确认合并已完成
- [ ] 确认测试数据路径正确
- [ ] 确认 GPU 可用

## 🔍 验证命令

```bash
# 1. 检查 checkpoint
bash check_checkpoint.sh /root/autodl-fs/output/classi_dual_20251019_131545

# 2. 检查合并后的模型
ls -la /root/autodl-fs/merged_models/llama_lora_merged_20251019/

# 3. 测试工作流程
bash test_merge_workflow.sh
```

## 📊 预期时间

| 步骤 | 时间 | 说明 |
|------|------|------|
| 训练 | 2-4 小时 | 取决于数据量和 GPU |
| 合并 | 5-10 分钟 | 一次性操作 |
| 评估 | 5-10 分钟 | 取决于测试集大小 |

## 🎉 完成！

评估完成后，结果保存在：
```
/root/autodl-fs/eval_results_merged_XXXXXX.json
```

查看结果：
```bash
cat /root/autodl-fs/eval_results_merged_XXXXXX.json | jq '.metrics'
```

## 📚 详细文档

- `FINAL_SOLUTION.md` - 完整解决方案
- `EVALUATION_GUIDE.md` - 评估指南
- `LORA_MERGE_WORKFLOW.md` - 工作流程详解

---

**就这么简单！** 🎊

