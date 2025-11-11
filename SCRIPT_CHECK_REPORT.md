# run_ft_culturemoe_gen.sh 脚本检查报告

## ✅ 检查结果：脚本整体良好，有几个小建议

---

## 📋 检查清单

### 1. Shell 脚本语法 ✅

- ✅ Shebang 正确：`#!/bin/bash`
- ✅ 所有反斜杠行续符正确
- ✅ 变量引用正确使用引号
- ✅ 条件判断语法正确
- ✅ Case 语句格式正确

### 2. 参数处理 ✅

- ✅ 参数默认值设置正确
- ✅ 参数验证完整
- ✅ 错误提示清晰

### 3. 路径检查 ✅

- ✅ Base 模型路径检查
- ✅ 训练文件检查
- ✅ LoRA 权重检查（使用通配符）
- ✅ 输出目录创建

### 4. Python 调用 ✅

- ✅ 所有参数正确传递
- ✅ 参数值正确引用
- ✅ 学习率设置合理（1e-4）

### 5. Python 文件 ✅

- ✅ 导入语句完整
- ✅ CultureMoE 模型正确导入
- ✅ 数据加载逻辑正确
- ✅ 训练循环完整
- ✅ 损失计算正确
- ✅ 设备管理正确
- ✅ 学习率调整合理

---

## ⚠️ 潜在问题和建议

### 问题 1：学习率可能仍然偏小

**当前设置**：
```python
# 在 ft_culturemoe_from_base_gen.py 第 782 行
learning_rate = args.learning_rate * 0.1  # 1e-4 * 0.1 = 1e-5
```

**问题**：
- Shell 脚本传入 `--learning_rate 1e-4`
- Python 代码又乘以 0.1，变成 1e-5
- 这可能仍然偏小

**建议**：
```python
# 选项 1：直接使用传入的学习率
learning_rate = args.learning_rate  # 1e-4

# 或选项 2：在 shell 脚本中调整
--learning_rate 2e-4  # 传入更大的值
```

---

### 问题 2：LoRA 权重路径使用硬编码时间戳

**当前设置**：
```bash
# 第 76-78 行
if [ "$BACKBONE" = "qwen" ]; then
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_20251110_2137/best_lora"
else
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_20251110_2135/best_lora"
fi
```

**问题**：
- 硬编码的时间戳可能不存在
- 如果重新训练 LoRA，路径会变化

**建议**：
```bash
# 使用通配符（已在第 37-41 行使用，但 DATA_ID=4 时被覆盖）
if [ "$BACKBONE" = "qwen" ]; then
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_*/best_lora"
else
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_*/best_lora"
fi
```

---

### 问题 3：数据文件名不一致

**当前设置**：
```bash
# 第 73 行
TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen_small.json"
```

**问题**：
- DATA_ID=2 和 3 使用完整数据集
- DATA_ID=4 使用 `_small.json`（小数据集）
- 可能导致训练结果不一致

**建议**：
```bash
# 统一使用完整数据集
TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"

# 或者明确说明使用小数据集
echo "Using CultureLLM dataset (small version for testing)"
```

---

### 问题 4：缺少梯度累积步数参数

**当前设置**：
```bash
# 第 158-181 行，没有传递 gradient_accumulation_steps
python ft_culturemoe_from_base_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    ...
    --device cuda
```

**问题**：
- Python 代码默认 `gradient_accumulation_steps=1`
- 对于小 batch size（4），可能需要梯度累积

**建议**：
```bash
# 添加梯度累积参数
python ft_culturemoe_from_base_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    ...
    --gradient_accumulation_steps 2 \  # 添加这行
    --device cuda
```

---

## 🔧 建议的修改

### 修改 1：调整学习率（推荐）

**在 ft_culturemoe_from_base_gen.py 第 782 行**：

```python
# 修改前
learning_rate = args.learning_rate * 0.1  # 1e-4 * 0.1 = 1e-5

# 修改后（选项 1：直接使用传入值）
learning_rate = args.learning_rate  # 1e-4

# 或修改后（选项 2：使用更大的值）
learning_rate = args.learning_rate  # 并在 shell 中传入 2e-4
```

### 修改 2：修复 LoRA 路径（推荐）

**在 run_ft_culturemoe_gen.sh 第 76-78 行**：

```bash
# 修改前
if [ "$BACKBONE" = "qwen" ]; then
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_20251110_2137/best_lora"
else
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_20251110_2135/best_lora"
fi

# 修改后
if [ "$BACKBONE" = "qwen" ]; then
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_*/best_lora"
else
    LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_*/best_lora"
fi
```

### 修改 3：统一数据文件（可选）

**在 run_ft_culturemoe_gen.sh 第 73 行**：

```bash
# 修改前
TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen_small.json"

# 修改后（使用完整数据集）
TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen.json"
```

### 修改 4：添加梯度累积（可选）

**在 run_ft_culturemoe_gen.sh 第 181 行后添加**：

```bash
python ft_culturemoe_from_base_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --lora_weights_path "$LORA_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --use_culture_loss "$USE_CULTURE_LOSS" \
    --culture_loss_lambda "$CULTURE_LOSS_WEIGHT" \
    --num_epochs 30 \
    --num_experts "$NUM_EXPERTS" \
    --shared_hidden_dim 4096 \
    --router_hidden_dim 2048 \
    --experts_hidden_dim 4096 \
    --moe_lora_rank 32 \
    --classification_hidden_dim 512 \
    --dropout 0.05 \
    --num_heads 8 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.001 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2 \
    --gradient_accumulation_steps 2 \  # 添加这行
    --device cuda
```

---

## 📊 检查总结

### 优点 ✅

1. **脚本结构清晰**
   - 参数处理完善
   - 错误检查充分
   - 输出信息详细

2. **Python 代码质量高**
   - 导入完整
   - 逻辑清晰
   - 错误处理完善

3. **已修复的问题**
   - ✅ 设备不匹配（已修复）
   - ✅ Logits 裁剪（已添加）
   - ✅ 学习率调整（已改进）

### 需要注意的问题 ⚠️

1. **学习率可能仍然偏小**
   - 当前：1e-5（1e-4 * 0.1）
   - 建议：1e-4 或 2e-4

2. **LoRA 路径硬编码**
   - 当前：使用固定时间戳
   - 建议：使用通配符

3. **数据文件不一致**
   - 当前：DATA_ID=4 使用 small 版本
   - 建议：统一使用完整版本

4. **缺少梯度累积**
   - 当前：默认为 1
   - 建议：设置为 2

---

## 🚀 运行建议

### 选项 1：直接运行（当前配置）

```bash
bash run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5
```

**预期**：
- 学习率：1e-5（可能偏小）
- 可能需要更多 epoch 才能收敛

### 选项 2：修改后运行（推荐）

```bash
# 1. 修改 ft_culturemoe_from_base_gen.py 第 782 行
learning_rate = args.learning_rate  # 不再乘以 0.1

# 2. 修改 run_ft_culturemoe_gen.sh 第 76-78 行
# 使用通配符而不是硬编码时间戳

# 3. 运行
bash run_ft_culturemoe_gen.sh llama 4 True 6 2 0.5
```

**预期**：
- 学习率：1e-4（合理）
- 更快收敛
- 更好的性能

---

## 🎉 总结

### 整体评价：✅ 良好

- ✅ 脚本语法正确
- ✅ 参数处理完善
- ✅ 错误检查充分
- ✅ Python 代码质量高
- ⚠️ 有几个小问题需要注意

### 建议优先级

1. **高优先级**：修改学习率（避免训练太慢）
2. **中优先级**：修复 LoRA 路径（避免路径不存在）
3. **低优先级**：统一数据文件、添加梯度累积

### 预期结果

修改后，训练应该能够：
- ✅ 正常启动
- ✅ Loss 正常下降
- ✅ Culture loss 正常计算
- ✅ 模型正常收敛

---

**脚本检查完成！可以运行，但建议先修改学习率** ✅

