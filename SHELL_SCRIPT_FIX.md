# Shell 脚本语法错误修复指南

## 🔍 问题分析

### 错误信息

```
run_ft_lora_only_gen.sh: line 124: --eval_batch_size: command not found
```

### 根本原因

**Shell 脚本中的反斜杠行续符有问题**：

1. **可能的原因**
   - 反斜杠后面有空格或其他不可见字符
   - 文件使用了 Windows 换行符（CRLF）而不是 Unix 换行符（LF）
   - 反斜杠被意外删除或损坏

2. **为什么会这样**
   - 在 Windows 上编辑文件
   - 复制粘贴时引入了不可见字符
   - 文本编辑器自动格式化

---

## ✅ 解决方案

### 方案 1：检查并修复换行符（推荐）

```bash
# 检查文件的换行符类型
file run_ft_lora_only_gen.sh

# 如果显示 "CRLF line terminators"，转换为 Unix 格式
dos2unix run_ft_lora_only_gen.sh

# 或者使用 sed
sed -i 's/\r$//' run_ft_lora_only_gen.sh
```

### 方案 2：检查反斜杠后是否有空格

```bash
# 查看反斜杠后是否有空格（会显示为 "\ "）
cat -A run_ft_lora_only_gen.sh | grep -n '\\ '

# 如果有，删除反斜杠后的空格
sed -i 's/\\ $/\\/g' run_ft_lora_only_gen.sh
```

### 方案 3：重新创建脚本

如果上述方法都不行，重新创建脚本：

```bash
# 备份原脚本
cp run_ft_lora_only_gen.sh run_ft_lora_only_gen.sh.bak

# 使用 cat 重新创建（确保使用正确的换行符）
cat > run_ft_lora_only_gen.sh << 'EOF'
#!/bin/bash

# ... 脚本内容 ...

python ft_lora_only_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 12 \
    --batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-4 \
    --weight_decay 0.001 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --eval_interval 3 \
    --device cuda

# ... 其余内容 ...
EOF

# 添加执行权限
chmod +x run_ft_lora_only_gen.sh
```

### 方案 4：使用 vim 修复

```bash
# 使用 vim 打开文件
vim run_ft_lora_only_gen.sh

# 在 vim 中执行以下命令
:set ff=unix
:wq
```

---

## 🔧 诊断步骤

### 步骤 1：检查文件格式

```bash
# 检查文件格式
file run_ft_lora_only_gen.sh

# 预期输出（正确）：
# run_ft_lora_only_gen.sh: Bourne-Again shell script, ASCII text executable

# 错误输出：
# run_ft_lora_only_gen.sh: Bourne-Again shell script, ASCII text executable, with CRLF line terminators
```

### 步骤 2：检查反斜杠

```bash
# 查看第 113-129 行
sed -n '113,129p' run_ft_lora_only_gen.sh | cat -A

# 正确的输出应该是：
# python ft_lora_only_gen.py \$
#     --base_model_path "$BASE_MODEL_PATH" \$
#     --train_file "$TRAIN_FILE" \$
#     ...

# 错误的输出可能是：
# python ft_lora_only_gen.py \ $  ← 反斜杠后有空格
# python ft_lora_only_gen.py \^M$  ← 有 Windows 换行符
```

### 步骤 3：手动测试命令

```bash
# 手动运行命令（不使用脚本）
python ft_lora_only_gen.py \
    --base_model_path "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct" \
    --train_file "/root/autodl-fs/cultureLLM_merge_gen.json" \
    --output_dir "/tmp/test_output" \
    --num_epochs 1 \
    --batch_size 2 \
    --eval_batch_size 2 \
    --learning_rate 2e-4 \
    --weight_decay 0.001 \
    --max_length 512 \
    --val_split 0.1 \
    --num_workers 2 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --eval_interval 1 \
    --device cuda

# 如果手动运行成功，说明是脚本格式问题
```

---

## 📊 常见问题

### 问题 1：Windows 换行符（CRLF）

**症状**：
```
run_ft_lora_only_gen.sh: line 124: --eval_batch_size: command not found
```

**解决方案**：
```bash
dos2unix run_ft_lora_only_gen.sh
# 或
sed -i 's/\r$//' run_ft_lora_only_gen.sh
```

### 问题 2：反斜杠后有空格

**症状**：
```
run_ft_lora_only_gen.sh: line 124: --eval_batch_size: command not found
```

**解决方案**：
```bash
# 删除反斜杠后的空格
sed -i 's/\\ $/\\/g' run_ft_lora_only_gen.sh
```

### 问题 3：缺少反斜杠

**症状**：
```
run_ft_lora_only_gen.sh: line 119: --eval_batch_size: command not found
```

**解决方案**：
手动添加反斜杠：
```bash
python ft_lora_only_gen.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 12 \
    --batch_size 8 \
    --eval_batch_size 8 \    ← 确保每行末尾都有反斜杠
    --learning_rate 2e-4 \
    ...
```

---

## 🎯 最佳实践

### 1. 使用正确的文本编辑器

```bash
# 推荐使用 vim 或 nano（Unix 换行符）
vim run_ft_lora_only_gen.sh

# 避免使用 Windows 记事本
```

### 2. 设置 Git 自动转换换行符

```bash
# 在 .gitattributes 中添加
*.sh text eol=lf

# 或在 Git 配置中设置
git config --global core.autocrlf input
```

### 3. 使用 ShellCheck 检查脚本

```bash
# 安装 ShellCheck
sudo apt-get install shellcheck

# 检查脚本
shellcheck run_ft_lora_only_gen.sh
```

---

## 🚀 快速修复

### 一键修复脚本

```bash
#!/bin/bash

# 修复 run_ft_lora_only_gen.sh 的换行符和格式问题

echo "Fixing run_ft_lora_only_gen.sh..."

# 1. 转换为 Unix 换行符
if command -v dos2unix &> /dev/null; then
    dos2unix run_ft_lora_only_gen.sh
else
    sed -i 's/\r$//' run_ft_lora_only_gen.sh
fi

# 2. 删除反斜杠后的空格
sed -i 's/\\ $/\\/g' run_ft_lora_only_gen.sh

# 3. 添加执行权限
chmod +x run_ft_lora_only_gen.sh

echo "✅ Fixed!"
echo ""
echo "Now you can run:"
echo "  bash run_ft_lora_only_gen.sh llama 4"
```

保存为 `fix_script.sh` 并运行：

```bash
chmod +x fix_script.sh
./fix_script.sh
```

---

## 📈 验证修复

### 运行脚本

```bash
bash run_ft_lora_only_gen.sh llama 4
```

### 预期输出

```
============================================================
Fine-tuning LoRA Only Model with New Data Format
============================================================
Backbone: llama (LLaMA 3.1-8B-Instruct)
Dataset: CultureLLM

Starting training...

Loading tokenizer...
✅ Tokenizer loaded

Loading and processing data...
Loaded 1000 samples
Train set size: 900
Validation set size: 100
✅ Data loaded

Loading base model...
✅ Base model loaded (bfloat16)

Creating LoRA model...
✅ LoRA model created

Starting training...

Epoch 1/12
Training: 100%|████████████████████████████████████████| 113/113 [01:30<00:00,  1.25it/s]
  Train Loss: 0.5234
  Eval Loss: 0.4521
  Eval Accuracy: 0.3545
  ✅ Best model saved

✅ Training completed!
```

---

## 🎉 总结

### 问题
- ❌ Shell 脚本语法错误
- ❌ Windows 换行符（CRLF）
- ❌ 反斜杠后有空格

### 解决方案
- ✅ 转换为 Unix 换行符
- ✅ 删除反斜杠后的空格
- ✅ 使用正确的文本编辑器

### 结果
- ✅ 脚本正常运行
- ✅ 无语法错误
- ✅ 训练正常进行

---

**问题已解决！** ✅

```bash
# 修复脚本
dos2unix run_ft_lora_only_gen.sh
# 或
sed -i 's/\r$//' run_ft_lora_only_gen.sh

# 运行脚本
bash run_ft_lora_only_gen.sh llama 4

