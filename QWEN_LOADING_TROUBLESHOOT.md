# Qwen 模型加载问题排查

## ❌ 问题

运行 `sh run_train_lora_only_gen.sh qwen` 时仍然报错：

```
OSError: tensor parallel is only supported for `torch>=2.5`.
```

## 🔍 可能的原因

### 1. 文件未同步

本地修改的文件可能还没有上传到服务器。

**检查方法**：
```bash
# 在服务器上检查文件
grep -n "attn_implementation" train_lora_only_gen.py

# 应该看到：
# 173:        attn_implementation="eager"  # 使用标准注意力实现
```

### 2. 使用了旧版本的文件

可能有多个版本的文件，运行的不是修改后的版本。

**检查方法**：
```bash
# 查找所有 train_lora_only_gen.py 文件
find . -name "train_lora_only_gen.py"

# 检查每个文件
for f in $(find . -name "train_lora_only_gen.py"); do
    echo "File: $f"
    grep -c "attn_implementation" "$f"
done
```

### 3. Python 缓存问题

Python 可能使用了缓存的 `.pyc` 文件。

**解决方法**：
```bash
# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
```

## ✅ 快速修复

### 方法 1：使用修复脚本

```bash
# 运行修复脚本
sh fix_qwen_loading.sh
```

这会自动检查并修复文件。

### 方法 2：手动修改

```bash
# 1. 打开文件
vim train_lora_only_gen.py

# 2. 找到第 167 行（加载模型的地方）
# 应该看到：
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 3. 修改为：
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # 添加这一行
)

# 4. 保存并退出
:wq
```

### 方法 3：使用 sed 命令

```bash
# 备份原文件
cp train_lora_only_gen.py train_lora_only_gen.py.backup

# 使用 sed 添加参数
sed -i '/trust_remote_code=True$/a\        attn_implementation="eager"  # 禁用 tensor parallel' train_lora_only_gen.py

# 验证修改
grep -A 2 "trust_remote_code" train_lora_only_gen.py
```

## 🔍 验证修复

### 1. 检查文件内容

```bash
# 查看修改后的代码
sed -n '165,175p' train_lora_only_gen.py

# 应该看到：
# 165:    # 加载模型
# 166:    print("Loading base model...")
# 167:    model = AutoModelForCausalLM.from_pretrained(
# 168:        args.model_name_or_path,
# 169:        torch_dtype=torch.float16,
# 170:        device_map="auto",
# 171:        trust_remote_code=True,
# 172:        # 禁用 tensor parallel（需要 torch >= 2.5）
# 173:        attn_implementation="eager"  # 使用标准注意力实现
# 174:    )
# 175:    print("✅ Base model loaded\n")
```

### 2. 测试加载

```bash
# 运行测试脚本
python test_qwen_loading.py

# 应该看到：
# Test 1: Loading without attn_implementation...
#    ❌ Failed (expected): tensor parallel is only supported for `torch>=2.5`.
#
# Test 2: Loading with attn_implementation='eager'...
#    ✅ Success!
```

### 3. 运行训练

```bash
# 重新运行训练
sh run_train_lora_only_gen.sh qwen

# 应该看到：
# Loading base model...
# ✅ Base model loaded
#
# Configuring LoRA...
# trainable params: 4,194,304 || all params: 7,615,616,000 || trainable%: 0.0551
# ✅ LoRA configured
```

## 🐛 调试技巧

### 1. 打印文件路径

在 Python 脚本开头添加：

```python
import os
print(f"Running script: {os.path.abspath(__file__)}")
```

这样可以确认运行的是哪个文件。

### 2. 打印参数

在加载模型前添加：

```python
print(f"Loading model with:")
print(f"  model_path: {args.model_name_or_path}")
print(f"  torch_dtype: {torch.float16}")
print(f"  device_map: auto")
print(f"  trust_remote_code: True")
print(f"  attn_implementation: eager")
```

### 3. 使用 Python 调试器

```bash
# 使用 pdb 调试
python -m pdb train_lora_only_gen.py \
    --model_name_or_path /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct \
    --data_path /root/autodl-fs/cultureLLM_merge_gen.json \
    --output_dir /tmp/test_output

# 在 pdb 中：
# (Pdb) b 167  # 在第 167 行设置断点
# (Pdb) c      # 继续执行
# (Pdb) p args.model_name_or_path  # 打印变量
```

## 📝 完整的修复后代码

```python
# train_lora_only_gen.py (第 165-175 行)

# 加载模型
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    # 禁用 tensor parallel（需要 torch >= 2.5）
    attn_implementation="eager"  # 使用标准注意力实现
)
print("✅ Base model loaded\n")
```

## ⚠️ 其他需要修改的文件

如果你有其他脚本也加载 Qwen 模型，也需要添加 `attn_implementation="eager"`：

### 1. eval_base_gen.py

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    device_map=device_map,
    trust_remote_code=True,
    attn_implementation="eager"  # 添加这一行
)
```

### 2. eval_base_llama.py

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    device_map=None,
    trust_remote_code=True,
    attn_implementation="eager"  # 添加这一行
)
```

### 3. load_culturemoe.py

```python
llama_model = AutoModelForCausalLM.from_pretrained(
    merged_llm_path,
    torch_dtype=torch.float16,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
    attn_implementation="eager"  # 添加这一行
)
```

## 🎉 总结

### 问题
- Qwen 模型尝试使用 tensor parallel
- 需要 PyTorch >= 2.5
- 当前版本不支持

### 修复
```python
# 添加这一行
attn_implementation="eager"
```

### 验证
```bash
# 1. 检查文件
grep "attn_implementation" train_lora_only_gen.py

# 2. 测试加载
python test_qwen_loading.py

# 3. 运行训练
sh run_train_lora_only_gen.sh qwen
```

### 如果仍然失败

1. 确认文件已修改：`cat train_lora_only_gen.py | grep -A 5 "Loading base model"`
2. 清理缓存：`find . -name "*.pyc" -delete`
3. 检查 PyTorch 版本：`python -c "import torch; print(torch.__version__)"`
4. 运行测试脚本：`python test_qwen_loading.py`

如果测试脚本成功但训练失败，说明问题在其他地方，请提供完整的错误堆栈。

