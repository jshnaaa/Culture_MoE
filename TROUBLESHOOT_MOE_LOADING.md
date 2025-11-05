# MoE 模型加载故障排查

## 🔍 问题

运行 `sh run_eval_vsm13.sh moe llama 2` 时，输出停在：

```
Loading model...
```

没有进一步的输出，可能的原因：

1. **模型加载时间长**：LLM 模型很大（7-8GB），加载需要时间
2. **缺少必要文件**：MoE 权重或配置文件缺失
3. **代码错误**：加载过程中出现异常但没有打印
4. **内存不足**：GPU 内存不够加载模型

## 🛠️ 诊断步骤

### 步骤 1：检查文件是否存在

```bash
sh check_moe_files.sh
```

这会检查：
- Merged LLM 目录
- MoE 权重目录
- `moe_config.json`
- `moe_weights.bin`

**预期输出**：
```
1. Checking Merged LLM: /root/autodl-tmp/.../llama_merge_2
   ✅ Directory exists
   Files: (列出模型文件)

2. Checking MoE Weights: /root/autodl-tmp/.../model_moe_llama_2
   ✅ Directory exists
   Files:
      - moe_config.json
      - moe_weights.bin
      - tokenizer files...

   ✅ moe_config.json exists
   Content: (显示配置)

   ✅ moe_weights.bin exists
```

### 步骤 2：测试 MoE 加载

```bash
python test_moe_loading.py
```

这会逐步测试：
1. 检查目录
2. 加载配置
3. 检查 merged LLM
4. 加载 tokenizer
5. **加载 LLM（这一步最慢）**
6. 创建 MoE 架构
7. 加载 MoE 权重
8. 移动到设备
9. 设置评估模式

**预期输出**：
```
1. Checking directory...
   ✅ Directory exists

2. Loading MoE config...
   ✅ Config loaded

3. Checking merged LLM...
   ✅ Merged LLM exists

4. Loading tokenizer...
   ✅ Tokenizer loaded

5. Loading merged LLM (this may take a while)...
   ✅ Merged LLM loaded and frozen  ← 这一步可能需要 1-2 分钟

6. Creating MoE architecture...
   ✅ MoE architecture created

7. Loading MoE weights...
   ✅ Weights loaded into model

8. Moving MoE layers to device...
   ✅ MoE layers moved to cuda

9. Setting evaluation mode...
   ✅ Model in evaluation mode

✅ MoE Model Loading Test Successful!
```

### 步骤 3：检查 GPU 内存

```bash
# 查看 GPU 使用情况
nvidia-smi

# 或持续监控
watch -n 1 nvidia-smi
```

**检查**：
- GPU 内存是否足够（至少需要 16GB）
- 是否有其他进程占用 GPU

### 步骤 4：查看详细日志

如果 `test_moe_loading.py` 在某一步卡住，说明问题在那一步。

**常见卡住的地方**：

#### 卡在步骤 5（加载 LLM）

```
5. Loading merged LLM (this may take a while)...
(卡住，没有进一步输出)
```

**原因**：
- 模型文件很大（7-8GB），加载需要时间
- 可能需要 1-2 分钟

**解决方法**：
- 耐心等待
- 检查 GPU 内存是否足够

#### 卡在步骤 7（加载权重）

```
7. Loading MoE weights...
(卡住)
```

**原因**：
- `moe_weights.bin` 文件损坏
- 文件格式不对

**解决方法**：
```bash
# 检查文件大小
ls -lh /root/autodl-tmp/.../model_moe_llama_2/moe_weights.bin

# 应该是几百 MB
```

## 🔧 常见问题

### 问题 1：moe_config.json 不存在

```
❌ Config not found: .../moe_config.json
```

**解决方法**：
- 重新训练 MoE 模型
- 确保训练时保存了配置

### 问题 2：merged_llm_path 不存在

```
❌ Merged LLM not found: .../llama_merge_2
```

**解决方法**：
```bash
# 检查配置中的路径
cat /root/autodl-tmp/.../model_moe_llama_2/moe_config.json | jq '.merged_llm_path'

# 如果路径错误，需要修改配置或重新训练
```

### 问题 3：moe_weights.bin 不存在

```
❌ Weights file not found: .../moe_weights.bin
```

**解决方法**：
- 检查是否有 `moe_weights.pt` 文件（旧版本）
- 如果有，重命名为 `moe_weights.bin`
- 或重新训练 MoE 模型

### 问题 4：GPU 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方法**：
```bash
# 清理 GPU 内存
nvidia-smi

# 杀死占用 GPU 的进程
kill -9 <PID>

# 或重启
```

### 问题 5：加载时间过长

如果加载 LLM 时卡住超过 5 分钟：

**检查**：
```bash
# 查看进程是否还在运行
ps aux | grep python

# 查看 GPU 使用情况
nvidia-smi

# 查看磁盘 I/O
iostat -x 1
```

**可能原因**：
- 磁盘 I/O 慢
- 模型文件损坏
- 内存不足，正在 swap

## 📝 调试技巧

### 1. 添加更多打印

在 `eval_vsm13.py` 中添加更多打印语句：

```python
print("Step 1: Loading config...")
# 代码
print("Step 1 done")

print("Step 2: Loading LLM...")
# 代码
print("Step 2 done")
```

### 2. 使用 Python 调试器

```bash
python -m pdb eval_vsm13.py --model moe --backbone llama --num_classes 2
```

### 3. 检查日志文件

```bash
# 重定向输出到文件
python eval_vsm13.py --model moe --backbone llama --num_classes 2 > moe_loading.log 2>&1

# 查看日志
tail -f moe_loading.log
```

### 4. 使用 strace 跟踪

```bash
# 跟踪系统调用
strace -o moe_loading.strace python eval_vsm13.py --model moe --backbone llama --num_classes 2

# 查看最后的系统调用
tail -100 moe_loading.strace
```

## ✅ 解决方案总结

### 如果是加载时间长

- **正常现象**：LLM 模型很大，加载需要 1-2 分钟
- **解决方法**：耐心等待

### 如果是文件缺失

- **检查文件**：运行 `sh check_moe_files.sh`
- **重新训练**：如果文件缺失，重新训练 MoE 模型

### 如果是内存不足

- **清理 GPU**：杀死其他进程
- **使用更小的模型**：或使用 CPU

### 如果是代码错误

- **运行测试**：`python test_moe_loading.py`
- **查看错误**：检查详细的错误信息

## 🚀 快速诊断命令

```bash
# 1. 检查文件
sh check_moe_files.sh

# 2. 测试加载
python test_moe_loading.py

# 3. 如果测试成功，运行完整评估
python eval_vsm13.py --model moe --backbone llama --num_classes 2

# 4. 查看 GPU 使用
nvidia-smi
```

## 💡 提示

如果 `test_moe_loading.py` 成功，但 `eval_vsm13.py` 失败，说明问题在 VSM13 评估逻辑中，而不是模型加载。

如果 `test_moe_loading.py` 在步骤 5 卡住超过 5 分钟，可能是：
- 磁盘 I/O 慢
- 模型文件损坏
- 需要检查系统资源

