# 磁盘空间不足问题诊断与修复

## 🔍 问题诊断

### 错误信息

```
RuntimeError: [enforce fail at inline_container.cc:778] . PytorchStreamWriter failed writing file data/0: file write failed
RuntimeError: [enforce fail at inline_container.cc:603] . unexpected pos 6272 vs 6166
```

### 根本原因

这个错误表明：
1. **磁盘空间不足** - PyTorch 无法完整写入模型文件
2. **文件系统错误** - 写入过程中出现不一致
3. **权限问题** - 可能无法写入输出目录

---

## ✅ 解决方案

### 方案 1：检查磁盘空间（最重要）

```bash
# 查看磁盘使用情况
df -h

# 查看具体目录的使用情况
du -sh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/
du -sh /root/autodl-fs/

# 查看 inode 使用情况
df -i
```

**预期输出**：
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       100G   85G   15G  85% /
```

如果 `Avail` 列小于 10GB，说明磁盘空间不足。

### 方案 2：清理磁盘空间

#### 2.1 删除旧的训练输出

```bash
# 查看训练输出目录
ls -lh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/

# 删除旧的实验（保留最近的几个）
# ⚠️ 谨慎操作，确保不删除重要的模型

# 删除特定的旧实验
rm -rf /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_qwen_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/

# 或者删除所有旧的 epoch 检查点（保留 best_moe）
find /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/ -name "epoch_*" -type d -exec rm -rf {} \;
```

#### 2.2 清理缓存

```bash
# 清理 PyTorch 缓存
rm -rf ~/.cache/huggingface/

# 清理 pip 缓存
pip cache purge

# 清理系统临时文件
rm -rf /tmp/*
rm -rf /var/tmp/*
```

#### 2.3 压缩或移动旧模型

```bash
# 压缩旧的模型文件
tar -czf /root/autodl-tmp/old_models_backup.tar.gz /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_*_20251108*/

# 删除原始文件
rm -rf /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_*_20251108*/

# 查看释放的空间
df -h
```

### 方案 3：修改训练脚本以减少内存占用

#### 3.1 减少 batch size

在 `run_train_culturemoe_from_base_gen.sh` 中修改：

```bash
# 修改前
--batch_size 4 \
--eval_batch_size 4 \

# 修改后（减少 batch size）
--batch_size 2 \
--eval_batch_size 2 \
```

#### 3.2 减少保存的检查点

在 `train_culturemoe_from_base_gen.py` 中修改：

```python
# 修改前：保存每个 epoch 的检查点
if val_accuracy > best_accuracy:
    best_accuracy = val_accuracy
    # 保存 best_moe

# 修改后：只保存最好的模型，不保存中间检查点
if val_accuracy > best_accuracy:
    best_accuracy = val_accuracy
    # 删除旧的 best_moe
    if os.path.exists(best_moe_dir):
        shutil.rmtree(best_moe_dir)
    # 保存新的 best_moe
```

#### 3.3 使用梯度累积而不是增加 batch size

```bash
# 在训练脚本中添加
--gradient_accumulation_steps 2 \
```

### 方案 4：修改输出目录到更大的磁盘

```bash
# 检查可用的磁盘
df -h /

# 如果 /root 空间不足，使用其他磁盘
# 修改 run_train_culturemoe_from_base_gen.sh 中的 OUTPUT_DIR

# 修改前
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/..."

# 修改后（使用 /autodl-fs 如果空间更大）
OUTPUT_DIR="/autodl-fs/CultureMoE/gen/..."
```

### 方案 5：修复 PyTorch 保存问题

在 `train_culturemoe_from_base_gen.py` 中修改保存逻辑：

```python
# 修改前
torch.save(moe_state_dict, os.path.join(best_moe_dir, "moe_state_dict.pt"))

# 修改后：添加错误处理和重试
import time

def save_with_retry(state_dict, path, max_retries=3):
    """
    带重试的保存函数
    """
    for attempt in range(max_retries):
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # 先保存到临时文件
            temp_path = path + ".tmp"
            torch.save(state_dict, temp_path)

            # 验证文件大小
            if os.path.getsize(temp_path) > 0:
                # 移动到最终位置
                import shutil
                shutil.move(temp_path, path)
                print(f"✅ Successfully saved to {path}")
                return True
            else:
                print(f"⚠️  Temp file is empty, retrying...")
                os.remove(temp_path)
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"❌ Failed to save after {max_retries} attempts")
                return False
    return False

# 使用
save_with_retry(moe_state_dict, os.path.join(best_moe_dir, "moe_state_dict.pt"))
```

---

## 🚀 快速修复步骤

### 步骤 1：检查磁盘空间

```bash
df -h
du -sh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/
```

### 步骤 2：清理旧文件

```bash
# 删除旧的实验（保留最近的 3 个）
cd /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/
ls -lt | head -20  # 查看最近的文件
rm -rf moe_*_20251108*/  # 删除 11 月 8 日的实验
```

### 步骤 3：清理缓存

```bash
rm -rf ~/.cache/huggingface/
pip cache purge
```

### 步骤 4：重新运行训练

```bash
# 确保有足够的空间（至少 10GB）
df -h

# 重新运行训练
bash run_train_culturemoe_from_base_gen.sh qwen 4
```

---

## 📊 磁盘空间需求

### 每个训练实验需要的空间

| 组件 | 大小 |
|------|------|
| Base 模型 | ~15GB |
| LoRA 权重 | ~500MB |
| MOE 权重 | ~1GB |
| 训练输出（每个 epoch） | ~2GB |
| 总计（30 epochs） | ~60GB |

### 推荐的可用空间

- **最小**：20GB
- **推荐**：50GB
- **舒适**：100GB+

---

## 🔧 修改训练脚本以减少磁盘占用

### 修改 1：减少 batch size

在 `run_train_culturemoe_from_base_gen.sh` 中：

```bash
# 修改前
--batch_size 4 \
--eval_batch_size 4 \

# 修改后
--batch_size 2 \
--eval_batch_size 2 \
```

### 修改 2：只保存最好的模型

在 `train_culturemoe_from_base_gen.py` 中，修改保存逻辑：

```python
# 在保存 best_moe 前，删除旧的
if os.path.exists(best_moe_dir):
    import shutil
    shutil.rmtree(best_moe_dir)

# 然后保存新的
os.makedirs(best_moe_dir, exist_ok=True)
torch.save(moe_state_dict, os.path.join(best_moe_dir, "pytorch_model.bin"))
```

### 修改 3：不保存中间检查点

```python
# 只保存最后一个 epoch 的结果
if epoch == num_epochs - 1:
    # 保存最终模型
    pass
```

---

## 💡 常见问题

### Q1：如何知道需要多少磁盘空间？

**A**：
```bash
# 查看当前使用情况
du -sh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/

# 查看可用空间
df -h /root/
```

### Q2：可以中断训练吗？

**A**：可以，但要确保：
1. 按 `Ctrl+C` 优雅地中断
2. 不要强制杀死进程
3. 清理临时文件

### Q3：如何恢复中断的训练？

**A**：
```bash
# 删除不完整的输出
rm -rf /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_*_incomplete/

# 重新运行训练
bash run_train_culturemoe_from_base_gen.sh qwen 4
```

### Q4：可以使用外部存储吗？

**A**：可以，修改输出目录：
```bash
# 在 run_train_culturemoe_from_base_gen.sh 中
OUTPUT_DIR="/mnt/external_drive/CultureMoE/gen/..."
```

---

## ✅ 验证修复

### 检查 1：磁盘空间充足

```bash
df -h /root/
# 应该显示至少 20GB 可用空间
```

### 检查 2：旧文件已清理

```bash
du -sh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/
# 应该显示合理的大小
```

### 检查 3：训练成功完成

```bash
# 查看输出目录
ls -lh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_*_latest/

# 应该包含 best_moe 目录
ls -lh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_*_latest/best_moe/
```

---

## 🎉 总结

### 问题
- PyTorch 无法完整写入模型文件
- 磁盘空间不足

### 解决方案
1. ✅ 检查磁盘空间
2. ✅ 清理旧文件
3. ✅ 清理缓存
4. ✅ 减少 batch size
5. ✅ 修改保存逻辑

### 现在可以继续训练了！ 🚀

