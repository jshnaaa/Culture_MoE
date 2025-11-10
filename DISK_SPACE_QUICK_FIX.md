# 磁盘空间不足 - 快速修复

## 🚨 错误信息

```
RuntimeError: [enforce fail at inline_container.cc:778] . PytorchStreamWriter failed writing file data/0: file write failed
RuntimeError: [enforce fail at inline_container.cc:603] . unexpected pos 6272 vs 6166
```

## 🔍 原因

**磁盘空间不足** - PyTorch 无法完整写入模型文件

## ⚡ 快速修复（3 步）

### 步骤 1：检查磁盘空间

```bash
df -h /root/
```

如果 `Avail` 列小于 10GB，需要清理。

### 步骤 2：清理旧文件

```bash
# 使用自动清理脚本
bash cleanup_disk_space.sh

# 或手动清理
rm -rf /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_*_20251108*/
```

### 步骤 3：清理缓存

```bash
rm -rf ~/.cache/huggingface/
pip cache purge
```

## 🚀 重新运行训练

```bash
# 确保有足够的空间
df -h /root/

# 重新运行
bash run_train_culturemoe_from_base_gen.sh qwen 4
```

## 📊 磁盘空间需求

| 项目 | 大小 |
|------|------|
| 最小可用空间 | 20GB |
| 推荐可用空间 | 50GB |
| 每个实验 | ~2GB |

## 💡 如果还是不够

### 方案 A：减少 batch size

在 `run_train_culturemoe_from_base_gen.sh` 中：

```bash
# 修改
--batch_size 2 \
--eval_batch_size 2 \
```

### 方案 B：使用其他磁盘

```bash
# 修改输出目录
OUTPUT_DIR="/autodl-fs/CultureMoE/gen/..."
```

### 方案 C：删除更多旧文件

```bash
# 查看所有实验
ls -lh /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/

# 删除特定实验
rm -rf /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_qwen_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/
```

## ✅ 验证修复

```bash
# 检查可用空间
df -h /root/

# 应该显示至少 20GB 可用空间
```

---

**现在可以继续训练了！** 🚀

