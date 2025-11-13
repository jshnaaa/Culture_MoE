# 缩进错误修复指南

## 🔍 问题

`src/llamafactory/model/CultureMoE.py` 文件中存在缩进错误：

```
IndentationError: unexpected indent at line 286
```

---

## ✅ 快速修复方法

### 方法 1：使用原始文件（推荐）

如果你有备份的原始 `CultureMoE.py` 文件，建议恢复原始文件，然后重新应用修改。

### 方法 2：手动修复缩进

打开 `src/llamafactory/model/CultureMoE.py`，检查以下位置的缩进：

#### 位置 1：Router 调用（约 273-283 行）

**正确的缩进**：
```python
        # Step 4: Router（基于 pooled representation）
        # ✅ 如果使用共享专家，基于 shared_out 计算路由；否则基于 h_all 计算
        if use_shared_experts:
            pooled = shared_out.mean(dim=1)  # [B, H]
        else:
            pooled = h_all.mean(dim=1)  # [B, H]

        # ✅ 使用温度参数调用 router（防止塌陷）
        expert_weights, router_logits = self.router(pooled, temperature=router_temperature)  # [B, E]

        # ✅ 保存专家权重
        self._last_expert_weights = expert_weights.detach()
```

**关键点**：
- 所有代码都应该有 **8 个空格**的缩进（2 个 tab）
- `if use_shared_experts:` 和 `expert_weights, router_logits = ...` 应该在同一缩进级别

---

#### 位置 2：else 分支（约 412-428 行）

**正确的缩进**：
```python
                # ✅ 总损失：生成损失 + 文化损失 + 负载均衡损失 + 熵损失
                total_loss = (generation_loss +
                             culture_loss_lambda * culture_loss +
                             load_balance_weight * load_balance_loss +
                             entropy_weight * entropy_loss)
            else:
                # ✅ 确保 culture_loss 与 generation_loss 在同一设备上
                outputs['culture_loss'] = torch.tensor(0.0, device=generation_loss.device)
                outputs['specialization_loss'] = torch.tensor(0.0, device=generation_loss.device)
                outputs['diversity_loss'] = torch.tensor(0.0, device=generation_loss.device)

                # ✅ 计算防塌陷损失
                load_balance_loss = self.router.compute_load_balancing_loss(router_logits)
                entropy_loss = self.router.entropy_regularization(expert_weights)

                outputs['load_balance_loss'] = load_balance_loss
                outputs['entropy_loss'] = entropy_loss

                # ✅ 总损失：只有生成损失 + 防塌陷损失
                total_loss = (generation_loss +
                             load_balance_weight * load_balance_loss +
                             entropy_weight * entropy_loss)

            outputs['loss'] = total_loss
```

**关键点**：
- `else:` 应该与上面的 `if use_culture_loss and culture_labels is not None:` 对齐（12 个空格）
- else 分支内的代码应该有 **16 个空格**的缩进（4 个 tab）
- `outputs['loss'] = total_loss` 应该有 **12 个空格**的缩进（3 个 tab）

---

### 方法 3：使用 Python 自动格式化工具

```bash
# 安装 autopep8
pip install autopep8

# 自动修复缩进
autopep8 --in-place --aggressive --aggressive src/llamafactory/model/CultureMoE.py
```

或者使用 `black`：

```bash
# 安装 black
pip install black

# 自动格式化
black src/llamafactory/model/CultureMoE.py
```

---

## 🔧 验证修复

修复后，运行以下命令验证：

```bash
python -m py_compile src/llamafactory/model/CultureMoE.py
```

如果没有输出，说明修复成功。

---

## 📋 完整的 forward 函数签名

确保 `forward` 函数的签名是：

```python
def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None, attention_mask_mask=None,
            labels=None, culture_labels=None, use_culture_loss=False, culture_loss_lambda=0.5,
            culture_loss_alpha=2.0, culture_loss_beta=1.0,
            use_shared_experts=True, router_temperature=2.0, load_balance_weight=0.01, entropy_weight=0.1, **kwargs):
```

---

## 💡 常见缩进错误

### 错误 1：混用 Tab 和空格

```python
# ❌ 错误：混用 tab 和空格
        if use_shared_experts:
	    pooled = shared_out.mean(dim=1)  # 这里用了 tab
```

**解决方案**：统一使用空格（推荐 4 个空格 = 1 个缩进级别）

### 错误 2：缩进不一致

```python
# ❌ 错误：缩进不一致
        if use_shared_experts:
            pooled = shared_out.mean(dim=1)
        else:
           pooled = h_all.mean(dim=1)  # 少了一个空格
```

**解决方案**：确保同一级别的代码有相同的缩进

### 错误 3：多余的缩进

```python
# ❌ 错误：多余的缩进
        expert_weights, router_logits = self.router(pooled)
         self._last_expert_weights = expert_weights.detach()  # 多了一个空格
```

**解决方案**：删除多余的空格

---

## ✅ 验证清单

- [ ] Router 调用部分的缩进正确（273-283 行）
- [ ] else 分支的缩进正确（412-428 行）
- [ ] 没有混用 tab 和空格
- [ ] 所有同级代码的缩进一致
- [ ] 运行 `python -m py_compile` 没有错误

---

## 🚀 修复后测试

修复缩进后，运行训练脚本测试：

```bash
bash run_ft_culturemoe_gen.sh qwen 3
```

如果没有 `IndentationError`，说明修复成功！

---

**祝修复顺利！** 🎉

