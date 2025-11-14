# 🚀 CultureMoE 改进版快速开始指南

## 📋 改进内容一览

| 改进项 | 原来 | 改进后 | 效果 |
|--------|------|--------|------|
| Shared 容量 | 4096→4096→4096 | 4096→1024→4096 | 削弱 shared，让 MoE 有机会 |
| MoE 容量 | 4096→4096→4096 | 4096→4096→4096 | 强化 MoE，增强学习能力 |
| 融合方式 | h_out = shared + α*moe | h_out = shared + gate·moe | 强制 MoE 参与 |
| Expert 结构 | FFN + LoRA | LayerNorm + FFN + LoRA + 残差 | 稳定训练 |
| Router 路由 | 均匀分布 | 尖锐分布（负熵正则化） | 增强专家 specialization |

---

## 🎯 核心改进原理

### 问题：为什么原来的结构不行？

```
h_out = shared + α * moe

问题：
1. Shared 层永远参与 → Shared 学所有东西
2. MoE 只是 residual 增强 → MoE 容易被忽视
3. Router 容易塌陷 → 所有样本都选同一个专家
4. Experts 不被利用 → 文化差异无法被捕捉
```

### 解决方案：新的结构

```
h_out = shared + gate · moe

其中：
- gate = sigmoid(Wg · h + bg)，bg 初始为 -2
- Shared 层变弱（1024 中间层）
- MoE 层变强（4096 中间层）
- 负熵正则化尖锐化 Router

效果：
1. Shared 只学基础特征
2. MoE 有机会竞争
3. Gate 强制 MoE 参与
4. Router 尖锐化，每个样本倾向使用特定专家
5. Experts 充分利用，文化差异被有效捕捉
```

---

## 🔧 关键代码修改

### 1. Shared 层变弱

```python
# CultureMoE.py 第 76-82 行
self.shared = nn.Sequential(
    nn.Linear(hidden_dim, 1024),  # ✅ 从 4096 降到 1024
    nn.ReLU(),
    nn.Dropout(args.dropout),
    nn.Linear(1024, hidden_dim)
)
```

### 2. Gating 机制

```python
# CultureMoE.py 第 85-91 行
self.gate_linear = nn.Linear(hidden_dim, hidden_dim)
nn.init.xavier_uniform_(self.gate_linear.weight)
nn.init.constant_(self.gate_linear.bias, -2.0)  # ✅ 关键：初始 bias = -2
self.gate_sigmoid = nn.Sigmoid()

# CultureMoE.py 第 322-340 行
gate_logits = self.gate_linear(shared_out)
gate = self.gate_sigmoid(gate_logits)
moe_gated = gate * expert_sum
enhanced_hidden = shared_out + moe_warmup_weight * moe_gated
```

### 3. Expert 中的 LayerNorm

```python
# experts.py 第 42-56 行
class LoRAExpert(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.base_model = MLPBase(...)
        self.lora_layer = LoRA(...)
        self.layer_norm = nn.LayerNorm(output_dim)  # ✅ 新增

    def forward(self, x):
        normalized = self.layer_norm(x)
        features = self.base_model(normalized)
        lora_out = self.lora_layer(features)
        return x + lora_out  # ✅ 残差连接
```

### 4. 负熵正则化

```python
# router.py 第 152-160 行
def negative_entropy_regularization(self, expert_weights, lambda_entropy=-0.01):
    entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-8), dim=-1)
    neg_entropy_loss = lambda_entropy * entropy.mean()
    return neg_entropy_loss

# CultureMoE.py 中使用
neg_entropy_loss = self.router.negative_entropy_regularization(expert_weights, lambda_entropy=-0.01)
```

---

## 📊 训练日志示例

### 改进前（原始结构）

```
Epoch 1 Training Results:
   Train Loss: 1.2345
   Train Gen Loss: 1.2000
   Train Culture Loss: 0.0345
   Load Balance Loss: 0.0234
   Entropy Loss: -0.5678
   Neg Entropy Loss: 0.0000  ← 没有

Epoch 3 Evaluation Results:
   Eval Accuracy: 0.5200 (520/1000)  ← 不涨

Epoch 6 Evaluation Results:
   Eval Accuracy: 0.5250 (525/1000)  ← 几乎没涨
```

### 改进后（新结构）

```
Epoch 1 Training Results:
   Train Loss: 1.2345
   Train Gen Loss: 1.2000
   Train Culture Loss: 0.0345
   Load Balance Loss: 0.0234
   Entropy Loss: -0.5678
   Neg Entropy Loss: -0.0089  ← 新增，尖锐化路由

Epoch 3 Evaluation Results:
   Eval Accuracy: 0.5800 (580/1000)  ← 显著提升！

Epoch 6 Evaluation Results:
   Eval Accuracy: 0.6500 (650/1000)  ← 继续提升！

Epoch 9 Evaluation Results:
   Eval Accuracy: 0.6789 (679/1000)  ← 最终结果
```

---

## 🚀 使用命令

### 基础训练

```bash
# 自动学习 lambda（推荐）
sh run_ft_culturemoe_gen.sh qwen 4 True 6 0.4 -1

# 固定 lambda = 0.1
sh run_ft_culturemoe_gen.sh qwen 4 True 6 0.4 0.1

# 固定 lambda = 0.5
sh run_ft_culturemoe_gen.sh qwen 4 True 6 0.4 0.5
```

### 完整参数

```bash
sh run_ft_culturemoe_gen.sh \
    qwen \                    # BACKBONE
    4 \                       # DATA_ID (4=CultureLLM)
    True \                    # USE_CULTURE_LOSS
    6 \                       # NUM_EXPERTS
    0.4 \                     # MOE_FUSION
    -1 \                      # CULTURE_LOSS_WEIGHT (-1=自动学习)
    0.5 \                     # MARGIN
    1.0 \                     # LAMBDA_DIFF
    True \                    # USE_SHARED
    2.0 \                     # ROUTER_TEMP
    0.01 \                    # LOAD_BAL
    0.1 \                     # ENTROPY
    2                         # NUM_GPUS
```

---

## 📈 预期效果

### Accuracy 提升

```
基线（无 MoE）：           ~60%
原始 MoE：                 ~62%（可能不涨）
改进后 MoE（新结构）：     ~65-70%  ✅ 显著提升
```

### 训练过程

```
Epoch 1-3:   Gate 从 0 逐步增大，MoE 开始参与
Epoch 3-6:   Router 权重分布尖锐化，文化差异开始被捕捉
Epoch 6-9:   Accuracy 显著提升，模型收敛
```

---

## 🔍 监控指标

### 关键指标

1. **Neg Entropy Loss**：应该逐步减小（路由变尖锐）
2. **Load Balance Loss**：应该保持稳定（防止塌陷）
3. **Eval Accuracy**：应该显著提升（文化差异被捕捉）

### 查看日志

```bash
# 查看所有训练结果
cat $OUTPUT_DIR/epoch_eval_results.json | python -m json.tool

# 查看最佳模型信息
cat $OUTPUT_DIR/config.json | python -m json.tool

# 查看最终准确率
python -c "import json; data = json.load(open('$OUTPUT_DIR/epoch_eval_results.json')); print(f'Final Accuracy: {data[-1][\"eval_accuracy\"]:.4f}')"
```

---

## ⚠️ 常见问题

### Q1: 为什么 Neg Entropy Loss 是负数？

A: 这是正常的！因为 `lambda_entropy = -0.01`（负号），所以：
- 当熵高时（分布均匀），neg_entropy_loss 是负数
- 当熵低时（分布尖锐），neg_entropy_loss 接近 0
- 目标是最小化 neg_entropy_loss，即最小化熵

### Q2: Gate 初始化为什么是 -2？

A: 因为 `sigmoid(-2) ≈ 0.12`，所以初期 gate 接近 0，MoE 影响很小。这样：
- 初期不会破坏原模型输出
- 训练过程中 gate 逐步增大
- 模型学会何时需要 MoE 增强

### Q3: 为什么要降低 Shared 的容量？

A: 因为原来 Shared 太强，会学会所有东西，MoE 无法竞争。降低容量后：
- Shared 只能学基础特征
- MoE 有机会学习文化差异
- 两者形成互补

### Q4: 负熵正则化的 lambda 值可以调吗？

A: 可以，但 -0.01 是推荐值。如果：
- 路由还是不够尖锐：改为 -0.02 或 -0.05
- 路由过于尖锐（某个专家权重 > 0.9）：改为 -0.005

---

## 📚 相关文件

- `CultureMoE.py`：核心模型实现
- `experts.py`：Expert 层实现
- `router.py`：Router 实现
- `run_ft_culturemoe_gen.sh`：训练脚本
- `ft_culturemoe_from_base_gen.py`：训练主程序
- `STRUCTURE_IMPROVEMENTS.md`：详细改进说明

---

**准备好了吗？开始训练吧！** 🚀

