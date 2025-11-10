# 消融实验参数实现总结

## 📋 修改概览

为了支持消融实验，我们在以下文件中添加了两个新参数：

1. **`culture_loss_weight`** - 文化损失权重（默认 0.5）
2. **`use_shared`** - 是否使用共享专家（默认 true）

---

## 🔧 修改详情

### 1. `run_train_culturemoe_from_base_gen.sh`

#### 添加的参数

```bash
CULTURE_LOSS_WEIGHT="${7:-0.5}"     # ✅ 新增：文化损失权重 (默认 0.5)
USE_SHARED="${8:-true}"             # ✅ 新增：是否使用共享专家 (默认 true)
```

#### 更新的输出目录

```bash
# 包含新参数的目录名称
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_${DATASET_TAG}_${BACKBONE}_experts${NUM_EXPERTS}_USE_CULTURE_LOSS${USE_CULTURE_LOSS}_CULTURE_LOSS_WEIGHT${CULTURE_LOSS_WEIGHT}_USE_SHARED${USE_SHARED}_MASK_USE${MASK_USE}_$(date +%Y%m%d_%H%M)"
```

#### 更新的训练命令

```bash
TRAIN_CMD="python train_culturemoe_from_base_gen.py \
    ...
    --culture_loss_lambda $CULTURE_LOSS_WEIGHT \
    --use_shared_experts $USE_SHARED \
    ..."
```

#### 更新的日志输出

```bash
echo "Culture loss weight: $CULTURE_LOSS_WEIGHT"  # ✅ 新增
echo "Use shared experts: $USE_SHARED"             # ✅ 新增
```

---

### 2. `train_culturemoe_from_base_gen.py`

#### 添加的命令行参数

```python
# ✅ 消融实验参数
parser.add_argument("--use_shared_experts", type=lambda x: x.lower() == 'true', default=True,
                    help="Whether to use shared experts layer (default: True)")
```

#### 修改的 train_epoch 函数签名

```python
def train_epoch(model, train_loader, optimizer, device, use_culture_loss, culture_loss_lambda,
                lambda_entropy=0.05, lambda_load=0.01, use_shared_experts=True):
    """
    训练一个 epoch

    Args:
        lambda_entropy: 熵正则化系数（推荐 0.01-0.1）
        lambda_load: 负载均衡系数（推荐 0.01）
        use_shared_experts: 是否使用共享专家层（消融实验）
    """
```

#### 修改的模型调用

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    input_ids_mask=input_ids_mask,
    attention_mask_mask=attention_mask_mask,
    labels=labels,
    culture_labels=culture_labels,
    use_culture_loss=use_culture_loss,
    culture_loss_lambda=culture_loss_lambda,
    use_shared_experts=use_shared_experts  # ✅ 新增
)
```

#### 修改的 train_epoch 调用

```python
train_metrics = train_epoch(
    model, train_loader, optimizer, args.device,
    args.use_culture_loss, args.culture_loss_lambda,
    lambda_entropy=args.lambda_entropy,
    lambda_load=args.lambda_load,
    use_shared_experts=args.use_shared_experts  # ✅ 新增：消融实验参数
)
```

---

### 3. `src/llamafactory/model/CultureMoE.py`

#### 修改的 forward 方法签名

```python
def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None, attention_mask_mask=None,
            labels=None, culture_labels=None, use_culture_loss=False, culture_loss_lambda=0.5,
            use_shared_experts=True, **kwargs):  # ✅ 新增参数
    """
    ✅ 生成式前向传播

    Args:
        ...
        use_shared_experts: 是否使用共享专家层（消融实验）
    """
```

#### 修改的 instruction_mask 处理

```python
# ✅ 如果不使用共享专家，则不需要处理 instruction_mask
if use_shared_experts and input_ids_mask is not None:
    outputs_no = self.llama_model.model(...)
    ...
else:
    h_no = h_all.clone()
```

#### 修改的 shared 层处理

```python
# ✅ 如果不使用共享专家，则跳过 shared 层
if use_shared_experts:
    h_no = h_no.to(device=device, dtype=dtype)
    shared_out = self.shared(h_no)  # [B, L, H]
else:
    # ✅ 消融实验：不使用共享专家，shared_out 为零
    shared_out = torch.zeros_like(h_all)
```

#### 修改的 Router 输入

```python
# ✅ 如果使用共享专家，基于 shared_out 计算路由；否则基于 h_all 计算
if use_shared_experts:
    pooled = shared_out.mean(dim=1)  # [B, H]
else:
    pooled = h_all.mean(dim=1)  # [B, H]
expert_weights, router_logits = self.router(pooled)  # [B, E]
```

---

## 📊 参数流向图

```
run_train_culturemoe_from_base_gen.sh
    ↓
    CULTURE_LOSS_WEIGHT="${7:-0.5}"
    USE_SHARED="${8:-true}"
    ↓
    python train_culturemoe_from_base_gen.py \
        --culture_loss_lambda $CULTURE_LOSS_WEIGHT \
        --use_shared_experts $USE_SHARED
    ↓
train_culturemoe_from_base_gen.py
    ↓
    args.culture_loss_lambda (已有)
    args.use_shared_experts (新增)
    ↓
    train_epoch(..., use_shared_experts=args.use_shared_experts)
    ↓
    model(..., use_shared_experts=use_shared_experts)
    ↓
CultureMoE.forward(..., use_shared_experts=True)
    ↓
    if use_shared_experts:
        # 使用共享专家
    else:
        # 不使用共享专家
```

---

## 🎯 功能说明

### culture_loss_weight 的作用

**在 train_culturemoe_from_base_gen.py 中**：
- 通过 `--culture_loss_lambda` 参数传递给模型
- 控制文化损失在总损失中的权重
- 公式：`total_loss = generation_loss + culture_loss_lambda * culture_loss`

**在 CultureMoE.py 中**：
- 在 forward 方法中使用 `culture_loss_lambda` 参数
- 计算总损失时应用权重

### use_shared_experts 的作用

**在 CultureMoE.py 中**：

1. **instruction_mask 处理**
   - 如果 `use_shared_experts=true`：处理 instruction_mask 获取 h_no
   - 如果 `use_shared_experts=false`：跳过 instruction_mask 处理

2. **shared 层处理**
   - 如果 `use_shared_experts=true`：使用 shared 层处理 h_no
   - 如果 `use_shared_experts=false`：shared_out 为零张量

3. **Router 输入**
   - 如果 `use_shared_experts=true`：基于 shared_out 计算路由
   - 如果 `use_shared_experts=false`：基于 h_all 计算路由

4. **融合方式**
   - 如果 `use_shared_experts=true`：`enhanced_hidden = shared_out + moe_warmup_weight * expert_sum`
   - 如果 `use_shared_experts=false`：`enhanced_hidden = 0 + moe_warmup_weight * expert_sum = moe_warmup_weight * expert_sum`

---

## ✅ 验证修改

### 检查参数是否正确传递

```bash
# 1. 检查脚本中的参数
grep "CULTURE_LOSS_WEIGHT\|USE_SHARED" run_train_culturemoe_from_base_gen.sh

# 2. 检查训练脚本中的参数
grep "use_shared_experts" train_culturemoe_from_base_gen.py

# 3. 检查模型中的参数
grep "use_shared_experts" src/llamafactory/model/CultureMoE.py
```

### 运行测试

```bash
# 测试默认参数
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true

# 测试自定义参数
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.1 false

# 检查输出目录名称是否包含新参数
ls -la /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/ | grep "CULTURE_LOSS_WEIGHT"
```

---

## 📈 预期行为

### 当 culture_loss_weight=0.0 时

- 文化损失不被使用
- 总损失 = 生成损失
- 模型只关注生成任务

### 当 culture_loss_weight=0.5 时（默认）

- 文化损失和生成损失权重相等
- 总损失 = 生成损失 + 0.5 * 文化损失
- 模型平衡生成和文化对齐

### 当 culture_loss_weight=1.0 时

- 文化损失权重最高
- 总损失 = 生成损失 + 文化损失
- 模型强调文化对齐

### 当 use_shared_experts=true 时（默认）

- 使用共享专家层
- 模型有两个信息流：shared 和 experts
- 更复杂但性能更好

### 当 use_shared_experts=false 时

- 不使用共享专家层
- 模型只有专家信息流
- 更简单但性能可能下降

---

## 🚀 使用示例

### 基础使用

```bash
# 使用所有默认参数
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true
```

### 消融实验 1：文化损失权重

```bash
# 不使用文化损失
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.0 true

# 低权重
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.1 true

# 高权重
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.8 true
```

### 消融实验 2：共享专家

```bash
# 有共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 true

# 无共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.5 false
```

### 消融实验 3：组合

```bash
# 低文化损失 + 无共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.1 false

# 高文化损失 + 无共享专家
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 2 true 0.8 false
```

---

## 📝 修改文件清单

- ✅ `run_train_culturemoe_from_base_gen.sh` - 添加参数和传递
- ✅ `train_culturemoe_from_base_gen.py` - 添加参数定义和传递
- ✅ `src/llamafactory/model/CultureMoE.py` - 实现 use_shared_experts 逻辑

---

## 🎉 总结

通过这些修改，你现在可以：

✅ 调整文化损失权重进行消融实验
✅ 启用/禁用共享专家进行对比实验
✅ 验证每个组件的有效性
✅ 找到最优的模型配置
✅ 为论文提供有力的证据

**所有修改都已完成，可以开始消融实验了！** 🚀

