# 错误修复总结

## ❌ 原始错误

```python
AttributeError: 'LoRATrainingArguments' object has no attribute 'save_model'
```

### 错误原因

在 `train_and_eval_lora_only.py` 中：
1. ✅ `main()` 函数中定义了 `--save_model` 命令行参数
2. ❌ `LoRATrainingArguments` 类中**没有** `save_model` 字段
3. ❌ 在 `train_lora_only()` 函数中尝试访问 `args.save_model`，导致 AttributeError

## ✅ 修复方案

### 修改 1：在 `LoRATrainingArguments` 类中添加 `save_model` 字段

**位置**：`train_and_eval_lora_only.py` 第 166-170 行

```python
eval_steps: int = field(
    default=500,
    metadata={"help": "评估步数"}
)
save_model: bool = field(
    default=False,
    metadata={"help": "是否保存模型权重（默认不保存，只保存评估结果）"}
)
```

### 修改 2：在创建 `LoRATrainingArguments` 时传递 `save_model` 参数

**位置**：`train_and_eval_lora_only.py` 第 492-505 行

```python
# 创建训练参数
training_args = LoRATrainingArguments(
    model_path=args.model_path,
    train_file=args.train_file,
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    learning_rate=args.learning_rate,
    lora_rank=args.lora_rank,
    max_length=args.max_length,
    val_split=args.val_split,
    num_classes=args.num_classes,
    save_model=args.save_model,  # ✅ 添加这一行
)
```

## 🔍 验证修复

### 方法 1：运行测试脚本

```bash
sh test_fix.sh
```

这个脚本会：
1. 检查 Python 文件语法
2. 验证 `--save_model` 参数存在
3. 验证 `--num_classes` 参数存在

### 方法 2：查看帮助信息

```bash
python train_and_eval_lora_only.py --help
```

应该能看到：
```
--save_model          是否保存模型（默认不保存，只保存评估结果）
--num_classes NUM_CLASSES
                      分类类别数（2/3/4/5）
```

### 方法 3：实际运行训练

```bash
# 不保存模型（默认）
sh run_train_lora_only.sh llama 3

# 保存模型
sh run_train_lora_only.sh llama 3 true
```

## 📊 修复前后对比

### 修复前

```python
@dataclass
class LoRATrainingArguments:
    # ... 其他字段 ...
    eval_steps: int = field(
        default=500,
        metadata={"help": "评估步数"}
    )
    # ❌ 缺少 save_model 字段
```

```python
# 创建训练参数
training_args = LoRATrainingArguments(
    # ... 其他参数 ...
    num_classes=args.num_classes,
    # ❌ 没有传递 save_model
)
```

### 修复后

```python
@dataclass
class LoRATrainingArguments:
    # ... 其他字段 ...
    eval_steps: int = field(
        default=500,
        metadata={"help": "评估步数"}
    )
    save_model: bool = field(  # ✅ 添加了 save_model 字段
        default=False,
        metadata={"help": "是否保存模型权重（默认不保存，只保存评估结果）"}
    )
```

```python
# 创建训练参数
training_args = LoRATrainingArguments(
    # ... 其他参数 ...
    num_classes=args.num_classes,
    save_model=args.save_model,  # ✅ 传递 save_model
)
```

## ✅ 完整的参数流程

### 1. 命令行参数定义（main 函数）

```python
parser.add_argument("--save_model", action="store_true",
                    help="是否保存模型（默认不保存，只保存评估结果）")
```

### 2. 解析命令行参数

```python
args = parser.parse_args()
# args.save_model = True 或 False
```

### 3. 创建 dataclass 对象

```python
training_args = LoRATrainingArguments(
    # ... 其他参数 ...
    save_model=args.save_model,  # 传递给 dataclass
)
```

### 4. 在训练函数中使用

```python
def train_lora_only(args: LoRATrainingArguments):
    # ...
    if args.save_model:  # ✅ 现在可以正常访问
        print(f"\n8. Saving model to {args.output_dir}...")
        trainer.save_model()
        # ...
    else:
        print(f"\n8. Skipping model saving (--save_model not set)")
```

## 🎯 使用示例

### 不保存模型（默认）

```bash
sh run_train_lora_only.sh llama 3
# 或
sh run_train_lora_only.sh llama 3 false
```

**效果**：
- ❌ 不保存模型权重
- ✅ 保存评估结果（`eval_results.json`）
- ✅ 保存训练日志

### 保存模型

```bash
sh run_train_lora_only.sh llama 3 true
```

**效果**：
- ✅ 保存模型权重
- ✅ 保存评估结果
- ✅ 保存训练日志

## 📝 相关文件

修改的文件：
- ✅ `train_and_eval_lora_only.py` - 添加 `save_model` 字段和参数传递
- ✅ `run_train_lora_only.sh` - 已经支持第三个参数

未修改的文件（已经正确）：
- ✅ `train_culturemoe.py` - 已经有 `--save_model` 参数支持
- ✅ `run_train_culturemoe.sh` - 已经支持第三个参数

## 🚀 现在可以正常使用

所有训练脚本现在都支持三个参数：

```bash
sh run_train_lora_only.sh [backbone] [num_classes] [save_model]
sh run_train_culturemoe.sh [backbone] [num_classes] [save_model]
```

示例：
```bash
# LoRA Only
sh run_train_lora_only.sh llama 2 false
sh run_train_lora_only.sh llama 3 true
sh run_train_lora_only.sh qwen 5 false

# CultureMoE
sh run_train_culturemoe.sh llama 2 false
sh run_train_culturemoe.sh llama 3 true
sh run_train_culturemoe.sh qwen 5 false
```

## ✨ 总结

**问题**：`LoRATrainingArguments` 缺少 `save_model` 字段

**解决方案**：
1. ✅ 在 `LoRATrainingArguments` 类中添加 `save_model` 字段
2. ✅ 在创建 `LoRATrainingArguments` 时传递 `save_model` 参数

**结果**：现在可以正常使用 `--save_model` 参数了！🎉

