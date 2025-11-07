# run_train_culturemoe_from_base_gen.sh 修复总结

## 修复内容

### ✅ 1. LORA_WEIGHTS_PATH 检查（你写的很好！）

你添加的条件判断是正确的：

```bash
# 第 123-134 行
if [ ! -d "$LORA_WEIGHTS_PATH" ]; then
    echo "❌ Error: LoRA weights not found: $LORA_WEIGHTS_PATH"
    echo ""
    echo "Please ensure you have completed:"
    echo "  sh run_train_lora_only_gen.sh $BACKBONE <DATA_ID>"
    echo ""
    echo "The training should create a 'best_lora' directory containing:"
    echo "  - adapter_config.json"
    echo "  - adapter_model.safetensors"
    echo "  - tokenizer files"
    exit 1
fi
```

**优点**：
- ✅ 检查目录是否存在
- ✅ 清晰的错误信息
- ✅ 提示用户如何解决

### ❌ 2. 但发现了其他问题（已修复）

#### 问题 1：NUM_CLASSES 未定义

**原问题**：
```bash
echo "Num classes: $NUM_CLASSES"  # ❌ 未定义！
```

**修复**：
```bash
# 根据 DATA_ID 设置 NUM_CLASSES
case $DATA_ID in
    2)
        NUM_CLASSES=2  # CulturalBench: TRUE/FALSE
        ;;
    3)
        NUM_CLASSES=3  # NormAD: yes/no/neutral
        ;;
    4)
        NUM_CLASSES=10  # CultureLLM: 1-10
        ;;
    *)
        NUM_CLASSES=10  # 默认
        ;;
esac
```

#### 问题 2：OUTPUT_DIR 被覆盖

**原问题**：
```bash
# 第 38、50、62 行在 case 中设置
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_CulturalBench_${BACKBONE}_..."

# 第 83 行又覆盖了！
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/moe_${BACKBONE}_..."
```

**修复**：
```bash
# 在 case 中设置 DATASET_TAG 而不是 OUTPUT_DIR
case $DATA_ID in
    2)
        DATASET_TAG="CulturalBench"
        ;;
    3)
        DATASET_TAG="normad"
        ;;
    4)
        DATASET_TAG="cultureLLM"
        ;;
esac

# 使用 DATASET_TAG 构建 OUTPUT_DIR
OUTPUT_DIR="/root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/moe_${DATASET_TAG}_${BACKBONE}_experts${NUM_EXPERTS}_..."
```

#### 问题 3：DATA_ID 参数说明不对

**原问题**：
```bash
DATA_ID="${2:-4}"  # 默认 5 分类  ❌ 错误！
```

**修复**：
```bash
DATA_ID="${2:-4}"  # 默认 CultureLLM (4)  ✅ 正确！
```

## 修复前后对比

| 功能 | 修复前 | 修复后 |
|------|--------|--------|
| **LORA_WEIGHTS_PATH 检查** | ✅ 正确 | ✅ 正确 |
| **NUM_CLASSES 定义** | ❌ 未定义 | ✅ 自动设置 |
| **OUTPUT_DIR** | ❌ 被覆盖 | ✅ 正确构建 |
| **参数说明** | ❌ 错误 | ✅ 正确 |

## 完整的参数流程

```
输入参数
  ↓
DATA_ID (2/3/4)
  ├─ 设置 NUM_CLASSES (2/3/10)
  ├─ 设置 DATASET_NAME
  ├─ 设置 TRAIN_FILE
  ├─ 设置 DATASET_TAG
  └─ 设置 LORA_WEIGHTS_PATH
  ↓
OUTPUT_DIR = moe_${DATASET_TAG}_${BACKBONE}_...
  ↓
检查 LORA_WEIGHTS_PATH 是否存在
  ├─ 存在 → 继续训练
  └─ 不存在 → 报错退出
```

## 使用示例

### 训练 CultureLLM（默认）

```bash
sh run_train_culturemoe_from_base_gen.sh llama
# 等价于
sh run_train_culturemoe_from_base_gen.sh llama 4 True 6 1 true
```

**输出**：
```
Backbone: llama (LLaMA 3.1-8B-Instruct)
Num classes: 10
Dataset: CultureLLM
...
Output: /root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_YYYYMMDD_HHMM
```

### 训练 NormAD

```bash
sh run_train_culturemoe_from_base_gen.sh llama 3 True 6 1 true
```

**输出**：
```
Backbone: llama (LLaMA 3.1-8B-Instruct)
Num classes: 3
Dataset: NormAD
...
Output: /root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/moe_normad_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_YYYYMMDD_HHMM
```

### 训练 CulturalBench

```bash
sh run_train_culturemoe_from_base_gen.sh llama 2 True 6 1 true
```

**输出**：
```
Backbone: llama (LLaMA 3.1-8B-Instruct)
Num classes: 2
Dataset: CulturalBench
...
Output: /root/autodl-tmp/CultureMoE/Culture_Alignment/gen_test_results/moe_CulturalBench_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_YYYYMMDD_HHMM
```

## 错误处理

### 错误 1：LoRA 权重不存在

```bash
$ sh run_train_culturemoe_from_base_gen.sh llama

❌ Error: LoRA weights not found: /root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora

Please ensure you have completed:
  sh run_train_lora_only_gen.sh llama 4

The training should create a 'best_lora' directory containing:
  - adapter_config.json
  - adapter_model.safetensors
  - tokenizer files
```

**解决方案**：
```bash
# 先训练 LoRA
sh run_train_lora_only_gen.sh llama 4

# 再训练 MoE
sh run_train_culturemoe_from_base_gen.sh llama 4
```

### 错误 2：无效的 DATA_ID

```bash
$ sh run_train_culturemoe_from_base_gen.sh llama 5

❌ Error: Invalid DATA_ID=5. Must be 2, 3, or 4.

DATA_ID options:
  2 - CulturalBench
  3 - NormAD
  4 - CultureLLM (default)
```

## 总结

✅ **你的 LORA_WEIGHTS_PATH 检查写得很好！**

我额外修复的问题：
1. ✅ 添加了 NUM_CLASSES 的自动设置
2. ✅ 修复了 OUTPUT_DIR 被覆盖的问题
3. ✅ 修正了参数说明

现在脚本更加健壮和易用了！🎉

