# 修复混合格式训练导致的输出混乱问题

## 🔍 问题诊断

### 现象

训练了三个不同格式的数据集：
1. **数据集 A**：数字类（1, 2, 3, 4）
2. **数据集 B**：文本类（yes, no, neutral）
3. **数据集 C**：布尔类（TRUE, FALSE）

测试时模型输出：
```json
{
  "raw_answer": "Code",
  "predicted_label": 5,
  "correct": false
}
```

### 根本原因

**输出空间混乱**：
- 模型在训练时看到了三种完全不同的输出格式
- LoRA 参数是共享的，无法区分当前任务类型
- 导致输出 token 的 logits 分布混乱
- 模型退回到"语言生成"模式，输出高频词（如 "Code", "Country"）

---

## ✅ 解决方案（按优先级）

### 方案 1：为每个任务训练独立的 LoRA（推荐 🔥）

**优点**：
- 每个 LoRA 的输出空间独立
- 不会互相污染
- 可以后续做 Router 自动选择

**实现**：

```bash
# 1. 分别训练三个 LoRA
python train_lora_only_gen.py \
    --model_name_or_path /path/to/llama \
    --data_path /path/to/numeric_data.json \
    --output_dir /path/to/lora_numeric \
    --lora_rank 8

python train_lora_only_gen.py \
    --model_name_or_path /path/to/llama \
    --data_path /path/to/yesno_data.json \
    --output_dir /path/to/lora_yesno \
    --lora_rank 8

python train_lora_only_gen.py \
    --model_name_or_path /path/to/llama \
    --data_path /path/to/truefalse_data.json \
    --output_dir /path/to/lora_truefalse \
    --lora_rank 8

# 2. 测试时根据数据类型加载对应 LoRA
python eval_lora_only_from_components.py \
    --base_model_path /path/to/llama \
    --lora_weights_path /path/to/lora_numeric/best_lora \
    --test_file /path/to/numeric_test.json \
    --output_dir /path/to/results
```

---

### 方案 2：在训练数据中添加任务类型标识

**修改数据格式**：

```json
// 数字类
{
  "instruction": "[Task: NUMERIC] Question: How important is family in your life? Country: Andorra",
  "input": "Option: 1. Very important\n2. Rather important\n3. Not very important\n4. Not at all important\nPlease select the most appropriate option by replying with the option number only (1, 2, 3, ...). Your answer must be a single number only. Your answer is:",
  "output": "1"
}

// 文本类
{
  "instruction": "[Task: SENTIMENT] Question: Do you approve of the government?",
  "input": "Please answer with one of: yes, no, neutral. Your answer is:",
  "output": "yes"
}

// 布尔类
{
  "instruction": "[Task: BOOLEAN] Question: Do you agree with the statement?",
  "input": "Please answer with TRUE or FALSE. Your answer is:",
  "output": "TRUE"
}
```

**修改 `train_lora_only_gen.py` 中的数据处理**：

```python
def load_and_process_data(data_path: str, tokenizer, max_length: int = 512, val_split: float = 0.1):
    """加载并处理生成式数据（添加任务类型标识）"""

    # ... 加载数据 ...

    # 检测输出类型并添加任务标识
    for item in data:
        output = str(item['output']).strip()

        if output.isdigit():
            # 数字类
            task_type = "[Task: NUMERIC]"
        elif output.lower() in ['yes', 'no', 'neutral']:
            # 文本类
            task_type = "[Task: SENTIMENT]"
        elif output.upper() in ['TRUE', 'FALSE']:
            # 布尔类
            task_type = "[Task: BOOLEAN]"
        else:
            task_type = ""

        # 在 instruction 前添加任务类型
        if task_type and not item['instruction'].startswith('[Task:'):
            item['instruction'] = f"{task_type} {item['instruction']}"

    # ... 继续处理 ...
```

---

### 方案 3：统一标签编码（最简单）

**映射规则**：

| 原始标签 | 统一编码 |
|---------|---------|
| 1 | A |
| 2 | B |
| 3 | C |
| 4 | D |
| yes | A |
| no | B |
| neutral | C |
| TRUE | A |
| FALSE | B |

**修改数据处理**：

```python
def load_and_process_data(data_path: str, tokenizer, max_length: int = 512, val_split: float = 0.1):
    """加载并处理生成式数据（统一标签编码）"""

    # 标签映射
    label_mapping = {
        # 数字类
        '1': 'A', '2': 'B', '3': 'C', '4': 'D',
        '5': 'E', '6': 'F', '7': 'G', '8': 'H',
        '9': 'I', '10': 'J',
        # 文本类
        'yes': 'A', 'no': 'B', 'neutral': 'C',
        # 布尔类
        'TRUE': 'A', 'FALSE': 'B',
        'True': 'A', 'False': 'B',
        'true': 'A', 'false': 'B',
    }

    # 反向映射（用于评估）
    reverse_mapping = {}

    # 转换标签
    for item in data:
        original_output = str(item['output']).strip()

        if original_output in label_mapping:
            # 保存原始标签（用于评估）
            item['original_output'] = original_output
            # 转换为统一编码
            item['output'] = label_mapping[original_output]
            # 记录反向映射
            reverse_mapping[label_mapping[original_output]] = original_output
        else:
            print(f"⚠️  Unknown label: {original_output}")

    # ... 继续处理 ...

    return {
        'train': train_dataset,
        'val': val_dataset,
        'output_type': 'unified',
        'label_mapping': label_mapping,
        'reverse_mapping': reverse_mapping
    }
```

---

### 方案 4：强化输出格式约束（立即可用）

**修改 `train_lora_only_gen.py` 中的 prompt 构建**：

```python
def preprocess_function(examples):
    """预处理函数 - 强化输出格式约束"""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i]
        output = str(examples['output'][i])

        # 检测输出类型
        if output.isdigit():
            # 数字类 - 强化约束
            format_constraint = "\n\n⚠️ IMPORTANT: Your answer MUST be a single digit number (1, 2, 3, 4, etc.). Do NOT write any words, letters, or explanations. Only output the number."
            prompt = f"{instruction}\n{input_text}{format_constraint}\nAnswer:"
        elif output.lower() in ['yes', 'no', 'neutral']:
            # 文本类 - 强化约束
            format_constraint = "\n\n⚠️ IMPORTANT: Your answer MUST be one of: yes, no, neutral. Do NOT write any numbers or explanations."
            prompt = f"{instruction}\n{input_text}{format_constraint}\nAnswer:"
        elif output.upper() in ['TRUE', 'FALSE']:
            # 布尔类 - 强化约束
            format_constraint = "\n\n⚠️ IMPORTANT: Your answer MUST be TRUE or FALSE. Do NOT write any numbers or explanations."
            prompt = f"{instruction}\n{input_text}{format_constraint}\nAnswer:"
        else:
            # 默认
            prompt = f"{instruction}\n{input_text}\nAnswer:"

        # ... 继续处理 ...
```

**修改 `eval_lora_only_from_components.py` 中的生成**：

```python
def generate_answer(model, tokenizer, instruction: str, input_text: str, num_classes: int = 10, max_new_tokens: int = 10):
    """生成答案 - 强化格式约束"""
    device = next(model.parameters()).device

    # 添加强约束
    format_constraint = "\n\n⚠️ IMPORTANT: Reply with ONLY the option number (1, 2, 3, 4). Do NOT write words like 'Code', 'Country', 'Option', or any explanations."

    # 构建 prompt
    full_input = f"{instruction}\n{input_text}{format_constraint}\nAnswer:"

    # Tokenize
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate - 使用确定性生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,           # ✅ 禁用采样
            temperature=None,          # ✅ 不使用 temperature
            top_p=None,                # ✅ 不使用 top_p
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,               # ✅ 贪婪解码
            repetition_penalty=1.0     # ✅ 不惩罚重复
        )

    # ... 继续处理 ...
```

---

### 方案 5：Curriculum Learning（渐进式训练）

**训练顺序**：

```bash
# Step 1: 先训练数字类（最简单）
python train_lora_only_gen.py \
    --model_name_or_path /path/to/llama \
    --data_path /path/to/numeric_data.json \
    --output_dir /path/to/lora_stage1 \
    --learning_rate 1e-4 \
    --num_train_epochs 3

# Step 2: 在 yes/no 上继续训练（使用更低学习率）
python train_lora_only_gen.py \
    --model_name_or_path /path/to/llama \
    --lora_weights_path /path/to/lora_stage1/best_lora \
    --data_path /path/to/yesno_data.json \
    --output_dir /path/to/lora_stage2 \
    --learning_rate 5e-5 \
    --num_train_epochs 2

# Step 3: 在 TRUE/FALSE 上继续训练（使用更低学习率）
python train_lora_only_gen.py \
    --model_name_or_path /path/to/llama \
    --lora_weights_path /path/to/lora_stage2/best_lora \
    --data_path /path/to/truefalse_data.json \
    --output_dir /path/to/lora_stage3 \
    --learning_rate 1e-5 \
    --num_train_epochs 2
```

---

## 🎯 推荐方案组合

### 短期修复（立即可用）

1. **修改评估脚本**：添加强格式约束（方案 4）
2. **使用确定性生成**：`temperature=0.0, do_sample=False`
3. **改进后处理**：更鲁棒的标签提取

### 中期优化（重新训练）

1. **统一标签编码**（方案 3）：最简单，效果好
2. **添加任务类型标识**（方案 2）：如果需要保留原始标签

### 长期方案（最佳实践）

1. **独立 LoRA**（方案 1）：最稳定，可扩展
2. **Curriculum Learning**（方案 5）：如果必须混训

---

## 📝 立即可用的修复代码

### 修改 `eval_lora_only_from_components.py`

```python
def generate_answer(model, tokenizer, instruction: str, input_text: str, num_classes: int = 10, max_new_tokens: int = 10):
    """生成答案 - 修复版"""
    device = next(model.parameters()).device

    # ✅ 添加强约束
    format_constraint = "\n\n⚠️ CRITICAL: You MUST reply with ONLY a single digit number (1, 2, 3, or 4). Do NOT write any words, letters, or explanations. Examples of CORRECT answers: '1', '2', '3', '4'. Examples of WRONG answers: 'Code', 'Country', 'Option 1', 'The answer is 1'."

    # 构建 prompt
    full_input = f"{instruction}\n{input_text}{format_constraint}\nAnswer:"

    # Tokenize
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ✅ 使用确定性生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,              # ✅ 减少生成长度
            min_new_tokens=1,
            do_sample=False,               # ✅ 禁用采样
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            repetition_penalty=1.0
        )

    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取生成的部分
    raw_answer = full_output[len(full_input):].strip()

    # ✅ 改进的标签提取
    predicted_label = extract_label_robust(raw_answer, num_classes)

    return raw_answer, predicted_label


def extract_label_robust(answer: str, num_classes: int = 10):
    """改进的标签提取 - 更鲁棒"""
    if not answer:
        return num_classes // 2

    # 1. 尝试直接转换为整数
    try:
        label = int(answer.strip())
        if 1 <= label <= num_classes:
            return label - 1  # 1-indexed -> 0-indexed
        elif 0 <= label < num_classes:
            return label
    except ValueError:
        pass

    # 2. 提取第一个数字
    import re
    numbers = re.findall(r'\d+', answer)
    if numbers:
        label = int(numbers[0])
        if 1 <= label <= num_classes:
            return label - 1
        elif 0 <= label < num_classes:
            return label

    # 3. 检查是否是文本答案（如果混训了）
    answer_lower = answer.lower().strip()
    text_mapping = {
        'yes': 0, 'no': 1, 'neutral': 2,
        'true': 0, 'false': 1,
        'a': 0, 'b': 1, 'c': 2, 'd': 3
    }
    if answer_lower in text_mapping:
        return text_mapping[answer_lower]

    # 4. 如果是无关词（如 "Code", "Country"），返回 None 而不是默认值
    if answer.isalpha() and len(answer) > 2:
        print(f"⚠️  Invalid answer: '{answer}' - returning None")
        return None  # 或者返回一个特殊值表示无效

    # 5. 默认返回中间类别
    return num_classes // 2
```

---

## 🔍 快速验证

运行以下脚本验证是否是混训问题：

```python
# test_output_format.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载模型
base_model = AutoModelForCausalLM.from_pretrained("/path/to/llama", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "/path/to/lora/best_lora")
tokenizer = AutoTokenizer.from_pretrained("/path/to/llama", trust_remote_code=True)

# 测试不同类型的 prompt
prompts = [
    # 数字类
    "Question: How important is family? Options: 1. Very important 2. Rather important 3. Not very important 4. Not at all important. Reply with only the number. Answer:",

    # 文本类
    "Question: Do you approve? Reply with yes, no, or neutral. Answer:",

    # 布尔类
    "Question: Do you agree? Reply with TRUE or FALSE. Answer:"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    answer = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(f"Prompt: {prompt[:50]}...")
    print(f"Answer: '{answer}'")
    print()
```

**预期结果**：
- 如果输出混乱（如数字类输出 "yes"，文本类输出 "1"），说明确实是混训问题
- 如果输出正确，说明是 prompt 格式问题

---

## 总结

✅ **立即修复**：
1. 修改 `eval_lora_only_from_components.py` 添加强格式约束
2. 使用确定性生成（`do_sample=False`）
3. 改进标签提取逻辑

✅ **重新训练**（推荐）：
1. **方案 1**：分别训练三个独立 LoRA ⭐⭐⭐⭐⭐
2. **方案 3**：统一标签编码（A/B/C/D）⭐⭐⭐⭐
3. **方案 2**：添加任务类型标识 ⭐⭐⭐

✅ **长期优化**：
- 使用 Router 自动选择 LoRA
- 实现多任务学习框架
- 添加输出格式验证

现在可以根据你的实际情况选择合适的方案了！🎉

