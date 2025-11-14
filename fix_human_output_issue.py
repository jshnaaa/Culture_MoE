#!/usr/bin/env python3
"""
修复 ft_lora_only_gen.py 中模型输出包含 "Human" 文本的问题

问题原因：
1. 训练时使用简单文本格式，但LLaMA模型期望聊天模板格式
2. 生成时格式不匹配，导致模型"泄露"训练时的模板token

解决方案：
1. 使用正确的聊天模板格式进行生成
2. 添加合适的停止词
3. 确保输入格式与模型期望一致
"""

import re

def generate_answer_fixed(model, tokenizer, instruction: str, input_text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    """
    修复后的生成函数 - 使用正确的聊天模板格式

    Args:
        model: 模型
        tokenizer: tokenizer
        instruction: 指令
        input_text: 输入文本
        device: 设备
        max_new_tokens: 最大生成 token 数

    Returns:
        生成的文本
    """
    # 方案1: 使用LLaMA3聊天模板格式
    if input_text:
        full_content = f"{instruction}\n{input_text}"
    else:
        full_content = instruction

    # 使用LLaMA3聊天模板格式
    formatted_input = f"<|start_header_id|>user<|end_header_id|>\n\n{full_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # Tokenize
    inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 设置停止词
    stop_token_ids = []
    for stop_word in ["<|eot_id|>", "<|eom_id|>", "Human", "\nHuman", "human", "\nhuman"]:
        stop_ids = tokenizer.encode(stop_word, add_special_tokens=False)
        if stop_ids:
            stop_token_ids.extend(stop_ids)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # 贪婪解码
            num_beams=1,      # 禁用 beam search
            repetition_penalty=1.0,
            # 添加停止词
            stopping_criteria=None  # 可以自定义停止条件
        )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 后处理：移除可能的"Human"文本
    generated_text = clean_generated_text(generated_text)

    return generated_text


def generate_answer_simple_fix(model, tokenizer, instruction: str, input_text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    """
    简单修复版本 - 只修改后处理逻辑

    这个版本保持原有的输入格式，但改进后处理和停止条件
    """
    # 构建输入（保持原格式）
    if input_text:
        full_input = f"{instruction}\n{input_text}"
    else:
        full_input = instruction

    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # 贪婪解码
            num_beams=1,      # 禁用 beam search
            repetition_penalty=1.0,
            # 设置更严格的停止条件
            early_stopping=True
        )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 强化的后处理
    generated_text = clean_generated_text(generated_text)

    return generated_text


def clean_generated_text(text: str) -> str:
    """
    清理生成的文本，移除不需要的内容

    Args:
        text: 原始生成文本

    Returns:
        清理后的文本
    """
    # 移除所有包含"Human"的内容
    text = re.sub(r'Human.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'human.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\nHuman.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\nhuman.*', '', text, flags=re.IGNORECASE)

    # 移除可能的聊天模板token
    text = re.sub(r'<\|.*?\|>', '', text)

    # 移除多余的换行和空格
    text = re.sub(r'\n+', '', text)
    text = text.strip()

    return text


def extract_answer_from_text_improved(text: str) -> str:
    """
    改进的答案提取函数

    Args:
        text: 生成的文本

    Returns:
        提取的答案（数字字符串）
    """
    # 首先清理文本
    text = clean_generated_text(text)

    # 查找第一个数字（优先匹配单个数字）
    match = re.search(r'^\s*(\d+)', text)
    if match:
        return match.group(1)

    # 如果没找到开头的数字，查找任意数字
    match = re.search(r'\d+', text)
    if match:
        return match.group(0)

    return ""


# 使用示例和测试函数
def test_fixes():
    """
    测试修复效果的函数
    """
    # 模拟一些有问题的输出
    problematic_outputs = [
        "0HumanHumanHuman\nHuman",
        "2HumanHumanHuman\nHuman",
        "3HumanHumanHumanHuman",
        "2HumanHumanHuman\nHuman",
        "3HumanHumanHuman\nHuman"
    ]

    print("🔧 测试修复效果:")
    print("=" * 50)

    for i, output in enumerate(problematic_outputs, 1):
        cleaned = clean_generated_text(output)
        extracted = extract_answer_from_text_improved(output)

        print(f"样本 {i}:")
        print(f"  原始输出: {output}")
        print(f"  清理后: '{cleaned}'")
        print(f"  提取答案: '{extracted}'")
        print()


if __name__ == "__main__":
    test_fixes()