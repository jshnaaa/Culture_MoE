#!/usr/bin/env python3
"""
修补 ft_lora_only_gen.py 中的 generate_answer 函数

使用方法：
1. 备份原文件: cp ft_lora_only_gen.py ft_lora_only_gen.py.backup
2. 运行此脚本: python patch_ft_lora_only_gen.py
3. 重新训练或评估
"""

import re

def patch_generate_answer_function():
    """
    修补 generate_answer 函数以解决 Human 输出问题
    """

    # 读取原文件
    with open('ft_lora_only_gen.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 新的 generate_answer 函数
    new_generate_answer = '''def generate_answer(model, tokenizer, instruction: str, input_text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    """
    使用模型生成答案 - 修复版本

    关键修复：
    - 使用正确的聊天模板格式
    - 添加强化的后处理逻辑
    - 移除"Human"等不需要的输出

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
    # 构建输入
    if input_text:
        full_content = f"{instruction}\\n{input_text}"
    else:
        full_content = instruction

    # 🔧 修复1: 使用LLaMA3聊天模板格式
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # 使用官方聊天模板
        messages = [{"role": "user", "content": full_content}]
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # 手动构建LLaMA3格式
        formatted_input = f"<|start_header_id|>user<|end_header_id|>\\n\\n{full_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"

    inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=512)
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
            # 🔧 修复2: 添加早停
            early_stopping=True
        )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 🔧 修复3: 强化后处理
    generated_text = _clean_generated_text(generated_text)

    return generated_text


def _clean_generated_text(text: str) -> str:
    """
    清理生成的文本，移除不需要的内容
    """
    # 移除所有包含"Human"的内容
    text = re.sub(r'Human.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'human.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\nHuman.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\nhuman.*', '', text, flags=re.IGNORECASE)

    # 移除可能的聊天模板token
    text = re.sub(r'<\\|.*?\\|>', '', text)

    # 移除多余的换行和空格
    text = re.sub(r'\\n+', '', text)
    text = text.strip()

    return text'''

    # 同时修复 extract_answer_from_text 函数
    new_extract_answer = '''def extract_answer_from_text(text: str) -> str:
    """
    从生成的文本中提取答案 - 改进版本

    使用正则表达式查找数字，增强了对"Human"文本的处理

    Args:
        text: 生成的文本

    Returns:
        提取的答案（数字字符串）
    """
    # 🔧 修复: 首先清理文本
    text = _clean_generated_text(text)

    # 查找第一个数字（优先匹配开头的数字）
    match = re.search(r'^\\s*(\\d+)', text)
    if match:
        return match.group(1)

    # 如果没找到开头的数字，查找任意数字
    match = re.search(r'\\d+', text)
    if match:
        return match.group(0)

    return ""'''

    # 查找并替换 generate_answer 函数
    pattern = r'def generate_answer\(.*?\n    return generated_text'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        content = content.replace(match.group(0), new_generate_answer)
        print("✅ 已修复 generate_answer 函数")
    else:
        print("❌ 未找到 generate_answer 函数")
        return False

    # 查找并替换 extract_answer_from_text 函数
    pattern = r'def extract_answer_from_text\(.*?\n    return ""'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        content = content.replace(match.group(0), new_extract_answer)
        print("✅ 已修复 extract_answer_from_text 函数")
    else:
        print("❌ 未找到 extract_answer_from_text 函数")

    # 写回文件
    with open('ft_lora_only_gen.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ 修补完成！")
    return True


def create_simple_fix():
    """
    创建一个简单的修复版本，只修改后处理逻辑
    """

    # 读取原文件
    with open('ft_lora_only_gen.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 只修改 extract_answer_from_text 函数和添加清理函数
    new_extract_answer = '''def extract_answer_from_text(text: str) -> str:
    """
    从生成的文本中提取答案 - 简单修复版本

    使用正则表达式查找数字，移除"Human"等干扰文本

    Args:
        text: 生成的文本

    Returns:
        提取的答案（数字字符串）
    """
    # 🔧 简单修复: 移除"Human"相关文本
    import re

    # 移除所有包含"Human"的内容
    text = re.sub(r'Human.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'human.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\nHuman.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\nhuman.*', '', text, flags=re.IGNORECASE)

    # 移除换行
    text = re.sub(r'\\n+', '', text)
    text = text.strip()

    # 查找第一个数字
    match = re.search(r'^\\s*(\\d+)', text)
    if match:
        return match.group(1)

    # 如果没找到开头的数字，查找任意数字
    match = re.search(r'\\d+', text)
    if match:
        return match.group(0)

    return ""'''

    # 查找并替换 extract_answer_from_text 函数
    pattern = r'def extract_answer_from_text\(.*?\n    return ""'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        content = content.replace(match.group(0), new_extract_answer)
        print("✅ 已应用简单修复到 extract_answer_from_text 函数")

        # 写回文件
        with open('ft_lora_only_gen.py', 'w', encoding='utf-8') as f:
            f.write(content)

        print("✅ 简单修复完成！")
        return True
    else:
        print("❌ 未找到 extract_answer_from_text 函数")
        return False


if __name__ == "__main__":
    import sys

    print("🔧 ft_lora_only_gen.py 修补工具")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        print("应用简单修复...")
        create_simple_fix()
    else:
        print("应用完整修复...")

        # 备份原文件
        import shutil
        shutil.copy2('ft_lora_only_gen.py', 'ft_lora_only_gen.py.backup')
        print("✅ 已备份原文件为 ft_lora_only_gen.py.backup")

        patch_generate_answer_function()

    print("\n💡 修复后的效果:")
    print("- 模型输出将使用正确的聊天模板格式")
    print("- 自动移除'Human'等干扰文本")
    print("- 提取答案时优先匹配开头的数字")
    print("\n🚀 现在可以重新运行训练或评估脚本！")