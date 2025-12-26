#!/usr/bin/env python3
"""
调试"生成为空"问题的专用脚本
"""

import torch
from transformers import AutoTokenizer

def debug_generation_issue():
    """调试生成为空的问题"""
    print("🔍 调试生成为空问题")
    print("=" * 60)

    # 在generate_answer函数中添加这些调试代码
    debug_code = '''
# 🔍 在generate_answer函数的解码部分添加详细调试
def generate_answer(model, tokenizer, instruction: str, input_text: str, device: str = 'cuda', max_new_tokens: int = 1) -> str:
    # ... 前面的代码保持不变 ...

    with torch.no_grad():
        outputs = model.generate(...)

    # 🔍 详细调试生成结果
    print(f"🔍 生成调试信息:")
    print(f"  输入长度: {inputs['input_ids'].shape[1]}")
    print(f"  输出总长度: {outputs[0].shape[0]}")
    print(f"  生成的token数量: {outputs[0].shape[0] - inputs['input_ids'].shape[1]}")

    # 检查生成的原始token IDs
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    print(f"  生成的token IDs: {generated_ids.tolist()}")

    # 检查每个token的解码
    for i, token_id in enumerate(generated_ids.tolist()):
        try:
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            token_text_with_special = tokenizer.decode([token_id], skip_special_tokens=False)
            print(f"    Token {i}: {token_id} -> '{token_text}' (with_special: '{token_text_with_special}')")
        except Exception as e:
            print(f"    Token {i}: {token_id} -> 解码失败: {e}")

    # 检查特殊token
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    print(f"  PAD token ID: {tokenizer.pad_token_id}")
    print(f"  生成的token中是否包含EOS: {tokenizer.eos_token_id in generated_ids.tolist()}")

    # 解码完整结果
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    generated_text_with_special = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()

    print(f"  解码结果(skip_special=True): '{generated_text}' (长度: {len(generated_text)})")
    print(f"  解码结果(skip_special=False): '{generated_text_with_special}' (长度: {len(generated_text_with_special)})")

    # 检查是否为空的具体原因
    if len(generated_text.strip()) == 0:
        print(f"  ⚠️ 生成为空的原因分析:")
        if len(generated_ids) == 0:
            print(f"    - 没有生成任何token")
        elif len(generated_ids) == 1 and generated_ids[0].item() == tokenizer.eos_token_id:
            print(f"    - 只生成了EOS token")
        elif generated_text_with_special.strip():
            print(f"    - 生成了特殊token但被skip_special_tokens过滤掉了")
        else:
            print(f"    - 生成了无法解码的内容")

    return generated_text
'''

    print("🔧 修复建议:")
    print("1. 在ft_lora_only_gen.py的generate_answer函数中添加上述调试代码")
    print("2. 重新运行训练，观察第二轮开始时的生成调试信息")
    print("3. 特别关注是否生成了过多的EOS token")

    print("\n📋 可能的修复方案:")
    print("方案1：检查EOS token设置")
    print("  - 确保eos_token_id设置正确")
    print("  - 避免模型过早学会生成EOS")

    print("方案2：调整生成参数")
    print("  - 尝试设置min_new_tokens=1强制至少生成1个token")
    print("  - 检查是否需要设置eos_token_id=None临时禁用EOS")

    print("方案3：检查训练标签")
    print("  - 确保训练时没有错误地将答案部分标记为-100")
    print("  - 检查labels masking是否正确")

if __name__ == "__main__":
    debug_generation_issue()