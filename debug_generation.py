#!/usr/bin/env python3
"""
调试生成重复数字问题的脚本
"""

import torch
from transformers import AutoTokenizer

def test_tokenizer_settings():
    """测试tokenizer设置"""
    print("=" * 60)
    print("🔍 调试tokenizer设置")
    print("=" * 60)

    # 加载tokenizer
    model_path = "/Users/yzl/models/Meta-Llama-3.1-8B-Instruct"  # 本地路径
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print(f"✅ Tokenizer加载成功")
        print(f"  - vocab_size: {tokenizer.vocab_size}")
        print(f"  - pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"  - eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"  - bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        print(f"  - unk_token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")

        # 测试数字token
        print(f"\n🔍 测试数字token:")
        for i in range(10):
            token_ids = tokenizer.encode(str(i), add_special_tokens=False)
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            print(f"  数字 '{i}' -> token_ids: {token_ids} -> decoded: '{decoded}'")

        # 测试问题格式
        print(f"\n🔍 测试问题格式:")
        test_instruction = "### Question: Give me the answer from 1 to 7: What sports are often broadcasted"
        test_answer = "7"

        # 训练格式
        full_input = f"{test_instruction}\n"  # 没有input
        train_format = f"{full_input} {test_answer}"

        print(f"  训练输入: '{full_input}'")
        print(f"  训练格式: '{train_format}'")

        # Tokenize
        input_tokens = tokenizer(full_input, add_special_tokens=True, return_tensors="pt")
        train_tokens = tokenizer(train_format, add_special_tokens=True, return_tensors="pt")

        print(f"  输入token数: {input_tokens['input_ids'].shape[1]}")
        print(f"  训练token数: {train_tokens['input_ids'].shape[1]}")
        print(f"  答案token数: {train_tokens['input_ids'].shape[1] - input_tokens['input_ids'].shape[1]}")

        # 检查答案部分的token
        answer_tokens = train_tokens['input_ids'][0][input_tokens['input_ids'].shape[1]:]
        print(f"  答案tokens: {answer_tokens.tolist()}")
        answer_decoded = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        print(f"  答案解码: '{answer_decoded}'")

    except Exception as e:
        print(f"❌ Tokenizer测试失败: {e}")

def test_generation_issue():
    """测试生成问题"""
    print("\n" + "=" * 60)
    print("🔍 分析生成重复数字问题")
    print("=" * 60)

    print("观察到的问题:")
    print("  - 生成文本: '22222', '22666'")
    print("  - 预期答案: '7', '5', '3', '5', '3'")
    print("  - 准确率: 22.92%")

    print("\n可能原因:")
    print("  1. EOS token问题: 模型没有学会何时停止生成")
    print("  2. 训练数据格式: 输入输出格式不匹配")
    print("  3. 生成参数: max_new_tokens=5可能还是太多")
    print("  4. Tokenizer设置: pad_token=eos_token可能导致混淆")

    print("\n建议修复:")
    print("  1. 减少max_new_tokens到1或2")
    print("  2. 检查训练时的labels masking是否正确")
    print("  3. 确保生成时使用正确的eos_token_id")
    print("  4. 考虑使用不同的pad_token")

if __name__ == "__main__":
    test_tokenizer_settings()
    test_generation_issue()