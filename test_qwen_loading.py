#!/usr/bin/env python3
"""
测试 Qwen 模型加载（验证 attn_implementation 参数）

使用方法：
    python test_qwen_loading.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_qwen_loading():
    """测试 Qwen 模型加载"""

    model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"

    print("="*80)
    print("Testing Qwen Model Loading")
    print("="*80)
    print(f"Model path: {model_path}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*80)
    print("")

    # 测试 1：不使用 attn_implementation（应该失败）
    print("Test 1: Loading without attn_implementation...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("   ✅ Success (unexpected!)")
    except Exception as e:
        print(f"   ❌ Failed (expected): {e}")
    print("")

    # 测试 2：使用 attn_implementation="eager"（应该成功）
    print("Test 2: Loading with attn_implementation='eager'...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # 禁用 tensor parallel
        )
        print("   ✅ Success!")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Model dtype: {next(model.parameters()).dtype}")

        # 清理
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    print("")

    print("="*80)
    print("Test Complete")
    print("="*80)


if __name__ == "__main__":
    test_qwen_loading()

