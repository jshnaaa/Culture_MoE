#!/usr/bin/env python3
"""
合并 LoRA 权重到 base 模型（生成式版本）

使用方法：
    python merge_lora_weights_gen.py \
        --base_model_path /path/to/base_model \
        --lora_model_path /path/to/lora_model \
        --output_path /path/to/output
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到 base 模型（生成式版本）")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="基础模型路径")
    parser.add_argument("--lora_model_path", type=str, required=True,
                        help="LoRA 模型路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出路径")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Merging LoRA Weights (Generative Version)")
    print("="*80)
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA model: {args.lora_model_path}")
    print(f"Output: {args.output_path}")
    print("="*80)
    print("")

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.lora_model_path,
        trust_remote_code=True
    )
    print("✅ Tokenizer loaded\n")

    # 加载 base 模型
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True
    )
    print("✅ Base model loaded\n")

    # 加载 LoRA 模型
    print("Loading LoRA model...")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_model_path,
        torch_dtype=torch.float16
    )
    print("✅ LoRA model loaded\n")

    # 合并权重
    print("Merging weights...")
    model = model.merge_and_unload()
    print("✅ Weights merged\n")

    # 保存合并后的模型
    print("Saving merged model...")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"✅ Merged model saved to: {args.output_path}\n")

    print("="*80)
    print("✅ Merging completed successfully!")
    print("="*80)
    print("")


if __name__ == "__main__":
    main()

