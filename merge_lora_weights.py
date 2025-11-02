#!/usr/bin/env python3
"""
合并 LoRA 权重到 base 模型
将 LoRA Only 训练的最佳模型合并成完整模型，供 CultureMoE 使用
"""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_lora_weights(
    base_model_path: str,
    lora_model_path: str,
    output_path: str,
    device: str = "auto"
):
    """
    合并 LoRA 权重到 base 模型

    Args:
        base_model_path: 基座模型路径
        lora_model_path: LoRA 模型路径（包含 adapter_config.json 和 adapter_model.bin）
        output_path: 输出路径
        device: 设备
    """
    print("="*80)
    print("Merging LoRA Weights to Base Model")
    print("="*80)
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")
    print(f"Output: {output_path}")
    print("="*80)
    print("")

    # 1. 加载 tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载 base model
    print("\n2. Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    print("   ✅ Base model loaded")
    print(f"   Parameters: {sum(p.numel() for p in base_model.parameters()):,}")

    # 3. 加载 LoRA 权重
    print("\n3. Loading LoRA weights...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map=device
    )
    print("   ✅ LoRA weights loaded")

    # 4. 合并权重
    print("\n4. Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    print("   ✅ Weights merged")

    # 5. 保存合并后的模型（保持 FP16 格式）
    print(f"\n5. Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    # ✅ 确保保存为 FP16 格式
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,  # 使用 safetensors 格式
        max_shard_size="5GB"      # 分片保存
    )
    tokenizer.save_pretrained(output_path)

    # 打印保存的模型信息
    print("   ✅ Merged model saved (FP16 format)")
    print(f"      Format: safetensors")
    print(f"      Precision: float16")
    print("   ✅ Merged model saved")

    print("\n" + "="*80)
    print("✅ Merging completed successfully!")
    print("="*80)
    print(f"Merged model saved to: {output_path}")
    print("")
    print("You can now use this merged model for:")
    print("  1. Inference")
    print("  2. CultureMoE training (as frozen LLM)")
    print("="*80)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到 base 模型")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="基座模型路径"
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        required=True,
        help="LoRA 模型路径（包含 adapter_config.json 和 adapter_model.bin）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备（auto/cuda/cpu）"
    )

    args = parser.parse_args()

    # 检查路径
    if not os.path.exists(args.base_model_path):
        print(f"❌ Error: Base model path not found: {args.base_model_path}")
        sys.exit(1)

    if not os.path.exists(args.lora_model_path):
        print(f"❌ Error: LoRA model path not found: {args.lora_model_path}")
        sys.exit(1)

    # 检查 LoRA 模型是否包含必要文件
    adapter_config = os.path.join(args.lora_model_path, "adapter_config.json")
    adapter_model = os.path.join(args.lora_model_path, "adapter_model.bin")

    if not os.path.exists(adapter_config):
        print(f"❌ Error: adapter_config.json not found in {args.lora_model_path}")
        print("   Make sure you're using a LoRA model directory")
        sys.exit(1)

    if not os.path.exists(adapter_model):
        # 尝试查找 safetensors 格式
        adapter_model_safetensors = os.path.join(args.lora_model_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_model_safetensors):
            print(f"❌ Error: adapter_model.bin or adapter_model.safetensors not found in {args.lora_model_path}")
            sys.exit(1)

    # 合并权重
    merge_lora_weights(
        base_model_path=args.base_model_path,
        lora_model_path=args.lora_model_path,
        output_path=args.output_path,
        device=args.device
    )


if __name__ == "__main__":
    main()

