#!/usr/bin/env python3
"""
合并 LoRA 权重到基础模型，并保存合并后的模型
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

def merge_lora_weights(
    base_model_path: str,
    lora_checkpoint_path: str,
    output_path: str,
    device: str = "cuda:0"
):
    """
    合并 LoRA 权重到基础模型

    Args:
        base_model_path: 原始 LLaMA 模型路径
        lora_checkpoint_path: 包含 LoRA 权重的 checkpoint 路径
        output_path: 合并后模型的保存路径
        device: 设备
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("="*60)
    print("Merging LoRA Weights")
    print("="*60)

    # 1. 加载基础模型
    print(f"\n1. Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True
    )
    base_model = base_model.to(device)
    print(f"   ✅ Base model loaded on {device}")

    # 2. 加载 tokenizer
    print(f"\n2. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_checkpoint_path, trust_remote_code=True)
        print(f"   ✅ Tokenizer loaded from checkpoint")
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print(f"   ✅ Tokenizer loaded from base model")

    # 3. 加载 LoRA 权重
    print(f"\n3. Loading LoRA weights from: {lora_checkpoint_path}")

    # 检查是否有 LoRA 权重
    adapter_config_path = os.path.join(lora_checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"   ❌ Error: adapter_config.json not found in {lora_checkpoint_path}")
        print(f"\n   Available files:")
        for f in os.listdir(lora_checkpoint_path):
            print(f"      - {f}")
        print(f"\n   This checkpoint does not contain LoRA weights!")
        print(f"   Please use a checkpoint that was saved with SaveFullModelCallback.")
        return False

    try:
        peft_model = PeftModel.from_pretrained(
            base_model,
            lora_checkpoint_path,
            is_trainable=False
        )
        print(f"   ✅ LoRA weights loaded")
    except Exception as e:
        print(f"   ❌ Error loading LoRA weights: {e}")
        return False

    # 4. 合并 LoRA 权重
    print(f"\n4. Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()
    print(f"   ✅ LoRA weights merged")

    # 5. 保存合并后的模型
    print(f"\n5. Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    merged_model.save_pretrained(
        output_path,
        safe_serialization=True  # 使用 safetensors 格式
    )
    tokenizer.save_pretrained(output_path)

    print(f"   ✅ Merged model saved")

    # 6. 验证保存的文件
    print(f"\n6. Verifying saved files...")
    saved_files = os.listdir(output_path)
    required_files = ["config.json", "tokenizer_config.json"]

    for f in required_files:
        if f in saved_files:
            print(f"   ✅ {f}")
        else:
            print(f"   ❌ {f} missing")

    # 检查模型权重文件
    if any(f.endswith('.safetensors') or f.endswith('.bin') for f in saved_files):
        print(f"   ✅ Model weights found")
    else:
        print(f"   ❌ Model weights missing")

    print("\n" + "="*60)
    print("✅ Merge completed successfully!")
    print("="*60)
    print(f"\nMerged model saved to: {output_path}")
    print(f"\nYou can now use this model for evaluation:")
    print(f"  MODEL_PATH=\"{output_path}\"")
    print("="*60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the original LLaMA base model"
    )
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint containing LoRA weights"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )

    args = parser.parse_args()

    # 检查输入路径
    if not os.path.exists(args.base_model_path):
        print(f"❌ Error: Base model path does not exist: {args.base_model_path}")
        return 1

    if not os.path.exists(args.lora_checkpoint_path):
        print(f"❌ Error: LoRA checkpoint path does not exist: {args.lora_checkpoint_path}")
        return 1

    # 合并权重
    success = merge_lora_weights(
        base_model_path=args.base_model_path,
        lora_checkpoint_path=args.lora_checkpoint_path,
        output_path=args.output_path,
        device=args.device
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

