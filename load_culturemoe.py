#!/usr/bin/env python3
"""
加载 CultureMoE 模型（从 MoE 权重 + 合并后的 LLM）

使用方法：
    python load_culturemoe.py --moe_weights_path /path/to/moe_weights

示例：
    python load_culturemoe.py \
        --moe_weights_path /root/autodl-fs/model/moe_llama_2 \
        --test_input "Your test input here"
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


def load_culturemoe(moe_weights_path: str, device: str = "cuda"):
    """
    从 MoE 权重加载完整的 CultureMoE 模型

    Args:
        moe_weights_path: MoE 权重目录路径
        device: 设备（cuda 或 cpu）

    Returns:
        model: 完整的 CultureMoE 模型
        tokenizer: Tokenizer
        config: MoE 配置
    """
    print("\n" + "="*80)
    print("Loading CultureMoE Model")
    print("="*80)
    print(f"MoE weights path: {moe_weights_path}")
    print("="*80)
    print("")

    # 1. 加载 MoE 配置
    print("1. Loading MoE configuration...")
    config_path = os.path.join(moe_weights_path, "moe_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"MoE config not found: {config_path}")

    with open(config_path, 'r') as f:
        moe_config = json.load(f)

    print(f"   ✅ MoE config loaded")
    print(f"      Model type: {moe_config['model_type']}")
    print(f"      Architecture: {moe_config['architecture']}")
    print(f"      Num experts: {moe_config['num_experts']}")
    print(f"      Num classes: {moe_config['num_classes']}")
    print(f"      Best accuracy: {moe_config['best_accuracy']:.4f}")

    # 2. 加载 Tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(moe_weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 3. 加载合并后的 LLM
    merged_llm_path = moe_config['merged_llm_path']
    print(f"\n3. Loading merged LLM from {merged_llm_path}...")

    if not os.path.exists(merged_llm_path):
        raise FileNotFoundError(
            f"Merged LLM not found: {merged_llm_path}\n"
            f"Please ensure the merged LLM is available at the specified path."
        )

    llama_model = AutoModelForCausalLM.from_pretrained(
        merged_llm_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # 冻结 LLM
    for param in llama_model.parameters():
        param.requires_grad = False

    print("   ✅ Merged LLM loaded and frozen")
    print(f"      LLM parameters: {sum(p.numel() for p in llama_model.parameters()):,}")

    # 4. 创建 CultureMoE 模型架构
    print("\n4. Creating CultureMoE architecture...")
    moe_args = ModelArgs(
        num_experts=moe_config['num_experts'],
        shared_hidden_dim=moe_config['shared_hidden_dim'],
        router_hidden_dim=moe_config['router_hidden_dim'],
        experts_hidden_dim=moe_config['experts_hidden_dim'],
        lora_rank=moe_config['moe_lora_rank'],
        num_classes=moe_config['num_classes'],
        classification_hidden_dim=moe_config['classification_hidden_dim'],
        dropout=moe_config['dropout'],
        num_heads=moe_config['num_heads']
    )

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    )

    print("   ✅ CultureMoE architecture created")

    # 5. 加载 MoE 权重
    print("\n5. Loading MoE weights...")
    moe_weights_file = os.path.join(moe_weights_path, "moe_weights.bin")
    if not os.path.exists(moe_weights_file):
        raise FileNotFoundError(f"MoE weights not found: {moe_weights_file}")

    moe_state_dict = torch.load(moe_weights_file, map_location="cpu")

    # 加载权重（strict=False 允许只加载 MoE 部分）
    missing_keys, unexpected_keys = model.load_state_dict(moe_state_dict, strict=False)

    # 验证：missing_keys 应该都是 llama_model 的键
    llama_missing = [k for k in missing_keys if k.startswith('llama_model.')]
    non_llama_missing = [k for k in missing_keys if not k.startswith('llama_model.')]

    if non_llama_missing:
        print(f"   ⚠️  Warning: Missing non-LLM keys: {non_llama_missing[:5]}...")

    print("   ✅ MoE weights loaded")
    print(f"      Loaded parameters: {len(moe_state_dict):,}")
    print(f"      Missing LLM keys: {len(llama_missing):,} (expected)")
    print(f"      Missing MoE keys: {len(non_llama_missing):,} (should be 0)")
    print(f"      Unexpected keys: {len(unexpected_keys):,} (should be 0)")

    # 6. 设置为评估模式
    model.eval()

    # 7. 打印模型统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n6. Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")

    print("\n" + "="*80)
    print("✅ CultureMoE Model Loaded Successfully!")
    print("="*80)
    print("")

    return model, tokenizer, moe_config


def test_model(model, tokenizer, test_input: str, num_classes: int):
    """
    测试模型推理

    Args:
        model: CultureMoE 模型
        tokenizer: Tokenizer
        test_input: 测试输入文本
        num_classes: 分类数量
    """
    print("\n" + "="*80)
    print("Testing Model Inference")
    print("="*80)
    print(f"Input: {test_input}")
    print("")

    # Tokenize
    inputs = tokenizer(
        test_input,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )

    # 移动到模型所在设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        logits = outputs  # CultureMoE 直接返回 logits

        # 获取预测
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(logits, dim=-1).item()

    print("Results:")
    print(f"   Predicted class: {pred_class}")
    print(f"   Probabilities:")
    for i in range(num_classes):
        print(f"      Class {i}: {probs[0, i].item():.4f}")

    print("="*80)
    print("")


def main():
    parser = argparse.ArgumentParser(description="加载 CultureMoE 模型")
    parser.add_argument("--moe_weights_path", type=str, required=True,
                        help="MoE 权重目录路径")
    parser.add_argument("--test_input", type=str, default=None,
                        help="测试输入文本（可选）")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="设备")

    args = parser.parse_args()

    # 加载模型
    model, tokenizer, moe_config = load_culturemoe(
        moe_weights_path=args.moe_weights_path,
        device=args.device
    )

    # 测试推理（如果提供了测试输入）
    if args.test_input:
        test_model(
            model=model,
            tokenizer=tokenizer,
            test_input=args.test_input,
            num_classes=moe_config['num_classes']
        )
    else:
        print("💡 Tip: Use --test_input to test model inference")
        print("   Example: python load_culturemoe.py --moe_weights_path /path/to/moe --test_input 'Your text here'")

    return model, tokenizer, moe_config


if __name__ == "__main__":
    model, tokenizer, config = main()

