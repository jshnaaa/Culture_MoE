#!/usr/bin/env python3
"""
测试 MoE 模型加载

使用方法：
    python test_moe_loading.py
"""

import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


def test_moe_loading():
    """测试 MoE 模型加载"""

    moe_weights_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2"

    print("="*80)
    print("Testing MoE Model Loading")
    print("="*80)
    print(f"MoE weights path: {moe_weights_path}")
    print("="*80)
    print("")

    # 1. 检查目录
    print("1. Checking directory...")
    if not os.path.exists(moe_weights_path):
        print(f"   ❌ Directory not found: {moe_weights_path}")
        return
    print(f"   ✅ Directory exists")

    # 列出文件
    print(f"   Files in directory:")
    for f in os.listdir(moe_weights_path):
        file_path = os.path.join(moe_weights_path, f)
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"      - {f} ({size:.2f} MB)")
    print("")

    # 2. 加载配置
    print("2. Loading MoE config...")
    config_path = os.path.join(moe_weights_path, "moe_config.json")
    if not os.path.exists(config_path):
        print(f"   ❌ Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        moe_config = json.load(f)

    print(f"   ✅ Config loaded")
    print(f"      Model type: {moe_config.get('model_type', 'N/A')}")
    print(f"      Num experts: {moe_config.get('num_experts', 'N/A')}")
    print(f"      Num classes: {moe_config.get('num_classes', 'N/A')}")
    print(f"      Merged LLM path: {moe_config.get('merged_llm_path', 'N/A')}")
    print("")

    # 3. 检查 merged LLM
    merged_llm_path = moe_config.get('merged_llm_path')
    print(f"3. Checking merged LLM: {merged_llm_path}")
    if not os.path.exists(merged_llm_path):
        print(f"   ❌ Merged LLM not found: {merged_llm_path}")
        return
    print(f"   ✅ Merged LLM exists")
    print("")

    # 4. 加载 tokenizer
    print("4. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(moe_weights_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"   ✅ Tokenizer loaded")
    except Exception as e:
        print(f"   ❌ Failed to load tokenizer: {e}")
        return
    print("")

    # 5. 加载 LLM
    print("5. Loading merged LLM (this may take a while)...")
    try:
        llama_model = AutoModelForCausalLM.from_pretrained(
            merged_llm_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # 冻结 LLM
        for param in llama_model.parameters():
            param.requires_grad = False

        print(f"   ✅ Merged LLM loaded and frozen")
        print(f"      LLM device: {next(llama_model.parameters()).device}")
    except Exception as e:
        print(f"   ❌ Failed to load LLM: {e}")
        import traceback
        traceback.print_exc()
        return
    print("")

    # 6. 创建 MoE 架构
    print("6. Creating MoE architecture...")
    try:
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

        print(f"   ✅ MoE architecture created")
    except Exception as e:
        print(f"   ❌ Failed to create MoE architecture: {e}")
        import traceback
        traceback.print_exc()
        return
    print("")

    # 7. 加载 MoE 权重
    print("7. Loading MoE weights...")
    weights_file = os.path.join(moe_weights_path, "moe_weights.bin")
    if not os.path.exists(weights_file):
        print(f"   ❌ Weights file not found: {weights_file}")
        return

    try:
        moe_state_dict = torch.load(weights_file, map_location='cpu')
        print(f"   ✅ Weights file loaded")
        print(f"      Number of keys: {len(moe_state_dict)}")

        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(moe_state_dict, strict=False)

        # 验证
        llama_missing = [k for k in missing_keys if k.startswith('llama_model.')]
        non_llama_missing = [k for k in missing_keys if not k.startswith('llama_model.')]

        print(f"   ✅ Weights loaded into model")
        print(f"      Missing LLM keys: {len(llama_missing)} (expected)")
        print(f"      Missing MoE keys: {len(non_llama_missing)} (should be 0)")
        print(f"      Unexpected keys: {len(unexpected_keys)}")

        if non_llama_missing:
            print(f"   ⚠️  Warning: Missing MoE keys:")
            for k in non_llama_missing[:5]:
                print(f"         - {k}")
    except Exception as e:
        print(f"   ❌ Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        return
    print("")

    # 8. 移动到设备
    print("8. Moving MoE layers to device...")
    try:
        llm_device = next(llama_model.parameters()).device

        if hasattr(model, 'shared'):
            model.shared = model.shared.to(llm_device)
        if hasattr(model, 'router'):
            model.router = model.router.to(llm_device)
        if hasattr(model, 'experts_layer'):
            model.experts_layer = model.experts_layer.to(llm_device)
        if hasattr(model, 'classifier'):
            model.classifier = model.classifier.to(llm_device)
        if hasattr(model, 'culture_classifier'):
            model.culture_classifier = model.culture_classifier.to(llm_device)

        print(f"   ✅ MoE layers moved to {llm_device}")
    except Exception as e:
        print(f"   ❌ Failed to move layers: {e}")
        import traceback
        traceback.print_exc()
        return
    print("")

    # 9. 设置评估模式
    print("9. Setting evaluation mode...")
    model.eval()
    print(f"   ✅ Model in evaluation mode")
    print("")

    print("="*80)
    print("✅ MoE Model Loading Test Successful!")
    print("="*80)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*80)


if __name__ == "__main__":
    test_moe_loading()

