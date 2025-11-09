#!/usr/bin/env python3
"""
诊断 NaN Loss 问题的脚本

检查以下几个方面：
1. 梯度爆炸
2. Logits 溢出
3. MoE gating 权重异常
4. Loss 计算方式
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_tensor_stats(name, tensor):
    """检查张量的统计信息"""
    if tensor is None:
        print(f"  {name}: None")
        return

    print(f"  {name}:")
    print(f"    Shape: {tensor.shape}")
    print(f"    Dtype: {tensor.dtype}")
    print(f"    Min: {tensor.min().item():.6f}")
    print(f"    Max: {tensor.max().item():.6f}")
    print(f"    Mean: {tensor.mean().item():.6f}")
    print(f"    Std: {tensor.std().item():.6f}")
    print(f"    Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"    Has Inf: {torch.isinf(tensor).any().item()}")

def diagnose_forward_pass(model, batch, device):
    """诊断前向传播"""
    print("\n" + "="*80)
    print("🔍 Diagnosing Forward Pass")
    print("="*80)

    model.eval()

    with torch.no_grad():
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        input_ids_mask = batch['input_ids_mask'].to(device)
        attention_mask_mask = batch['attention_mask_mask'].to(device)
        labels = batch['labels'].to(device)
        culture_labels = batch['culture_labels']

        print("\n1️⃣ Input Tensors:")
        check_tensor_stats("input_ids", input_ids)
        check_tensor_stats("labels", labels)

        # 前向传播
        print("\n2️⃣ Forward Pass:")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids_mask=input_ids_mask,
            attention_mask_mask=attention_mask_mask,
            labels=labels,
            culture_labels=culture_labels,
            use_culture_loss=False
        )

        print("\n3️⃣ Output Tensors:")
        check_tensor_stats("logits", outputs['logits'])
        check_tensor_stats("loss", outputs['generation_loss'])

        # 检查 MoE 权重
        if hasattr(model, '_last_expert_weights'):
            print("\n4️⃣ MoE Expert Weights:")
            check_tensor_stats("expert_weights", model._last_expert_weights)

        # 检查 llama_model 的输出
        print("\n5️⃣ LLaMA Model Outputs:")
        llama_outputs = model.llama_model.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        check_tensor_stats("hidden_states", llama_outputs.last_hidden_state)

        # 检查 lm_head 的输出
        print("\n6️⃣ LM Head Outputs:")
        lm_head_output = model.llama_model.lm_head(llama_outputs.last_hidden_state)
        check_tensor_stats("lm_head_output", lm_head_output)

def diagnose_gradients(model, batch, device):
    """诊断梯度"""
    print("\n" + "="*80)
    print("🔍 Diagnosing Gradients")
    print("="*80)

    model.train()

    # 前向传播
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    input_ids_mask = batch['input_ids_mask'].to(device)
    attention_mask_mask = batch['attention_mask_mask'].to(device)
    labels = batch['labels'].to(device)
    culture_labels = batch['culture_labels']

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_ids_mask=input_ids_mask,
        attention_mask_mask=attention_mask_mask,
        labels=labels,
        culture_labels=culture_labels,
        use_culture_loss=False
    )

    loss = outputs['loss']

    print(f"\n1️⃣ Loss: {loss.item():.6f}")
    print(f"   Has NaN: {torch.isnan(loss).item()}")
    print(f"   Has Inf: {torch.isinf(loss).item()}")

    # 反向传播
    print("\n2️⃣ Backward Pass:")
    loss.backward()

    # 检查梯度
    print("\n3️⃣ Gradient Statistics:")

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm

            if grad_norm > 100:
                print(f"  ⚠️  {name}: {grad_norm:.6f} (LARGE!)")
            elif torch.isnan(torch.tensor(grad_norm)):
                print(f"  ❌ {name}: NaN")
            elif torch.isinf(torch.tensor(grad_norm)):
                print(f"  ❌ {name}: Inf")

    # 统计梯度
    if grad_norms:
        grad_values = list(grad_norms.values())
        print(f"\n4️⃣ Gradient Summary:")
        print(f"   Min: {min(grad_values):.6f}")
        print(f"   Max: {max(grad_values):.6f}")
        print(f"   Mean: {sum(grad_values)/len(grad_values):.6f}")
        print(f"   Num NaN: {sum(1 for v in grad_values if torch.isnan(torch.tensor(v)))}")
        print(f"   Num Inf: {sum(1 for v in grad_values if torch.isinf(torch.tensor(v)))}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="诊断 NaN Loss 问题")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # 加载模型
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
    from src.llamafactory.model.moe_args import ModelArgs
    from train_culturemoe_from_base_gen import load_and_process_generative_data, generative_data_collator
    from functools import partial

    print("Loading model...")

    # 加载 Base 模型
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 加载 LoRA 权重
    tokenizer = AutoTokenizer.from_pretrained(args.lora_weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_with_lora = PeftModel.from_pretrained(
        base_model,
        args.lora_weights_path,
        is_trainable=False,
        torch_dtype=torch.float16
    )

    merged_model = model_with_lora.merge_and_unload()

    # 创建 CultureMoE
    moe_args = ModelArgs(
        num_experts=6,
        shared_hidden_dim=2048,
        router_hidden_dim=1024,
        experts_hidden_dim=2048,
        lora_rank=16,
        classification_hidden_dim=512,
        dropout=0.1,
        num_heads=8
    )

    model = LlamaSharedRouterExpertsModel(
        llama_model=merged_model,
        config=merged_model.config,
        args=moe_args
    )

    model = model.to(args.device)

    # 加载数据
    print("Loading data...")
    datasets = load_and_process_generative_data(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_length=512,
        val_split=0.1,
        use_instruction_mask=True
    )

    train_dataset = datasets['train']
    data_collator = partial(generative_data_collator, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0
    )

    # 获取第一个 batch
    batch = next(iter(train_loader))

    # 诊断
    diagnose_forward_pass(model, batch, args.device)
    diagnose_gradients(model, batch, args.device)

    print("\n" + "="*80)
    print("✅ Diagnosis Complete")
    print("="*80)

if __name__ == "__main__":
    main()

