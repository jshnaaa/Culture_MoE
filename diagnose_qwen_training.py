#!/usr/bin/env python3
"""
Qwen 训练问题诊断脚本

用法：
    python diagnose_qwen_training.py \
        --model_path /path/to/qwen \
        --data_path /path/to/data.json
"""

import argparse
import json

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM


def check_lora_injection(model):
    """Step 1: 检查 LoRA 是否真正注入"""
    print("\n" + "="*80)
    print("Step 1: Checking LoRA Injection")
    print("="*80)

    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Frozen parameters: {len(frozen_params)}")

    if trainable_params:
        print("\n✅ LoRA layers found:")
        for name in trainable_params[:10]:  # 只打印前 10 个
            print(f"   {name}")
        if len(trainable_params) > 10:
            print(f"   ... and {len(trainable_params) - 10} more")

        # 检查是否包含 lora_A 和 lora_B
        has_lora_a = any('lora_A' in name for name in trainable_params)
        has_lora_b = any('lora_B' in name for name in trainable_params)

        if has_lora_a and has_lora_b:
            print("\n✅ LoRA injection successful (found lora_A and lora_B)")
            return True
        else:
            print("\n⚠️  Warning: LoRA layers found but no lora_A/lora_B")
            return False
    else:
        print("\n❌ No trainable parameters found!")
        print("   This means LoRA was not injected successfully.")
        print("   Loss will remain constant at ~13.")
        return False


def check_labels(tokenizer, data_path, max_length=512):
    """Step 2: 确认标签是否正确"""
    print("\n" + "="*80)
    print("Step 2: Checking Labels")
    print("="*80)

    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 取第一个样本
    sample = data[0]
    instruction = sample['instruction']
    input_text = sample['input']
    output = str(sample['output'])

    # 构建 prompt
    prompt = f"{instruction}\n{input_text}\nAnswer:"
    full_text = f"{prompt} {output}"

    # Tokenize
    prompt_tokens = tokenizer(prompt, add_special_tokens=True, truncation=False)
    answer_tokens = tokenizer(output, add_special_tokens=False, truncation=False)

    # 合并
    full_input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
    full_attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"]

    # 创建 labels
    prompt_len = len(prompt_tokens["input_ids"])
    labels = [-100] * prompt_len + answer_tokens["input_ids"]

    print(f"Sample:")
    print(f"   Instruction: {instruction[:50]}...")
    print(f"   Input: {input_text[:50]}...")
    print(f"   Output: {output}")
    print(f"\nTokenization:")
    print(f"   Prompt length: {prompt_len}")
    print(f"   Answer length: {len(answer_tokens['input_ids'])}")
    print(f"   Total length: {len(full_input_ids)}")
    print(f"\nFirst 20 input_ids: {full_input_ids[:20]}")
    print(f"First 20 labels: {labels[:20]}")

    # 检查 labels 是否全是 -100
    non_ignore_labels = [l for l in labels if l != -100]
    if not non_ignore_labels:
        print("\n❌ All labels are -100!")
        print("   This means there's no supervision signal.")
        print("   Loss will remain constant at ~13.")
        return False
    else:
        print(f"\n✅ Found {len(non_ignore_labels)} non-ignore labels")
        print(f"   Non-ignore labels: {non_ignore_labels[:10]}")
        return True


def check_optimizer_lr(model, learning_rate=1e-4):
    """Step 3: 检查优化器学习率"""
    print("\n" + "="*80)
    print("Step 3: Checking Optimizer Learning Rate")
    print("="*80)

    # 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Optimizer learning rate: {current_lr}")

    if current_lr == 0:
        print("\n❌ Learning rate is 0!")
        print("   This means the scheduler is not initialized correctly.")
        return False
    else:
        print(f"\n✅ Learning rate is correct: {current_lr}")
        return True


def check_gradients(model, tokenizer, data_path, max_length=512):
    """Step 4: 测试梯度是否为 NaN"""
    print("\n" + "="*80)
    print("Step 4: Checking Gradients")
    print("="*80)

    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 取第一个样本
    sample = data[0]
    instruction = sample['instruction']
    input_text = sample['input']
    output = str(sample['output'])

    # 构建 prompt
    prompt = f"{instruction}\n{input_text}\nAnswer:"

    # Tokenize
    prompt_tokens = tokenizer(prompt, add_special_tokens=True, truncation=False)
    answer_tokens = tokenizer(output, add_special_tokens=False, truncation=False)

    # 合并
    full_input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]

    # 创建 labels
    prompt_len = len(prompt_tokens["input_ids"])
    labels = [-100] * prompt_len + answer_tokens["input_ids"]

    # 转换为 tensor
    input_ids = torch.tensor([full_input_ids]).to(model.device)
    labels_tensor = torch.tensor([labels]).to(model.device)

    # 前向传播
    model.train()
    outputs = model(input_ids=input_ids, labels=labels_tensor)
    loss = outputs.loss

    print(f"Loss: {loss.item()}")

    # 反向传播
    loss.backward()

    # 检查梯度
    nan_gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            nan_gradients.append(name)

    if nan_gradients:
        print(f"\n❌ Found {len(nan_gradients)} parameters with NaN gradients:")
        for name in nan_gradients[:10]:
            print(f"   {name}")
        if len(nan_gradients) > 10:
            print(f"   ... and {len(nan_gradients) - 10} more")
        return False
    else:
        print("\n✅ No NaN gradients found")

        # 打印一些梯度的范数
        print("\nGradient norms (first 5 trainable parameters):")
        count = 0
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_norm = param.grad.norm().item()
                print(f"   {name}: {grad_norm:.6f}")
                count += 1
                if count >= 5:
                    break

        return True


def check_config(model):
    """Step 5: 重新确认 config 与 checkpoint 一致"""
    print("\n" + "="*80)
    print("Step 5: Checking Model Config")
    print("="*80)

    config = model.config

    print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
    print(f"RoPE scaling: {getattr(config, 'rope_scaling', None)}")
    print(f"RMS norm eps: {getattr(config, 'rms_norm_eps', None)}")
    print(f"Max position embeddings: {getattr(config, 'max_position_embeddings', None)}")
    print(f"Pad token id: {getattr(config, 'pad_token_id', None)}")
    print(f"EOS token id: {getattr(config, 'eos_token_id', None)}")

    # 检查 rope_scaling
    rope_scaling = getattr(config, 'rope_scaling', None)
    if rope_scaling is not None:
        print(f"\n⚠️  RoPE scaling is enabled: {rope_scaling}")
        print("   This may cause position encoding issues.")
    else:
        print("\n✅ No RoPE scaling")

    # 检查 pad_token_id
    pad_token_id = getattr(config, 'pad_token_id', None)
    if pad_token_id is None:
        print("\n❌ pad_token_id is None!")
        print("   This will cause loss calculation issues.")
        return False
    else:
        print(f"\n✅ pad_token_id is set: {pad_token_id}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Qwen 训练问题诊断")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据路径")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Qwen Training Diagnosis")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print("="*80)

    # 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")

    # 加载模型
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    print("✅ Model loaded")

    # 设置 pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        print(f"✅ Set model.config.pad_token_id = {model.config.pad_token_id}")

    # 检测模型类型
    model_type = model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
    print(f"Model type: {model_type}")

    # 配置 LoRA
    print("\nConfiguring LoRA...")
    if model_type == 'qwen':
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            init_lora_weights="gaussian"
        )
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )

    model = get_peft_model(model, lora_config)
    print("✅ LoRA configured")

    # 运行诊断
    results = {}

    # Step 1: 检查 LoRA 注入
    results['lora_injection'] = check_lora_injection(model)

    # Step 2: 检查标签
    results['labels'] = check_labels(tokenizer, args.data_path)

    # Step 3: 检查优化器学习率
    results['optimizer_lr'] = check_optimizer_lr(model)

    # Step 4: 检查梯度
    results['gradients'] = check_gradients(model, tokenizer, args.data_path)

    # Step 5: 检查配置
    results['config'] = check_config(model)

    # 总结
    print("\n" + "="*80)
    print("Diagnosis Summary")
    print("="*80)

    all_passed = all(results.values())

    for step, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{step.replace('_', ' ').title()}: {status}")

    if all_passed:
        print("\n✅ All checks passed!")
        print("   Your configuration should work correctly.")
    else:
        print("\n❌ Some checks failed!")
        print("   Please fix the issues above before training.")

    print("="*80)


if __name__ == "__main__":
    main()

