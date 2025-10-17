# examples/test_classification.py
# !/usr/bin/env python3
"""
分类任务测试脚本
测试数据加载、模型前向传播、损失计算
"""

import sys
import os
import torch
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llamafactory.data.classification_processor import load_and_process_classification_data
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


def test_data_loading(data_file: str):
    """测试数据加载"""
    print("=" * 60)
    print("Test 1: Data Loading and Processing")
    print("=" * 60)

    # 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    print(f"\nLoading data from {data_file}...")
    data = load_and_process_classification_data(
        data_path=data_file,
        tokenizer=tokenizer,
        max_length=512,
        val_split=0.1,
        num_proc=4
    )

    train_dataset = data['train']
    val_dataset = data['validation']

    print(f"\n✅ Train dataset size: {len(train_dataset)}")
    print(f"✅ Validation dataset size: {len(val_dataset)}")

    # 查看第一个样本
    sample = train_dataset[0]
    print(f"\n✅ First sample:")
    print(f"   - input_ids length: {len(sample['input_ids'])}")
    print(f"   - attention_mask length: {len(sample['attention_mask'])}")
    print(f"   - label: {sample['labels']} (type: {type(sample['labels'])})")

    # 解码查看文本
    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"\n✅ Decoded text (first 200 chars):")
    print(f"   {decoded_text[:200]}...")

    # 统计标签分布
    train_labels = [sample['labels'] for sample in train_dataset]
    val_labels = [sample['labels'] for sample in val_dataset]

    print(f"\n✅ Train label distribution:")
    label_names = {0: "no", 1: "neutral", 2: "yes"}
    for label, count in sorted(Counter(train_labels).items()):
        label_name = label_names.get(label, "unknown")
        print(f"   {label_name} ({label}): {count} ({count / len(train_labels) * 100:.1f}%)")

    print(f"\n✅ Validation label distribution:")
    for label, count in sorted(Counter(val_labels).items()):
        label_name = label_names.get(label, "unknown")
        print(f"   {label_name} ({label}): {count} ({count / len(val_labels) * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("✅ Test 1 PASSED: Data loading successful")
    print("=" * 60 + "\n")

    return data


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 60)
    print("Test 2: Model Forward Pass")
    print("=" * 60)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Using device: {device}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建测试数据
    test_texts = [
        "Do you agree with this statement?",
        "What is your opinion on this matter?",
    ]

    inputs = tokenizer(
        test_texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)  # Ensure inputs are on the same device as the model

    # Ensure input_ids are LongTensor (important for embedding layer)
    inputs['input_ids'] = inputs['input_ids'].long()

    # 将所有输入数据都转换为 float16（但保持 input_ids 为 long 类型）
    inputs = {key: value.to(torch.float16) if key != 'input_ids' else value for key, value in inputs.items()}

    # 加载模型
    llama_model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    ).to(device)  # Ensure model is on the same device

    # 创建 MoE 模型
    moe_args = ModelArgs(
        num_experts=6,
        shared_hidden_dim=2048,
        router_hidden_dim=1024,
        experts_hidden_dim=2048,
        lora_rank=16,
        num_classes=3,
        classification_hidden_dim=512,
        dropout=0.1,
        num_heads=8
    )

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    ).to(device)  # Ensure model is on the same device

    # 将整个模型转换为 float16
    model = model.half()

    model.eval()

    print(f"✅ Model loaded on: {next(model.parameters()).device}")

    # 前向传播
    print("\nRunning forward pass...")
    with torch.no_grad():
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    print(f"✅ Output shape: {logits.shape}")
    print(f"✅ Expected shape: [batch_size={len(test_texts)}, num_classes=3]")
    print(f"✅ Output device: {logits.device}")

    # 计算预测
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)

    label_map = {0: "no", 1: "neutral", 2: "yes"}

    print(f"\n✅ Predictions:")
    for i, text in enumerate(test_texts):
        print(f"\n   Text: {text}")
        print(f"   Prediction: {label_map[preds[i].item()]}")
        print(f"   Probabilities:")
        print(f"     - no:      {probs[i, 0].item():.4f}")
        print(f"     - neutral: {probs[i, 1].item():.4f}")
        print(f"     - yes:     {probs[i, 2].item():.4f}")

    print("\n" + "=" * 60)
    print("✅ Test 2 PASSED: Model forward pass successful")
    print("=" * 60 + "\n")


def test_loss_computation():
    """测试损失计算"""
    print("=" * 60)
    print("Test 3: Loss Computation")
    print("=" * 60)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0")
    print(f"\n✅ Using GPU: {torch.cuda.get_device_name(0)}")

    # 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建测试数据
    test_texts = ["Test text 1", "Test text 2", "Test text 3"]
    test_labels = [0, 1, 2]  # no, neutral, yes

    inputs = tokenizer(
        test_texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    # Ensure input_ids are LongTensor (important for embedding layer)
    inputs['input_ids'] = inputs['input_ids'].long()

    # 将所有输入数据都转换为 float16（但保持 input_ids 为 long 类型）
    inputs = {key: value.to(torch.float16) if key != 'input_ids' else value for key, value in inputs.items()}

    labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    print(f"✅ Input shape: {inputs['input_ids'].shape}")
    print(f"✅ Labels: {labels.tolist()}")
    print(f"✅ Labels device: {labels.device}")

    # 加载模型
    print("\nLoading model...")
    llama_model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )

    moe_args = ModelArgs(
        num_experts=6,
        shared_hidden_dim=2048,
        router_hidden_dim=1024,
        experts_hidden_dim=2048,
        lora_rank=16,
        num_classes=3,
        classification_hidden_dim=512,
        dropout=0.1,
        num_heads=8
    )

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    ).to(device)

    # 将整个模型转换为 float16
    model = model.half()

    model.train()

    # 前向传播
    print("\nRunning forward pass...")
    logits = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )

    print(f"✅ Logits shape: {logits.shape}")
    print(f"✅ Logits device: {logits.device}")

    # 计算交叉熵损失
    print("\nComputing CrossEntropyLoss...")
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits, labels)

    print(f"✅ Loss: {loss.item():.4f}")
    print(f"✅ Loss device: {loss.device}")
    print(f"✅ Loss requires_grad: {loss.requires_grad}")

    # 反向传播
    print("\nRunning backward pass...")
    loss.backward()

    # 检查梯度
    print("\nChecking gradients...")
    grad_count = 0
    total_grad_norm = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm

            if grad_count <= 5:  # 只打印前5个
                print(f"   {name[:50]:50s} grad_norm={grad_norm:.4f}")

    print(f"\n✅ Total parameters with gradients: {grad_count}")
    print(f"✅ Average gradient norm: {total_grad_norm / grad_count:.4f}")

    print("\n" + "=" * 60)
    print("✅ Test 3 PASSED: Loss computation and backprop successful")
    print("=" * 60 + "\n")


def main():
    """运行所有测试"""
    import argparse

    parser = argparse.ArgumentParser(description="测试分类任务")
    parser.add_argument("--data_file", type=str,
                        default="/root/autodl-fs/normad_ed.json",
                        help="测试数据文件路径")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "data", "model", "loss"],
                        help="要运行的测试")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Classification Task Testing")
    print("=" * 60 + "\n")

    try:
        if args.test in ["all", "data"]:
            test_data_loading(args.data_file)

        if args.test in ["all", "model"]:
            test_model_forward()

        if args.test in ["all", "loss"]:
            test_loss_computation()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()