# examples/predict_classification.py
# !/usr/bin/env python3
"""
分类任务推理脚本
所有操作在 GPU 上进行
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


def load_model(model_path: str, device: str = "cuda"):
    """
    加载训练好的模型
    直接加载到 GPU
    """
    print(f"Loading model from {model_path}...")

    # 检查 GPU 可用性
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please check your GPU setup.")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ✅ 直接加载到 GPU
    llama_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # ✅ 自动分配到 GPU
        trust_remote_code=True
    )

    # 重建 MoE 模型（需要与训练时的参数一致）
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
    )

    model.eval()

    print(f"Model loaded successfully on {next(model.parameters()).device}")

    return model, tokenizer


def predict(model, tokenizer, text: str):
    """
    对单个文本进行预测
    所有操作在 GPU 上
    """
    # ✅ Tokenize 并直接发送到模型所在设备
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)  # ✅ 直接发送到 GPU

    # 推理
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # 获取预测结果（在 GPU 上计算）
    probs = torch.softmax(logits, dim=-1)
    pred_class = torch.argmax(logits, dim=-1).item()
    confidence = probs[0, pred_class].item()

    # 映射到标签
    label_map = {0: "no", 1: "neutral", 2: "yes"}
    pred_label = label_map[pred_class]

    return {
        "label": pred_label,
        "class_id": pred_class,
        "confidence": confidence,
        "probabilities": {
            "no": probs[0, 0].item(),
            "neutral": probs[0, 1].item(),
            "yes": probs[0, 2].item()
        }
    }


def batch_predict(model, tokenizer, texts: list, batch_size: int = 16):
    """
    批量预测
    所有操作在 GPU 上
    """
    results = []
    device = next(model.parameters()).device

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # ✅ Tokenize 并直接发送到 GPU
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        # 推理
        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        # 处理结果（在 GPU 上）
        probs = torch.softmax(logits, dim=-1)
        pred_classes = torch.argmax(logits, dim=-1)

        label_map = {0: "no", 1: "neutral", 2: "yes"}

        for j in range(len(batch_texts)):
            pred_class = pred_classes[j].item()
            results.append({
                "text": batch_texts[j],
                "label": label_map[pred_class],
                "class_id": pred_class,
                "confidence": probs[j, pred_class].item(),
                "probabilities": {
                    "no": probs[j, 0].item(),
                    "neutral": probs[j, 1].item(),
                    "yes": probs[j, 2].item()
                }
            })

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="分类任务推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--text", type=str, help="要预测的文本")
    parser.add_argument("--input_file", type=str, help="输入文件（每行一个样本）")
    parser.add_argument("--output_file", type=str, help="输出文件")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")

    args = parser.parse_args()

    # 检查 GPU
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA is not available! This script requires GPU.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB\n")

    # 加载模型
    model, tokenizer = load_model(args.model_path)

    if args.text:
        # 单个文本预测
        print(f"Input text: {args.text}\n")
        result = predict(model, tokenizer, args.text)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")

    elif args.input_file:
        # 批量预测
        print(f"Reading from {args.input_file}...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"Predicting {len(texts)} samples...")
        results = batch_predict(model, tokenizer, texts, args.batch_size)

        # 保存结果
        if args.output_file:
            import json
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output_file}")
        else:
            # 打印结果
            for i, result in enumerate(results):
                print(f"\n[{i + 1}] {result['text'][:50]}...")
                print(f"    Prediction: {result['label']} (confidence: {result['confidence']:.4f})")

    else:
        print("Please provide either --text or --input_file")


if __name__ == "__main__":
    main()