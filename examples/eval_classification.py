# examples/evaluate_classification.py
# !/usr/bin/env python3
"""
分类任务评估脚本
在测试集上评估训练好的模型
"""

import sys
import os
import torch
import json
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llamafactory.data.classification_processor import load_and_process_classification_data
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs
from src.llamafactory.train.classification.metrics import compute_classification_metrics


def load_model(model_path: str):
    """加载训练好的模型"""
    print(f"Loading model from {model_path}...")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    llama_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 重建 MoE 模型
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

    print(f"Model loaded on: {next(model.parameters()).device}")

    return model, tokenizer


def evaluate_model(
        model,
        tokenizer,
        test_file: str,
        batch_size: int = 16,
        max_length: int = 512,
        output_file: str = None
):
    """
    评估模型

    Args:
        model: 训练好的模型
        tokenizer: tokenizer
        test_file: 测试数据文件
        batch_size: 批次大小
        max_length: 最大序列长度
        output_file: 输出文件路径（可选）
    """
    print(f"\nLoading test data from {test_file}...")

    # 加载测试数据
    data = load_and_process_classification_data(
        data_path=test_file,
        tokenizer=tokenizer,
        max_length=max_length,
        val_split=0,  # 不分割
        num_proc=4
    )

    test_dataset = data['train']  # 因为 val_split=0，所有数据都在 train 中
    print(f"Test dataset size: {len(test_dataset)}")

    # 准备评估
    device = next(model.parameters()).device
    all_predictions = []
    all_labels = []
    all_logits = []

    print(f"\nEvaluating on {len(test_dataset)} samples...")

    # 批量评估
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i:i + batch_size]

        # 准备输入
        input_ids = torch.tensor([item['input_ids'] for item in batch]).to(device)
        attention_mask = torch.tensor([item['attention_mask'] for item in batch]).to(device)
        labels = torch.tensor([item['labels'] for item in batch])

        # 前向传播
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # 获取预测
        preds = torch.argmax(logits, dim=-1).cpu()

        all_predictions.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_logits.extend(logits.cpu().tolist())

    # 转换为 numpy 数组
    import numpy as np
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    # 计算指标
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    metrics = compute_classification_metrics((predictions, labels))

    # 打印主要指标
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:        {metrics['accuracy']:.4f}")
    print(f"   F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"   F1 (weighted):   {metrics['f1_weighted']:.4f}")

    print(f"\n📊 Per-class Metrics:")
    for class_name in ["no", "neutral", "yes"]:
        print(f"\n   {class_name.upper()}:")
        print(f"      Precision: {metrics[f'precision_{class_name}']:.4f}")
        print(f"      Recall:    {metrics[f'recall_{class_name}']:.4f}")
        print(f"      F1:        {metrics[f'f1_{class_name}']:.4f}")

    # 保存结果
    if output_file:
        results = {
            "metrics": metrics,
            "predictions": all_predictions,
            "labels": all_labels,
            "logits": all_logits
        }

        print(f"\n💾 Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ Results saved!")

    print("\n" + "=" * 60)

    return metrics


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="评估分类模型")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型路径")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出文件路径")

    args = parser.parse_args()

    # 检查 GPU
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available!")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

    # 加载模型
    model, tokenizer = load_model(args.model_path)

    # 评估模型
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_file=args.output_file
    )

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()