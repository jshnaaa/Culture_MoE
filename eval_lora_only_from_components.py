#!/usr/bin/env python3
"""
从 Base 模型 + LoRA 权重还原模型并评估

使用方法：
    python eval_lora_only_from_components.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --test_file /path/to/test_data.json \
        --output_dir /path/to/output \
        --num_classes 2
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_model_from_components(
    base_model_path: str,
    lora_weights_path: str,
    device: str = "cuda"
):
    """
    从 Base 模型 + LoRA 权重还原完整模型

    Args:
        base_model_path: Base 模型路径
        lora_weights_path: LoRA 权重路径
        device: 设备

    Returns:
        model: 合并后的完整模型
        tokenizer: Tokenizer
    """
    print("\n" + "="*80)
    print("Loading Model from Components")
    print("="*80)
    print(f"Base model: {base_model_path}")
    print(f"LoRA weights: {lora_weights_path}")
    print("="*80)
    print("")

    # 1. 加载 Tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载 Base 模型
    print("\n2. Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("   ✅ Base model loaded")
    print(f"   Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")

    # 3. 加载 LoRA 权重并合并
    print("\n3. Loading and merging LoRA weights...")
    from peft import PeftModel

    # 加载 LoRA
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        is_trainable=False
    )
    print("   ✅ LoRA weights loaded")

    # 合并 LoRA 权重到 base 模型
    print("   Merging LoRA weights into base model...")
    merged_model = model_with_lora.merge_and_unload()
    print("   ✅ LoRA weights merged")

    # 4. 移动到设备
    print(f"\n4. Moving model to {device}...")
    merged_model = merged_model.to(device)
    merged_model.eval()
    print("   ✅ Model ready")

    # 打印模型信息
    total_params = sum(p.numel() for p in merged_model.parameters())
    print(f"\n📊 Model Info:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Device: {device}")

    print("\n" + "="*80)
    print("✅ Model Loaded and Merged Successfully!")
    print("="*80)
    print("")

    return merged_model, tokenizer


class ClassificationModel(torch.nn.Module):
    """带分类头的模型"""

    def __init__(self, llama_model, num_classes=2):
        super().__init__()
        self.llama_model = llama_model
        self.num_classes = num_classes

        # 分类头
        hidden_size = llama_model.config.hidden_size
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 获取 LLaMA 输出
        outputs = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 使用最后一层的 [CLS] token（第一个 token）
        hidden_states = outputs.hidden_states[-1]
        cls_hidden = hidden_states[:, 0, :]  # [batch_size, hidden_size]

        # 分类
        logits = self.classifier(cls_hidden)

        return logits


def load_test_data(data_path: str, tokenizer, max_length: int = 512):
    """加载测试数据"""
    print(f"Loading test data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} test samples")

    processed_data = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item['output']

        # 组合文本
        if input_text and input_text.strip():
            full_text = f"{instruction}\n{input_text}"
        else:
            full_text = instruction

        # Tokenize
        encoded = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        processed_data.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': output
        })

    return processed_data


def evaluate_model(model, tokenizer, test_dataset, num_classes: int, batch_size: int = 8, device: str = "cuda"):
    """评估模型"""
    print(f"\nEvaluating on {len(test_dataset)} samples...")

    # 创建分类模型
    classification_model = ClassificationModel(model, num_classes).to(device)
    classification_model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch = test_dataset[i:i+batch_size]

            # 准备批次数据
            input_ids = [item['input_ids'] for item in batch]
            attention_mask = [item['attention_mask'] for item in batch]
            labels = [item['label'] for item in batch]

            # Padding
            max_len = max(len(ids) for ids in input_ids)
            input_ids_padded = []
            attention_mask_padded = []

            for ids, mask in zip(input_ids, attention_mask):
                padding_length = max_len - len(ids)
                input_ids_padded.append(ids + [tokenizer.pad_token_id] * padding_length)
                attention_mask_padded.append(mask + [0] * padding_length)

            # 转换为 tensor
            input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long).to(device)
            attention_mask_tensor = torch.tensor(attention_mask_padded, dtype=torch.long).to(device)

            # 前向传播
            logits = classification_model(input_ids_tensor, attention_mask_tensor)

            # 获取预测
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    # 计算指标
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    accuracy = accuracy_score(labels, predictions)

    # 根据类别数选择平均方式
    if num_classes == 2:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', pos_label=1, zero_division=0
        )
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )

    # 混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

    # 每个类别的指标
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(num_classes)), zero_division=0
    )

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(labels),
        'num_classes': num_classes,
        'per_class_metrics': {
            f'class_{i}': {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i]),
                'support': int(per_class_support[i])
            }
            for i in range(num_classes)
        }
    }

    return results


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"Samples:   {results['num_samples']}")
    print("="*80)

    # 打印混淆矩阵
    print("\n📊 Confusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    num_classes = results['num_classes']

    # 打印表头
    print("\n           Predicted")
    header = "           " + "  ".join([f"C{i}" for i in range(num_classes)])
    print(header)

    # 打印每一行
    for i in range(num_classes):
        row = f"Actual C{i}  " + "  ".join([f"{cm[i][j]:3d}" for j in range(num_classes)])
        print(row)

    # 打印每个类别的指标
    print("\n📊 Per-Class Metrics:")
    for i in range(num_classes):
        metrics = results['per_class_metrics'][f'class_{i}']
        print(f"\nClass {i}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Support:   {metrics['support']}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="从组件加载模型并评估")

    # 模型路径
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Base 模型路径")
    parser.add_argument("--lora_weights_path", type=str, required=True,
                        help="LoRA 权重路径（best_lora 目录）")

    # 数据和评估
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="评估结果输出目录")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="分类数量")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")

    args = parser.parse_args()

    # 检查路径
    if not os.path.exists(args.base_model_path):
        print(f"❌ Error: Base model not found: {args.base_model_path}")
        sys.exit(1)

    if not os.path.exists(args.lora_weights_path):
        print(f"❌ Error: LoRA weights not found: {args.lora_weights_path}")
        sys.exit(1)

    if not os.path.exists(args.test_file):
        print(f"❌ Error: Test file not found: {args.test_file}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LoRA Only Model Evaluation (From Components)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.lora_weights_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    print("")

    # 加载模型
    model, tokenizer = load_model_from_components(
        args.base_model_path,
        args.lora_weights_path,
        args.device
    )

    # 加载测试数据
    test_dataset = load_test_data(args.test_file, tokenizer, args.max_length)

    # 评估
    results = evaluate_model(
        model, tokenizer, test_dataset,
        args.num_classes, args.batch_size, args.device
    )

    # 打印结果
    print_results(results)

    # 保存详细结果
    output_data = {
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_model_path': args.base_model_path,
        'lora_weights_path': args.lora_weights_path,
        'test_file': args.test_file,
        'num_classes': args.num_classes,
        'results': results
    }

    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed results saved to: {results_file}")

    # 保存摘要
    summary = {
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'LoRA_Only_From_Components',
        'base_model': args.base_model_path,
        'lora_weights': args.lora_weights_path,
        'num_classes': args.num_classes,
        'num_samples': results['num_samples'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1']
    }

    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Summary saved to: {summary_file}")
    print("")


if __name__ == "__main__":
    main()

