#!/usr/bin/env python3
"""
灵活的 Base 模型评估（支持不同类别数和标签名称）
支持 LLaMA 3.1 和 Qwen 2.5
适用于 WVS 等标签不统一的数据集
"""

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_label_info_from_text(text: str):
    """
    从文本中提取标签信息

    例如：
    "1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree"
    返回：["Strongly agree", "agree", "Disagree", "Strongly disagree"]
    """
    # 先移除干扰文本
    text = re.sub(r'You can only choose one option[.\s]*', '', text)
    text = re.sub(r'###[.\s]*', '', text)

    # 匹配模式：数字. 文本（直到下一个数字或结束）
    pattern = r'(\d+)\.\s*([^0-9]+?)(?=\s*\d+\.|$)'
    matches = re.findall(pattern, text)

    if matches:
        # 清理标签文本
        labels = []
        for num, label_text in matches:
            # 移除末尾的标点和空格
            cleaned = label_text.strip().rstrip('.,;:\n')
            if cleaned:  # 确保不是空字符串
                labels.append(cleaned)
        return labels if labels else None

    return None


def load_wvs_data(data_path: str, tokenizer, max_length: int = 512):
    """
    加载 WVS 数据集

    数据格式：
    {
        "instruction": "Question: ... Options: 1. Strongly agree 2. agree ...",
        "input": "...",
        "output": 0,  # 0-based index
        "label": "0"
    }
    """
    print(f"Loading data from {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # 处理数据
    processed_data = []
    label_info_stats = defaultdict(int)

    for item in data:
        instruction = item['instruction']
        input_text = item.get('input', '')
        output = item['output']

        # ✅ 优先从 input 字段提取标签，如果没有则从 instruction 提取
        label_options = None
        if input_text:
            label_options = extract_label_info_from_text(input_text)

        if label_options is None and instruction:
            label_options = extract_label_info_from_text(instruction)

        if label_options is None:
            print(f"Warning: Cannot extract labels from instruction or input")
            print(f"  Instruction: {instruction[:100]}...")
            print(f"  Input: {input_text[:100]}...")
            continue

        num_classes = len(label_options)
        label_info_stats[num_classes] += 1

        # 组合文本
        if input_text:
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
            'label': output,
            'label_options': label_options,
            'num_classes': num_classes,
            'instruction': instruction,
            'input': input_text
        })

    print(f"\nLabel distribution:")
    for num_classes, count in sorted(label_info_stats.items()):
        print(f"  {num_classes}-class: {count} samples ({count/len(processed_data)*100:.1f}%)")

    return processed_data


def evaluate_base_llama_flexible(
    model_path: str,
    test_file: str,
    batch_size: int = 8,
    max_length: int = 512,
    output_file: str = None,
    device: str = "cuda:0",
    backbone: str = "llama"
):
    """
    灵活评估 Base 模型（支持不同类别数和标签名称）

    Args:
        backbone: 基座模型类型，"llama" 或 "qwen"
    """
    print("="*60)
    if backbone == "qwen":
        print("Evaluating Base Qwen 2.5 Model (Flexible Labels)")
    else:
        print("Evaluating Base LLaMA 3.1 Model (Flexible Labels)")
    print("="*60)

    # 1. 加载 tokenizer
    print(f"\n1. Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载模型
    print(f"\n2. Loading base LLaMA model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map=None,
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    print(f"   ✅ Model loaded on {device}")

    # 3. 加载测试数据
    print(f"\n3. Loading test data from {test_file}...")
    test_data = load_wvs_data(test_file, tokenizer, max_length)
    print(f"   ✅ Test dataset size: {len(test_data)}")

    # 4. 评估
    print(f"\n4. Evaluating on {len(test_data)} samples...")

    all_predictions = []
    all_labels = []
    all_num_classes = []
    all_correct = []

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch_data = test_data[i:i+batch_size]

        # 准备输入
        from torch.nn.utils.rnn import pad_sequence

        input_ids_list = [torch.tensor(item['input_ids']) for item in batch_data]
        attention_mask_list = [torch.tensor(item['attention_mask']) for item in batch_data]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

        labels = [item['label'] for item in batch_data]
        label_options_batch = [item['label_options'] for item in batch_data]
        num_classes_batch = [item['num_classes'] for item in batch_data]

        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # 获取下一个 token 的 logits
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            # 对每个样本单独处理（因为标签不同）
            for j, (label_options, true_label, num_classes) in enumerate(zip(label_options_batch, labels, num_classes_batch)):
                # 获取每个标签选项的 token ID
                label_token_ids = []
                for option in label_options:
                    # 尝试多种方式编码标签
                    # 1. 直接编码
                    tokens = tokenizer.encode(option, add_special_tokens=False)
                    if len(tokens) > 0:
                        label_token_ids.append(tokens[0])
                    else:
                        # 2. 尝试小写
                        tokens = tokenizer.encode(option.lower(), add_special_tokens=False)
                        if len(tokens) > 0:
                            label_token_ids.append(tokens[0])
                        else:
                            # 3. 使用默认值
                            label_token_ids.append(0)

                # 提取对应的 logits
                sample_logits = logits[j]  # [vocab_size]
                class_logits = torch.tensor([sample_logits[tid].item() for tid in label_token_ids])

                # 预测
                pred = torch.argmax(class_logits).item()

                all_predictions.append(pred)
                all_labels.append(true_label)
                all_num_classes.append(num_classes)
                all_correct.append(pred == true_label)

    # 5. 计算指标
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    num_classes_array = np.array(all_num_classes)

    # 总体准确率
    overall_accuracy = accuracy_score(labels, predictions)
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy: {overall_accuracy:.4f}")

    # 按类别数分组统计
    print(f"\n📊 Accuracy by Number of Classes:")
    for nc in sorted(set(num_classes_array)):
        mask = num_classes_array == nc
        if mask.sum() > 0:
            acc = accuracy_score(labels[mask], predictions[mask])
            print(f"   {nc}-class: {acc:.4f} ({mask.sum()} samples)")

    # 6. 保存结果
    if output_file:
        results = {
            "model": "base_llama_3.1_8b_flexible",
            "overall_accuracy": float(overall_accuracy),
            "predictions": all_predictions,
            "labels": all_labels,
            "num_classes": all_num_classes,
            "correct": all_correct
        }

        print(f"\n💾 Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Results saved!")

    print("\n" + "="*60)

    return {
        "overall_accuracy": overall_accuracy
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="灵活评估 Base LLaMA 3.1 模型")
    parser.add_argument("--model_path", type=str, required=True,
                        help="LLaMA 模型路径")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    parser.add_argument("--backbone", type=str, default="llama", choices=["llama", "qwen"],
                        help="基座模型类型：llama 或 qwen")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("⚠️  Warning: CUDA not available, using CPU")
        args.device = "cpu"

    if args.device.startswith("cuda"):
        print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

    metrics = evaluate_base_llama_flexible(
        model_path=args.model_path,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_file=args.output_file,
        device=args.device,
        backbone=args.backbone
    )

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

