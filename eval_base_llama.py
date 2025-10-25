#!/usr/bin/env python3
"""
直接使用 Base LLaMA 3.1 模型评估（无训练，无 MoE）
全部数据作为测试集，二分类任务
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data


def evaluate_base_llama(
    model_path: str,
    test_file: str,
    batch_size: int = 8,
    max_length: int = 512,
    output_file: str = None,
    device: str = "cuda:0",
    use_dual_input: bool = True
):
    """
    使用 Base LLaMA 3.1 模型评估

    数据格式：
    {
        "instruction": "...",
        "instruction_mask": "...",
        "input": "...",
        "output": 0,  # 整数：0, 1, 2
        "label": "0"  # 字符串：文化维度标签
    }

    Args:
        model_path: LLaMA 模型路径
        test_file: 测试数据文件
        batch_size: 批次大小
        max_length: 最大序列长度
        output_file: 输出文件路径
        device: 设备
        use_dual_input: 是否使用双路输入
    """
    print("="*60)
    print("Evaluating Base LLaMA 3.1 Model (No Training, No MoE)")
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

    # 3. 加载测试数据（使用相同的数据处理器）
    print(f"\n3. Loading test data from {test_file}...")
    print(f"   Data format: instruction/instruction_mask/input/output/label")
    data = load_and_process_dual_classification_data(
        data_path=test_file,
        tokenizer=tokenizer,
        max_length=max_length,
        val_split=0,  # 全部作为测试集
        num_proc=4
    )
    test_dataset = data['train']
    print(f"   ✅ Test dataset size: {len(test_dataset)}")

    # 4. 评估
    print(f"\n4. Evaluating on {len(test_dataset)} samples...")
    print(f"   Using {'dual input' if use_dual_input else 'single input'} mode")

    all_predictions = []
    all_labels = []
    all_logits = []
    all_culture_labels = []

    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch_indices = list(range(i, min(i + batch_size, len(test_dataset))))
        batch = test_dataset.select(batch_indices)

        # 准备输入
        from torch.nn.utils.rnn import pad_sequence

        # 使用第一路输入（instruction + input）
        input_ids_list = [torch.tensor(item) for item in batch['input_ids']]
        attention_mask_list = [torch.tensor(item) for item in batch['attention_mask']]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

        labels = torch.tensor(batch['labels'])
        culture_labels = batch['culture_labels']  # List[List[int]]

        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # 使用最后一层的隐藏状态
            hidden_states = outputs.hidden_states[-1]  # [B, L, H]

            # 使用最后一个 token 的表示
            last_hidden = hidden_states[:, -1, :]  # [B, H]

            # 简单的二分类：使用线性投影
            # 注意：这是一个简化的方法，实际上 Base LLaMA 没有分类头
            # 我们使用隐藏状态的均值作为特征
            pooled = hidden_states.mean(dim=1)  # [B, H]

            # 使用一个简单的启发式方法：
            # 计算 "yes" 和 "no" token 的概率
            yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = tokenizer.encode("no", add_special_tokens=False)[0]

            # 获取下一个 token 的 logits
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            # 提取 yes 和 no 的 logits
            yes_logits = logits[:, yes_token_id]  # [B]
            no_logits = logits[:, no_token_id]   # [B]

            # 组合成二分类 logits
            binary_logits = torch.stack([no_logits, yes_logits], dim=1)  # [B, 2]

            # 预测
            preds = torch.argmax(binary_logits, dim=-1).cpu()  # [B]

        all_predictions.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_logits.extend(binary_logits.cpu().tolist())
        all_culture_labels.extend(culture_labels)

    # 5. 计算指标
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )

    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:   {accuracy:.4f}")
    print(f"   Precision:  {precision:.4f}")
    print(f"   Recall:     {recall:.4f}")
    print(f"   F1:         {f1:.4f}")

    print(f"\n📊 Classification Report:")
    print(classification_report(
        labels,
        predictions,
        target_names=["no (0)", "yes (1)"],
        digits=4
    ))

    cm = confusion_matrix(labels, predictions)
    print(f"\n📊 Confusion Matrix:")
    print("           Predicted")
    print("           no   yes")
    print(f"Actual no  {cm[0][0]:3d}  {cm[0][1]:3d}")
    print(f"Actual yes {cm[1][0]:3d}  {cm[1][1]:3d}")

    # 6. 保存结果
    if output_file:
        results = {
            "model": "base_llama_3.1_8b",
            "data_format": "instruction/instruction_mask/input/output/label",
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            },
            "predictions": all_predictions,
            "labels": all_labels,
            "culture_labels": all_culture_labels,
            "logits": all_logits,
            "confusion_matrix": cm.tolist()
        }

        print(f"\n💾 Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Results saved!")

    print("\n" + "="*60)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="评估 Base LLaMA 3.1 模型（无训练）")
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

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("⚠️  Warning: CUDA not available, using CPU")
        args.device = "cpu"

    if args.device.startswith("cuda"):
        print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

    metrics = evaluate_base_llama(
        model_path=args.model_path,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_file=args.output_file,
        device=args.device
    )

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

