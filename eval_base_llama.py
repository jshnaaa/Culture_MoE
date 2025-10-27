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


def load_simple_classification_data(data_path: str, tokenizer, max_length: int = 512):
    """
    简单加载分类数据（只读取 instruction, input, output）

    Args:
        data_path: 数据文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度

    Returns:
        处理后的数据列表
    """
    print(f"Loading data from {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    processed_data = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item['output']  # 必须字段

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


def evaluate_base_llama(
    model_path: str,
    test_file: str,
    batch_size: int = 8,
    max_length: int = 512,
    output_file: str = None,
    device: str = "cuda:0",
    use_dual_input: bool = True,
    num_classes: int = 2,
    backbone: str = "llama"
):
    """
    使用 Base LLaMA 3.1 模型评估

    数据格式：
    {
        "instruction": "...",
        "instruction_mask": "...",
        "input": "...",
        "output": 0,  # 整数：0, 1, 2, 3, ...
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
        num_classes: 分类数量（2, 3, 4, ...）
    """
    print("="*60)
    if backbone == "qwen":
        print("Evaluating Base Qwen 2.5 Model (No Training, No MoE)")
    else:
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

    # 3. 加载测试数据（只读取 instruction, input, output）
    print(f"\n3. Loading test data from {test_file}...")
    print(f"   Data format: instruction/input/output")
    test_dataset = load_simple_classification_data(
        data_path=test_file,
        tokenizer=tokenizer,
        max_length=max_length
    )
    print(f"   ✅ Test dataset size: {len(test_dataset)}")

    # 4. 评估
    print(f"\n4. Evaluating on {len(test_dataset)} samples...")
    print(f"   Using {'dual input' if use_dual_input else 'single input'} mode")

    all_predictions = []
    all_labels = []
    all_logits = []

    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch_data = test_dataset[i:i+batch_size]

        # 准备输入
        from torch.nn.utils.rnn import pad_sequence

        # 提取 input_ids 和 attention_mask
        input_ids_list = [torch.tensor(item['input_ids']) for item in batch_data]
        attention_mask_list = [torch.tensor(item['attention_mask']) for item in batch_data]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

        labels = [item['label'] for item in batch_data]

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

            # ✅ 使用启发式方法：根据类别数量提取对应 token 的 logits
            # 定义类别标签映射
            if num_classes == 2:
                class_tokens = ["no", "yes"]
            elif num_classes == 3:
                class_tokens = ["no", "neutral", "yes"]
            elif num_classes == 4:
                class_tokens = ["strongly_disagree", "disagree", "agree", "strongly_agree"]
            elif num_classes == 5:
                class_tokens = ["strongly_disagree", "disagree", "neutral", "agree", "strongly_agree"]
            else:
                # 通用方案：使用数字
                class_tokens = [str(i) for i in range(num_classes)]

            # 获取每个类别 token 的 ID
            class_token_ids = []
            for token in class_tokens:
                token_ids = tokenizer.encode(token, add_special_tokens=False)
                if len(token_ids) > 0:
                    class_token_ids.append(token_ids[0])
                else:
                    # 如果 token 不存在，使用一个默认值
                    class_token_ids.append(0)

            # 获取下一个 token 的 logits
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            # 提取每个类别的 logits
            class_logits_list = []
            for token_id in class_token_ids:
                class_logits_list.append(logits[:, token_id])  # [B]

            # 组合成多分类 logits
            multi_class_logits = torch.stack(class_logits_list, dim=1)  # [B, num_classes]

            # 预测
            preds = torch.argmax(multi_class_logits, dim=-1).cpu()  # [B]

        all_predictions.extend(preds.tolist())
        all_labels.extend(labels)
        all_logits.extend(multi_class_logits.cpu().tolist())

    # 5. 计算指标
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    accuracy = accuracy_score(labels, predictions)

    # ✅ 根据分类数量选择平均方式
    if num_classes == 2:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='binary', pos_label=1, zero_division=0
        )
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )

    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:   {accuracy:.4f}")
    print(f"   Precision:  {precision:.4f}")
    print(f"   Recall:     {recall:.4f}")
    print(f"   F1:         {f1:.4f}")

    # ✅ 根据分类数量设置标签名称
    if num_classes == 2:
        target_names = ["no (0)", "yes (1)"]
    elif num_classes == 3:
        target_names = ["no (0)", "neutral (1)", "yes (2)"]
    elif num_classes == 4:
        target_names = ["strongly_disagree (0)", "disagree (1)", "agree (2)", "strongly_agree (3)"]
    elif num_classes == 5:
        target_names = ["strongly_disagree (0)", "disagree (1)", "neutral (2)", "agree (3)", "strongly_agree (4)"]
    else:
        target_names = [f"class_{i} ({i})" for i in range(num_classes)]

    print(f"\n📊 Classification Report:")
    print(classification_report(
        labels,
        predictions,
        target_names=target_names,
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    print(f"\n📊 Confusion Matrix:")
    print("           Predicted")

    # ✅ 动态打印混淆矩阵
    if num_classes == 2:
        print("           no   yes")
        print(f"Actual no  {cm[0][0]:3d}  {cm[0][1]:3d}")
        print(f"Actual yes {cm[1][0]:3d}  {cm[1][1]:3d}")
    elif num_classes == 3:
        print("              no  neutral  yes")
        print(f"Actual no      {cm[0][0]:3d}  {cm[0][1]:3d}  {cm[0][2]:3d}")
        print(f"Actual neutral {cm[1][0]:3d}  {cm[1][1]:3d}  {cm[1][2]:3d}")
        print(f"Actual yes     {cm[2][0]:3d}  {cm[2][1]:3d}  {cm[2][2]:3d}")
    elif num_classes == 4:
        print("                    SD   D    A    SA")
        for i, label in enumerate(["SD", "D", "A", "SA"]):
            row_str = f"Actual {label:2s}         "
            for j in range(num_classes):
                row_str += f"{cm[i][j]:3d}  "
            print(row_str)
    elif num_classes == 5:
        print("                    SD   D    N    A    SA")
        for i, label in enumerate(["SD", "D", "N", "A", "SA"]):
            row_str = f"Actual {label:2s}         "
            for j in range(num_classes):
                row_str += f"{cm[i][j]:3d}  "
            print(row_str)
    else:
        # 通用打印
        header = "           " + "  ".join([f"C{i}" for i in range(num_classes)])
        print(header)
        for i in range(num_classes):
            row_str = f"Actual C{i}  "
            for j in range(num_classes):
                row_str += f"{cm[i][j]:3d}  "
            print(row_str)

    # 6. 保存结果
    if output_file:
        results = {
            "model": "base_llama_3.1_8b",
            "data_format": "instruction/input/output",
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            },
            "predictions": all_predictions,
            "labels": all_labels,
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
    parser.add_argument("--num_classes", type=int, default=2,
                        help="分类数量（2, 3, 4, ...）")
    parser.add_argument("--backbone", type=str, default="llama", choices=["llama", "qwen"],
                        help="基座模型类型：llama 或 qwen")

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
        device=args.device,
        num_classes=args.num_classes,
        backbone=args.backbone
    )

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

