#!/usr/bin/env python3
"""
评估生成式 LoRA Only 模型
只生成数字（选项编号），方便计算准确率
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_single_digit(model, tokenizer, prompt: str, num_classes: int = 5, device: str = "cuda"):
    """
    生成单个数字（0-num_classes-1）

    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入 prompt
        num_classes: 类别数量
        device: 设备

    Returns:
        predicted_class: 预测的类别（整数）
        raw_answer: 原始生成的答案（字符串）
    """
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 获取数字 token 的 ID
    digit_tokens = [str(i) for i in range(num_classes)]
    digit_token_ids = []
    for digit in digit_tokens:
        token_ids = tokenizer.encode(digit, add_special_tokens=False)
        if len(token_ids) > 0:
            digit_token_ids.append(token_ids[0])
        else:
            digit_token_ids.append(tokenizer.unk_token_id)

    # 生成（只生成 1 个 token）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            min_new_tokens=1,
            do_sample=False,  # 贪婪解码
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True
        )

    # 解码
    generated_ids = outputs.sequences[0]
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    raw_answer = full_output[len(prompt):].strip()

    # 方法1：从生成的 token 中提取
    if len(generated_ids) > len(inputs['input_ids'][0]):
        generated_token_id = generated_ids[len(inputs['input_ids'][0])].item()

        # 检查是否是数字 token
        if generated_token_id in digit_token_ids:
            predicted_class = digit_token_ids.index(generated_token_id)
            return predicted_class, raw_answer

    # 方法2：从 logits 中选择最可能的数字
    if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
        logits = outputs.scores[0][0]  # [vocab_size]

        # 只看数字 token 的 logits
        digit_logits = torch.tensor([logits[tid].item() for tid in digit_token_ids])
        predicted_class = torch.argmax(digit_logits).item()

        return predicted_class, raw_answer

    # 方法3：从原始答案中提取数字
    import re
    match = re.search(r'([0-9])', raw_answer)
    if match:
        digit = int(match.group(1))
        if 0 <= digit < num_classes:
            return digit, raw_answer

    # 如果都失败，返回默认值（中间类别）
    print(f"⚠️  Warning: Failed to extract digit from '{raw_answer}', using default {num_classes // 2}")
    return num_classes // 2, raw_answer


def evaluate_generative_model(
    model_path: str,
    lora_path: str,
    test_file: str,
    output_dir: str,
    num_classes: int = 5,
    batch_size: int = 1,  # 生成式模型通常 batch_size=1
    device: str = "cuda:0",
    save_answers: bool = True
):
    """
    评估生成式模型

    Args:
        model_path: 基座模型路径
        lora_path: LoRA 权重路径
        test_file: 测试数据文件
        output_dir: 输出目录
        num_classes: 类别数量
        batch_size: 批次大小（生成式通常为1）
        device: 设备
        save_answers: 是否保存生成的答案
    """
    print("="*80)
    print("Evaluating Generative LoRA Model")
    print("="*80)
    print(f"Base model: {model_path}")
    print(f"LoRA path: {lora_path}")
    print(f"Test file: {test_file}")
    print(f"Num classes: {num_classes}")
    print(f"Output dir: {output_dir}")
    print("="*80)
    print("")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载模型
    print("\n2. Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True
    )

    # 加载 LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.to(device)
    model.eval()
    print("   ✅ Model loaded")

    # 3. 加载测试数据
    print("\n3. Loading test data...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"   ✅ Loaded {len(test_data)} samples")

    # 4. 评估
    print("\n4. Generating predictions...")
    all_predictions = []
    all_labels = []
    all_answers = []  # 保存原始答案

    for item in tqdm(test_data, desc="Evaluating"):
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        label = item['output']  # 真实标签

        # 构建 prompt
        if input_text and input_text.strip():
            full_text = f"{instruction}\n{input_text}"
        else:
            full_text = instruction

        # 添加明确的指令
        prompt = f"{full_text}\n\nPlease answer with ONLY ONE NUMBER (0-{num_classes-1}).\nYour answer:"

        # 生成
        predicted_class, raw_answer = generate_single_digit(
            model, tokenizer, prompt, num_classes, device
        )

        all_predictions.append(predicted_class)
        all_labels.append(label)

        # 保存答案详情
        if save_answers:
            all_answers.append({
                "instruction": instruction,
                "input": input_text,
                "true_label": label,
                "predicted_label": predicted_class,
                "raw_answer": raw_answer,
                "correct": predicted_class == label
            })

    # 5. 计算指标
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    accuracy = accuracy_score(labels, predictions)

    if num_classes == 2:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', pos_label=1, zero_division=0
        )
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )

    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:   {accuracy:.4f}")
    print(f"   Precision:  {precision:.4f}")
    print(f"   Recall:     {recall:.4f}")
    print(f"   F1:         {f1:.4f}")

    # 分类报告
    if num_classes == 2:
        target_names = ["no (0)", "yes (1)"]
    elif num_classes == 3:
        target_names = ["no (0)", "neutral (1)", "yes (2)"]
    elif num_classes == 4:
        target_names = [f"class_{i} ({i})" for i in range(4)]
    elif num_classes == 5:
        target_names = [f"class_{i} ({i})" for i in range(5)]
    else:
        target_names = [f"class_{i} ({i})" for i in range(num_classes)]

    print(f"\n📊 Classification Report:")
    print(classification_report(
        labels, predictions,
        labels=list(range(num_classes)),
        target_names=target_names,
        digits=4,
        zero_division=0
    ))

    # 混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    print(f"\n📊 Confusion Matrix:")
    print("           Predicted")
    header = "           " + "  ".join([f"{i:3d}" for i in range(num_classes)])
    print(header)
    for i in range(num_classes):
        row_str = f"Actual {i:3d}  "
        for j in range(num_classes):
            row_str += f"{cm[i][j]:3d}  "
        print(row_str)

    # 6. 保存结果
    print(f"\n5. Saving results...")

    # 保存指标
    metrics = {
        "model_type": "generative_lora",
        "base_model": model_path,
        "lora_path": lora_path,
        "num_classes": num_classes,
        "test_samples": len(test_data),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        },
        "confusion_matrix": cm.tolist(),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_file = os.path.join(output_dir, "eval_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Metrics saved to: {metrics_file}")

    # 保存详细答案
    if save_answers:
        answers_file = os.path.join(output_dir, "generated_answers.json")
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(all_answers, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Answers saved to: {answers_file}")

        # 保存错误案例
        errors = [ans for ans in all_answers if not ans['correct']]
        if errors:
            errors_file = os.path.join(output_dir, "error_cases.json")
            with open(errors_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Error cases saved to: {errors_file}")
            print(f"      Total errors: {len(errors)}/{len(all_answers)} ({len(errors)/len(all_answers)*100:.1f}%)")

    # 保存预测结果（简洁版）
    predictions_file = os.path.join(output_dir, "predictions.txt")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        f.write("Index\tTrue\tPred\tCorrect\tRaw_Answer\n")
        for i, (true, pred, ans) in enumerate(zip(all_labels, all_predictions, all_answers)):
            correct = "✓" if true == pred else "✗"
            f.write(f"{i}\t{true}\t{pred}\t{correct}\t{ans['raw_answer']}\n")
    print(f"   ✅ Predictions saved to: {predictions_file}")

    print("\n" + "="*80)
    print("✅ Evaluation completed!")
    print("="*80)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="评估生成式 LoRA 模型")
    parser.add_argument("--model_path", type=str, required=True, help="基座模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA 权重路径")
    parser.add_argument("--test_file", type=str, required=True, help="测试数据文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_classes", type=int, default=5, help="类别数量")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--save_answers", action="store_true", default=True, help="保存生成的答案")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("⚠️  Warning: CUDA not available, using CPU")
        args.device = "cpu"

    metrics = evaluate_generative_model(
        model_path=args.model_path,
        lora_path=args.lora_path,
        test_file=args.test_file,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        device=args.device,
        save_answers=args.save_answers
    )

    print("\n✅ Done!")


if __name__ == "__main__":
    main()

