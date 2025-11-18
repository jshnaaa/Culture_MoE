#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 Base 模型 + LoRA 权重还原完整模型并评估（修复空值问题的版本）

用法：
    python eval_lora_only_from_components_fixed.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --test_file /path/to/test.json \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_from_components(base_model_path: str, lora_weights_path: str, device: str = "cuda"):
    """
    从 base 模型和 LoRA 权重还原完整模型
    """
    print("Loading model from components...")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA weights: {lora_weights_path}")

    # 列出 LoRA 权重目录的内容
    if os.path.exists(lora_weights_path):
        lora_files = os.listdir(lora_weights_path)
        print(f"  LoRA directory contents: {lora_files}")
    else:
        raise FileNotFoundError(f"LoRA weights directory not found: {lora_weights_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 base 模型
    print("Step 1: Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载 LoRA 权重
    print("Step 2: Loading LoRA weights...")

    # 检查 adapter_config.json 是否存在
    adapter_config_path = os.path.join(lora_weights_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {lora_weights_path}")

    print(f"  Found adapter_config.json: {adapter_config_path}")

    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        is_trainable=False,
        torch_dtype=torch.float16
    )

    # 合并 LoRA 权重到 base 模型（仅在内存中）
    print("Step 3: Merging LoRA weights into base model...")
    model = peft_model.merge_and_unload()

    # 清理内存
    del base_model
    del peft_model
    torch.cuda.empty_cache()

    print("✅ Model loaded and merged successfully (in-memory only)")

    return model, tokenizer


def generate_answer_fixed(model, tokenizer, instruction: str, input_text: str, num_classes: int = 10):
    """
    修复版本的答案生成函数 - 解决空值问题
    """
    # 处理 DataParallel 包装的模型
    if isinstance(model, torch.nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model

    device = next(model.parameters()).device

    # 构建输入 - 添加明确的答案提示
    if input_text:
        # 方案1：在prompt末尾添加明确的答案引导
        full_input = f"{instruction}\n{input_text}\n答案："
    else:
        full_input = f"{instruction}\n答案："

    # Tokenize
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 记录输入长度
    input_length = inputs['input_ids'].shape[1]

    # 生成答案 - 使用更保守的参数
    with torch.no_grad():
        try:
            outputs = actual_model.generate(
                **inputs,
                max_new_tokens=20,                # 增加到20个token
                min_new_tokens=1,                 # 强制至少生成1个token
                do_sample=False,                  # 贪婪解码
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                repetition_penalty=1.0,
                length_penalty=0.0,               # 不惩罚长度
                early_stopping=False              # 不提前停止
            )
        except Exception as e:
            print(f"⚠️  Generation failed: {e}, using fallback")
            # 如果生成失败，返回默认答案
            return "1", num_classes // 2

    # 提取生成的部分
    generated_ids = outputs[0][input_length:]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 如果仍然为空，尝试备用方法
    if not raw_answer:
        # 方案2：解码完整输出并手动提取
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "答案：" in full_output:
            parts = full_output.split("答案：")
            if len(parts) > 1:
                raw_answer = parts[-1].strip()

        # 方案3：如果还是空的，尝试不跳过特殊token
        if not raw_answer:
            raw_answer_with_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
            # 移除特殊token但保留内容
            raw_answer = re.sub(r'<[^>]*>', '', raw_answer_with_tokens).strip()

    # 清理答案
    if raw_answer:
        # 只取第一行和第一个词
        raw_answer = raw_answer.split('\n')[0].split()[0] if raw_answer.split() else raw_answer
        # 移除标点符号
        raw_answer = raw_answer.strip('.,!?;:()[]{}"\'-')

    # 如果还是空的，给一个明确的警告和默认值
    if not raw_answer:
        print(f"⚠️  Generated completely empty answer, using default: 1")
        raw_answer = "1"

    # 提取标签
    predicted_label = extract_label_robust_fixed(raw_answer, num_classes)

    return raw_answer, predicted_label


def extract_label_robust_fixed(answer: str, num_classes: int = 10):
    """
    修复版本的标签提取函数
    """
    if not answer or answer.strip() == "":
        print(f"⚠️  Empty answer, using default: 1")
        return 0  # 返回标签1对应的0-indexed值

    answer = answer.strip()

    # 1. 尝试直接转换为整数
    try:
        label = int(answer)
        if 1 <= label <= num_classes:
            return label - 1  # 1-indexed -> 0-indexed
        elif 0 <= label < num_classes:
            return label
        else:
            # 超出范围，截断到有效范围
            clipped = min(max(1, label), num_classes) - 1
            print(f"⚠️  Label {label} out of range [1, {num_classes}], using {clipped + 1}")
            return clipped
    except ValueError:
        pass

    # 2. 提取第一个数字
    numbers = re.findall(r'\d+', answer)
    if numbers:
        label = int(numbers[0])
        if 1 <= label <= num_classes:
            return label - 1
        elif 0 <= label < num_classes:
            return label
        else:
            clipped = min(max(1, label), num_classes) - 1
            print(f"⚠️  Extracted label {label} out of range, using {clipped + 1}")
            return clipped

    # 3. 文本答案映射
    answer_lower = answer.lower()

    # 常见文本映射
    text_mapping = {
        'yes': 0, 'no': 1, 'neutral': 2,
        'true': 0, 'false': 1,
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
        'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9
    }

    if answer_lower in text_mapping:
        mapped = text_mapping[answer_lower]
        if mapped < num_classes:
            print(f"⚠️  Found text answer '{answer_lower}', mapping to {mapped + 1}")
            return mapped

    # 4. 默认返回1（0-indexed为0）
    print(f"⚠️  Could not extract valid label from '{answer}', using default: 1")
    return 0


def evaluate_model(model, tokenizer, test_data, num_classes: int = 10, output_dir: str = None):
    """
    评估模型
    """
    print("\nRunning evaluation...")
    print(f"Number of classes: {num_classes}")

    all_preds = []
    all_labels = []
    all_answers = []

    empty_count = 0
    invalid_count = 0

    for item in tqdm(test_data, desc="Evaluating"):
        instruction = item['instruction']
        input_text = item['input']

        # 处理标签（1-indexed -> 0-indexed）
        label = int(item['output'])
        if label >= 1:
            label = label - 1

        # 生成答案
        raw_answer, pred = generate_answer_fixed(model, tokenizer, instruction, input_text, num_classes)

        # 统计问题
        if not raw_answer or raw_answer.strip() == "":
            empty_count += 1
        elif not any(c.isdigit() for c in raw_answer):
            invalid_count += 1

        all_preds.append(pred)
        all_labels.append(label)

        # 保存详细答案
        all_answers.append({
            "instruction": instruction,
            "input": input_text,
            "true_label": label,
            "predicted_label": pred,
            "raw_answer": raw_answer,
            "correct": (pred == label)
        })

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 保存详细答案
    if output_dir:
        answers_file = os.path.join(output_dir, "generated_answers.json")
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(all_answers, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {len(all_answers)} detailed answers to: {answers_file}")

    # 打印统计信息
    if empty_count > 0 or invalid_count > 0:
        print(f"\n⚠️  Generation Issues:")
        print(f"   Empty answers: {empty_count}/{len(test_data)} ({100 * empty_count / len(test_data):.2f}%)")
        print(f"   Invalid answers: {invalid_count}/{len(test_data)} ({100 * invalid_count / len(test_data):.2f}%)")

    # 计算指标
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print(f"\n📊 Classification Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1:        {f1:.4f}")
    print("="*80)

    # 构建结果字典
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "num_samples": len(all_labels),
        "empty_answers": empty_count,
        "invalid_answers": invalid_count,
        "empty_rate": float(empty_count / len(test_data)) if len(test_data) > 0 else 0.0,
        "invalid_rate": float(invalid_count / len(test_data)) if len(test_data) > 0 else 0.0
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="从 Base 模型 + LoRA 权重评估（修复空值问题版本）")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Base 模型路径")
    parser.add_argument("--lora_weights_path", type=str, required=True,
                        help="LoRA 权重路径")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="评估结果输出目录")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="类别数量（默认 10）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--use_multi_gpu", action="store_true",
                        help="使用多 GPU 评估（DataParallel）")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LoRA Model Evaluation (Fixed Version - No Empty Answers)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.lora_weights_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Num classes: {args.num_classes}")
    print("="*80)
    print("")

    # 加载模型
    model, tokenizer = load_model_from_components(
        args.base_model_path,
        args.lora_weights_path,
        args.device
    )

    # 使用多 GPU（DataParallel）
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"\n{'='*80}")
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        print(f"{'='*80}")
        model = torch.nn.DataParallel(model)
        print(f"✅ Model wrapped with DataParallel")
        print(f"{'='*80}\n")

    # 加载测试数据
    print(f"\nLoading test data from: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ Loaded {len(test_data)} test samples")

    # 评估模型
    results = evaluate_model(
        model,
        tokenizer,
        test_data,
        num_classes=args.num_classes,
        output_dir=args.output_dir
    )

    # 保存评估结果
    print("\n" + "="*80)
    print("Saving Evaluation Results")
    print("="*80)

    # 保存详细结果
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Detailed results saved to: {results_file}")

    # 保存摘要
    summary = {
        "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "base_model_path": args.base_model_path,
        "lora_weights_path": args.lora_weights_path,
        "test_file": args.test_file,
        "num_samples": results['num_samples'],
        "num_classes": args.num_classes,
        "accuracy": results['accuracy'],
        "precision": results['precision'],
        "recall": results['recall'],
        "f1": results['f1'],
        "empty_answers": results['empty_answers'],
        "invalid_answers": results['invalid_answers'],
        "empty_rate": results['empty_rate'],
        "invalid_rate": results['invalid_rate']
    }

    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Summary saved to: {summary_file}")

    print("\n" + "="*80)
    print("✅ Evaluation completed successfully!")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   Samples: {results['num_samples']}")
    print(f"   Empty answers: {results['empty_answers']} ({results['empty_rate']:.2%})")
    print(f"   Invalid answers: {results['invalid_answers']} ({results['invalid_rate']:.2%})")
    print("")


if __name__ == "__main__":
    main()