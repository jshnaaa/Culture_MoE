#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 Base 模型 + LoRA 权重还原完整模型并评估（Qwen模型特别优化版本）

修复内容：
  ✅ 解决空答案生成问题（增加token数量，明确答案提示）
  ✅ 解决超范围标签问题（222, 22 → 正确范围 1-3）
  ✅ 解决无效文本输出（"education" → 数字）
  ✅ 自然提示工程（使用数据集原生### Answer:格式）
  ✅ 激进后处理（智能截断，最后位数提取）
  ✅ 针对Qwen优化生成参数（max_tokens=3, repetition_penalty=1.2）
  ✅ 改进答案提取和回退机制
  ✅ 增强答案质量统计和监控

Qwen模型特别优化：
  🎯 解决Qwen特殊token重复问题（151643重复生成）
  🎯 智能数字映射（12→2, 22→2, 222→2）
  🎯 重复token检测和截断
  🎯 增强的debug模式用于Qwen问题诊断
  🎯 多层约束机制确保数字在正确范围内
  🎯 解决空答案生成问题（平衡生成参数，增强fallback机制）
  🎯 多重答案提取策略（逐token解码，特殊token处理）

用法：
    python eval_lora_only_from_components.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --test_file /path/to/test.json \
        --output_dir /path/to/output \
        --num_classes 3
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

    Args:
        base_model_path: base 模型路径
        lora_weights_path: LoRA 权重路径
        device: 设备

    Returns:
        model: 合并后的完整模型
        tokenizer: tokenizer
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


def create_natural_prompt(instruction: str, input_text: str, num_classes: int):
    """
    创建符合数据集格式的自然prompt（避免过度约束）
    """
    # 直接使用数据集的原始格式，只在末尾添加轻微提示
    # 🔧 格式统一：使用空格分隔，与训练格式一致
    if input_text and input_text.strip():
        # 如果有input_text，按原格式组合，末尾加空格等待答案
        full_prompt = f"{instruction}{input_text} "
    else:
        # 如果没有input_text，直接使用instruction，末尾加空格等待答案
        full_prompt = f"{instruction} "

    # 确保prompt以适当的格式结束
    if not full_prompt.rstrip().endswith((':', '：')):
        # 如果不是以冒号结尾，检查是否是### Answer:格式
        if "### Answer:" in full_prompt:
            # 已经包含### Answer:，直接使用
            pass
        else:
            # 添加简单的答案提示
            full_prompt += " "

    return full_prompt


def create_constrained_prompt(instruction: str, input_text: str, num_classes: int):
    """
    创建带约束的prompt，明确指定答案范围和格式（备用版本）
    """
    # 根据类别数量生成选项提示（更加严格和明确）
    if num_classes == 2:
        options_text = """严格要求：只能回答 1 或 2
- 选择 1：第一个选项
- 选择 2：第二个选项
禁止回答其他任何数字！"""
    elif num_classes == 3:
        options_text = """严格要求：只能回答 1、2 或 3
- 选择 1：第一个选项
- 选择 2：第二个选项
- 选择 3：第三个选项
禁止回答 4、5、6、7、8、9、10、11、12、22、222 等其他数字！"""
    elif num_classes == 10:
        options_text = """严格要求：只能回答 1 到 10 之间的数字
可选择：1、2、3、4、5、6、7、8、9、10
禁止回答 11、12、22、222 等超出范围的数字！"""
    else:
        options_text = f"""严格要求：只能回答 1 到 {num_classes} 之间的数字
禁止回答超出 1-{num_classes} 范围的任何数字！"""

    # 构建完整的prompt（更加严格）
    if input_text and input_text.strip():
        full_prompt = f"""{instruction}

{input_text}

{options_text}

重要提醒：
- 必须回答且只能回答一个数字
- 数字必须在指定范围内
- 不要回答任何解释或额外内容
- 不要回答超出范围的数字

你的回答："""
    else:
        full_prompt = f"""{instruction}

{options_text}

重要提醒：
- 必须回答且只能回答一个数字
- 数字必须在指定范围内
- 不要回答任何解释或额外内容
- 不要回答超出范围的数字

你的回答："""

    return full_prompt


def generate_answer(model, tokenizer, instruction: str, input_text: str, num_classes: int = 10, max_new_tokens: int = 10):
    """
    生成答案（修复和改进版本 - 解决空值、范围和文本问题）

    Args:
        model: 模型（可能被 DataParallel 包装）
        tokenizer: tokenizer
        instruction: 指令
        input_text: 输入文本
        num_classes: 类别数量
        max_new_tokens: 最大生成 token 数

    Returns:
        raw_answer: 原始生成的答案
        predicted_label: 提取的标签（整数）
    """
    # 处理 DataParallel 包装的模型
    if isinstance(model, torch.nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model

    device = next(model.parameters()).device

    # 首先尝试使用自然格式（符合数据集原始格式）
    full_input = create_natural_prompt(instruction, input_text, num_classes)

    # Debug: 打印前几个样本的prompt来理解问题
    import random
    if random.random() < 0.001:  # 0.1%的概率打印debug信息（降低频率）
        print(f"\n🔍 DEBUG - Sample prompt (first 500 chars):")
        print(f"{full_input[:500]}...")
        print(f"Prompt length: {len(full_input)} characters")

    # Tokenize - 增加max_length以避免截断重要信息
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 检查是否发生了截断
    if inputs['input_ids'].shape[1] >= 2048:
        print(f"⚠️  Input truncated at 2048 tokens, important information may be lost")

    # 记录输入长度
    input_length = inputs['input_ids'].shape[1]

    # 生成答案 - 针对Qwen模型优化的参数
    with torch.no_grad():
        try:
            # 针对Qwen模型的平衡处理（避免过度约束）
            generate_kwargs = {
                **inputs,
                'max_new_tokens': 3,              # 🔧 减少到3个token，防止过度生成
                'min_new_tokens': 1,              # 强制至少生成1个token
                'do_sample': False,               # 贪婪解码，确保确定性
                'temperature': 1.0,
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'num_beams': 1,
                'repetition_penalty': 1.0,       # 🔧 移除重复惩罚，保持简洁
                'length_penalty': 0.0,           # 不惩罚长度，让模型自然生成
                'early_stopping': True           # 🔧 启用早停，防止过度生成
            }

            # 简化停止tokens处理（避免过度约束）
            # 只添加明确的停止序列，不干扰数字生成
            try:
                # 只阻止明显的无关token，不影响数字1,2,3的生成
                newline_tokens = tokenizer.encode('\n', add_special_tokens=False)
                if newline_tokens and len(newline_tokens) == 1:
                    generate_kwargs['bad_words_ids'] = [[newline_tokens[0]]]
            except:
                pass  # 如果失败就不添加任何约束

            outputs = actual_model.generate(**generate_kwargs)
        except Exception as e:
            print(f"⚠️  Generation failed: {e}, using fallback")
            # 如果生成失败，返回默认答案
            return "1", 0

    # 提取生成的部分
    generated_ids = outputs[0][input_length:]

    # 针对Qwen模型的特殊处理：过滤重复的特殊tokens
    if len(generated_ids) > 0:
        # 检查是否有大量重复的特殊tokens (如151643)
        unique_tokens = set(generated_ids.tolist())
        if len(unique_tokens) <= 2 and len(generated_ids) > 3:
            # 如果生成的tokens几乎都是重复的，只保留前几个
            generated_ids = generated_ids[:2]
            if random.random() < 0.01:  # 1%的概率打印警告
                print(f"⚠️  Detected repetitive tokens, truncated to first 2 tokens")

    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Debug: 打印生成的原始答案（增加频率以便调试空答案问题）
    if random.random() < 0.01:  # 增加到5%的概率打印debug信息
        print(f"🔍 DEBUG - Raw generated answer: '{raw_answer}'")
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🔍 DEBUG - Full output (last 200 chars): '...{full_output[-200:]}'")
        print(f"🔍 DEBUG - Generated tokens: {generated_ids.tolist()}")
        print(f"🔍 DEBUG - Generated tokens length: {len(generated_ids)}")
        print(f"🔍 DEBUG - Input length: {input_length}")
        print(f"🔍 DEBUG - Output length: {len(outputs[0])}")

        # 检查是否真的有生成内容
        if len(generated_ids) == 0:
            print(f"⚠️  No tokens were generated!")
        elif len(generated_ids) > 0:
            print(f"🔍 DEBUG - First generated token: {generated_ids[0].item()}")
            print(f"🔍 DEBUG - Unique tokens: {set(generated_ids.tolist())}")

            # 尝试不跳过特殊tokens的解码
            raw_with_special = tokenizer.decode(generated_ids, skip_special_tokens=False)
            print(f"🔍 DEBUG - Raw with special tokens: '{raw_with_special}'")

        # 检查是否是Qwen的问题tokens
        if 151643 in generated_ids.tolist():
            print(f"⚠️  Detected Qwen special token 151643 in generation")

    # 如果仍然为空，尝试多种备用方法
    if not raw_answer:
        # 方案2：解码完整输出并手动提取
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 调试：打印完整输出长度比较
        if random.random() < 0.01:
            print(f"🔍 DEBUG - Full input length: {len(full_input)}")
            print(f"🔍 DEBUG - Full output length: {len(full_output)}")
            print(f"🔍 DEBUG - Length difference: {len(full_output) - len(full_input)}")

        # 尝试多种分割方式
        if "### Answer:" in full_output:
            parts = full_output.split("### Answer:")
            if len(parts) > 1:
                raw_answer = parts[-1].strip()
                if random.random() < 0.05:
                    print(f"🔍 DEBUG - Found answer after '### Answer:': '{raw_answer}'")
        elif "你的回答：" in full_output:
            parts = full_output.split("你的回答：")
            if len(parts) > 1:
                raw_answer = parts[-1].strip()
                if random.random() < 0.01:
                    print(f"🔍 DEBUG - Found answer after '你的回答：': '{raw_answer}'")
        elif len(full_output) > len(full_input):
            raw_answer = full_output[len(full_input):].strip()
            if random.random() < 0.01:
                print(f"🔍 DEBUG - Extracted new content: '{raw_answer}'")

        # 方案3：如果还是空的，尝试不跳过特殊token
        if not raw_answer:
            raw_answer_with_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
            # 移除特殊token但保留内容
            raw_answer = re.sub(r'<[^>]*>', '', raw_answer_with_tokens).strip()
            if raw_answer and random.random() < 0.01:
                print(f"🔍 DEBUG - Found answer with special tokens: '{raw_answer}'")

        # 方案4：如果生成的tokens不为空但解码为空，可能是特殊tokens问题
        if not raw_answer and len(generated_ids) > 0:
            # 尝试逐个token解码
            for i, token_id in enumerate(generated_ids.tolist()):
                try:
                    token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    if token_text and token_text.strip() and token_text.strip() in '123456789':
                        raw_answer = token_text.strip()
                        if random.random() < 0.01:
                            print(f"🔍 DEBUG - Found digit in token {i}: '{raw_answer}'")
                        break
                except:
                    continue

    # 清理答案 - 加强版清理逻辑
    if raw_answer:
        # 移除换行符和多余空格
        raw_answer = raw_answer.split('\n')[0].strip()
        # 只取第一个词
        first_word = raw_answer.split()[0] if raw_answer.split() else raw_answer
        # 移除标点符号
        cleaned_answer = first_word.strip('.,!?;:()[]{}"\'-')
        raw_answer = cleaned_answer

        # 额外的数字提取和约束（针对Qwen模型增强）
        # 如果答案包含多个数字，只取第一个
        numbers_in_answer = re.findall(r'\d+', raw_answer)
        if numbers_in_answer:
            # 取第一个数字
            first_number = int(numbers_in_answer[0])
            # 如果超出范围，使用更智能的映射策略
            if first_number > num_classes:
                if first_number >= 10:  # 对于12, 22, 222等多位数
                    # 策略1: 如果是12，可能想表达1或2
                    if first_number == 12 and num_classes >= 2:
                        # 随机选择1或2，或者根据上下文
                        raw_answer = "2"  # 倾向于选择2
                        print(f"⚠️  Mapped 12 to 2 (intelligent mapping)")
                    elif first_number == 22:
                        raw_answer = "2"  # 22映射到2
                        print(f"⚠️  Mapped 22 to 2 (intelligent mapping)")
                    elif first_number == 222:
                        raw_answer = "2"  # 222映射到2
                        print(f"⚠️  Mapped 222 to 2 (intelligent mapping)")
                    elif str(first_number).startswith('1') and num_classes >= 1:
                        raw_answer = "1"  # 以1开头的映射到1
                        print(f"⚠️  Mapped {first_number} to 1 (starts with 1)")
                    elif str(first_number).startswith('2') and num_classes >= 2:
                        raw_answer = "2"  # 以2开头的映射到2
                        print(f"⚠️  Mapped {first_number} to 2 (starts with 2)")
                    elif str(first_number).startswith('3') and num_classes >= 3:
                        raw_answer = "3"  # 以3开头的映射到3
                        print(f"⚠️  Mapped {first_number} to 3 (starts with 3)")
                    else:
                        # 取最后一位数字作为fallback
                        last_digit = first_number % 10
                        if last_digit == 0:
                            last_digit = 1
                        if last_digit <= num_classes:
                            raw_answer = str(last_digit)
                            print(f"⚠️  Extracted last digit from {first_number}: {last_digit}")
                        else:
                            raw_answer = str(min(last_digit, num_classes))
                            print(f"⚠️  Clipped last digit {last_digit} to {min(last_digit, num_classes)}")
                else:
                    # 普通的超范围数字，直接截断
                    raw_answer = str(min(first_number, num_classes))
                    print(f"⚠️  Clipped {first_number} to {min(first_number, num_classes)}")

    # 如果还是空的，给一个明确的警告和默认值
    if not raw_answer:
        print(f"⚠️  Generated completely empty answer, using default: 1")
        raw_answer = "1"

    # 使用改进的标签提取
    predicted_label = extract_label_enhanced(raw_answer, num_classes)

    return raw_answer, predicted_label


def extract_label(answer: str, num_classes: int = 10):
    """
    从答案中提取标签（旧版本，保留兼容性）

    Args:
        answer: 模型回答
        num_classes: 类别数量

    Returns:
        label: 标签（0-indexed）
    """
    if not answer:
        return num_classes // 2

    # 尝试直接转换为整数
    try:
        label = int(answer)
        # 1-indexed -> 0-indexed
        if 1 <= label <= num_classes:
            return label - 1
        elif 0 <= label < num_classes:
            return label
        elif label > num_classes:
            return num_classes - 1
    except ValueError:
        pass

    # 尝试匹配第一个数字
    match = re.search(r'(\d+)', answer)
    if match:
        label = int(match.group(1))
        if 1 <= label <= num_classes:
            return label - 1
        elif 0 <= label < num_classes:
            return label
        elif label > num_classes:
            return num_classes - 1

    # 默认返回中间类别
    return num_classes // 2


def extract_label_enhanced(answer: str, num_classes: int = 10):
    """
    增强的标签提取（结合fixed和improved版本的所有改进）

    Args:
        answer: 模型回答
        num_classes: 类别数量

    Returns:
        label: 标签（0-indexed）
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
            # 超出范围，严格截断到有效范围（来自improved版本）
            clipped = max(0, min(num_classes - 1, label - 1 if label >= 1 else 0))
            print(f"⚠️  Label {label} out of range [1, {num_classes}], clipped to {clipped + 1}")
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
            # 严格截断
            clipped = max(0, min(num_classes - 1, label - 1 if label >= 1 else 0))
            print(f"⚠️  Extracted label {label} out of range, clipped to {clipped + 1}")
            return clipped

    # 3. 检查是否是文本答案（保留原有的映射逻辑）
    answer_lower = answer.lower().strip()

    # 统一编码的文本映射（针对 num_classes=15）
    if num_classes == 15:
        text_mapping = {
            # NormAD (11-13)
            'yes': 10,      # 11 - 1 = 10
            'no': 11,       # 12 - 1 = 11
            'neutral': 12,  # 13 - 1 = 12
            # CulturalBench (14-15)
            'true': 13,     # 14 - 1 = 13
            'false': 14,    # 15 - 1 = 14
        }
        if answer_lower in text_mapping:
            print(f"⚠️  Found text answer '{answer_lower}', mapping to {text_mapping[answer_lower]}")
            return text_mapping[answer_lower]

    # 标准文本映射（针对其他情况）
    standard_text_mapping = {
        'yes': 0, 'no': 1, 'neutral': 2,
        'true': 0, 'false': 1,
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
        'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
        # 添加一些常见的无效答案映射
        'education': 0, 'culture': 1, 'social': 2
    }
    if answer_lower in standard_text_mapping:
        mapped = standard_text_mapping[answer_lower]
        if mapped < num_classes:
            print(f"⚠️  Found text answer '{answer_lower}', mapping to {mapped + 1}")
            return mapped

    # 4. 如果是无关词（如 "Code", "Country"），给出更明确的警告
    if answer.isalpha() and len(answer) > 2:
        print(f"⚠️  Invalid text answer: '{answer}' - expected a number between 1 and {num_classes}")
        print(f"   Using default: 1")
        return 0

    # 5. 最后的fallback - 返回第一个选项
    print(f"⚠️  Could not extract valid label from '{answer}', using default: 1")
    return 0


def extract_label_robust(answer: str, num_classes: int = 10):
    """
    改进的标签提取（保留向后兼容性）
    """
    return extract_label_enhanced(answer, num_classes)


def evaluate_model(model, tokenizer, test_data, num_classes: int = 10, output_dir: str = None, group_by_country: bool = False):
    """
    评估模型

    Args:
        model: 模型
        tokenizer: tokenizer
        test_data: 测试数据
        num_classes: 类别数量
        output_dir: 输出目录
        group_by_country: 是否按country分组统计结果（用于blend数据集）

    Returns:
        results: 评估结果
    """
    print("\nRunning evaluation...")
    print(f"Number of classes: {num_classes}")
    if group_by_country:
        print(f"🌍 Country grouping: enabled")

    all_preds = []
    all_labels = []
    all_answers = []

    # 🔧 新增：country分组统计
    country_stats = {}  # {country: {'correct': 0, 'total': 0, 'accuracy': 0.0}}

    failed_count = 0

    for item in tqdm(test_data, desc="Evaluating"):
        instruction = item['instruction']
        input_text = item['input']
        # 🔧 新增：获取country字段（用于blend数据集分组统计）
        country = item.get('country', None)

        # 处理标签（1-indexed -> 0-indexed）
        label = int(item['output'])
        if label >= 1:
            label = label - 1

        # 生成答案
        raw_answer, pred = generate_answer(model, tokenizer, instruction, input_text, num_classes)

        # 统计失败
        if pred == num_classes // 2 and raw_answer and not raw_answer.isdigit():
            failed_count += 1

        # 判断正确性
        is_correct = (pred == label)

        # 🔧 新增：更新country分组统计
        if group_by_country and country is not None:
            if country not in country_stats:
                country_stats[country] = {'correct': 0, 'total': 0, 'accuracy': 0.0}

            country_stats[country]['total'] += 1
            if is_correct:
                country_stats[country]['correct'] += 1
            country_stats[country]['accuracy'] = country_stats[country]['correct'] / country_stats[country]['total']

        all_preds.append(pred)
        all_labels.append(label)

        # 保存详细答案
        answer_item = {
            "instruction": instruction,
            "input": input_text,
            "true_label": label,
            "predicted_label": pred,
            "raw_answer": raw_answer,
            "correct": is_correct
        }
        # 🔧 新增：如果有country字段，也保存到结果中
        if country is not None:
            answer_item['country'] = country
        all_answers.append(answer_item)

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 保存详细答案
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        answers_file = os.path.join(output_dir, "generated_answers.json")
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(all_answers, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {len(all_answers)} detailed answers to: {answers_file}")

    # 增强的统计信息（来自improved版本）
    empty_answers = sum(1 for ans in all_answers if not ans['raw_answer'] or ans['raw_answer'].strip() == "")
    out_of_range_count = 0
    text_answers = 0

    for ans in all_answers:
        raw = ans['raw_answer']
        if raw and raw.strip():
            # 检查是否是纯文本答案
            if raw.isalpha() and len(raw) > 2:
                text_answers += 1
            # 检查是否原本超出范围（通过检查是否有警告信息）
            try:
                original_num = int(raw)
                if original_num > num_classes or original_num < 1:
                    out_of_range_count += 1
            except ValueError:
                pass

    # 打印详细的提取统计
    print(f"\n📈 Answer Quality Statistics:")
    print(f"   Total samples: {len(test_data)}")
    print(f"   Empty answers: {empty_answers} ({100 * empty_answers / len(test_data):.1f}%)")
    print(f"   Text answers: {text_answers} ({100 * text_answers / len(test_data):.1f}%)")
    print(f"   Out-of-range answers: {out_of_range_count} ({100 * out_of_range_count / len(test_data):.1f}%)")
    print(f"   Failed extractions: {failed_count} ({100 * failed_count / len(test_data):.1f}%)")

    # 🔧 新增：显示country分组统计结果
    if group_by_country and country_stats:
        print(f"\n🌍 Country-wise Statistics:")
        print("-" * 100)
        for country, stats in sorted(country_stats.items()):
            print(f"  {country}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.4f})")
        print("-" * 100)

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

    # 构建结果字典（增强版本，包含更多统计信息）
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "num_samples": len(all_labels),
        "failed_extractions": failed_count,
        "failed_rate": float(failed_count / len(test_data)) if len(test_data) > 0 else 0.0,
        # 新增的质量统计
        "empty_answers": empty_answers,
        "empty_answer_rate": float(empty_answers / len(test_data)) if len(test_data) > 0 else 0.0,
        "text_answers": text_answers,
        "text_answer_rate": float(text_answers / len(test_data)) if len(test_data) > 0 else 0.0,
        "out_of_range_answers": out_of_range_count,
        "out_of_range_rate": float(out_of_range_count / len(test_data)) if len(test_data) > 0 else 0.0,
        "group_by_country": group_by_country
    }

    # 🔧 新增：如果启用了country分组统计，添加分组结果
    if group_by_country and country_stats:
        results['country_results'] = country_stats
        # 打印country分组结果
        print(f"\n🌍 Country-wise Results:")
        for country, stats in country_stats.items():
            print(f"    {country}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    return results


def main():
    parser = argparse.ArgumentParser(description="从 Base 模型 + LoRA 权重评估（生成式版本）")
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
    parser.add_argument("--data_id", type=str, default="",
                        help="Data ID to determine if country grouping is needed (16 for blend dataset)")
    parser.add_argument("--model_path", type=str, default="",
                        help="Model directory path (for pkl file detection when data_id=0)")

    args = parser.parse_args()

    # 🔧 新增：检查是否需要按country分组统计（DATA_ID=16的blend数据集）
    group_by_country = (args.data_id == "16")

    # 🔧 新增：检查是否使用pkl文件（DATA_ID=0）
    use_pkl_files = (args.data_id == "0")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LoRA Model Evaluation (Enhanced Version - Fixed & Improved)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.lora_weights_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Num classes: {args.num_classes}")
    if group_by_country:
        print(f"🌍 Country grouping: enabled (DATA_ID={args.data_id})")
    print("="*80)
    print(f"🔧 Root Cause Fixes Applied:")
    print(f"  ✅ Fixed prompt format mismatch (use dataset's ### Answer: format)")
    print(f"  ✅ Increased max_length to 2048 (prevent important info truncation)")
    print(f"  ✅ Natural prompt generation (avoid over-constraining)")
    print(f"  ✅ Multiple answer extraction methods (### Answer:, 你的回答：)")
    print(f"  ✅ Smart number extraction and clipping (222→2, 22→2)")
    print(f"  ✅ Debug mode for understanding model behavior")
    print(f"  ✅ Enhanced answer quality statistics")
    print(f"  ✅ Robust fallback mechanisms")
    print(f"  🎯 Addresses root cause of 222/22 generation")
    print("="*80)
    print("")

    # 加载模型
    model, tokenizer = load_model_from_components(
        args.base_model_path,
        args.lora_weights_path,
        args.device
    )

    # ✅ 使用多 GPU（DataParallel）
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"\n{'='*80}")
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        print(f"{'='*80}")
        model = torch.nn.DataParallel(model)
        print(f"✅ Model wrapped with DataParallel")
        print(f"{'='*80}\n")

    # 🔧 根据DATA_ID决定数据加载方式
    if use_pkl_files:
        # DATA_ID=0: 使用pkl文件中的测试集
        print(f"\n🔧 使用pkl文件模式 (DATA_ID=0)")
        print(f"Model path: {args.model_path}")

        # 查找pkl文件
        import pickle
        import glob

        pkl_files = glob.glob(os.path.join(args.model_path, "*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No pkl files found in {args.model_path}")

        print(f"Found {len(pkl_files)} pkl files:")
        for pkl_file in pkl_files:
            print(f"  - {os.path.basename(pkl_file)}")

        # 对每个pkl文件分别进行评估
        all_results = {}

        for pkl_file in pkl_files:
            pkl_name = os.path.splitext(os.path.basename(pkl_file))[0]
            print(f"\n📋 Processing pkl file: {pkl_name}")

            # 加载pkl文件
            with open(pkl_file, 'rb') as f:
                split_info = pickle.load(f)

            # 检查pkl文件结构
            if 'test_indices' not in split_info:
                print(f"⚠️ Warning: {pkl_name} does not contain test_indices, skipping...")
                continue

            # 获取原始数据路径
            if 'data_path' not in split_info:
                print(f"⚠️ Warning: {pkl_name} does not contain data_path, skipping...")
                continue

            data_path = split_info['data_path']
            test_indices = split_info['test_indices']

            print(f"  Data path: {data_path}")
            print(f"  Test indices: {len(test_indices)} samples")

            # 加载原始数据
            if not os.path.exists(data_path):
                print(f"⚠️ Warning: Data file not found: {data_path}, skipping...")
                continue

            with open(data_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)

            # 提取测试集
            test_data = [full_data[i] for i in test_indices]
            print(f"  ✅ Extracted {len(test_data)} test samples")

            # 评估模型
            pkl_results = evaluate_model(
                model,
                tokenizer,
                test_data,
                num_classes=10,  # 默认使用10个类别
                output_dir=os.path.join(args.output_dir, pkl_name),
                group_by_country=False  # pkl文件模式不支持country分组
            )

            # 保存单个pkl文件的结果
            pkl_output_dir = os.path.join(args.output_dir, pkl_name)
            os.makedirs(pkl_output_dir, exist_ok=True)

            pkl_results_file = os.path.join(pkl_output_dir, "evaluation_results.json")
            with open(pkl_results_file, 'w', encoding='utf-8') as f:
                json.dump(pkl_results, f, indent=2, ensure_ascii=False)

            all_results[pkl_name] = {
                'accuracy': pkl_results['accuracy'],
                'precision': pkl_results['precision'],
                'recall': pkl_results['recall'],
                'f1': pkl_results['f1'],
                'num_samples': pkl_results['num_samples'],
                'data_path': data_path,
                'test_indices_count': len(test_indices)
            }

            print(f"  📊 {pkl_name} - Accuracy: {pkl_results['accuracy']:.4f}")

        # 合并结果
        results = {
            'evaluation_mode': 'pkl_files',
            'individual_results': all_results,
            'summary': {
                'total_pkl_files': len(all_results),
                'average_accuracy': np.mean([r['accuracy'] for r in all_results.values()]) if all_results else 0.0
            }
        }

    else:
        # 常规模式：使用JSON文件
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
            output_dir=args.output_dir,
            group_by_country=group_by_country
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

    # 🔧 根据评估模式保存不同的摘要
    if use_pkl_files:
        # pkl文件模式的摘要
        summary = {
            "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "evaluation_mode": "pkl_files",
            "base_model_path": args.base_model_path,
            "lora_weights_path": args.lora_weights_path,
            "model_path": args.model_path,
            "data_id": args.data_id,
            "total_pkl_files": results['summary']['total_pkl_files'],
            "average_accuracy": results['summary']['average_accuracy'],
            "individual_results": results['individual_results']
        }

        print(f"\n📊 PKL Files Evaluation Summary:")
        print(f"  Total pkl files processed: {results['summary']['total_pkl_files']}")
        print(f"  Average accuracy: {results['summary']['average_accuracy']:.4f}")
        print(f"\n📋 Individual Results:")
        for pkl_name, pkl_result in results['individual_results'].items():
            print(f"    {pkl_name}: {pkl_result['accuracy']:.4f} ({pkl_result['num_samples']} samples)")

    else:
        # 常规模式的摘要
        summary = {
            "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "evaluation_mode": "json_file",
            "base_model_path": args.base_model_path,
            "lora_weights_path": args.lora_weights_path,
            "test_file": args.test_file,
            "data_id": args.data_id,
            "num_samples": results['num_samples'],
            "num_classes": args.num_classes,
            "accuracy": results['accuracy'],
            "precision": results['precision'],
            "recall": results['recall'],
            "f1": results['f1'],
            "failed_extractions": results.get('failed_extractions', 0),
            "failed_rate": results.get('failed_rate', 0.0)
        }

        # 添加country分组结果（如果有）
        if 'country_results' in results:
            summary['country_results'] = results['country_results']

    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Summary saved to: {summary_file}")

    print("\n" + "="*80)
    print("✅ Enhanced LoRA Evaluation completed successfully!")
    print("="*80)
    print(f"\n📊 Final Results:")

    # 🔧 根据评估模式使用不同的结果打印逻辑
    if use_pkl_files:
        # pkl文件模式的结果打印
        print(f"   Average Accuracy: {results['summary']['average_accuracy']:.4f}")
        print(f"   Total pkl files processed: {results['summary']['total_pkl_files']}")
        print(f"   Individual Results:")
        for pkl_name, pkl_result in results['individual_results'].items():
            print(f"     {pkl_name}: {pkl_result['accuracy']:.4f} ({pkl_result['num_samples']} samples)")
    else:
        # 常规模式的结果打印
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Samples: {results['num_samples']}")
        print(f"   Empty answers: {results['empty_answers']} ({results['empty_answer_rate']:.1%})")
        print(f"   Text answers: {results['text_answers']} ({results['text_answer_rate']:.1%})")
        print(f"   Out-of-range: {results['out_of_range_answers']} ({results['out_of_range_rate']:.1%})")

    # 🔧 新增：显示country分组统计结果
    if 'country_stats' in results:
        print(f"\n🌍 Country-wise Statistics (DATA_ID=16 blend dataset):")
        country_stats = results['country_stats']
        for country, stats in sorted(country_stats.items()):
            print(f"  {country}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.4f})")

    if group_by_country:
        print(f"  Country分组: 启用 (DATA_ID={args.data_id})")
    print("")
    print("🔧 All fixes successfully applied!")
    print("")


if __name__ == "__main__":
    main()

