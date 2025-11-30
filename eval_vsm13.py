#!/usr/bin/env python3
"""
VSM13 评估脚本 - 生成式版本

用于评估 LoRA Only 和 CultureMoE 模型在 VSM13 数据集上的表现
计算每个国家在每个维度上的分数，并与霍夫斯泰德标准分数进行比较

使用方法：
    python eval_vsm13.py \
        --model_path /path/to/model \
        --data_path /path/to/vsm13.json \
        --output_dir /path/to/output \
        --model_type lora_only  # 或 culturemoe
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ 添加项目路径以导入 CultureMoE
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs

# ✅ 10 个国家及其形容词形式
COUNTRIES = {
    'China': 'Chinese',
    'South Korea': 'Korean',
    'Turkey': 'Turkish',
    'Saudi Arabia': 'Saudi',
    'Bangladesh': 'Bangladeshi',
    'Germany': 'German',
    'Portugal': 'Portuguese',
    'Spain': 'Spanish',
    'England': 'English',
    'Greece': 'Greek'
}

# ✅ 霍夫斯泰德标准分数（PDI, IDV, MAS, UAI, LTO, IVR）
# 格式：[PDI, IDV, MAS, UAI, LTO, IVR]
HOFSTEDE_SCORES = {
    'China': {'PDI': 80, 'IDV': 43, 'MAS': 66, 'UAI': 30, 'LTO': 77, 'IVR': 24},
    'South Korea': {'PDI': 60, 'IDV': 58, 'MAS': 39, 'UAI': 85, 'LTO': 86, 'IVR': 29},
    'Turkey': {'PDI': 66, 'IDV': 46, 'MAS': 45, 'UAI': 85, 'LTO': 35, 'IVR': 49},
    'Saudi Arabia': {'PDI': 72, 'IDV': 48, 'MAS': 43, 'UAI': 64, 'LTO': 27, 'IVR': 14},
    'Bangladesh': {'PDI': 80, 'IDV': 5, 'MAS': 55, 'UAI': 60, 'LTO': 38, 'IVR': 20},
    'Germany': {'PDI': 35, 'IDV': 79, 'MAS': 66, 'UAI': 65, 'LTO': 57, 'IVR': 40},
    'Portugal': {'PDI': 63, 'IDV': 59, 'MAS': 31, 'UAI': 99, 'LTO': 42, 'IVR': 33},
    'Spain': {'PDI': 57, 'IDV': 67, 'MAS': 42, 'UAI': 86, 'LTO': 47, 'IVR': 44},
    'England': {'PDI': 35, 'IDV': 76, 'MAS': 66, 'UAI': 35, 'LTO': 60, 'IVR': 69},
    'Greece': {'PDI': 60, 'IDV': 59, 'MAS': 57, 'UAI': 100, 'LTO': 51, 'IVR': 50}
}

# ✅ VSM24 问题编号（从 1 开始）
# 用于霍夫斯泰德公式计算
# 注意：题号从 1 开始，但在代码中使用 0-based 索引
# VSM24 一共 24 个问题
QUESTION_NUMBERS = {
    0: 1,    # 第 0 个问题 = Q1
    1: 2,    # 第 1 个问题 = Q2
    2: 3,    # 第 2 个问题 = Q3
    3: 4,    # 第 3 个问题 = Q4
    4: 5,    # 第 4 个问题 = Q5
    5: 6,    # 第 5 个问题 = Q6
    6: 7,    # 第 6 个问题 = Q7
    7: 8,    # 第 7 个问题 = Q8
    8: 9,    # 第 8 个问题 = Q9
    9: 10,   # 第 9 个问题 = Q10
    10: 11,  # 第 10 个问题 = Q11
    11: 12,  # 第 11 个问题 = Q12
    12: 13,  # 第 12 个问题 = Q13
    13: 14,  # 第 13 个问题 = Q14
    14: 15,  # 第 14 个问题 = Q15
    15: 16,  # 第 15 个问题 = Q16
    16: 17,  # 第 16 个问题 = Q17
    17: 18,  # 第 17 个问题 = Q18
    18: 19,  # 第 18 个问题 = Q19
    19: 20,  # 第 19 个问题 = Q20
    20: 21,  # 第 20 个问题 = Q21
    21: 22,  # 第 21 个问题 = Q22
    22: 23,  # 第 22 个问题 = Q23
    23: 24   # 第 23 个问题 = Q24
}

# ✅ 霍夫斯泰德维度计算公式中的常量
HOFSTEDE_CONSTANTS = {
    'PDI': 3,
    'IDV': 3,
    'MAS': 3,
    'UAI': 3,
    'LTO': 3,
    'IVR': 3
}


def load_model_and_tokenizer(
    model_type: str = 'lora_only',
    backbone: str = 'qwen',
    base_model_path: str = None,
    lora_weights_path: str = None,
    moe_weights_path: str = None,
    device: str = 'cuda'
):
    """
    加载模型和 tokenizer

    Args:
        model_type: 模型类型 ('base', 'lora_only', 'moe' 或 'culturemoe')
        backbone: 基座模型 ('qwen' 或 'llama')
        base_model_path: Base 模型路径（可选，如果不提供则自动推断）
        lora_weights_path: LoRA 权重路径（仅用于 lora_only 和 moe）
        moe_weights_path: MOE 权重路径（仅用于 moe）
        device: 设备

    Returns:
        model, tokenizer
    """
    # ✅ 如果没有提供路径，根据 backbone 自动推断
    if base_model_path is None:
        if backbone == 'qwen':
            base_model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
        else:  # llama
            base_model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"

    if model_type == 'lora_only' and lora_weights_path is None:
        if backbone == 'qwen':
            lora_weights_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora"
        else:  # llama
            lora_weights_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora"

    if model_type == 'moe' and lora_weights_path is None:
        if backbone == 'qwen':
            lora_weights_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_qwen_20251109_1549/best_lora"
        else:  # llama
            lora_weights_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/lora_only_gen_cultureLLM_llama_20251107_2124/best_lora"

    if model_type == 'moe' and moe_weights_path is None:
        if backbone == 'qwen':
            moe_weights_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_qwen_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe"
        else:  # llama
            moe_weights_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/gen/moe_cultureLLM_llama_experts6_USE_CULTURE_LOSSTrue_MASK_USEtrue_20251109_1323/best_moe"

    print(f"\n{'='*80}")
    print("Loading Model")
    print(f"{'='*80}")
    print(f"Model type: {model_type}")
    print(f"Backbone: {backbone}")
    print(f"Base model path: {base_model_path}")
    if model_type == 'lora_only':
        print(f"LoRA weights path: {lora_weights_path}")
    if model_type == 'moe':
        print(f"LoRA weights path: {lora_weights_path}")
        print(f"MOE weights path: {moe_weights_path}")
    print(f"{'='*80}\n")

    # 加载 tokenizer（从 base 模型或 lora 权重）
    if model_type in ['lora_only', 'moe']:
        tokenizer_path = lora_weights_path
    else:
        tokenizer_path = base_model_path

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")

    # 加载模型
    if model_type == 'base':
        # Base 模型：直接加载
        print(f"\nLoading base model from {base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Base model loaded")

    elif model_type == 'lora_only':
        # LoRA Only 模型：加载 base 模型 + LoRA 权重
        print(f"\nLoading base model from {base_model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Base model loaded")

        print(f"\nLoading LoRA weights from {lora_weights_path}...")
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights_path,
            is_trainable=False,
            torch_dtype=torch.float16
        )
        print("✅ LoRA weights loaded")

        print(f"\nMerging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("✅ LoRA weights merged")

    elif model_type == 'moe':
        # MOE 模型：加载 base 模型 + LoRA 权重 + MOE 权重
        print(f"\nLoading base model from {base_model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Base model loaded")

        print(f"\nLoading LoRA weights from {lora_weights_path}...")
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights_path,
            is_trainable=False,
            torch_dtype=torch.float16
        )
        print("✅ LoRA weights loaded")

        print(f"\nMerging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("✅ LoRA weights merged")

        print(f"\nLoading MOE weights from {moe_weights_path}...")
        moe_state_dict = torch.load(
            os.path.join(moe_weights_path, 'pytorch_model.bin'),
            map_location='cpu'
        )
        model.load_state_dict(moe_state_dict, strict=False)
        print("✅ MOE weights loaded and merged")

    else:  # culturemoe
        # ⚠️ CultureMoE 模型：需要从 Base + LoRA + MOE 权重还原
        # 这与 'moe' 类型相同，但用于评估已训练的完整 CultureMoE 模型
        print(f"\n⚠️  CultureMoE 模型需要从 Base + LoRA + MOE 权重还原")
        print(f"如果你想评估完整的 CultureMoE 模型，请使用 --model_type moe")
        print(f"或者提供 --moe_weights_path 参数")

        # 尝试从 base_model_path 加载（如果是完整的 CultureMoE 模型）
        print(f"\nAttempting to load CultureMoE model from {base_model_path}...")
        try:
            model = LlamaSharedRouterExpertsModel.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("✅ CultureMoE model loaded from pretrained")
        except Exception as e:
            print(f"❌ Failed to load CultureMoE from pretrained: {e}")
            print(f"Falling back to loading as Base model...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("⚠️  Loaded as Base model (not CultureMoE)")

    model.eval()
    print("\n✅ Model ready for evaluation")

    return model, tokenizer


def extract_number_from_text(text: str, min_val: int = 1, max_val: int = 5) -> int:
    """
    从生成的文本中提取数字答案

    Args:
        text: 生成的文本
        min_val: 最小值
        max_val: 最大值

    Returns:
        提取的数字，如果没有找到则返回 -1
    """
    # 提取所有数字
    numbers = re.findall(r'\d+', text)

    if not numbers:
        return -1

    # 返回第一个在范围内的数字
    for num_str in numbers:
        num = int(num_str)
        if min_val <= num <= max_val:
            return num

    return -1


def generate_answer(model, tokenizer, instruction: str, input_text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    """
    使用模型生成答案

    Args:
        model: 模型
        tokenizer: tokenizer
        instruction: 指令
        input_text: 输入文本
        device: 设备
        max_new_tokens: 最大生成 token 数

    Returns:
        生成的文本
    """
    # 构建完整的输入 - 与训练时格式保持一致
    # 🔧 格式统一：使用空格分隔，避免tokenizer自动格式化
    if input_text:
        full_input = f"{instruction}\n{input_text} "
    else:
        full_input = f"{instruction} "

    # Tokenize
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None
        )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text


def evaluate_vsm13(
    model,
    tokenizer,
    data_path: str,
    output_dir: str,
    device: str = 'cuda'
) -> Dict:
    """
    在 VSM13 数据集上评估模型

    Args:
        model: 模型
        tokenizer: tokenizer
        data_path: 数据集路径
        output_dir: 输出目录
        device: 设备

    Returns:
        评估结果字典
    """
    print(f"\nLoading VSM13 dataset from {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # 存储每个国家的答案和完整数据
    country_answers = defaultdict(lambda: defaultdict(list))  # country -> question_idx -> [answers]
    country_generated_data = defaultdict(list)  # country -> [完整数据集（包含生成的答案）]

    # 对每个国家进行评估
    for country, country_adj in COUNTRIES.items():
        print(f"\n{'='*80}")
        print(f"Evaluating {country}...")
        print(f"{'='*80}")

        # 为每个问题生成答案
        for sample_idx, sample in enumerate(tqdm(data, desc=f"Generating answers for {country}")):
            # ✅ 获取问题索引（从样本中提取或使用样本索引）
            # 假设数据中有 'question_idx' 字段，否则使用 sample_idx
            question_idx = sample.get('question_idx', sample_idx)

            # 添加国家文化提示词
            instruction = sample['instruction']

            # 在最前面添加国家提示词
            instruction_with_country = f"You are a {country_adj} culture chatbot that knows {country_adj} culture very well. {instruction}"

            # 在最后面添加国家提示词
            instruction_with_country += f" This question is for a country or language that is {country}."

            input_text = sample['input']

            # 生成答案
            generated_text = generate_answer(model, tokenizer, instruction_with_country, input_text, device)

            # 提取数字答案
            answer = extract_number_from_text(generated_text)

            # ✅ 调试输出：显示生成的答案
            if sample_idx < 3:  # 只显示前 3 个样本
                print(f"  Sample {sample_idx}: Generated='{generated_text}' → Answer={answer}")

            if answer == -1:
                # 如果没有提取到有效的数字，跳过
                if sample_idx < 10:  # 只显示前 10 个失败的样本
                    print(f"  ⚠️  Sample {sample_idx}: Failed to extract number from: '{generated_text}'")
                continue

            # 存储答案
            country_answers[country][question_idx].append(answer)

            # ✅ 创建完整的数据记录（包含生成的答案）
            sample_with_answer = sample.copy()
            sample_with_answer['output'] = str(answer)
            sample_with_answer['generated_text'] = generated_text
            sample_with_answer['question_idx'] = question_idx
            country_generated_data[country].append(sample_with_answer)

    # ✅ 计算每个国家在每个维度上的分数（使用霍夫斯泰德公式）
    print(f"\n{'='*80}")
    print("Computing dimension scores using Hofstede formulas...")
    print(f"{'='*80}")

    country_dimension_scores = {}  # country -> dimension -> score

    for country in COUNTRIES.keys():
        country_dimension_scores[country] = {}

        # ✅ 获取该国家的所有问题答案
        # 创建一个字典：问题编号（从 1 开始）-> 平均答案
        question_means = {}
        for q_idx in range(len(data)):
            if q_idx in country_answers[country]:
                answers = country_answers[country][q_idx]
                if answers:
                    question_means[QUESTION_NUMBERS[q_idx]] = np.mean(answers)

        # ✅ 使用霍夫斯泰德公式计算各维度分数（完整的 VSM24 公式）
        # PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + C_PDI
        pdi = 35 * (question_means.get(7, 0) - question_means.get(2, 0)) + \
              25 * (question_means.get(20, 0) - question_means.get(23, 0)) + HOFSTEDE_CONSTANTS['PDI']
        country_dimension_scores[country]['PDI'] = max(0, min(100, pdi))

        # IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + C_IDV
        idv = 35 * (question_means.get(4, 0) - question_means.get(1, 0)) + \
              35 * (question_means.get(9, 0) - question_means.get(6, 0)) + HOFSTEDE_CONSTANTS['IDV']
        country_dimension_scores[country]['IDV'] = max(0, min(100, idv))

        # MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + C_MAS
        mas = 35 * (question_means.get(5, 0) - question_means.get(3, 0)) + \
              25 * (question_means.get(8, 0) - question_means.get(10, 0)) + HOFSTEDE_CONSTANTS['MAS']
        country_dimension_scores[country]['MAS'] = max(0, min(100, mas))

        # UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + C_UAI
        uai = 40 * (question_means.get(18, 0) - question_means.get(15, 0)) + \
              25 * (question_means.get(21, 0) - question_means.get(24, 0)) + HOFSTEDE_CONSTANTS['UAI']
        country_dimension_scores[country]['UAI'] = max(0, min(100, uai))

        # LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + C_LTO
        lto = 40 * (question_means.get(13, 0) - question_means.get(14, 0)) + \
              25 * (question_means.get(19, 0) - question_means.get(22, 0)) + HOFSTEDE_CONSTANTS['LTO']
        country_dimension_scores[country]['LTO'] = max(0, min(100, lto))

        # IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + C_IVR
        ivr = 35 * (question_means.get(12, 0) - question_means.get(11, 0)) + \
              40 * (question_means.get(17, 0) - question_means.get(16, 0)) + HOFSTEDE_CONSTANTS['IVR']
        country_dimension_scores[country]['IVR'] = max(0, min(100, ivr))

    # 计算欧式距离
    print(f"\n{'='*80}")
    print("Computing Euclidean distances...")
    print(f"{'='*80}")

    euclidean_distances = {}

    for country in COUNTRIES.keys():
        # 获取模型预测的分数
        predicted_scores = country_dimension_scores[country]

        # 获取霍夫斯泰德标准分数
        hofstede_scores = HOFSTEDE_SCORES[country]

        # 计算欧式距离
        distance = 0
        for dimension in ['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR']:
            predicted = predicted_scores.get(dimension, 0)
            standard = hofstede_scores.get(dimension, 0)
            distance += (predicted - standard) ** 2

        euclidean_distances[country] = np.sqrt(distance)

    # ✅ 保存结果
    results = {
        'country_dimension_scores': country_dimension_scores,
        'hofstede_scores': HOFSTEDE_SCORES,
        'euclidean_distances': euclidean_distances,
        'average_euclidean_distance': np.mean(list(euclidean_distances.values()))
    }

    # ✅ 创建详细的分数和距离报告
    detailed_report = {}
    for country in sorted(COUNTRIES.keys()):
        detailed_report[country] = {
            'dimension_scores': country_dimension_scores[country],
            'hofstede_scores': HOFSTEDE_SCORES[country],
            'euclidean_distance': euclidean_distances[country]
        }

    # 保存到文件
    os.makedirs(output_dir, exist_ok=True)

    # ✅ 保存主要结果
    with open(os.path.join(output_dir, 'vsm13_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ✅ 保存详细的分数和距离报告
    with open(os.path.join(output_dir, 'vsm13_detailed_scores.json'), 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, indent=2, ensure_ascii=False)

    # ✅ 保存每个国家的生成数据（包含生成的答案）
    for country in COUNTRIES.keys():
        country_data = country_generated_data[country]
        output_file = os.path.join(output_dir, f'vsm13_generated_{country}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(country_data, f, indent=2, ensure_ascii=False)

    # ✅ 保存汇总的所有国家生成数据
    all_generated_data = {}
    for country in COUNTRIES.keys():
        all_generated_data[country] = country_generated_data[country]

    with open(os.path.join(output_dir, 'vsm13_all_generated.json'), 'w', encoding='utf-8') as f:
        json.dump(all_generated_data, f, indent=2, ensure_ascii=False)

    # 打印结果
    print(f"\n{'='*80}")
    print("VSM13 Evaluation Results")
    print(f"{'='*80}")

    print("\n📊 Dimension Scores by Country:")
    print("-" * 80)
    print(f"{'Country':<20} {'PDI':<10} {'IDV':<10} {'MAS':<10} {'UAI':<10} {'LTO':<10} {'IVR':<10}")
    print("-" * 80)

    for country in sorted(COUNTRIES.keys()):
        scores = country_dimension_scores[country]
        print(f"{country:<20} {scores.get('PDI', 0):<10.2f} {scores.get('IDV', 0):<10.2f} "
              f"{scores.get('MAS', 0):<10.2f} {scores.get('UAI', 0):<10.2f} "
              f"{scores.get('LTO', 0):<10.2f} {scores.get('IVR', 0):<10.2f}")

    print("\n📊 Hofstede Standard Scores:")
    print("-" * 80)
    print(f"{'Country':<20} {'PDI':<10} {'IDV':<10} {'MAS':<10} {'UAI':<10} {'LTO':<10} {'IVR':<10}")
    print("-" * 80)

    for country in sorted(COUNTRIES.keys()):
        scores = HOFSTEDE_SCORES[country]
        print(f"{country:<20} {scores.get('PDI', 0):<10} {scores.get('IDV', 0):<10} "
              f"{scores.get('MAS', 0):<10} {scores.get('UAI', 0):<10} "
              f"{scores.get('LTO', 0):<10} {scores.get('IVR', 0):<10}")

    print("\n📊 Euclidean Distances:")
    print("-" * 80)

    for country in sorted(euclidean_distances.keys(), key=lambda x: euclidean_distances[x]):
        distance = euclidean_distances[country]
        print(f"{country:<20} {distance:<10.2f}")

    print("-" * 80)
    print(f"{'Average':<20} {results['average_euclidean_distance']:<10.2f}")

    print(f"\n✅ Results saved to {output_dir}")
    print(f"\n📁 Generated files:")
    print(f"   - vsm13_results.json (主要结果)")
    print(f"   - vsm13_detailed_scores.json (详细的分数和距离报告)")
    print(f"   - vsm13_all_generated.json (所有国家的生成数据汇总)")
    for country in sorted(COUNTRIES.keys()):
        print(f"   - vsm13_generated_{country}.json (国家: {country})")

    return results


def main():
    parser = argparse.ArgumentParser(description="VSM13 Evaluation Script")

    parser.add_argument("--model_type", type=str, default='lora_only',
                        choices=['base', 'lora_only', 'moe', 'culturemoe'],
                        help="Model type: 'base', 'lora_only', 'moe' or 'culturemoe'")
    parser.add_argument("--backbone", type=str, default='qwen',
                        choices=['qwen', 'llama'],
                        help="Backbone model: 'qwen' or 'llama'")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to base model (optional, auto-inferred if not provided)")
    parser.add_argument("--lora_weights_path", type=str, default=None,
                        help="Path to LoRA weights (only for lora_only and moe model types)")
    parser.add_argument("--moe_weights_path", type=str, default=None,
                        help="Path to MOE weights (only for moe model type)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to VSM13 dataset (vsm13.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("VSM13 Evaluation")
    print("="*80)
    print(f"Model type: {args.model_type}")
    print(f"Backbone: {args.backbone}")
    if args.base_model_path:
        print(f"Base model path: {args.base_model_path}")
    if args.lora_weights_path:
        print(f"LoRA weights path: {args.lora_weights_path}")
    if args.moe_weights_path:
        print(f"MOE weights path: {args.moe_weights_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80)
    print("")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_type=args.model_type,
        backbone=args.backbone,
        base_model_path=args.base_model_path,
        lora_weights_path=args.lora_weights_path,
        moe_weights_path=args.moe_weights_path,
        device=args.device
    )

    # 评估
    results = evaluate_vsm13(model, tokenizer, args.data_path, args.output_dir, args.device)

    print("\n" + "="*80)
    print("✅ Evaluation completed!")
    print("="*80)


if __name__ == "__main__":
    main()

