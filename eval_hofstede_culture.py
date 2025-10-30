#!/usr/bin/env python3
"""
评估 Base LLaMA 模型的霍夫斯泰德文化维度一致性
基于 VSM2013 的 24 个问题评估模型的文化倾向
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 国家名称到形容词的映射
COUNTRY_TO_ADJECTIVE = {
    "China": "Chinese",
    "South Korea": "South Korean",
    "Turkey": "Turkish",
    "Saudi Arabia": "Saudi Arabian",
    "Bangladesh": "Bangladeshi",
    "Germany": "German",
    "Portugal": "Portuguese",
    "Spain": "Spanish"
}

# 霍夫斯泰德给定的各国文化维度得分
# 顺序：[PDI, IDV, MAS, UAI, LTO, IVR]
HOFSTEDE_SCORES = {
    "China": [80, 43, 66, 30, 77, 24],
    "South Korea": [60, 58, 39, 85, 86, 29],
    "Turkey": [66, 46, 45, 85, 35, 49],
    "Saudi Arabia": [72, 48, 43, 64, 27, 14],
    "Bangladesh": [80, 5, 55, 60, 38, 20],
    "Germany": [35, 79, 66, 65, 57, 40],
    "Portugal": [63, 59, 31, 99, 42, 33],
    "Spain": [57, 67, 42, 86, 47, 44]
}

# 维度名称
DIMENSION_NAMES = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]

# 常量 C
C_PDI = 3
C_IDV = 3
C_MAS = 3
C_UAI = 3
C_LTO = 3
C_IVR = 3


def load_vsm2013_questions(data_path: str) -> List[Dict]:
    """加载 VSM2013 问题"""
    with open(data_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions


def create_prompt(country_adjective: str, question: str) -> str:
    """创建提示词"""
    system_prompt = f"You are a {country_adjective} culture chatbot that knows {country_adjective} culture very well."
    full_prompt = f"{system_prompt}\n\n{question}\n\nPlease answer with only a number (1-5):"
    return full_prompt


def extract_answer(response: str) -> int:
    """从模型回答中提取数字答案"""
    # 尝试提取第一个数字
    import re
    numbers = re.findall(r'\b[1-5]\b', response)
    if numbers:
        return int(numbers[0])

    # 如果没找到，尝试提取任何数字
    numbers = re.findall(r'\d+', response)
    if numbers:
        num = int(numbers[0])
        if 1 <= num <= 5:
            return num

    # 默认返回 3（中间值）
    print(f"  ⚠️  Warning: Could not extract valid answer from response, using default 3")
    return 3


def generate_answer(model, tokenizer, prompt: str, device: str = "cuda") -> str:
    """使用模型生成答案"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 只取生成的新 token
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def calculate_dimensions(answers: Dict[str, int]) -> Dict[str, float]:
    """
    根据 VSM2013 公式计算六个文化维度得分

    PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + C_PDI
    IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + C_IDV
    MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + C_MAS
    UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + C_UAI
    LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + C_LTO
    IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + C_IVR
    """
    dimensions = {}

    # PDI (Power Distance Index)
    dimensions["PDI"] = 35 * (answers["Q7"] - answers["Q2"]) + 25 * (answers["Q20"] - answers["Q23"]) + C_PDI

    # IDV (Individualism vs Collectivism)
    dimensions["IDV"] = 35 * (answers["Q4"] - answers["Q1"]) + 35 * (answers["Q9"] - answers["Q6"]) + C_IDV

    # MAS (Masculinity vs Femininity)
    dimensions["MAS"] = 35 * (answers["Q5"] - answers["Q3"]) + 25 * (answers["Q8"] - answers["Q10"]) + C_MAS

    # UAI (Uncertainty Avoidance Index)
    dimensions["UAI"] = 40 * (answers["Q18"] - answers["Q15"]) + 25 * (answers["Q21"] - answers["Q24"]) + C_UAI

    # LTO (Long-Term Orientation)
    dimensions["LTO"] = 40 * (answers["Q13"] - answers["Q14"]) + 25 * (answers["Q19"] - answers["Q22"]) + C_LTO

    # IVR (Indulgence vs Restraint)
    dimensions["IVR"] = 35 * (answers["Q12"] - answers["Q11"]) + 40 * (answers["Q17"] - answers["Q16"]) + C_IVR

    return dimensions


def calculate_euclidean_distance(model_scores: List[float], hofstede_scores: List[float]) -> float:
    """计算欧氏距离"""
    model_array = np.array(model_scores)
    hofstede_array = np.array(hofstede_scores)
    distance = np.sqrt(np.sum((model_array - hofstede_array) ** 2))
    return distance


def evaluate_country(
    model,
    tokenizer,
    questions: List[Dict],
    country: str,
    output_dir: str,
    device: str = "cuda"
) -> Tuple[Dict[str, float], float]:
    """评估单个国家的文化维度"""

    country_adjective = COUNTRY_TO_ADJECTIVE[country]
    print(f"\n{'='*80}")
    print(f"Evaluating country: {country} ({country_adjective})")
    print(f"{'='*80}")

    # 存储答案
    answers = {}
    answered_questions = []

    # 对每个问题生成答案
    for i, q in enumerate(questions, 1):
        question_id = q["id"]
        question_text = q["Question"]

        print(f"\n[{i}/24] Question {question_id}:")
        print(f"  {question_text[:100]}...")

        # 创建提示词
        prompt = create_prompt(country_adjective, question_text)

        # 生成答案
        response = generate_answer(model, tokenizer, prompt, device)
        answer = extract_answer(response)

        print(f"  Response: {response}")
        print(f"  Answer: {answer}")

        # 保存答案
        answers[question_id] = answer

        # 更新问题数据
        q_copy = q.copy()
        q_copy["Answer"] = answer
        answered_questions.append(q_copy)

    # 保存带答案的问题文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"VSM2013_{country.replace(' ', '_')}_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(answered_questions, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved answered questions to: {output_file}")

    # 计算文化维度得分
    dimensions = calculate_dimensions(answers)

    # 获取霍夫斯泰德得分
    hofstede_scores = HOFSTEDE_SCORES[country]
    model_scores = [dimensions[dim] for dim in DIMENSION_NAMES]

    # 计算欧氏距离
    distance = calculate_euclidean_distance(model_scores, hofstede_scores)

    # 打印结果
    print(f"\n{'='*80}")
    print(f"Results for {country}:")
    print(f"{'='*80}")
    print(f"{'Dimension':<15} {'Model Score':<15} {'Hofstede Score':<15} {'Difference':<15}")
    print(f"{'-'*80}")
    for i, dim in enumerate(DIMENSION_NAMES):
        model_score = model_scores[i]
        hofstede_score = hofstede_scores[i]
        diff = model_score - hofstede_score
        print(f"{dim:<15} {model_score:<15.2f} {hofstede_score:<15} {diff:<15.2f}")
    print(f"{'-'*80}")
    print(f"Euclidean Distance: {distance:.2f}")
    print(f"{'='*80}")

    return dimensions, distance


def main():
    import argparse

    parser = argparse.ArgumentParser(description="评估模型的霍夫斯泰德文化维度一致性")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="VSM2013 问题文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"\n{'='*80}")
    print(f"Loading model: {args.model_path}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True
    )

    if args.device == "cpu":
        model = model.to(args.device)

    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Model name: {args.model_path.split('/')[-1]}")

    # 加载问题
    print(f"\nLoading VSM2013 questions from: {args.data_path}")
    questions = load_vsm2013_questions(args.data_path)
    print(f"✅ Loaded {len(questions)} questions")

    # 评估所有国家
    all_results = {}

    for country in COUNTRY_TO_ADJECTIVE.keys():
        dimensions, distance = evaluate_country(
            model, tokenizer, questions, country, args.output_dir, args.device
        )
        all_results[country] = {
            "dimensions": dimensions,
            "distance": distance
        }

    # 保存总结结果
    summary_file = os.path.join(args.output_dir, f"hofstede_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    summary = {
        "model_path": args.model_path,
        "model_name": args.model_path.split('/')[-1],
        "evaluation_time": datetime.now().isoformat(),
        "results": {}
    }

    for country, result in all_results.items():
        summary["results"][country] = {
            "model_scores": {dim: result["dimensions"][dim] for dim in DIMENSION_NAMES},
            "hofstede_scores": dict(zip(DIMENSION_NAMES, HOFSTEDE_SCORES[country])),
            "euclidean_distance": result["distance"]
        }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"✅ Evaluation completed!")
    print(f"{'='*80}")
    print(f"Summary saved to: {summary_file}")

    # 打印总体统计
    print(f"\n{'='*80}")
    print(f"Overall Statistics:")
    print(f"{'='*80}")
    print(f"{'Country':<20} {'Euclidean Distance':<20}")
    print(f"{'-'*80}")
    for country, result in all_results.items():
        print(f"{country:<20} {result['distance']:<20.2f}")
    print(f"{'-'*80}")
    avg_distance = np.mean([r["distance"] for r in all_results.values()])
    print(f"{'Average':<20} {avg_distance:<20.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

