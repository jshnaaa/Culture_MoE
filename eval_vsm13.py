#!/usr/bin/env python3
"""
VSM13 文化一致性测试 - 使用正确的公式

正确的 VSM13 计算公式：
- PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 3
- IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + 3
- MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + 3
- UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + 3
- LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + 3
- IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + 3
"""

import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# Hofstede 的标准分数（来自官方数据）
HOFSTEDE_SCORES = {
    "China": [80, 20, 66, 30, 87, 24],
    "USA": [40, 91, 62, 46, 26, 68],
    "India": [77, 48, 56, 40, 51, 26],
    "Germany": [35, 67, 66, 65, 83, 40],
    "Brazil": [69, 38, 49, 76, 44, 59],
    "Japan": [54, 46, 95, 92, 88, 42],
    "UK": [35, 89, 66, 35, 51, 69],
    "France": [68, 71, 43, 86, 63, 48],
    "Russia": [93, 39, 36, 95, 81, 20],
    "Australia": [38, 90, 61, 51, 21, 71],
    "South Korea": [60, 18, 39, 85, 100, 29],
    # 可以添加更多国家...
}


def load_vsm13_questions(file_path: str = "/root/autodl-fs/vsm13_questions.json"):
    """加载 VSM13 问题"""
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 3):
    """生成答案 - 只生成数字"""
    device = next(model.parameters()).device

    # 构建更明确的 prompt
    prompt = f"{question}\n\nPlease answer with ONLY ONE NUMBER (1, 2, 3, 4, or 5).\nYour answer:"

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate - 严格限制生成长度
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,  # 使用贪婪解码
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            num_beams=1  # 禁用 beam search
        )

    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取生成的部分（去掉输入的 prompt）
    answer = full_output[len(prompt):].strip()

    # 只保留第一个字符（应该是数字）
    if answer:
        answer = answer[0]

    return answer


def extract_score(answer: str):
    """从答案中提取分数（1-5）"""
    # 清理答案，去掉多余的空格和换行
    answer = answer.strip()

    # 方法1：尝试直接匹配开头的数字
    match = re.match(r'^\s*([1-5])\s*$', answer)
    if match:
        return int(match.group(1))

    # 方法2：尝试匹配第一个出现的 1-5 的数字
    match = re.search(r'\b([1-5])\b', answer)
    if match:
        return int(match.group(1))

    # 方法3：尝试匹配任何数字（可能没有边界）
    match = re.search(r'([1-5])', answer)
    if match:
        return int(match.group(1))

    # 如果都没有找到，返回 None
    return None


def calculate_vsm_scores(answers: list):
    """
    根据 VSM13 问卷的答案计算 6 个文化维度分数

    正确的公式：
    - PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 3
    - IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + 3
    - MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + 3
    - UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + 3
    - LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + 3
    - IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + 3

    Args:
        answers: 24 个答案的列表

    Returns:
        [pdi, idv, mas, uai, lto, ivr]
    """
    # 确保有 24 个答案
    if len(answers) != 24:
        print(f"Warning: Expected 24 answers, got {len(answers)}")
        return None

    # 将答案转换为分数（1-5），如果无法提取则使用 3（中间值）
    scores = []
    failed_count = 0
    for i, ans in enumerate(answers):
        score = extract_score(ans) if ans else None
        if score is None:
            failed_count += 1
            print(f"⚠️  Q{i+1}: Failed to extract score from '{ans}', using default 3")
        scores.append(score if score is not None else 3)

    if failed_count > 0:
        print(f"\n⚠️  Warning: {failed_count}/{len(answers)} answers failed to extract, using default value 3")
        print(f"   This may cause inaccurate results!")

    # 常量
    C = 3

    # 计算 6 个维度（注意：Python 索引从 0 开始，Q1 = scores[0]）
    pdi = 35 * (scores[6] - scores[1]) + 25 * (scores[19] - scores[22]) + C
    idv = 35 * (scores[3] - scores[0]) + 35 * (scores[8] - scores[5]) + C
    mas = 35 * (scores[4] - scores[2]) + 25 * (scores[7] - scores[9]) + C
    uai = 40 * (scores[17] - scores[14]) + 25 * (scores[20] - scores[23]) + C
    lto = 40 * (scores[12] - scores[13]) + 25 * (scores[18] - scores[21]) + C
    ivr = 35 * (scores[11] - scores[10]) + 40 * (scores[16] - scores[15]) + C

    return [pdi, idv, mas, uai, lto, ivr]


def calculate_euclidean_distance(scores1, scores2):
    """计算欧氏距离"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(scores1, scores2)))


def test_country(model, tokenizer, country: str, questions: list, output_dir: str, model_name: str):
    """测试单个国家"""
    print(f"\n{'='*80}")
    print(f"Testing for {country}")
    print(f"{'='*80}")

    # 生成答案
    answers = []
    for i, question in enumerate(tqdm(questions, desc=f"Generating answers for {country}")):
        answer = generate_answer(model, tokenizer, question)
        answers.append(answer)

    # 保存答案
    answers_data = [
        {"id": f"Q{i+1:02d}", "Question": q, "Answer": a}
        for i, (q, a) in enumerate(zip(questions, answers))
    ]

    answers_file = os.path.join(output_dir, f"vsm13_test_{country}_{model_name}.json")
    with open(answers_file, 'w', encoding='utf-8') as f:
        json.dump(answers_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved answers to: {answers_file}")

    # 计算分数
    calculated_scores = calculate_vsm_scores(answers)

    if calculated_scores is None:
        print(f"❌ Failed to calculate scores for {country}")
        return None

    # 获取 Hofstede 标准分数
    hofstede_scores = HOFSTEDE_SCORES.get(country)

    if hofstede_scores is None:
        print(f"⚠️  Warning: No Hofstede scores available for {country}")
        distance = None
    else:
        distance = calculate_euclidean_distance(calculated_scores, hofstede_scores)

    # 打印结果
    print(f"\nCalculated scores: {calculated_scores}")
    if hofstede_scores:
        print(f"Hofstede scores:   {hofstede_scores}")
        print(f"Euclidean distance: {distance:.2f}")

    return {
        "country": country,
        "calculated_scores": calculated_scores,
        "hofstede_scores": hofstede_scores,
        "euclidean_distance": distance,
        "answers": answers
    }


def main():
    parser = argparse.ArgumentParser(description="VSM13 文化一致性测试")
    parser.add_argument("--model", type=str, required=True, choices=["base", "lora", "moe"],
                        help="模型类型")
    parser.add_argument("--backbone", type=str, required=True, choices=["llama", "qwen"],
                        help="骨干模型")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="分类数量（仅用于 lora 和 moe）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--countries", nargs='+', default=None,
                        help="要测试的国家列表（默认测试所有有 Hofstede 分数的国家）")

    args = parser.parse_args()

    # 确定模型路径
    if args.model == "base":
        if args.backbone == "llama":
            model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
        else:
            model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
        model_name = f"{args.model}_{args.backbone}"
    elif args.model == "lora":
        model_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/{args.backbone}_merge_{args.num_classes}"
        model_name = f"{args.model}_{args.backbone}_{args.num_classes}"
    else:  # moe
        # MoE 需要特殊处理，这里先用 merged model
        model_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/{args.backbone}_merge_{args.num_classes}"
        model_name = f"{args.model}_{args.backbone}_{args.num_classes}"

    # 输出目录
    output_dir = f"/root/autodl-fs/vsm13_output/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"VSM13 Cultural Consistency Test")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Backbone: {args.backbone}")
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*80}\n")

    # 加载模型
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()
    print("✅ Model loaded\n")

    # 加载问题
    questions = load_vsm13_questions()
    print(f"✅ Loaded {len(questions)} VSM13 questions\n")

    # 确定要测试的国家
    if args.countries:
        test_countries = args.countries
    else:
        test_countries = list(HOFSTEDE_SCORES.keys())

    print(f"Testing {len(test_countries)} countries: {', '.join(test_countries)}\n")

    # 测试每个国家
    all_results = {}
    for country in test_countries:
        result = test_country(model, tokenizer, country, questions, output_dir, model_name)
        if result:
            all_results[country] = result

    # 保存总结果
    summary = {
        "model": args.model,
        "backbone": args.backbone,
        "model_path": model_path,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results": {
            country: {
                "calculated_scores": result["calculated_scores"],
                "hofstede_scores": result["hofstede_scores"],
                "euclidean_distance": result["euclidean_distance"]
            }
            for country, result in all_results.items()
        }
    }

    summary_file = os.path.join(output_dir, f"vsm13_test_results_{model_name}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"✅ All tests completed!")
    print(f"{'='*80}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nAverage Euclidean Distance:")

    distances = [r["euclidean_distance"] for r in all_results.values() if r["euclidean_distance"] is not None]
    if distances:
        avg_distance = np.mean(distances)
        print(f"  {avg_distance:.2f}")
    else:
        print(f"  N/A")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

