#!/usr/bin/env python3
"""
VSM13 文化一致性测试 - 使用正确的公式

正确的 VSM13 计算公式：
- PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 50
- IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + 50
- MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + 50
- UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + 50
- LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + 50
- IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + 50

常量设为 50 确保分数在合理范围内（通常 0-100）
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
    "China": [80, 43, 66, 30, 77, 24],
    "South Korea": [60, 58, 39, 85, 86, 29],
    "Turkey": [66, 46, 45, 85, 35, 49],
    "Saudi Arabia": [72, 48, 43, 64, 27, 14],
    "Bangladesh": [80, 5, 55, 60, 38, 20],
    "Germany": [35, 79, 66, 65, 57, 40],
    "Portugal": [63, 59, 31, 99, 42, 33],
    "Spain": [57, 67, 42, 86, 47, 44],
    "England": [35, 76, 66, 35, 60, 69],
    "Greece": [60, 59, 57, 100, 51, 50],
}

# 国家名到形容词的映射
COUNTRY_TO_ADJECTIVE = {
    "China": "Chinese",
    "South Korea": "South Korean",
    "Turkey": "Turkish",
    "Saudi Arabia": "Saudi Arabian",
    "Bangladesh": "Bangladeshi",
    "Germany": "German",
    "Portugal": "Portuguese",
    "Spain": "Spanish",
    "England": "English",
    "Greece": "Greek",
}


def load_vsm13_questions(file_path: str = "/root/autodl-fs/vsm13_test.json"):
    """
    加载 VSM13 问题

    文件格式：
    [
        {"id": "Q01", "Question": "...", "Answer": null},
        ...
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取问题列表
    questions = [item['Question'] for item in data]

    return questions


def generate_answer(model, tokenizer, question: str, country: str, max_new_tokens: int = 3):
    """生成答案 - 只生成数字"""
    device = next(model.parameters()).device

    # 获取国家的形容词形式
    country_adj = COUNTRY_TO_ADJECTIVE.get(country, country)

    # 构建带有文化背景的 prompt
    prompt = f"You are a {country_adj} culture chatbot that knows {country_adj} culture very well. Please answer with ONLY ONE NUMBER (1, 2, 3, 4, or 5) based on the {country}'s cultural background.\n\n{question}\n\nYour answer:"

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

    正确的公式（常量 = 50）：
    - PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 50
    - IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + 50
    - MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + 50
    - UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + 50
    - LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + 50
    - IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + 50

    常量设为 50 确保分数在合理范围内（0-100）

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

    # 常量（根据 Hofstede VSM13 官方文档）
    # 这些常量确保分数在合理范围内（通常 0-100）
    C_PDI = 50
    C_IDV = 50
    C_MAS = 50
    C_UAI = 50
    C_LTO = 50
    C_IVR = 50

    # 计算 6 个维度（注意：Python 索引从 0 开始，Q1 = scores[0]）
    pdi = 35 * (scores[6] - scores[1]) + 25 * (scores[19] - scores[22]) + C_PDI
    idv = 35 * (scores[3] - scores[0]) + 35 * (scores[8] - scores[5]) + C_IDV
    mas = 35 * (scores[4] - scores[2]) + 25 * (scores[7] - scores[9]) + C_MAS
    uai = 40 * (scores[17] - scores[14]) + 25 * (scores[20] - scores[23]) + C_UAI
    lto = 40 * (scores[12] - scores[13]) + 25 * (scores[18] - scores[21]) + C_LTO
    ivr = 35 * (scores[11] - scores[10]) + 40 * (scores[16] - scores[15]) + C_IVR

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
        answer = generate_answer(model, tokenizer, question, country)
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
        use_moe = False
    elif args.model == "lora":
        model_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/{args.backbone}_merge_{args.num_classes}"
        model_name = f"{args.model}_{args.backbone}_{args.num_classes}"
        use_moe = False
    else:  # moe
        # MoE 需要加载 merged model + MoE 权重
        merged_model_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/{args.backbone}_merge_{args.num_classes}"
        moe_weights_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_{args.backbone}_{args.num_classes}"
        model_path = merged_model_path  # 先用这个加载 LLM
        model_name = f"{args.model}_{args.backbone}_{args.num_classes}"
        use_moe = True

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

    # 确定设备
    if args.device.startswith("cuda"):
        device = args.device if ":" in args.device else "cuda:0"
        device_map = "auto"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        device_map = None
        torch_dtype = torch.float32

    if use_moe:
        # 加载 CultureMoE 模型（LLM + MoE）
        print(f"Loading CultureMoE model...")
        print(f"  LLM (merged): {model_path}")
        print(f"  MoE weights: {moe_weights_path}")

        # 导入 CultureMoE 相关模块
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
        from src.llamafactory.model.moe_args import ModelArgs

        # 1. 加载 LLM
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        # 2. 创建 MoE 模型结构
        # 需要从保存的配置中读取 MoE 参数
        moe_config_path = os.path.join(moe_weights_path, "moe_config.json")
        if os.path.exists(moe_config_path):
            with open(moe_config_path, 'r') as f:
                moe_config = json.load(f)

            moe_args = ModelArgs(
                num_experts=moe_config.get('num_experts', 6),
                shared_hidden_dim=moe_config.get('shared_hidden_dim', 2048),
                router_hidden_dim=moe_config.get('router_hidden_dim', 1024),
                experts_hidden_dim=moe_config.get('experts_hidden_dim', 2048),
                lora_rank=moe_config.get('lora_rank', 16),
                num_classes=args.num_classes,
                classification_hidden_dim=moe_config.get('classification_hidden_dim', 512),
                dropout=moe_config.get('dropout', 0.1),
                num_heads=moe_config.get('num_heads', 8)
            )
        else:
            # 使用默认配置
            print("⚠️  Warning: moe_config.json not found, using default config")
            moe_args = ModelArgs(
                num_experts=6,
                shared_hidden_dim=2048,
                router_hidden_dim=1024,
                experts_hidden_dim=2048,
                lora_rank=16,
                num_classes=args.num_classes,
                classification_hidden_dim=512,
                dropout=0.1,
                num_heads=8
            )

        # 3. 创建 CultureMoE 模型
        model = LlamaSharedRouterExpertsModel(
            llama_model=llama_model,
            config=llama_model.config,
            args=moe_args
        )

        # 4. 加载 MoE 权重
        moe_weights_file = os.path.join(moe_weights_path, "moe_weights.pt")
        if os.path.exists(moe_weights_file):
            print(f"  Loading MoE weights from: {moe_weights_file}")
            moe_state_dict = torch.load(moe_weights_file, map_location='cpu')
            model.load_state_dict(moe_state_dict, strict=False)
            print("  ✅ MoE weights loaded")
        else:
            print(f"  ❌ Error: MoE weights not found: {moe_weights_file}")
            print(f"  Available files in {moe_weights_path}:")
            if os.path.exists(moe_weights_path):
                print(f"    {os.listdir(moe_weights_path)}")
            raise FileNotFoundError(f"MoE weights not found: {moe_weights_file}")

        # 5. 移动到设备
        if device_map is None:
            model = model.to(device)

        print(f"✅ CultureMoE model loaded on {device}")
    else:
        # 加载普通模型（base 或 lora）
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        # 如果没有使用 device_map，手动移动到设备
        if device_map is None:
            model = model.to(device)

        print(f"✅ Model loaded on {device}")

    model.eval()
    print("")

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

