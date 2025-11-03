#!/usr/bin/env python3
"""
VSM13 文化一致性测试

使用方法：
    python eval_vsm13.py --model base --backbone llama
    python eval_vsm13.py --model lora --backbone qwen
    python eval_vsm13.py --model moe --backbone llama
"""

import argparse
import json
import os
import sys
import re
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs

# 霍夫斯泰德给定的10个国家的6个维度指数
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
    "Greece": [60, 59, 57, 100, 51, 50]
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
    "Greece": "Greek"
}


def load_base_model(backbone: str, device: str = "cuda"):
    """加载 Base 模型"""
    if backbone == "llama":
        model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
    elif backbone == "qwen":
        model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    print(f"Loading base model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    return model, tokenizer


def load_lora_model(backbone: str, num_classes: int, device: str = "cuda"):
    """加载 LoRA 合并后的模型"""
    model_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/{backbone}_merge_{num_classes}"

    print(f"Loading LoRA model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    return model, tokenizer


def load_moe_model(backbone: str, num_classes: int, device: str = "cuda"):
    """加载 CultureMoE 模型"""
    merged_llm_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/{backbone}_merge_{num_classes}"
    moe_weights_path = f"/root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_{backbone}_{num_classes}"

    print(f"Loading MoE model...")
    print(f"  Merged LLM: {merged_llm_path}")
    print(f"  MoE weights: {moe_weights_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(moe_weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 MoE 配置
    config_path = os.path.join(moe_weights_path, "moe_config.json")
    with open(config_path, 'r') as f:
        moe_config = json.load(f)

    # 加载合并后的 LLM
    llama_model = AutoModelForCausalLM.from_pretrained(
        merged_llm_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    for param in llama_model.parameters():
        param.requires_grad = False

    # 创建 CultureMoE 模型
    moe_args = ModelArgs(
        num_experts=moe_config['num_experts'],
        shared_hidden_dim=moe_config['shared_hidden_dim'],
        router_hidden_dim=moe_config['router_hidden_dim'],
        experts_hidden_dim=moe_config['experts_hidden_dim'],
        lora_rank=moe_config['moe_lora_rank'],
        num_classes=moe_config['num_classes'],
        classification_hidden_dim=moe_config['classification_hidden_dim'],
        dropout=moe_config['dropout'],
        num_heads=moe_config['num_heads']
    )

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    )

    # 加载 MoE 权重
    weights_path = os.path.join(moe_weights_path, "moe_weights.bin")
    moe_state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(moe_state_dict, strict=False)

    # 将 MoE 层移到正确的设备
    llm_device = next(llama_model.parameters()).device
    if hasattr(model, 'shared'):
        model.shared = model.shared.to(llm_device)
    if hasattr(model, 'router'):
        model.router = model.router.to(llm_device)
    if hasattr(model, 'experts_layer'):
        model.experts_layer = model.experts_layer.to(llm_device)
    if hasattr(model, 'classifier'):
        model.classifier = model.classifier.to(llm_device)
    if hasattr(model, 'culture_classifier'):
        model.culture_classifier = model.culture_classifier.to(llm_device)

    model.eval()

    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 50):
    """生成答案"""
    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪婪解码
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取生成的部分（去掉输入的问题）
    answer = answer[len(question):].strip()

    return answer


def extract_score(answer: str):
    """从答案中提取分数（1-5）"""
    # 尝试匹配数字
    match = re.search(r'\b([1-5])\b', answer)
    if match:
        return int(match.group(1))

    # 如果没有找到，返回 None
    return None


def calculate_vsm_scores(answers: list):
    """
    根据 VSM13 问卷的答案计算 6 个文化维度分数

    VSM13 包含 24 个问题，计算公式：
    - PDI (Power Distance Index): Q2 + Q5 + Q8 - Q11 - Q14 - Q17 + 50
    - IDV (Individualism): Q1 + Q4 + Q7 - Q10 - Q13 - Q16 + 50
    - MAS (Masculinity): Q3 + Q6 + Q9 - Q12 - Q15 - Q18 + 50
    - UAI (Uncertainty Avoidance): Q20 + Q23 - Q19 - Q22 + 50
    - LTO (Long Term Orientation): Q21 + Q24 - Q19 - Q22 + 50
    - IVR (Indulgence vs Restraint): Q19 + Q22 - Q20 - Q23 + 50
    """
    # 确保有 24 个答案
    if len(answers) != 24:
        print(f"Warning: Expected 24 answers, got {len(answers)}")
        return None

    # 将答案转换为分数（1-5），如果无法提取则使用 3（中间值）
    scores = []
    for ans in answers:
        score = extract_score(ans) if ans else None
        scores.append(score if score is not None else 3)

    # 计算 6 个维度（注意：Python 索引从 0 开始，所以 Q1 = scores[0]）
    pdi = scores[1] + scores[4] + scores[7] - scores[10] - scores[13] - scores[16] + 50
    idv = scores[0] + scores[3] + scores[6] - scores[9] - scores[12] - scores[15] + 50
    mas = scores[2] + scores[5] + scores[8] - scores[11] - scores[14] - scores[17] + 50
    uai = scores[19] + scores[22] - scores[18] - scores[21] + 50
    lto = scores[20] + scores[23] - scores[18] - scores[21] + 50
    ivr = scores[18] + scores[21] - scores[19] - scores[22] + 50

    return [pdi, idv, mas, uai, lto, ivr]


def calculate_euclidean_distance(scores1, scores2):
    """计算欧氏距离"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(scores1, scores2)))


def main():
    parser = argparse.ArgumentParser(description="VSM13 文化一致性测试")
    parser.add_argument("--model", type=str, required=True, choices=["base", "lora", "moe"],
                        help="模型类型：base/lora/moe")
    parser.add_argument("--backbone", type=str, required=True, choices=["llama", "qwen"],
                        help="骨干模型：llama/qwen")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="分类数量（仅用于 lora 和 moe）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("VSM13 Cultural Consistency Test")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Backbone: {args.backbone}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("")

    # 加载模型
    print("Loading model...")
    if args.model == "base":
        model, tokenizer = load_base_model(args.backbone, args.device)
    elif args.model == "lora":
        model, tokenizer = load_lora_model(args.backbone, args.num_classes, args.device)
    elif args.model == "moe":
        model, tokenizer = load_moe_model(args.backbone, args.num_classes, args.device)
    print("✅ Model loaded\n")

    # 加载 VSM13 测试数据
    test_file = "/root/autodl-fs/vsm13_test.json"
    print(f"Loading test data from: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ Loaded {len(test_data)} questions\n")

    # 创建输出目录
    output_dir = f"/root/autodl-fs/vsm13_output/{args.model}_{args.backbone}"
    os.makedirs(output_dir, exist_ok=True)

    # 存储所有国家的结果
    all_results = {}

    # 对每个国家进行测试
    for country, hofstede_scores in HOFSTEDE_SCORES.items():
        print("="*80)
        print(f"Testing for {country}")
        print("="*80)

        adjective = COUNTRY_TO_ADJECTIVE[country]
        prefix = f"You are a {adjective} culture chatbot that knows {adjective} culture very well. "

        # 复制测试数据并添加前缀
        country_data = []
        answers = []

        for item in tqdm(test_data, desc=f"Generating answers for {country}"):
            question_with_prefix = prefix + item['Question']

            # 生成答案
            answer = generate_answer(model, tokenizer, question_with_prefix)
            answers.append(answer)

            # 保存结果
            country_data.append({
                "id": item['id'],
                "Question": item['Question'],
                "Answer": answer
            })

        # 保存该国家的完整数据集
        country_file = os.path.join(output_dir, f"vsm13_test_{country.replace(' ', '_')}_{args.model}_{args.backbone}.json")
        with open(country_file, 'w', encoding='utf-8') as f:
            json.dump(country_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved answers to: {country_file}")

        # 计算 VSM 分数
        calculated_scores = calculate_vsm_scores(answers)

        if calculated_scores:
            # 计算欧氏距离
            distance = calculate_euclidean_distance(calculated_scores, hofstede_scores)

            all_results[country] = {
                "calculated_scores": calculated_scores,
                "hofstede_scores": hofstede_scores,
                "euclidean_distance": float(distance)
            }

            print(f"Calculated scores: {calculated_scores}")
            print(f"Hofstede scores:   {hofstede_scores}")
            print(f"Euclidean distance: {distance:.2f}")
        else:
            print(f"⚠️  Failed to calculate scores for {country}")

        print("")

    # 保存所有结果
    results_file = os.path.join(output_dir, f"vsm13_test_results_{args.model}_{args.backbone}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("="*80)
    print("✅ All tests completed!")
    print("="*80)
    print(f"Results saved to: {results_file}")
    print("")

    # 打印汇总
    print("Summary:")
    print("-" * 80)
    print(f"{'Country':<20} {'Distance':<15} {'Calculated Scores'}")
    print("-" * 80)
    for country, result in all_results.items():
        print(f"{country:<20} {result['euclidean_distance']:<15.2f} {result['calculated_scores']}")
    print("-" * 80)

    # 计算平均距离
    avg_distance = np.mean([r['euclidean_distance'] for r in all_results.values()])
    print(f"Average Euclidean Distance: {avg_distance:.2f}")
    print("")


if __name__ == "__main__":
    main()

