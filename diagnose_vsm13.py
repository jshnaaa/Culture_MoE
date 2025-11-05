#!/usr/bin/env python3
"""
诊断 VSM13 结果为什么相同
"""

import json
import numpy as np
import argparse


def extract_score(answer: str):
    """从答案中提取分数"""
    import re
    answer = answer.strip()

    # 尝试匹配数字
    match = re.search(r'([1-5])', answer)
    if match:
        return int(match.group(1))
    return None


def calculate_vsm_scores(answers: list):
    """
    计算 VSM13 分数

    正确的公式：
    - PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 3
    - IDV = 35(μQ4 − μQ1) + 35(μQ9 − μQ6) + 3
    - MAS = 35(μQ5 − μQ3) + 25(μQ8 − μQ10) + 3
    - UAI = 40(μQ18 − μQ15) + 25(μQ21 − μQ24) + 3
    - LTO = 40(μQ13 − μQ14) + 25(μQ19 − μQ22) + 3
    - IVR = 35(μQ12 − μQ11) + 40(μQ17 − μQ16) + 3
    """
    if len(answers) != 24:
        print(f"Warning: Expected 24 answers, got {len(answers)}")
        return None

    # 提取分数
    scores = []
    for ans in answers:
        score = extract_score(ans) if ans else None
        scores.append(score if score is not None else 3)

    # 常量
    C = 3

    # 计算 6 个维度（注意：Python 索引从 0 开始，Q1 = scores[0]）
    pdi = 35 * (scores[6] - scores[1]) + 25 * (scores[19] - scores[22]) + C
    idv = 35 * (scores[3] - scores[0]) + 35 * (scores[8] - scores[5]) + C
    mas = 35 * (scores[4] - scores[2]) + 25 * (scores[7] - scores[9]) + C
    uai = 40 * (scores[17] - scores[14]) + 25 * (scores[20] - scores[23]) + C
    lto = 40 * (scores[12] - scores[13]) + 25 * (scores[18] - scores[21]) + C
    ivr = 35 * (scores[11] - scores[10]) + 40 * (scores[16] - scores[15]) + C

    return [pdi, idv, mas, uai, lto, ivr], scores


def analyze_answers(answer_file: str, model_name: str):
    """分析答案文件"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}")

    # 读取答案
    with open(answer_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取答案
    answers = [item['Answer'] for item in data]

    # 计算分数
    vsm_scores, raw_scores = calculate_vsm_scores(answers)

    # 1. 答案分布
    print(f"\n1. Answer Distribution:")
    score_counts = {}
    for ans in answers:
        score = extract_score(ans)
        if score:
            score_counts[score] = score_counts.get(score, 0) + 1

    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        print(f"   {score}: {count:2d} ({count/24*100:5.1f}%)")

    # 统计
    mean_score = np.mean([s for s in raw_scores if s is not None])
    std_score = np.std([s for s in raw_scores if s is not None])
    print(f"\n   Mean: {mean_score:.2f}")
    print(f"   Std:  {std_score:.2f}")

    # 2. 每个维度的详细计算
    print(f"\n2. Dimension Calculations:")

    # PDI = 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 3
    q7, q2 = raw_scores[6], raw_scores[1]
    q20, q23 = raw_scores[19], raw_scores[22]
    pdi_term1 = 35 * (q7 - q2)
    pdi_term2 = 25 * (q20 - q23)
    print(f"\n   PDI (Power Distance):")
    print(f"      Formula: 35(μQ7 − μQ2) + 25(μQ20 − μQ23) + 3")
    print(f"      Q7={q7}, Q2={q2}  → 35×({q7}-{q2}) = {pdi_term1}")
    print(f"      Q20={q20}, Q23={q23} → 25×({q20}-{q23}) = {pdi_term2}")
    print(f"      Final: {pdi_term1} + {pdi_term2} + 3 = {vsm_scores[0]}")

    # IDV
    idv_pos = [raw_scores[0], raw_scores[3], raw_scores[6]]
    idv_neg = [raw_scores[9], raw_scores[12], raw_scores[15]]
    idv_sum_pos = sum(idv_pos)
    idv_sum_neg = sum(idv_neg)
    print(f"\n   IDV (Individualism):")
    print(f"      Positive terms (Q1, Q4, Q7):    {idv_pos} → sum = {idv_sum_pos}")
    print(f"      Negative terms (Q10, Q13, Q16): {idv_neg} → sum = {idv_sum_neg}")
    print(f"      Difference: {idv_sum_pos} - {idv_sum_neg} = {idv_sum_pos - idv_sum_neg}")
    print(f"      Final: {idv_sum_pos - idv_sum_neg} + 50 = {vsm_scores[1]}")

    # MAS
    mas_pos = [raw_scores[2], raw_scores[5], raw_scores[8]]
    mas_neg = [raw_scores[11], raw_scores[14], raw_scores[17]]
    mas_sum_pos = sum(mas_pos)
    mas_sum_neg = sum(mas_neg)
    print(f"\n   MAS (Masculinity):")
    print(f"      Positive terms (Q3, Q6, Q9):    {mas_pos} → sum = {mas_sum_pos}")
    print(f"      Negative terms (Q12, Q15, Q18): {mas_neg} → sum = {mas_sum_neg}")
    print(f"      Difference: {mas_sum_pos} - {mas_sum_neg} = {mas_sum_pos - mas_sum_neg}")
    print(f"      Final: {mas_sum_pos - mas_sum_neg} + 50 = {vsm_scores[2]}")

    # UAI
    uai_pos = [raw_scores[19], raw_scores[22]]
    uai_neg = [raw_scores[18], raw_scores[21]]
    uai_sum_pos = sum(uai_pos)
    uai_sum_neg = sum(uai_neg)
    print(f"\n   UAI (Uncertainty Avoidance):")
    print(f"      Positive terms (Q20, Q23): {uai_pos} → sum = {uai_sum_pos}")
    print(f"      Negative terms (Q19, Q22): {uai_neg} → sum = {uai_sum_neg}")
    print(f"      Difference: {uai_sum_pos} - {uai_sum_neg} = {uai_sum_pos - uai_sum_neg}")
    print(f"      Final: {uai_sum_pos - uai_sum_neg} + 50 = {vsm_scores[3]}")

    # LTO
    lto_pos = [raw_scores[20], raw_scores[23]]
    lto_neg = [raw_scores[18], raw_scores[21]]
    lto_sum_pos = sum(lto_pos)
    lto_sum_neg = sum(lto_neg)
    print(f"\n   LTO (Long Term Orientation):")
    print(f"      Positive terms (Q21, Q24): {lto_pos} → sum = {lto_sum_pos}")
    print(f"      Negative terms (Q19, Q22): {lto_neg} → sum = {lto_sum_neg}")
    print(f"      Difference: {lto_sum_pos} - {lto_sum_neg} = {lto_sum_pos - lto_sum_neg}")
    print(f"      Final: {lto_sum_pos - lto_sum_neg} + 50 = {vsm_scores[4]}")

    # IVR
    ivr_pos = [raw_scores[18], raw_scores[21]]
    ivr_neg = [raw_scores[19], raw_scores[22]]
    ivr_sum_pos = sum(ivr_pos)
    ivr_sum_neg = sum(ivr_neg)
    print(f"\n   IVR (Indulgence vs Restraint):")
    print(f"      Positive terms (Q19, Q22): {ivr_pos} → sum = {ivr_sum_pos}")
    print(f"      Negative terms (Q20, Q23): {ivr_neg} → sum = {ivr_sum_neg}")
    print(f"      Difference: {ivr_sum_pos} - {ivr_sum_neg} = {ivr_sum_pos - ivr_sum_neg}")
    print(f"      Final: {ivr_sum_pos - ivr_sum_neg} + 50 = {vsm_scores[5]}")

    # 3. 最终分数
    print(f"\n3. Final VSM13 Scores:")
    dimension_names = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]
    for name, score in zip(dimension_names, vsm_scores):
        print(f"   {name}: {score:3d}")

    return vsm_scores, raw_scores, score_counts


def compare_models(files: dict):
    """对比多个模型"""
    print(f"\n{'='*80}")
    print("Comparing Models")
    print(f"{'='*80}")

    all_results = {}

    for model_name, file_path in files.items():
        vsm_scores, raw_scores, score_counts = analyze_answers(file_path, model_name)
        all_results[model_name] = {
            'vsm_scores': vsm_scores,
            'raw_scores': raw_scores,
            'score_counts': score_counts
        }

    # 对比 VSM 分数
    print(f"\n{'='*80}")
    print("VSM13 Scores Comparison")
    print(f"{'='*80}")

    dimension_names = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]

    # 打印表头
    header = "Dimension  " + "  ".join([f"{name:>8s}" for name in files.keys()])
    print(header)
    print("-" * len(header))

    # 打印每个维度
    for i, dim_name in enumerate(dimension_names):
        row = f"{dim_name:9s}  "
        for model_name in files.keys():
            score = all_results[model_name]['vsm_scores'][i]
            row += f"{score:8d}  "
        print(row)

    # 计算差异
    print(f"\n{'='*80}")
    print("Analysis")
    print(f"{'='*80}")

    model_names = list(files.keys())
    if len(model_names) >= 2:
        model1 = model_names[0]
        model2 = model_names[1]

        scores1 = all_results[model1]['vsm_scores']
        scores2 = all_results[model2]['vsm_scores']

        print(f"\nDifference ({model1} vs {model2}):")
        for i, dim_name in enumerate(dimension_names):
            diff = scores1[i] - scores2[i]
            print(f"   {dim_name}: {diff:+4d}")

        # 欧氏距离
        euclidean = np.sqrt(sum((a - b) ** 2 for a, b in zip(scores1, scores2)))
        print(f"\nEuclidean Distance: {euclidean:.2f}")

        # 如果距离很小，分析原因
        if euclidean < 10:
            print(f"\n⚠️  Warning: Very small distance ({euclidean:.2f})")
            print(f"   This suggests the models are giving very similar answers.")
            print(f"\n   Possible reasons:")
            print(f"   1. Both models have similar answer distributions")
            print(f"   2. The VSM13 formula causes cancellation")
            print(f"   3. Models haven't learned cultural differences")

            # 对比原始答案
            print(f"\n   Comparing raw answers:")
            raw1 = all_results[model1]['raw_scores']
            raw2 = all_results[model2]['raw_scores']

            same_count = sum(1 for a, b in zip(raw1, raw2) if a == b)
            print(f"   Same answers: {same_count}/24 ({same_count/24*100:.1f}%)")

            if same_count > 18:  # 超过 75% 相同
                print(f"\n   ❌ Models are giving almost identical answers!")
                print(f"      This is the main problem.")
            else:
                print(f"\n   ✓ Answers are different, but VSM13 formula causes cancellation")
                print(f"     This is a limitation of the VSM13 metric.")


def main():
    parser = argparse.ArgumentParser(description="诊断 VSM13 结果")
    parser.add_argument("--files", nargs='+', required=True,
                        help="答案文件路径，格式：model_name:file_path")

    args = parser.parse_args()

    # 解析文件
    files = {}
    for item in args.files:
        if ':' in item:
            name, path = item.split(':', 1)
            files[name] = path
        else:
            # 如果没有指定名称，使用文件名
            import os
            name = os.path.basename(item).replace('.json', '')
            files[name] = item

    if len(files) == 1:
        # 单个文件，只分析
        model_name, file_path = list(files.items())[0]
        analyze_answers(file_path, model_name)
    else:
        # 多个文件，对比
        compare_models(files)


if __name__ == "__main__":
    main()

