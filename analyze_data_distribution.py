#!/usr/bin/env python3
"""
分析训练数据中的答案分布，特别关注是否存在对特定数字的偏向
"""

import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt

def analyze_answer_distribution(data_file, output_dir=None):
    """分析答案分布"""
    print(f"🔍 分析数据文件: {data_file}")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ 成功加载 {len(data)} 条数据")

        # 提取所有答案
        answers = []
        labels = []

        for item in data:
            output = item.get('output', '').strip()
            label = item.get('label', '').strip()

            answers.append(output)
            labels.append(label)

        # 分析答案分布
        print(f"\n📊 答案(output)分布:")
        answer_counter = Counter(answers)

        # 按频率排序显示前20个
        for answer, count in answer_counter.most_common(20):
            percentage = count / len(answers) * 100
            print(f"  '{answer}': {count} 次 ({percentage:.2f}%)")

        # 分析标签分布
        print(f"\n📊 标签(label)分布:")
        label_counter = Counter(labels)

        for label, count in label_counter.most_common():
            percentage = count / len(labels) * 100
            print(f"  '{label}': {count} 次 ({percentage:.2f}%)")

        # 检查是否有偏向特定数字的问题
        print(f"\n🔍 数字答案分析:")
        digit_answers = {}
        for answer in answers:
            # 提取数字
            digits = [c for c in answer if c.isdigit()]
            if digits:
                digit = digits[0]  # 取第一个数字
                digit_answers[digit] = digit_answers.get(digit, 0) + 1

        total_digit_answers = sum(digit_answers.values())
        print(f"总共有 {total_digit_answers} 个包含数字的答案")

        for digit in sorted(digit_answers.keys()):
            count = digit_answers[digit]
            percentage = count / total_digit_answers * 100
            print(f"  数字 '{digit}': {count} 次 ({percentage:.2f}%)")

        # 检查是否存在明显偏向
        if digit_answers:
            max_digit = max(digit_answers.items(), key=lambda x: x[1])
            min_digit = min(digit_answers.items(), key=lambda x: x[1])

            max_percentage = max_digit[1] / total_digit_answers * 100
            min_percentage = min_digit[1] / total_digit_answers * 100

            print(f"\n⚠️ 偏向分析:")
            print(f"  最常见数字: '{max_digit[0]}' ({max_percentage:.2f}%)")
            print(f"  最少见数字: '{min_digit[0]}' ({min_percentage:.2f}%)")

            # 如果差异超过20%，认为存在明显偏向
            if max_percentage - min_percentage > 20:
                print(f"  🚨 检测到明显偏向数字 '{max_digit[0]}'！")
                print(f"     这可能导致模型倾向于生成 '{max_digit[0]}'")
            else:
                print(f"  ✅ 数字分布相对均衡")

        # 分析instruction模式
        print(f"\n📝 Instruction模式分析:")
        instruction_patterns = {}

        for item in data:
            instruction = item.get('instruction', '')
            # 提取关键模式
            if 'Give me the answer from' in instruction:
                # 提取数字范围
                import re
                range_match = re.search(r'from (\d+) to (\d+)', instruction)
                if range_match:
                    range_pattern = f"from {range_match.group(1)} to {range_match.group(2)}"
                    instruction_patterns[range_pattern] = instruction_patterns.get(range_pattern, 0) + 1

        for pattern, count in instruction_patterns.items():
            percentage = count / len(data) * 100
            print(f"  '{pattern}': {count} 次 ({percentage:.2f}%)")

        # 保存统计结果
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)

            stats = {
                'total_samples': len(data),
                'answer_distribution': dict(answer_counter),
                'label_distribution': dict(label_counter),
                'digit_distribution': digit_answers,
                'instruction_patterns': instruction_patterns
            }

            stats_file = os.path.join(output_dir, 'data_distribution_analysis.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            print(f"\n✅ 统计结果已保存: {stats_file}")

        return {
            'answer_distribution': answer_counter,
            'label_distribution': label_counter,
            'digit_distribution': digit_answers
        }

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="分析训练数据答案分布")
    parser.add_argument("--data_file", type=str, required=True, help="数据文件路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")

    args = parser.parse_args()

    analyze_answer_distribution(args.data_file, args.output_dir)

if __name__ == "__main__":
    main()