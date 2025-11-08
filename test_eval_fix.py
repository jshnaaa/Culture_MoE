#!/usr/bin/env python3
"""
测试评估脚本的修复效果

用法：
    python test_eval_fix.py
"""

import re


def extract_label_robust(answer: str, num_classes: int = 10):
    """
    改进的标签提取（更鲁棒）
    """
    if not answer:
        print(f"⚠️  Empty answer, using default: {num_classes // 2}")
        return num_classes // 2

    # 1. 尝试直接转换为整数
    try:
        label = int(answer.strip())
        if 1 <= label <= num_classes:
            return label - 1  # 1-indexed -> 0-indexed
        elif 0 <= label < num_classes:
            return label
        else:
            print(f"⚠️  Label {label} out of range [1, {num_classes}], clipping")
            return min(max(0, label - 1), num_classes - 1)
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
            print(f"⚠️  Extracted label {label} out of range, clipping")
            return min(max(0, label - 1), num_classes - 1)

    # 3. 检查是否是文本答案（如果混训了）
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
        'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9
    }
    if answer_lower in standard_text_mapping:
        mapped = standard_text_mapping[answer_lower]
        if mapped < num_classes:
            print(f"⚠️  Found text answer '{answer_lower}', mapping to {mapped}")
            return mapped

    # 4. 如果是无关词（如 "Code", "Country"），打印警告
    if answer.isalpha() and len(answer) > 2:
        print(f"⚠️  Invalid answer: '{answer}' - this is likely a word, not a number")
        print(f"   Using default: {num_classes // 2}")
        return num_classes // 2

    # 5. 默认返回中间类别
    print(f"⚠️  Could not extract label from '{answer}', using default: {num_classes // 2}")
    return num_classes // 2


def test_extract_label():
    """测试标签提取函数"""
    print("="*80)
    print("Testing Label Extraction")
    print("="*80)
    print()

    # 测试用例
    test_cases = [
        # (answer, num_classes, expected_label, description)
        ("1", 10, 0, "正常数字 1"),
        ("5", 10, 4, "正常数字 5"),
        ("10", 10, 9, "正常数字 10"),
        ("Code", 10, 5, "无关词 Code"),
        ("Country", 10, 5, "无关词 Country"),
        ("Option", 10, 5, "无关词 Option"),
        ("11", 15, 10, "统一编码 11"),
        ("14", 15, 13, "统一编码 14"),
        ("15", 15, 14, "统一编码 15"),
        ("yes", 15, 10, "文本答案 yes (统一编码)"),
        ("no", 15, 11, "文本答案 no (统一编码)"),
        ("neutral", 15, 12, "文本答案 neutral (统一编码)"),
        ("TRUE", 15, 13, "文本答案 TRUE (统一编码)"),
        ("FALSE", 15, 14, "文本答案 FALSE (统一编码)"),
        ("", 10, 5, "空字符串"),
        ("abc123", 10, 5, "混合字符串（提取数字）"),
    ]

    passed = 0
    failed = 0

    for answer, num_classes, expected, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Input: '{answer}' (num_classes={num_classes})")
        print(f"  Expected: {expected}")

        result = extract_label_robust(answer, num_classes)

        if result == expected:
            print(f"  Result: {result} ✅ PASS")
            passed += 1
        else:
            print(f"  Result: {result} ❌ FAIL (expected {expected})")
            failed += 1

    print()
    print("="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80)


if __name__ == "__main__":
    test_extract_label()

