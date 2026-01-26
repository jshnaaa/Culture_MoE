#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-R1 API评测脚本
使用DeepSeek-R1模型对文化数据集进行评测
"""

import os
import json
import argparse
import re
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepSeek-R1 API评测脚本')
    parser.add_argument('--api_key', type=str, required=True,
                        help='OpenRouter API KEY')
    parser.add_argument('--data_file', type=str, required=True,
                        help='数据集JSON文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--model_name', type=str, default='deepseek/deepseek-r1-0528:free',
                        help='模型名称 (默认: deepseek/deepseek-r1-0528:free)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大评测样本数（用于测试，默认评测全部）')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='API调用失败最大重试次数 (默认: 3)')
    parser.add_argument('--retry_delay', type=float, default=2.0,
                        help='重试延迟秒数 (默认: 2.0)')

    return parser.parse_args()


def load_dataset(data_file: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    加载JSON数据集

    Args:
        data_file: 数据集文件路径
        max_samples: 最大加载样本数

    Returns:
        数据样本列表
    """
    print(f"📂 加载数据集: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples is not None and max_samples < len(data):
        data = data[:max_samples]
        print(f"  - 限制样本数: {max_samples}")

    print(f"  - 总样本数: {len(data)}")
    return data


def extract_answer(response: str) -> Optional[str]:
    """
    从模型响应中提取答案（阿拉伯数字）

    Args:
        response: 模型的原始响应

    Returns:
        提取的答案（数字字符串），如果无法提取则返回None
    """
    # 去除首尾空白
    response = response.strip()

    # 1. 如果响应就是单个数字
    if response in ['1', '2', '3', '4']:
        return response

    # 2. 提取第一个出现的数字（1-4）
    match = re.search(r'\b([1-4])\b', response)
    if match:
        return match.group(1)

    # 3. 如果都无法匹配，返回None
    return None


def call_deepseek_api(
    client: OpenAI,
    question: str,
    model_name: str,
    max_retries: int,
    retry_delay: float
) -> Tuple[Optional[str], bool]:
    """
    调用DeepSeek API

    Args:
        client: OpenAI客户端
        question: 问题文本
        model_name: 模型名称
        max_retries: 最大重试次数
        retry_delay: 重试延迟

    Returns:
        (模型响应, 是否成功)
    """
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/your-repo",
                    "X-Title": "Culture-Moe-Eval",
                },
                extra_body={},
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )

            answer = completion.choices[0].message.content
            return answer, True

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
            else:
                print(f"  ❌ API调用失败，已达最大重试次数: {str(e)}")
                return None, False

    return None, False


def evaluate_dataset(
    data: List[Dict],
    api_key: str,
    model_name: str,
    max_retries: int,
    retry_delay: float
) -> Tuple[List[Dict], Dict]:
    """
    评测整个数据集

    Args:
        data: 数据样本列表
        api_key: API KEY
        model_name: 模型名称
        max_retries: 最大重试次数
        retry_delay: 重试延迟

    Returns:
        (详细结果列表, 汇总指标字典)
    """
    # 创建OpenAI客户端
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    detailed_results = []
    correct_count = 0
    total_count = len(data)
    failed_count = 0

    print(f"\n🚀 开始评测，共 {total_count} 个样本...")

    for idx, sample in enumerate(tqdm(data, desc="评测进度")):
        # 获取问题
        question = sample.get('instruction', '')

        # 调用DeepSeek API
        response, success = call_deepseek_api(
            client, question, model_name, max_retries, retry_delay
        )

        # 提取答案
        if success and response:
            extracted_answer = extract_answer(response)
        else:
            extracted_answer = None
            failed_count += 1

        # 获取标准答案
        expected_answer = str(sample.get('output', '')).strip()

        # 判断是否正确
        is_correct = (extracted_answer == expected_answer) if extracted_answer else False
        if is_correct:
            correct_count += 1

        # 保存详细结果
        result = {
            'sample_id': idx,
            'instruction': question,
            'expected_answer': expected_answer,
            'model_response': response if success else 'API_CALL_FAILED',
            'extracted_answer': extracted_answer if extracted_answer else 'EXTRACTION_FAILED',
            'is_correct': is_correct
        }
        detailed_results.append(result)

    # 计算汇总指标
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    summary = {
        'model': model_name,
        'total_samples': total_count,
        'correct_predictions': correct_count,
        'wrong_predictions': total_count - correct_count - failed_count,
        'failed_api_calls': failed_count,
        'accuracy': accuracy
    }

    return detailed_results, summary


def save_results(
    detailed_results: List[Dict],
    summary: Dict,
    output_dir: str
):
    """
    保存评测结果

    Args:
        detailed_results: 详细结果列表
        summary: 汇总指标
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细答案
    answers_file = os.path.join(output_dir, 'generated_answers.json')
    with open(answers_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 详细答案已保存: {answers_file}")

    # 保存汇总结果
    summary_file = os.path.join(output_dir, 'eval_result.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 评测结果已保存: {summary_file}")


def main():
    """主函数"""
    args = parse_args()

    print("=" * 50)
    print("DeepSeek-R1 API 评测脚本")
    print("=" * 50)
    print(f"模型: {args.model_name}")
    print(f"数据集: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 50)

    # 加载数据集
    try:
        data = load_dataset(args.data_file, args.max_samples)
    except Exception as e:
        print(f"❌ 加载数据集失败: {str(e)}")
        return

    # 评测数据集
    detailed_results, summary = evaluate_dataset(
        data,
        args.api_key,
        args.model_name,
        args.max_retries,
        args.retry_delay
    )

    # 保存结果
    save_results(detailed_results, summary, args.output_dir)

    # 显示结果摘要
    print("\n" + "=" * 50)
    print("📊 评测结果摘要")
    print("=" * 50)
    print(f"模型: {summary['model']}")
    print(f"总样本数: {summary['total_samples']}")
    print(f"正确预测数: {summary['correct_predictions']}")
    print(f"错误预测数: {summary['wrong_predictions']}")
    print(f"API调用失败数: {summary['failed_api_calls']}")
    print(f"准确率: {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
    print("=" * 50)


if __name__ == '__main__':
    main()
