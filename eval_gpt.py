#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT API评测脚本
使用OpenAI GPT模型对文化数据集进行评测，作为baseline对比
"""

import os
import json
import argparse
import re
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GPT API评测脚本')
    parser.add_argument('--data_file', type=str, required=True,
                        help='数据集JSON文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo',
                        help='GPT模型名称 (默认: gpt-3.5-turbo)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大评测样本数（用于测试，默认评测全部）')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='GPT温度参数 (默认: 0.0，确定性输出)')
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


def construct_prompt(sample: Dict) -> str:
    """
    构造GPT提示词

    Args:
        sample: 数据样本，包含instruction和input字段

    Returns:
        完整的提示词
    """
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')

    # 组合问题
    if input_text:
        question = f"{instruction}\n{input_text}"
    else:
        question = instruction

    # 添加明确的答案格式要求
    prompt = f"""{question}

Please provide your answer as a single number (1, 2, 3, or 4) without any explanation."""

    return prompt


def extract_answer(gpt_response: str) -> Optional[str]:
    """
    从GPT响应中提取答案

    Args:
        gpt_response: GPT的原始响应

    Returns:
        提取的答案（1-4的数字字符串），如果无法提取则返回None
    """
    # 去除首尾空白
    response = gpt_response.strip()

    # 1. 如果响应就是单个数字
    if response in ['1', '2', '3', '4']:
        return response

    # 2. 如果响应是"答案是X"或"Answer: X"等格式
    patterns = [
        r'(?:answer|答案)(?:\s*is|\s*:|：)\s*([1-4])',
        r'^([1-4])\s*[.。]',  # 以数字开头，后跟句号
        r'\b([1-4])\b',  # 任何位置的独立数字
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1)

    # 3. 如果都无法匹配，返回None
    return None


def call_gpt_api(
    prompt: str,
    model_name: str,
    temperature: float,
    max_retries: int,
    retry_delay: float
) -> Tuple[Optional[str], bool]:
    """
    调用GPT API

    Args:
        prompt: 提示词
        model_name: 模型名称
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟

    Returns:
        (GPT响应, 是否成功)
    """
    try:
        from openai import OpenAI
        import httpx
    except ImportError as e:
        print(f"❌ 错误: 缺少依赖库，请运行: pip install openai httpx")
        print(f"   详细错误: {e}")
        return None, False

    # 从环境变量获取API KEY
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置OPENAI_API_KEY环境变量")
        return None, False

    # 🔧 配置代理和超时
    # 1. 从环境变量读取代理配置
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    # 2. 从环境变量读取自定义base_url（用于中转服务）
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # 3. 配置超时时间（默认60秒）
    timeout = float(os.environ.get("OPENAI_TIMEOUT", "60.0"))

    # 4. 创建OpenAI客户端
    try:
        if http_proxy or https_proxy:
            # 使用代理
            http_client = httpx.Client(
                proxies={
                    "http://": http_proxy,
                    "https://": https_proxy or http_proxy
                },
                timeout=timeout
            )
            client = OpenAI(api_key=api_key, http_client=http_client, base_url=base_url)
        else:
            # 不使用代理
            client = OpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
    except Exception as e:
        print(f"❌ 错误: 创建OpenAI客户端失败: {e}")
        return None, False

    # 重试机制
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers cultural survey questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=50  # 限制输出长度，因为只需要一个数字
            )

            answer = response.choices[0].message.content
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
    model_name: str,
    temperature: float,
    max_retries: int,
    retry_delay: float
) -> Tuple[List[Dict], Dict]:
    """
    评测整个数据集

    Args:
        data: 数据样本列表
        model_name: GPT模型名称
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟

    Returns:
        (详细结果列表, 汇总指标字典)
    """
    detailed_results = []
    correct_count = 0
    total_count = len(data)
    failed_count = 0

    print(f"\n🚀 开始评测，共 {total_count} 个样本...")

    for idx, sample in enumerate(tqdm(data, desc="评测进度")):
        # 构造提示词
        prompt = construct_prompt(sample)

        # 调用GPT API
        gpt_response, success = call_gpt_api(
            prompt, model_name, temperature, max_retries, retry_delay
        )

        # 提取答案
        if success and gpt_response:
            extracted_answer = extract_answer(gpt_response)
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
            'instruction': sample.get('instruction', ''),
            'input': sample.get('input', ''),
            'expected_answer': expected_answer,
            'gpt_response': gpt_response if success else 'API_CALL_FAILED',
            'extracted_answer': extracted_answer if extracted_answer else 'EXTRACTION_FAILED',
            'is_correct': is_correct,
            'label': sample.get('label', '')
        }
        detailed_results.append(result)

    # 计算汇总指标
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    summary = {
        'model': model_name,
        'temperature': temperature,
        'total_samples': total_count,
        'correct_predictions': correct_count,
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

    # 保存详细结果
    detailed_file = os.path.join(output_dir, 'detailed_results.json')
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 详细结果已保存: {detailed_file}")

    # 保存汇总结果
    summary_file = os.path.join(output_dir, 'eval_results.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 汇总结果已保存: {summary_file}")


def main():
    """主函数"""
    args = parse_args()

    print("=" * 50)
    print("GPT API 评测脚本")
    print("=" * 50)
    print(f"模型: {args.model_name}")
    print(f"温度: {args.temperature}")
    print(f"数据集: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 50)

    # 检查API KEY
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ 错误: 未设置OPENAI_API_KEY环境变量")
        print("请运行: export OPENAI_API_KEY='your-api-key'")
        return

    # 加载数据集
    try:
        data = load_dataset(args.data_file, args.max_samples)
    except Exception as e:
        print(f"❌ 加载数据集失败: {str(e)}")
        return

    # 评测数据集
    detailed_results, summary = evaluate_dataset(
        data,
        args.model_name,
        args.temperature,
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
    print(f"API调用失败数: {summary['failed_api_calls']}")
    print(f"准确率: {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
    print("=" * 50)


if __name__ == '__main__':
    main()
