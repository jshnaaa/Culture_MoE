#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiLoRA (Mixture of LoRA) 评估脚本

功能：
1. 从 Base 模型 + MiLoRA 权重还原完整模型
2. 测试提示感知路由机制的效果
3. 在测试集上进行评估并生成答案
4. 分析专家使用模式和路由效率
5. 保存详细的评估结果和专家统计

使用方法：
    python eval_milora_gen.py \
        --base_model_path /path/to/base_model \
        --milora_weights_path /path/to/milora_weights \
        --test_file /path/to/test_data.json \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.milora import MiLoRAModel


class MiLoRAEvaluator:
    """MiLoRA 模型评估器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 设置随机种子
        set_seed(args.seed)

        # 初始化模型
        self._load_model()

        # 加载测试数据
        self._load_test_data()

    def _load_model(self):
        """加载 MiLoRA 模型"""
        logging.info("Loading MiLoRA model...")

        # 1. 加载分词器
        logging.info("1. Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.base_model_path,
            trust_remote_code=True,
            padding_side='right'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 加载基础模型
        logging.info("2. Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path,
            torch_dtype=torch.float16,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        # 3. 加载 MiLoRA 配置
        logging.info("3. Loading MiLoRA configuration...")
        config_path = os.path.join(self.args.milora_weights_path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.milora_config = json.load(f)

        model_config = self.milora_config['model_config']

        # 4. 创建 MiLoRA 模型
        logging.info("4. Creating MiLoRA model...")
        config = self.base_model.config
        hidden_dim = config.hidden_size
        intermediate_dim = getattr(config, 'intermediate_size', hidden_dim * 4)
        num_attention_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers

        self.model = MiLoRAModel(
            base_model=self.base_model,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_attention_heads=num_attention_heads,
            lora_rank=model_config['lora_rank'],
            lora_alpha=model_config['lora_alpha'],
            top_k=model_config['top_k'],
            pooling_type=model_config['pooling_type'],
            dropout=model_config['dropout'],
            load_balance_weight=model_config['load_balance_weight']
        )

        # 5. 加载 MiLoRA 权重
        logging.info("5. Loading MiLoRA weights...")
        milora_weights_file = os.path.join(self.args.milora_weights_path, "milora_weights.pth")

        if not os.path.exists(milora_weights_file):
            raise FileNotFoundError(f"MiLoRA weights not found: {milora_weights_file}")

        milora_state_dict = torch.load(milora_weights_file, map_location="cpu")

        # 确保权重类型匹配
        model_dtype = next(self.model.parameters()).dtype
        for key in milora_state_dict:
            if isinstance(milora_state_dict[key], torch.Tensor):
                milora_state_dict[key] = milora_state_dict[key].to(model_dtype)

        # 加载权重到模型
        missing_keys, unexpected_keys = self.model.load_state_dict(milora_state_dict, strict=False)

        if missing_keys:
            logging.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys}")

        # 6. 设置为评估模式
        self.model.eval()
        self.model.to(self.device)

        logging.info("MiLoRA model loaded successfully!")

    def _load_test_data(self):
        """加载测试数据"""
        logging.info(f"Loading test data from {self.args.test_file}")

        with open(self.args.test_file, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

        logging.info(f"Loaded {len(self.test_data)} test samples")

    def _extract_answer(self, generated_text: str, question_text: str) -> str:
        """从生成的文本中提取答案"""
        # 移除问题部分
        if "### Answer:" in generated_text:
            answer_part = generated_text.split("### Answer:")[-1].strip()
        else:
            answer_part = generated_text.strip()

        # 提取数字答案
        import re
        numbers = re.findall(r'\b([1-9]|10)\b', answer_part)

        if numbers:
            return numbers[0]

        # 如果没有找到数字，返回第一个字符
        if answer_part:
            return answer_part[0]

        return "1"  # 默认答案

    @torch.no_grad()
    def evaluate_sample(
        self,
        instruction: str,
        true_answer: str,
        use_prompt_routing: bool = True
    ) -> Dict[str, Any]:
        """
        评估单个样本
        Args:
            instruction: 输入指令
            true_answer: 真实答案
            use_prompt_routing: 是否使用提示感知路由
        """
        # 构建输入
        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # 分词
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_length,
            padding=False
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # 设置提示感知路由模式
        self.model.set_prompt_routing_mode(use_prompt_routing)

        # 记录路由计算时间
        routing_start_time = time.time()

        # 生成答案
        with torch.no_grad():
            generated_ids = self.model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,  # 贪婪解码
                temperature=1.0,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        routing_time = time.time() - routing_start_time

        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 提取答案
        predicted_answer = self._extract_answer(generated_text, instruction)

        # 判断正确性
        is_correct = predicted_answer.strip() == true_answer.strip()

        # 获取专家统计（如果可用）
        expert_stats = {}
        try:
            model_stats = self.model.get_model_statistics()
            expert_stats = model_stats.get('layer_statistics', {})
        except Exception as e:
            logging.warning(f"Failed to get expert statistics: {e}")

        return {
            'instruction': instruction,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'generated_text': generated_text,
            'is_correct': is_correct,
            'routing_time': routing_time,
            'use_prompt_routing': use_prompt_routing,
            'expert_statistics': expert_stats
        }

    def evaluate(self) -> Dict[str, Any]:
        """执行完整评估"""
        logging.info("Starting MiLoRA evaluation...")

        results = []
        correct_count = 0
        total_routing_time = 0.0

        # 评估模式对比
        routing_modes = [
            ('with_prompt_routing', True),
            ('without_prompt_routing', False)
        ]

        all_results = {}

        for mode_name, use_prompt_routing in routing_modes:
            logging.info(f"Evaluating with {mode_name}...")

            mode_results = []
            mode_correct = 0
            mode_routing_time = 0.0

            progress_bar = tqdm(self.test_data, desc=f"Evaluating ({mode_name})")

            for item in progress_bar:
                instruction = item.get('instruction', '')
                true_answer = str(item.get('output', ''))

                # 评估样本
                result = self.evaluate_sample(
                    instruction=instruction,
                    true_answer=true_answer,
                    use_prompt_routing=use_prompt_routing
                )

                mode_results.append(result)

                if result['is_correct']:
                    mode_correct += 1

                mode_routing_time += result['routing_time']

                # 更新进度条
                current_accuracy = mode_correct / len(mode_results)
                progress_bar.set_postfix({
                    'Accuracy': f'{current_accuracy:.3f}',
                    'Correct': f'{mode_correct}/{len(mode_results)}'
                })

            # 计算模式统计
            mode_accuracy = mode_correct / len(mode_results)
            avg_routing_time = mode_routing_time / len(mode_results)

            all_results[mode_name] = {
                'results': mode_results,
                'accuracy': mode_accuracy,
                'correct_count': mode_correct,
                'total_samples': len(mode_results),
                'avg_routing_time': avg_routing_time,
                'total_routing_time': mode_routing_time
            }

            logging.info(f"{mode_name} - Accuracy: {mode_accuracy:.4f}, "
                         f"Avg Routing Time: {avg_routing_time:.4f}s")

        # 计算路由效率提升
        prompt_routing_time = all_results['with_prompt_routing']['avg_routing_time']
        normal_routing_time = all_results['without_prompt_routing']['avg_routing_time']
        efficiency_gain = (normal_routing_time - prompt_routing_time) / normal_routing_time * 100

        # 生成评估报告
        evaluation_summary = {
            'model_config': self.milora_config['model_config'],
            'evaluation_args': vars(self.args),
            'results_comparison': {
                'with_prompt_routing': {
                    'accuracy': all_results['with_prompt_routing']['accuracy'],
                    'avg_routing_time': all_results['with_prompt_routing']['avg_routing_time']
                },
                'without_prompt_routing': {
                    'accuracy': all_results['without_prompt_routing']['accuracy'],
                    'avg_routing_time': all_results['without_prompt_routing']['avg_routing_time']
                }
            },
            'efficiency_analysis': {
                'routing_time_reduction': prompt_routing_time - normal_routing_time,
                'efficiency_gain_percent': efficiency_gain,
                'prompt_routing_advantage': efficiency_gain > 0
            },
            'overall_statistics': {
                'total_samples': len(self.test_data),
                'model_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'evaluation_time': time.time()
            }
        }

        return {
            'evaluation_summary': evaluation_summary,
            'detailed_results': all_results
        }

    def save_results(self, results: Dict[str, Any]):
        """保存评估结果"""
        os.makedirs(self.args.output_dir, exist_ok=True)

        # 保存评估摘要
        summary_path = os.path.join(self.args.output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results['evaluation_summary'], f, indent=2, ensure_ascii=False)

        # 保存详细结果
        for mode_name, mode_data in results['detailed_results'].items():
            results_path = os.path.join(self.args.output_dir, f'{mode_name}_results.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(mode_data['results'], f, indent=2, ensure_ascii=False)

        # 保存配置信息
        config_path = os.path.join(self.args.output_dir, 'evaluation_config.json')
        eval_config = {
            'model_config': self.milora_config,
            'evaluation_args': vars(self.args),
            'timestamp': datetime.now().isoformat()
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(eval_config, f, indent=2, ensure_ascii=False)

        logging.info(f"Results saved to {self.args.output_dir}")

    def run_evaluation(self):
        """运行完整的评估流程"""
        # 执行评估
        results = self.evaluate()

        # 保存结果
        self.save_results(results)

        # 打印摘要
        summary = results['evaluation_summary']
        logging.info("\n" + "=" * 60)
        logging.info("MILORA EVALUATION SUMMARY")
        logging.info("=" * 60)

        logging.info(f"Model Configuration:")
        model_config = summary['model_config']
        logging.info(f"  LoRA Rank: {model_config['lora_rank']}")
        logging.info(f"  Top-k: {model_config['top_k']}")
        logging.info(f"  Pooling: {model_config['pooling_type']}")

        logging.info(f"\nResults Comparison:")
        comparison = summary['results_comparison']
        logging.info(f"  With Prompt Routing:")
        logging.info(f"    Accuracy: {comparison['with_prompt_routing']['accuracy']:.4f}")
        logging.info(f"    Avg Routing Time: {comparison['with_prompt_routing']['avg_routing_time']:.4f}s")

        logging.info(f"  Without Prompt Routing:")
        logging.info(f"    Accuracy: {comparison['without_prompt_routing']['accuracy']:.4f}")
        logging.info(f"    Avg Routing Time: {comparison['without_prompt_routing']['avg_routing_time']:.4f}s")

        logging.info(f"\nEfficiency Analysis:")
        efficiency = summary['efficiency_analysis']
        logging.info(f"  Efficiency Gain: {efficiency['efficiency_gain_percent']:.2f}%")
        logging.info(f"  Time Reduction: {efficiency['routing_time_reduction']:.4f}s")
        logging.info(f"  Prompt Routing Advantage: {efficiency['prompt_routing_advantage']}")

        logging.info("=" * 60)

        return results


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MiLoRA Evaluation Script")

    # 模型参数
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Path to the base model')
    parser.add_argument('--milora_weights_path', type=str, required=True,
                        help='Path to the MiLoRA weights directory')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to the test data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for saving results')

    # 生成参数
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum input sequence length')
    parser.add_argument('--max_new_tokens', type=int, default=5,
                        help='Maximum number of new tokens to generate')

    # 系统参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def setup_logging(output_dir: str):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'evaluation.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def main():
    # 解析参数
    args = parse_arguments()

    # 设置日志
    setup_logging(args.output_dir)

    # 记录参数
    logging.info("MiLoRA Evaluation Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

    # 创建评估器并运行评估
    evaluator = MiLoRAEvaluator(args)
    results = evaluator.run_evaluation()

    return results


if __name__ == "__main__":
    main()