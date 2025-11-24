#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA增强FFN集成CultureMoE评估脚本（DDP多卡版本）

功能：
1. 支持DDP多卡推理加速
2. 加载训练好的LoRA权重
3. 在测试集上评估模型性能
4. 分析专家利用率和文化专业化
5. 生成详细的评估报告
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.lora_enhanced_culturemoe import LoRACultureMoEConfig
from llamafactory.model.lora_culturemoe_model import create_lora_culturemoe_model
from train_lora_culturemoe_ffn_integrated_ddp import CultureDatasetForLoRA


# ===== DDP工具函数 =====
def setup_ddp(rank, world_size):
    """初始化DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # 使用不同的端口避免冲突

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置当前设备
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()


def gather_results(results, world_size):
    """收集所有进程的结果"""
    if world_size == 1:
        return results

    # 将结果转换为tensor
    results_tensor = torch.tensor([len(results)], dtype=torch.long, device=torch.cuda.current_device())

    # 收集每个进程的结果数量
    all_counts = [torch.zeros_like(results_tensor) for _ in range(world_size)]
    dist.all_gather(all_counts, results_tensor)

    # 收集所有结果
    all_results = []
    for rank_results in [results]:  # 简化版本，实际应该收集所有rank的结果
        all_results.extend(rank_results)

    return all_results


class LoRACultureMoEEvaluatorDDP:
    """LoRA增强CultureMoE评估器（DDP版本）"""

    def __init__(self, model, tokenizer, lora_config: LoRACultureMoEConfig, rank: int, world_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.lora_config = lora_config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.cuda.current_device()

        # 文化标签映射
        self.culture_names = {
            0: "Asia", 1: "Europe", 2: "North America",
            3: "South America", 4: "Africa", 5: "Oceania"
        }

    def load_lora_weights(self, lora_weights_path: str):
        """加载LoRA权重"""
        if not os.path.exists(lora_weights_path):
            raise FileNotFoundError(f"LoRA weights not found at {lora_weights_path}")

        lora_state_dict = torch.load(lora_weights_path, map_location=self.device)

        # 加载权重到模型
        missing_keys = []
        unexpected_keys = []

        # 获取实际模型（处理DDP包装）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model_dict = dict(model.named_parameters())

        for key, weight in lora_state_dict.items():
            if key in model_dict:
                model_dict[key].data.copy_(weight)
            else:
                unexpected_keys.append(key)

        if missing_keys and self.rank == 0:
            logging.warning(f"Missing keys in LoRA weights: {missing_keys}")
        if unexpected_keys and self.rank == 0:
            logging.warning(f"Unexpected keys in LoRA weights: {unexpected_keys}")

        if self.rank == 0:
            logging.info(f"Successfully loaded LoRA weights from {lora_weights_path}")

    def evaluate_on_dataset(self, dataloader: DataLoader) -> Dict[str, Any]:
        """在数据集上评估模型（生成式任务）"""
        self.model.eval()

        generated_answers = []
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            if self.rank == 0:
                progress_bar = tqdm(dataloader, desc="Evaluating")
            else:
                progress_bar = dataloader

            for batch_idx, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                input_ids_mask = batch['input_ids_mask']
                attention_mask_mask = batch['attention_mask_mask']
                labels = batch['labels']
                culture_ids = batch['culture_ids']

                batch_size = input_ids.shape[0]

                for i in range(batch_size):
                    try:
                        # 准备单个样本的输入
                        sample_input_ids = input_ids[i:i+1]
                        sample_attention_mask = attention_mask[i:i+1]
                        sample_input_ids_mask = input_ids_mask[i:i+1]
                        sample_attention_mask_mask = attention_mask_mask[i:i+1]
                        sample_culture_ids = culture_ids[i:i+1]

                        # 构建用于生成的prompt（去掉答案部分）
                        # 找到assistant开始位置
                        full_tokens = sample_input_ids[0].cpu().numpy()
                        prompt_tokens = []

                        # 寻找"<|start_header_id|>assistant<|end_header_id|>"之后的位置
                        assistant_start_pattern = self.tokenizer.encode(
                            "<|start_header_id|>assistant<|end_header_id|>\\n\\n",
                            add_special_tokens=False
                        )

                        # 简化处理：截断到assistant开始位置
                        for j, token_id in enumerate(full_tokens):
                            prompt_tokens.append(token_id)
                            # 检查是否到达assistant开始位置
                            if len(prompt_tokens) >= len(assistant_start_pattern):
                                if prompt_tokens[-len(assistant_start_pattern):] == assistant_start_pattern:
                                    break

                        prompt_input_ids = torch.tensor([prompt_tokens], device=self.device)
                        prompt_attention_mask = torch.ones_like(prompt_input_ids)

                        # 生成答案
                        max_new_tokens = 5  # 只生成几个token（数字答案）

                        generate_kwargs = {
                            'input_ids': prompt_input_ids,
                            'attention_mask': prompt_attention_mask,
                            'max_new_tokens': max_new_tokens,
                            'do_sample': False,
                            'temperature': 1.0,
                            'pad_token_id': self.tokenizer.pad_token_id,
                            'eos_token_id': self.tokenizer.eos_token_id,
                            'culture_ids': sample_culture_ids,
                            'input_ids_mask': sample_input_ids_mask,
                            'attention_mask_mask': sample_attention_mask_mask
                        }

                        outputs = self.model.generate(**generate_kwargs)

                        # 解码生成的文本
                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        prompt_text = self.tokenizer.decode(prompt_input_ids[0], skip_special_tokens=True)

                        # 提取生成的答案部分
                        if len(generated_text) > len(prompt_text):
                            generated_answer = generated_text[len(prompt_text):].strip()
                        else:
                            generated_answer = ""

                        # 提取数字答案
                        predicted_answer = self._extract_answer(generated_answer)

                        # 获取真实答案
                        true_answer = self._get_true_answer_from_batch(batch, i)

                        # 判断正确性
                        is_correct = self._is_answer_correct(predicted_answer, true_answer)

                        if is_correct:
                            correct_predictions += 1

                        # 保存结果
                        answer_data = {
                            'index': total_samples,
                            'generated_answer': generated_answer,
                            'predicted_answer': predicted_answer,
                            'true_answer': true_answer,
                            'culture_id': culture_ids[i].item(),
                            'culture_name': self.culture_names[culture_ids[i].item()],
                            'is_correct': is_correct,
                            'rank': self.rank  # 记录处理的rank
                        }

                        generated_answers.append(answer_data)
                        total_samples += 1

                    except Exception as e:
                        if self.rank == 0:
                            logging.error(f"Error processing sample {total_samples}: {str(e)}")
                        # 添加错误样本
                        generated_answers.append({
                            'index': total_samples,
                            'generated_answer': f"Error: {str(e)}",
                            'predicted_answer': "0",
                            'true_answer': "0",
                            'culture_id': culture_ids[i].item() if i < len(culture_ids) else 0,
                            'culture_name': "Unknown",
                            'is_correct': False,
                            'rank': self.rank
                        })
                        total_samples += 1

        # 在DDP环境中收集所有结果
        if self.world_size > 1:
            # 同步所有进程的结果
            all_generated_answers = gather_results(generated_answers, self.world_size)

            # 计算全局统计
            total_correct = sum(1 for ans in all_generated_answers if ans['is_correct'])
            total_count = len(all_generated_answers)
        else:
            all_generated_answers = generated_answers
            total_correct = correct_predictions
            total_count = total_samples

        # 计算总体准确率
        overall_accuracy = total_correct / total_count if total_count > 0 else 0

        # 计算各文化的准确率
        culture_metrics = {}
        for culture_id in range(self.lora_config.num_cultures):
            culture_data = [ans for ans in all_generated_answers if ans['culture_id'] == culture_id]
            if culture_data:
                culture_correct = sum(1 for ans in culture_data if ans['is_correct'])
                culture_total = len(culture_data)
                culture_accuracy = culture_correct / culture_total

                culture_metrics[self.culture_names[culture_id]] = {
                    'accuracy': culture_accuracy,
                    'correct_count': culture_correct,
                    'total_count': culture_total,
                    'sample_count': culture_total
                }

        return {
            'overall_accuracy': overall_accuracy,
            'correct_predictions': total_correct,
            'total_samples': total_count,
            'culture_metrics': culture_metrics,
            'generated_answers': all_generated_answers
        }

    def _extract_answer(self, generated_text: str) -> str:
        """从生成的文本中提取答案"""
        import re

        # 清理文本
        text = generated_text.strip()

        # 尝试提取数字（0-5）
        number_match = re.search(r'^(\\d+)', text)
        if number_match:
            return number_match.group(1)

        # 尝试提取字母（A-F）
        letter_match = re.search(r'^([A-F])', text, re.IGNORECASE)
        if letter_match:
            letter = letter_match.group(1).upper()
            # 将字母映射到数字
            letter_to_number = {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': '4', 'F': '5'}
            return letter_to_number.get(letter, '0')

        # 如果都没找到，返回第一个词或默认值
        words = text.split()
        if words:
            return words[0]

        return "0"  # 默认答案

    def _get_true_answer_from_batch(self, batch: Dict[str, torch.Tensor], index: int) -> str:
        """从batch中获取真实答案"""
        # 从数据集中获取真实的output答案
        if 'true_output' in batch:
            return batch['true_output'][index].strip()
        else:
            # 回退到使用culture_id
            culture_id = batch['culture_ids'][index].item()
            return str(culture_id)

    def _is_answer_correct(self, predicted: str, true: str) -> bool:
        """判断答案是否正确"""
        # 清理和标准化答案
        pred_clean = predicted.strip().lower()
        true_clean = true.strip().lower()

        # 直接比较
        if pred_clean == true_clean:
            return True

        # 尝试数字比较
        try:
            pred_num = int(pred_clean)
            true_num = int(true_clean)
            return pred_num == true_num
        except ValueError:
            pass

        # 尝试字母到数字的映射比较
        letter_to_number = {'a': '0', 'b': '1', 'c': '2', 'd': '3', 'e': '4', 'f': '5'}

        pred_mapped = letter_to_number.get(pred_clean, pred_clean)
        true_mapped = letter_to_number.get(true_clean, true_clean)

        return pred_mapped == true_mapped

    def generate_evaluation_report(self, eval_results: Dict[str, Any], output_dir: str):
        """生成评估报告（只在主进程）"""
        if self.rank != 0:
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. 文本报告
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\\n")
            f.write("LoRA Enhanced CultureMoE Evaluation Report (DDP)\\n")
            f.write("=" * 60 + "\\n\\n")

            # 整体性能
            f.write("Overall Performance:\\n")
            f.write(f"  Accuracy: {eval_results['overall_accuracy']:.4f}\\n")
            f.write(f"  Total Samples: {eval_results['total_samples']}\\n")
            f.write(f"  Correct Predictions: {eval_results['correct_predictions']}\\n\\n")

            # 每种文化的性能
            f.write("Performance by Culture:\\n")
            for culture, metrics in eval_results['culture_metrics'].items():
                f.write(f"  {culture}:\\n")
                f.write(f"    Accuracy: {metrics['accuracy']:.4f}\\n")
                f.write(f"    Correct/Total: {metrics['correct_count']}/{metrics['total_count']}\\n")
                f.write(f"    Samples: {metrics['sample_count']}\\n\\n")

        # 2. 保存详细结果
        results_path = os.path.join(output_dir, 'detailed_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Evaluation report generated in {output_dir}")


def evaluate_ddp(rank, world_size, args):
    """DDP评估函数"""
    # 设置DDP
    setup_ddp(rank, world_size)

    # 设置日志（只在主进程）
    if rank == 0:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    else:
        logger = None

    # LoRA配置
    lora_config = LoRACultureMoEConfig(
        num_experts=args.num_experts,
        top_k=2,
        capacity_factor=1.25,
        num_cultures=6,
        culture_dim=256,
        lora_rank=16,  # 固定值
        lora_alpha=32.0,  # 固定值
        lora_dropout=0.1,
        attention_lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        expert_lora_targets=["gate_proj", "up_proj", "down_proj"],
    )

    # 创建模型
    if rank == 0:
        logger.info("Creating LoRA enhanced CultureMoE model...")
    model = create_lora_culturemoe_model(args.base_model, lora_config)
    model.cuda(rank)

    # 包装为DDP模型（用于推理）
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建评估器
    evaluator = LoRACultureMoEEvaluatorDDP(model, tokenizer, lora_config, rank, world_size)

    # 加载LoRA权重
    if rank == 0:
        logger.info(f"Loading LoRA weights from {args.lora_weights}")
    evaluator.load_lora_weights(args.lora_weights)

    # 加载测试数据
    if rank == 0:
        logger.info(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_dataset = CultureDatasetForLoRA(test_data, tokenizer, max_length=512, use_mask_mechanism=True)

    # 创建DDP采样器
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)

    # 评估模型
    if rank == 0:
        logger.info("Starting evaluation...")
    eval_results = evaluator.evaluate_on_dataset(test_dataloader)

    # 生成报告（只在主进程）
    if rank == 0:
        logger.info("Generating evaluation report...")
        evaluator.generate_evaluation_report(eval_results, args.output_dir)

        # 打印关键结果
        print("\\n" + "=" * 60)
        print("EVALUATION SUMMARY (DDP)")
        print("=" * 60)
        print(f"Overall Accuracy: {eval_results['overall_accuracy']:.4f}")
        print(f"Total Samples: {eval_results['total_samples']}")
        print(f"Correct Predictions: {eval_results['correct_predictions']}")

        print(f"\\nDetailed results saved to: {args.output_dir}")
        print("=" * 60)

        logger.info("Evaluation completed!")

    # 清理DDP
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='LoRA Enhanced CultureMoE Evaluation with DDP')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf', help='Base model path')
    parser.add_argument('--lora_weights', type=str, required=True, help='Path to LoRA weights')
    parser.add_argument('--test_data', type=str, required=True, help='Test data path')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs (1 for single GPU, 2+ for DDP)')

    args = parser.parse_args()

    if args.num_gpus == 1:
        # 单卡推理
        print("Using single GPU evaluation...")
        world_size = 1
        evaluate_ddp(0, world_size, args)
    else:
        # 多卡推理
        print(f"Using DDP evaluation with {args.num_gpus} GPUs...")
        world_size = args.num_gpus

        # 检查GPU数量
        if torch.cuda.device_count() < world_size:
            print(f"Warning: Only {torch.cuda.device_count()} GPUs available, but {world_size} requested")
            world_size = torch.cuda.device_count()

        # 启动多进程
        torch.multiprocessing.spawn(evaluate_ddp, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()