#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA增强FFN集成CultureMoE评估脚本

功能：
1. 加载训练好的LoRA权重
2. 在测试集上评估模型性能
3. 分析专家利用率和文化专业化
4. 生成详细的评估报告
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
from torch.utils.data import DataLoader
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
from train_lora_culturemoe_ffn_integrated import CultureDatasetForLoRA, generate_cultural_mask


class LoRACultureMoEEvaluator:
    """LoRA增强CultureMoE评估器"""

    def __init__(self, model, tokenizer, lora_config: LoRACultureMoEConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.lora_config = lora_config
        self.device = next(model.parameters()).device

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

        model_dict = dict(self.model.named_parameters())

        for key, weight in lora_state_dict.items():
            if key in model_dict:
                model_dict[key].data.copy_(weight)
            else:
                unexpected_keys.append(key)

        if missing_keys:
            logging.warning(f"Missing keys in LoRA weights: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys in LoRA weights: {unexpected_keys}")

        logging.info(f"Successfully loaded LoRA weights from {lora_weights_path}")

    def evaluate_on_dataset(self, dataloader: DataLoader) -> Dict[str, Any]:
        """在数据集上评估模型"""
        self.model.eval()

        all_predictions = []
        all_true_labels = []
        all_culture_ids = []
        all_expert_weights = []
        all_losses = []

        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                culture_ids = batch['culture_ids']
                use_mask = batch.get('use_mask_mechanism', True)

                # 生成mask（如果启用）
                hidden_states_mask = None
                if use_mask:
                    temp_hidden_states = self.model.embed_tokens(input_ids)
                    hidden_states_mask = generate_cultural_mask(temp_hidden_states, culture_ids)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    culture_ids=culture_ids,
                    hidden_states_mask=hidden_states_mask,
                    return_dict=True
                )

                # 计算预测（基于生成概率）
                logits = outputs.last_hidden_state
                predictions = self._get_predictions_from_generation(
                    input_ids, attention_mask, culture_ids, hidden_states_mask
                )

                # 收集结果
                all_predictions.extend(predictions)
                all_true_labels.extend(culture_ids.cpu().numpy())
                all_culture_ids.extend(culture_ids.cpu().numpy())

                # 收集专家权重信息
                if hasattr(outputs, 'moe_aux_info') and outputs.moe_aux_info:
                    layer_expert_weights = []
                    for layer_aux in outputs.moe_aux_info:
                        if 'expert_weights' in layer_aux:
                            layer_expert_weights.append(layer_aux['expert_weights'].cpu().numpy())
                    all_expert_weights.append(layer_expert_weights)

                # 统计准确率
                batch_correct = sum(p == t for p, t in zip(predictions, culture_ids.cpu().numpy()))
                correct_predictions += batch_correct
                total_samples += len(predictions)

        # 计算评估指标
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_predictions, average='weighted'
        )

        # 每个文化的详细指标
        culture_metrics = {}
        for culture_id in range(self.lora_config.num_cultures):
            culture_mask = np.array(all_true_labels) == culture_id
            if culture_mask.sum() > 0:
                culture_predictions = np.array(all_predictions)[culture_mask]
                culture_true = np.array(all_true_labels)[culture_mask]

                culture_accuracy = accuracy_score(culture_true, culture_predictions)
                culture_precision, culture_recall, culture_f1, _ = precision_recall_fscore_support(
                    culture_true, culture_predictions, average='weighted', zero_division=0
                )

                culture_metrics[self.culture_names[culture_id]] = {
                    'accuracy': culture_accuracy,
                    'precision': culture_precision,
                    'recall': culture_recall,
                    'f1': culture_f1,
                    'sample_count': culture_mask.sum()
                }

        return {
            'overall_accuracy': accuracy,
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1': f1,
            'culture_metrics': culture_metrics,
            'predictions': all_predictions,
            'true_labels': all_true_labels,
            'expert_weights': all_expert_weights,
            'confusion_matrix': confusion_matrix(all_true_labels, all_predictions)
        }

    def _get_predictions_from_generation(self, input_ids, attention_mask, culture_ids, hidden_states_mask=None):
        """基于生成任务获取预测"""
        # 这是一个简化版本，实际应该基于生成的文本内容进行文化分类
        # 这里我们使用专家权重的模式作为预测
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            culture_ids=culture_ids,
            hidden_states_mask=hidden_states_mask,
            return_dict=True
        )

        predictions = []
        if hasattr(outputs, 'moe_aux_info') and outputs.moe_aux_info:
            for i in range(input_ids.shape[0]):
                # 使用最后一层的专家权重进行预测
                last_layer_aux = outputs.moe_aux_info[-1]
                if 'expert_weights' in last_layer_aux:
                    expert_weights = last_layer_aux['expert_weights'][i]

                    # 简单的启发式：选择权重最大的专家对应的文化
                    max_expert = torch.argmax(expert_weights).item()

                    # 将专家映射到文化（简化映射）
                    predicted_culture = max_expert % self.lora_config.num_cultures
                    predictions.append(predicted_culture)
                else:
                    predictions.append(0)  # 默认预测
        else:
            predictions = [0] * input_ids.shape[0]  # 默认预测

        return predictions

    def analyze_expert_utilization(self, expert_weights_data: List[List[np.ndarray]]) -> Dict[str, Any]:
        """分析专家利用率"""
        if not expert_weights_data:
            return {}

        num_layers = len(expert_weights_data[0])
        num_experts = self.lora_config.num_experts

        layer_stats = {}

        for layer_idx in range(num_layers):
            layer_weights = []
            for batch_weights in expert_weights_data:
                if layer_idx < len(batch_weights):
                    layer_weights.append(batch_weights[layer_idx])

            if layer_weights:
                # 合并所有批次的权重
                all_weights = np.concatenate(layer_weights, axis=0)  # [total_samples, num_experts]

                # 计算统计信息
                mean_weights = np.mean(all_weights, axis=0)
                std_weights = np.std(all_weights, axis=0)
                max_weights = np.max(all_weights, axis=0)

                # 计算专家被选为top-1的频率
                top1_selections = np.argmax(all_weights, axis=1)
                top1_frequencies = np.bincount(top1_selections, minlength=num_experts) / len(top1_selections)

                # 计算负载均衡指标
                load_balance = 1.0 - np.std(top1_frequencies)  # 越接近1越均衡

                # 计算熵
                entropy = -np.sum(top1_frequencies * np.log(top1_frequencies + 1e-8))
                max_entropy = np.log(num_experts)
                normalized_entropy = entropy / max_entropy

                layer_stats[f'layer_{layer_idx}'] = {
                    'mean_weights': mean_weights.tolist(),
                    'std_weights': std_weights.tolist(),
                    'max_weights': max_weights.tolist(),
                    'top1_frequencies': top1_frequencies.tolist(),
                    'load_balance': load_balance,
                    'entropy': entropy,
                    'normalized_entropy': normalized_entropy
                }

        return layer_stats

    def analyze_cultural_specialization(self, expert_weights_data: List[List[np.ndarray]],
                                      culture_ids: List[int]) -> Dict[str, Any]:
        """分析文化专业化"""
        if not expert_weights_data or not culture_ids:
            return {}

        num_layers = len(expert_weights_data[0])
        num_experts = self.lora_config.num_experts
        num_cultures = self.lora_config.num_cultures

        specialization_stats = {}

        for layer_idx in range(num_layers):
            # 收集该层的专家权重
            layer_weights = []
            layer_cultures = []

            sample_idx = 0
            for batch_idx, batch_weights in enumerate(expert_weights_data):
                if layer_idx < len(batch_weights):
                    batch_size = batch_weights[layer_idx].shape[0]
                    layer_weights.append(batch_weights[layer_idx])
                    layer_cultures.extend(culture_ids[sample_idx:sample_idx + batch_size])
                    sample_idx += batch_size

            if layer_weights:
                all_weights = np.concatenate(layer_weights, axis=0)

                # 分析每个专家对每种文化的专业化程度
                expert_culture_specialization = np.zeros((num_experts, num_cultures))

                for culture_id in range(num_cultures):
                    culture_mask = np.array(layer_cultures) == culture_id
                    if culture_mask.sum() > 0:
                        culture_weights = all_weights[culture_mask]
                        expert_culture_specialization[:, culture_id] = np.mean(culture_weights, axis=0)

                # 计算专家的文化偏好
                expert_preferences = np.argmax(expert_culture_specialization, axis=1)

                # 计算文化分离度
                culture_separation = []
                for expert_id in range(num_experts):
                    expert_weights_by_culture = expert_culture_specialization[expert_id]
                    max_weight = np.max(expert_weights_by_culture)
                    mean_weight = np.mean(expert_weights_by_culture)
                    separation = (max_weight - mean_weight) / (max_weight + 1e-8)
                    culture_separation.append(separation)

                specialization_stats[f'layer_{layer_idx}'] = {
                    'expert_culture_specialization': expert_culture_specialization.tolist(),
                    'expert_preferences': expert_preferences.tolist(),
                    'culture_separation': culture_separation,
                    'mean_separation': np.mean(culture_separation)
                }

        return specialization_stats

    def generate_evaluation_report(self, eval_results: Dict[str, Any],
                                 expert_stats: Dict[str, Any],
                                 specialization_stats: Dict[str, Any],
                                 output_dir: str):
        """生成评估报告"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 文本报告
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("LoRA Enhanced CultureMoE Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            # 整体性能
            f.write("Overall Performance:\n")
            f.write(f"  Accuracy: {eval_results['overall_accuracy']:.4f}\n")
            f.write(f"  Precision: {eval_results['overall_precision']:.4f}\n")
            f.write(f"  Recall: {eval_results['overall_recall']:.4f}\n")
            f.write(f"  F1-Score: {eval_results['overall_f1']:.4f}\n\n")

            # 每种文化的性能
            f.write("Performance by Culture:\n")
            for culture, metrics in eval_results['culture_metrics'].items():
                f.write(f"  {culture}:\n")
                f.write(f"    Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall: {metrics['recall']:.4f}\n")
                f.write(f"    F1-Score: {metrics['f1']:.4f}\n")
                f.write(f"    Samples: {metrics['sample_count']}\n\n")

            # 专家利用率
            if expert_stats:
                f.write("Expert Utilization Summary:\n")
                for layer_name, stats in expert_stats.items():
                    f.write(f"  {layer_name}:\n")
                    f.write(f"    Load Balance: {stats['load_balance']:.4f}\n")
                    f.write(f"    Normalized Entropy: {stats['normalized_entropy']:.4f}\n")
                    f.write(f"    Top-1 Frequencies: {[f'{freq:.3f}' for freq in stats['top1_frequencies']]}\n\n")

            # 文化专业化
            if specialization_stats:
                f.write("Cultural Specialization Summary:\n")
                for layer_name, stats in specialization_stats.items():
                    f.write(f"  {layer_name}:\n")
                    f.write(f"    Mean Separation: {stats['mean_separation']:.4f}\n")
                    f.write(f"    Expert Preferences: {stats['expert_preferences']}\n\n")

        # 2. 混淆矩阵可视化
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            eval_results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.culture_names.values()),
            yticklabels=list(self.culture_names.values())
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Culture')
        plt.ylabel('True Culture')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        # 3. 专家利用率可视化
        if expert_stats:
            self._plot_expert_utilization(expert_stats, output_dir)

        # 4. 文化专业化可视化
        if specialization_stats:
            self._plot_cultural_specialization(specialization_stats, output_dir)

        # 5. 保存详细结果
        results_path = os.path.join(output_dir, 'detailed_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            serializable_results = self._make_json_serializable({
                'evaluation_results': eval_results,
                'expert_utilization': expert_stats,
                'cultural_specialization': specialization_stats
            })
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Evaluation report generated in {output_dir}")

    def _plot_expert_utilization(self, expert_stats: Dict[str, Any], output_dir: str):
        """绘制专家利用率图表"""
        layer_names = list(expert_stats.keys())

        # 负载均衡图
        load_balances = [stats['load_balance'] for stats in expert_stats.values()]
        entropies = [stats['normalized_entropy'] for stats in expert_stats.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 负载均衡
        ax1.bar(layer_names, load_balances)
        ax1.set_title('Load Balance by Layer')
        ax1.set_ylabel('Load Balance Score')
        ax1.set_xlabel('Layer')
        ax1.tick_params(axis='x', rotation=45)

        # 熵
        ax2.bar(layer_names, entropies)
        ax2.set_title('Normalized Entropy by Layer')
        ax2.set_ylabel('Normalized Entropy')
        ax2.set_xlabel('Layer')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'expert_utilization.png'))
        plt.close()

    def _plot_cultural_specialization(self, specialization_stats: Dict[str, Any], output_dir: str):
        """绘制文化专业化图表"""
        # 选择几个代表性层进行可视化
        representative_layers = list(specialization_stats.keys())[:4]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, layer_name in enumerate(representative_layers):
            if i >= 4:
                break

            stats = specialization_stats[layer_name]
            specialization_matrix = np.array(stats['expert_culture_specialization'])

            sns.heatmap(
                specialization_matrix,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=list(self.culture_names.values()),
                yticklabels=[f'Expert {j}' for j in range(specialization_matrix.shape[0])],
                ax=axes[i]
            )
            axes[i].set_title(f'Cultural Specialization - {layer_name}')
            axes[i].set_xlabel('Culture')
            axes[i].set_ylabel('Expert')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cultural_specialization.png'))
        plt.close()

    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(description='LoRA Enhanced CultureMoE Evaluation')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf', help='Base model path')
    parser.add_argument('--lora_weights', type=str, required=True, help='Path to LoRA weights')
    parser.add_argument('--test_data', type=str, required=True, help='Test data path')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32.0, help='LoRA alpha')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # LoRA配置
    lora_config = LoRACultureMoEConfig(
        num_experts=args.num_experts,
        top_k=2,
        capacity_factor=1.25,
        num_cultures=6,
        culture_dim=256,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        attention_lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        expert_lora_targets=["gate_proj", "up_proj", "down_proj"],
    )

    # 创建模型
    logger.info("Creating LoRA enhanced CultureMoE model...")
    model = create_lora_culturemoe_model(args.base_model, lora_config)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建评估器
    evaluator = LoRACultureMoEEvaluator(model, tokenizer, lora_config)

    # 加载LoRA权重
    logger.info(f"Loading LoRA weights from {args.lora_weights}")
    evaluator.load_lora_weights(args.lora_weights)

    # 加载测试数据
    logger.info(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_dataset = CultureDatasetForLoRA(test_data, tokenizer, max_length=512, use_mask_mechanism=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 评估模型
    logger.info("Starting evaluation...")
    eval_results = evaluator.evaluate_on_dataset(test_dataloader)

    # 分析专家利用率
    logger.info("Analyzing expert utilization...")
    expert_stats = evaluator.analyze_expert_utilization(eval_results['expert_weights'])

    # 分析文化专业化
    logger.info("Analyzing cultural specialization...")
    specialization_stats = evaluator.analyze_cultural_specialization(
        eval_results['expert_weights'],
        eval_results['true_labels']
    )

    # 生成报告
    logger.info("Generating evaluation report...")
    evaluator.generate_evaluation_report(
        eval_results, expert_stats, specialization_stats, args.output_dir
    )

    # 打印关键结果
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {eval_results['overall_accuracy']:.4f}")
    print(f"Overall F1-Score: {eval_results['overall_f1']:.4f}")
    print(f"Test Samples: {len(eval_results['true_labels'])}")

    if expert_stats:
        avg_load_balance = np.mean([stats['load_balance'] for stats in expert_stats.values()])
        avg_entropy = np.mean([stats['normalized_entropy'] for stats in expert_stats.values()])
        print(f"Average Load Balance: {avg_load_balance:.4f}")
        print(f"Average Normalized Entropy: {avg_entropy:.4f}")

    print(f"\nDetailed results saved to: {args.output_dir}")
    print("=" * 60)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()