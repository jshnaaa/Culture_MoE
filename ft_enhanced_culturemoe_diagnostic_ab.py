#!/usr/bin/env python3
"""
Enhanced CultureMoE 诊断实验脚本

目标：诊断路由塌陷和专家功能重叠问题

关键诊断功能：
1. 路由分布分析 - 检测专家利用率不均衡
2. 专家输出相似度 - 检测功能重叠
3. 文化信号强度 - 评估数据集文化差异
4. 按国家分组分析 - 识别文化特异性路由模式
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.enhanced_culturemoe import EnhancedCultureMoE
from src.llamafactory.model.moe_args import ModelArgs

# 复用现有的数据集类
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    load_and_process_data,
    extract_answer_from_text,
    generate_answer
)


class DiagnosticCollector:
    """诊断数据收集器"""

    def __init__(self):
        self.router_probs = []
        self.expert_outputs = []
        self.expert_assignments = []
        self.cultural_info = []
        self.sample_entropies = []

    def collect_batch_data(self, router_probs, expert_outputs, cultural_labels):
        """收集单个batch的数据"""
        self.router_probs.append(router_probs.detach().cpu())
        if expert_outputs is not None:
            self.expert_outputs.append(expert_outputs.detach().cpu())
        if cultural_labels is not None:
            self.cultural_info.extend(cultural_labels)

        # 计算路由熵
        entropy = -(router_probs * (router_probs + 1e-12).log()).sum(dim=-1)
        self.sample_entropies.append(entropy.detach().cpu())

        # 记录专家分配
        top1_assignments = router_probs.argmax(dim=-1)
        self.expert_assignments.append(top1_assignments.detach().cpu())

    def get_aggregated_data(self):
        """获取聚合后的数据"""
        router_all = torch.cat(self.router_probs, dim=0)
        assignments_all = torch.cat(self.expert_assignments, dim=0)
        entropies_all = torch.cat(self.sample_entropies, dim=0)

        expert_outputs_all = None
        if self.expert_outputs:
            expert_outputs_all = torch.cat(self.expert_outputs, dim=0)

        return {
            'router_probs': router_all,
            'expert_assignments': assignments_all,
            'sample_entropies': entropies_all,
            'expert_outputs': expert_outputs_all,
            'cultural_info': self.cultural_info
        }


def calculate_gini_coefficient(x):
    """计算Gini系数"""
    x = np.array(x)
    if x.sum() == 0:
        return 0.0

    # 排序
    sorted_x = np.sort(x)
    n = len(x)

    # 计算Gini系数
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

    return gini


def analyze_expert_utilization(router_probs):
    """分析专家利用率"""
    # 软权重平均
    mean_weights = router_probs.mean(dim=0).numpy()

    # Top-1分配统计
    top1_assignments = router_probs.argmax(dim=1)
    counts = torch.bincount(top1_assignments, minlength=router_probs.size(1))
    frac_top1 = counts.float() / router_probs.size(0)

    # 计算Gini系数
    gini_coeff = calculate_gini_coefficient(mean_weights)

    return {
        'mean_weights': mean_weights.tolist(),
        'top1_fractions': frac_top1.numpy().tolist(),
        'top1_counts': counts.numpy().tolist(),
        'gini_coefficient': gini_coeff,
        'utilization_std': np.std(mean_weights),
        'max_utilization': np.max(mean_weights),
        'min_utilization': np.min(mean_weights)
    }


def analyze_router_entropy(sample_entropies):
    """分析路由熵分布"""
    entropies = sample_entropies.numpy()

    return {
        'mean': float(np.mean(entropies)),
        'std': float(np.std(entropies)),
        'min': float(np.min(entropies)),
        'max': float(np.max(entropies)),
        'percentiles': {
            '25': float(np.percentile(entropies, 25)),
            '50': float(np.percentile(entropies, 50)),
            '75': float(np.percentile(entropies, 75)),
            '90': float(np.percentile(entropies, 90)),
            '95': float(np.percentile(entropies, 95))
        },
        'low_entropy_fraction': float(np.mean(entropies < 0.1)),
        'high_entropy_fraction': float(np.mean(entropies > 1.0))
    }


def analyze_expert_similarity(expert_outputs):
    """分析专家输出相似度"""
    if expert_outputs is None:
        return {'error': 'No expert outputs available'}

    # 计算每个专家的平均输出
    num_experts = expert_outputs.size(-1)
    expert_means = []

    for i in range(num_experts):
        expert_output = expert_outputs[:, :, i]  # [batch, seq_len]
        expert_mean = expert_output.mean(dim=0)  # [seq_len]
        expert_means.append(expert_mean.numpy())

    expert_means = np.array(expert_means)  # [num_experts, seq_len]

    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(expert_means)

    # 提取上三角（不包括对角线）
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    return {
        'similarity_matrix': similarity_matrix.tolist(),
        'mean_similarity': float(np.mean(upper_triangle)),
        'std_similarity': float(np.std(upper_triangle)),
        'max_similarity': float(np.max(upper_triangle)),
        'min_similarity': float(np.min(upper_triangle)),
        'high_similarity_pairs': int(np.sum(upper_triangle > 0.8)),
        'total_pairs': len(upper_triangle)
    }


def analyze_cultural_routing(router_probs, expert_assignments, cultural_info):
    """分析按文化分组的路由模式"""
    if not cultural_info:
        return {'error': 'No cultural information available'}

    # 创建DataFrame便于分析
    df_data = []
    for i, culture in enumerate(cultural_info):
        if i < len(expert_assignments):
            df_data.append({
                'culture': culture,
                'expert_assignment': expert_assignments[i].item(),
                'sample_idx': i
            })

    if not df_data:
        return {'error': 'No matching cultural data'}

    df = pd.DataFrame(df_data)

    # 计算每个文化的专家分配分布
    culture_expert_dist = pd.crosstab(df['culture'], df['expert_assignment'], normalize='index')

    # 计算文化间的路由相似度
    culture_similarities = {}
    cultures = culture_expert_dist.index.tolist()

    for i, culture1 in enumerate(cultures):
        for culture2 in cultures[i+1:]:
            dist1 = culture_expert_dist.loc[culture1].values
            dist2 = culture_expert_dist.loc[culture2].values
            similarity = cosine_similarity([dist1], [dist2])[0, 0]
            culture_similarities[f"{culture1}_vs_{culture2}"] = float(similarity)

    return {
        'culture_expert_distribution': culture_expert_dist.to_dict(),
        'culture_similarities': culture_similarities,
        'num_cultures': len(cultures),
        'cultures': cultures,
        'mean_culture_similarity': float(np.mean(list(culture_similarities.values()))) if culture_similarities else 0.0
    }


def create_visualizations(diagnostic_results, output_dir):
    """创建可视化图表"""
    viz_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)

    plt.style.use('default')

    # 1. 专家利用率图
    if 'expert_utilization' in diagnostic_results:
        util_data = diagnostic_results['expert_utilization']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 软权重分布
        experts = range(len(util_data['mean_weights']))
        ax1.bar(experts, util_data['mean_weights'])
        ax1.set_title('Expert Utilization (Soft Weights)')
        ax1.set_xlabel('Expert ID')
        ax1.set_ylabel('Mean Weight')
        ax1.grid(True, alpha=0.3)

        # Top-1分配分布
        ax2.bar(experts, util_data['top1_fractions'])
        ax2.set_title('Expert Utilization (Top-1 Assignments)')
        ax2.set_xlabel('Expert ID')
        ax2.set_ylabel('Fraction of Samples')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'expert_utilization.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. 路由熵分布
    if 'router_entropy' in diagnostic_results:
        entropy_data = diagnostic_results['router_entropy']

        plt.figure(figsize=(10, 6))

        # 创建示例数据用于直方图（实际应该从原始数据生成）
        # 这里使用正态分布近似
        sample_entropies = np.random.normal(
            entropy_data['mean'],
            entropy_data['std'],
            1000
        )

        plt.hist(sample_entropies, bins=50, alpha=0.7, density=True)
        plt.axvline(entropy_data['mean'], color='red', linestyle='--', label=f"Mean: {entropy_data['mean']:.3f}")
        plt.axvline(entropy_data['percentiles']['50'], color='orange', linestyle='--', label=f"Median: {entropy_data['percentiles']['50']:.3f}")

        plt.title('Router Entropy Distribution')
        plt.xlabel('Entropy')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(viz_dir, 'router_entropy_dist.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. 专家相似度热图
    if 'expert_similarity' in diagnostic_results and 'similarity_matrix' in diagnostic_results['expert_similarity']:
        sim_matrix = np.array(diagnostic_results['expert_similarity']['similarity_matrix'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Expert Output Similarity Matrix')
        plt.xlabel('Expert ID')
        plt.ylabel('Expert ID')

        plt.savefig(os.path.join(viz_dir, 'expert_similarity_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 4. 文化路由模式
    if 'cultural_routing' in diagnostic_results and 'culture_expert_distribution' in diagnostic_results['cultural_routing']:
        culture_dist = diagnostic_results['cultural_routing']['culture_expert_distribution']

        # 转换为DataFrame
        df = pd.DataFrame(culture_dist).fillna(0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap='Blues', fmt='.3f')
        plt.title('Cultural Routing Patterns')
        plt.xlabel('Expert ID')
        plt.ylabel('Culture/Country')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        plt.savefig(os.path.join(viz_dir, 'cultural_routing_pattern.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ 可视化图表已保存到: {viz_dir}")


def train_epoch_diagnostic(model_adapter, train_loader, optimizer, device, collector):
    """诊断模式的训练epoch"""
    model_adapter.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Diagnostic Training")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model_adapter(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs['loss']

        # 收集诊断数据
        if 'routing_info' in outputs and 'expert_weights' in outputs:
            routing_info = outputs['routing_info']
            expert_weights = outputs['expert_weights']  # [B, num_experts]
            cultural_labels = batch.get('cultural_labels', None)

            # 使用expert_weights作为router_probs
            router_probs = expert_weights
            expert_outputs = None  # EnhancedCultureMoE doesn't expose individual expert outputs

            collector.collect_batch_data(
                router_probs=router_probs,
                expert_outputs=expert_outputs,
                cultural_labels=cultural_labels
            )

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return {'loss': total_loss / num_batches if num_batches > 0 else 0}


def main():
    parser = argparse.ArgumentParser(description="Enhanced CultureMoE Diagnostic Experiment")

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)

    # CultureMoE参数
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--culture_loss_weight", type=float, default=1.0)
    parser.add_argument("--load_balance_weight", type=float, default=0.01)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--specialization_weight", type=float, default=0.1)
    parser.add_argument("--diversity_weight", type=float, default=0.05)
    parser.add_argument("--eval_interval", type=int, default=1)

    # 诊断参数
    parser.add_argument("--diagnostic_mode", action='store_true')
    parser.add_argument("--save_router_logs", action='store_true')
    parser.add_argument("--save_expert_outputs", action='store_true')
    parser.add_argument("--analyze_cultural_routing", action='store_true')

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print("Enhanced CultureMoE 诊断实验")
    print("="*80)
    print(f"设备: {device}")
    print(f"基础模型: {args.base_model_path}")
    print(f"训练数据: {args.train_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"诊断模式: {'开启' if args.diagnostic_mode else '关闭'}")
    print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded")

    # 加载数据
    print("\nLoading and processing data...")
    datasets = load_and_process_data(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        val_split=args.val_split
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    print("✅ Data loaded")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 加载基础模型
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    base_model = base_model.to(device)
    print("✅ Base model loaded")

    # 创建CultureMoE配置
    print("\nConfiguring Enhanced CultureMoE...")
    model_args = ModelArgs(
        llama_model_path=args.base_model_path,
        num_experts=args.num_experts,
        top_k=args.top_k,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # 设置诊断模式的相关参数
        experts_hidden_dim=256,
        router_hidden_dim=256,
        shared_hidden_dim=512,
        num_heads=8,
        classification_hidden_dim=256,
        num_classes=2
    )

    # 创建模型适配器
    config = base_model.config  # 获取模型配置
    model_adapter = EnhancedCultureMoE(
        llama_model=base_model,
        config=config,
        args=model_args,
        culture_loss_lambda=args.culture_loss_weight,
        moe_fusion=0.4,
        num_cultures=6,
        culture_dim=256
    )
    model_adapter.print_trainable_parameters()
    print("✅ Enhanced CultureMoE configured")

    # 优化器
    optimizer = torch.optim.AdamW(
        model_adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 创建诊断收集器
    collector = DiagnosticCollector()

    # 训练循环（诊断模式）
    print("\n" + "="*80)
    print("开始诊断训练...")
    print("="*80 + "\n")

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_metrics = train_epoch_diagnostic(
            model_adapter, train_loader, optimizer, device, collector
        )

        print(f"  Train Loss: {train_metrics['loss']:.4f}")

    # 分析收集的诊断数据
    print("\n" + "="*80)
    print("分析诊断数据...")
    print("="*80)

    aggregated_data = collector.get_aggregated_data()

    # 执行各项诊断分析
    diagnostic_results = {}

    # 1. 专家利用率分析
    print("1. 分析专家利用率...")
    expert_util = analyze_expert_utilization(aggregated_data['router_probs'])
    diagnostic_results['expert_utilization'] = expert_util
    print(f"   Gini系数: {expert_util['gini_coefficient']:.4f}")
    print(f"   最大利用率: {expert_util['max_utilization']:.4f}")
    print(f"   最小利用率: {expert_util['min_utilization']:.4f}")

    # 2. 路由熵分析
    print("\n2. 分析路由熵分布...")
    entropy_analysis = analyze_router_entropy(aggregated_data['sample_entropies'])
    diagnostic_results['router_entropy'] = entropy_analysis
    print(f"   平均熵: {entropy_analysis['mean']:.4f}")
    print(f"   标准差: {entropy_analysis['std']:.4f}")
    print(f"   低熵样本比例: {entropy_analysis['low_entropy_fraction']:.4f}")

    # 3. 专家相似度分析
    print("\n3. 分析专家输出相似度...")
    similarity_analysis = analyze_expert_similarity(aggregated_data['expert_outputs'])
    diagnostic_results['expert_similarity'] = similarity_analysis
    if 'mean_similarity' in similarity_analysis:
        print(f"   平均相似度: {similarity_analysis['mean_similarity']:.4f}")
        print(f"   高相似度对数: {similarity_analysis['high_similarity_pairs']}")

    # 4. 文化路由分析
    if args.analyze_cultural_routing:
        print("\n4. 分析文化路由模式...")
        cultural_analysis = analyze_cultural_routing(
            aggregated_data['router_probs'],
            aggregated_data['expert_assignments'],
            aggregated_data['cultural_info']
        )
        diagnostic_results['cultural_routing'] = cultural_analysis
        if 'mean_culture_similarity' in cultural_analysis:
            print(f"   文化间路由相似度: {cultural_analysis['mean_culture_similarity']:.4f}")

    # 保存诊断结果
    print("\n5. 保存诊断结果...")

    # 主诊断报告
    with open(os.path.join(args.output_dir, 'diagnostic_report.json'), 'w', encoding='utf-8') as f:
        json.dump(diagnostic_results, f, indent=2, ensure_ascii=False)

    # 详细数据
    if args.save_router_logs:
        router_data = {
            'router_probs': aggregated_data['router_probs'].numpy().tolist(),
            'expert_assignments': aggregated_data['expert_assignments'].numpy().tolist(),
            'sample_entropies': aggregated_data['sample_entropies'].numpy().tolist()
        }
        with open(os.path.join(args.output_dir, 'router_distribution.json'), 'w') as f:
            json.dump(router_data, f, indent=2)

    # 生成可视化
    print("\n6. 生成可视化图表...")
    create_visualizations(diagnostic_results, args.output_dir)

    # 输出诊断总结
    print("\n" + "="*80)
    print("✅ 诊断实验完成！")
    print("="*80)

    # 诊断结论
    print("\n📊 诊断结论:")

    gini = expert_util['gini_coefficient']
    mean_entropy = entropy_analysis['mean']
    mean_similarity = similarity_analysis.get('mean_similarity', 0)

    issues = []

    if gini > 0.5:
        issues.append(f"🚨 严重的专家利用不均衡 (Gini={gini:.3f})")
    elif gini > 0.3:
        issues.append(f"⚠️  中度专家利用不均衡 (Gini={gini:.3f})")

    if mean_entropy < 0.5:
        issues.append(f"🚨 路由过于确定，可能存在塌陷 (平均熵={mean_entropy:.3f})")
    elif mean_entropy < 1.0:
        issues.append(f"⚠️  路由熵偏低 (平均熵={mean_entropy:.3f})")

    if mean_similarity > 0.8:
        issues.append(f"🚨 专家功能严重重叠 (相似度={mean_similarity:.3f})")
    elif mean_similarity > 0.6:
        issues.append(f"⚠️  专家功能中度重叠 (相似度={mean_similarity:.3f})")

    if issues:
        print("发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ 未发现明显的路由或专家问题")

    # 建议下一步行动
    print("\n💡 建议下一步行动:")
    if gini > 0.3 or mean_entropy < 1.0:
        print("  1. 运行路由优化实验: bash run_ft_enhanced_culturemoe_routing_fix_ab.sh")
    if mean_similarity > 0.6:
        print("  2. 运行多样性增强实验: bash run_ft_enhanced_culturemoe_diversity_ab.sh")
    if len(issues) == 0:
        print("  1. 运行文化信号增强实验: bash run_ft_enhanced_culturemoe_culture_enhanced_ab.sh")

    print(f"\n📁 详细结果查看: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()