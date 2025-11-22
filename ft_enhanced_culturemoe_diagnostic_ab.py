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


def save_epoch_diagnostic_results(collector, output_dir, epoch):
    """保存单个epoch的诊断结果"""
    try:
        # 分析当前收集的数据
        if not collector.router_probs_list:
            print(f"⚠️  Epoch {epoch}: 没有收集到路由数据")
            return

        # 计算基础统计
        all_router_probs = np.concatenate(collector.router_probs_list, axis=0)  # [N, num_experts]
        expert_usage = np.mean(all_router_probs, axis=0)  # [num_experts]

        # 路由熵分析
        router_entropies = []
        for probs in all_router_probs:
            # 避免log(0)
            probs_safe = np.clip(probs, 1e-8, 1.0)
            entropy = -np.sum(probs_safe * np.log(probs_safe))
            router_entropies.append(entropy)

        # 专家相似度分析（如果有专家输出）
        expert_similarity = None
        if collector.expert_outputs_list:
            try:
                all_expert_outputs = np.concatenate(collector.expert_outputs_list, axis=0)
                expert_similarity = cosine_similarity(all_expert_outputs.T).tolist()
            except:
                expert_similarity = None

        # 文化路由分析
        cultural_routing = {}
        if collector.cultural_labels_list:
            try:
                all_cultural_labels = np.concatenate(collector.cultural_labels_list, axis=0)
                unique_cultures = np.unique(all_cultural_labels)

                for culture in unique_cultures:
                    culture_mask = all_cultural_labels == culture
                    culture_router_probs = all_router_probs[culture_mask]
                    cultural_routing[f'culture_{int(culture)}'] = {
                        'expert_usage': np.mean(culture_router_probs, axis=0).tolist(),
                        'sample_count': int(np.sum(culture_mask))
                    }
            except:
                cultural_routing = {}

        # 构建epoch诊断报告
        epoch_diagnostic = {
            'epoch': epoch,
            'timestamp': pd.Timestamp.now().isoformat(),
            'expert_utilization': {
                'expert_usage': expert_usage.tolist(),
                'gini_coefficient': float(compute_gini_coefficient(expert_usage)),
                'max_usage': float(np.max(expert_usage)),
                'min_usage': float(np.min(expert_usage)),
                'usage_std': float(np.std(expert_usage))
            },
            'router_entropy': {
                'mean': float(np.mean(router_entropies)),
                'std': float(np.std(router_entropies)),
                'min': float(np.min(router_entropies)),
                'max': float(np.max(router_entropies))
            },
            'expert_similarity': expert_similarity,
            'cultural_routing': cultural_routing,
            'total_samples': len(all_router_probs)
        }

        # 保存epoch结果
        epoch_file = os.path.join(output_dir, f'diagnostic_epoch_{epoch}.json')
        with open(epoch_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_diagnostic, f, indent=2, ensure_ascii=False)

        # 保存简化的可读报告
        summary_file = os.path.join(output_dir, f'diagnostic_summary_epoch_{epoch}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Epoch {epoch} 诊断报告 ===\n\n")
            f.write(f"📊 专家利用率分析:\n")
            f.write(f"  - Gini系数: {epoch_diagnostic['expert_utilization']['gini_coefficient']:.4f}\n")
            f.write(f"  - 最大使用率: {epoch_diagnostic['expert_utilization']['max_usage']:.4f}\n")
            f.write(f"  - 最小使用率: {epoch_diagnostic['expert_utilization']['min_usage']:.4f}\n")
            f.write(f"  - 使用率标准差: {epoch_diagnostic['expert_utilization']['usage_std']:.4f}\n\n")

            f.write(f"🎯 路由熵分析:\n")
            f.write(f"  - 平均熵: {epoch_diagnostic['router_entropy']['mean']:.4f}\n")
            f.write(f"  - 熵标准差: {epoch_diagnostic['router_entropy']['std']:.4f}\n\n")

            f.write(f"🌍 文化路由分析:\n")
            for culture, data in cultural_routing.items():
                f.write(f"  - {culture}: {data['sample_count']} samples\n")

            f.write(f"\n📈 诊断建议:\n")
            gini = epoch_diagnostic['expert_utilization']['gini_coefficient']
            avg_entropy = epoch_diagnostic['router_entropy']['mean']

            if gini > 0.5:
                f.write(f"  ⚠️  专家利用不均衡 (Gini={gini:.3f} > 0.5)\n")
            if avg_entropy < 0.5:
                f.write(f"  ⚠️  路由过于确定，可能存在塌陷 (熵={avg_entropy:.3f} < 0.5)\n")
            if gini <= 0.3 and avg_entropy >= 1.0:
                f.write(f"  ✅ 专家利用均衡且路由多样性良好\n")

        print(f"✅ Epoch {epoch} 诊断结果已保存:")
        print(f"   - 详细数据: {epoch_file}")
        print(f"   - 可读报告: {summary_file}")

        # 打印关键指标
        print(f"📊 Epoch {epoch} 关键指标:")
        print(f"   - Gini系数: {epoch_diagnostic['expert_utilization']['gini_coefficient']:.4f}")
        print(f"   - 平均路由熵: {epoch_diagnostic['router_entropy']['mean']:.4f}")
        print(f"   - 样本数量: {epoch_diagnostic['total_samples']}")

    except Exception as e:
        print(f"❌ Epoch {epoch} 诊断保存失败: {e}")


def compute_gini_coefficient(values):
    """计算基尼系数"""
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def train_epoch_diagnostic(model_adapter, train_loader, optimizer, device, collector, scaler=None, gradient_accumulation_steps=8):
    """诊断模式的训练epoch - 完整架构 + 梯度累积 + 混合精度"""
    model_adapter.train()
    total_loss = 0
    num_batches = 0
    accumulation_loss = 0

    pbar = tqdm(train_loader, desc="Diagnostic Training (Full Architecture + Grad Accumulation)")

    for batch_idx, batch in enumerate(pbar):
        # 每隔10个batch清理一次内存
        if batch_idx % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # 前向传播 (混合精度)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model_adapter(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
        else:
            outputs = model_adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']

        # 梯度累积：除以累积步数
        loss = loss / gradient_accumulation_steps
        accumulation_loss += loss.item()

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

        # 反向传播 (混合精度 + 梯度累积)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积：每accumulation_steps步更新一次
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            # 记录累积损失
            total_loss += accumulation_loss
            num_batches += 1
            accumulation_loss = 0

        # 立即清理中间变量
        del input_ids, attention_mask, labels, outputs

        # 每个batch后清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 更新进度条信息 (包含内存使用和累积信息)
        postfix_info = {
            'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",  # 显示真实损失
            'acc_loss': f"{accumulation_loss:.4f}",  # 显示累积损失
            'step': f"{(batch_idx + 1) % gradient_accumulation_steps}/{gradient_accumulation_steps}"
        }
        if device.type == 'cuda':
            try:
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                postfix_info['mem_gb'] = f"{memory_allocated:.1f}/{memory_reserved:.1f}"
            except:
                postfix_info['mem_gb'] = "N/A"

        pbar.set_postfix(postfix_info)

    # 处理剩余的梯度累积步骤
    if accumulation_loss > 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulation_loss
        num_batches += 1

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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")

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

    # 设置设备 (强制单卡运行)
    if torch.cuda.is_available():
        # 清理GPU内存
        torch.cuda.empty_cache()

        # 只使用第一个GPU，避免多卡问题
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

        # 显示GPU信息
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"🚀 强制使用单卡: {device}")
        print(f"   GPU: {gpu_name}")
        print(f"   总内存: {gpu_memory:.1f} GB")
    else:
        device = torch.device("cpu")
        print("🖥️  使用CPU设备")

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
        num_workers=0,  # 减少内存使用: 2 -> 0
        pin_memory=False,  # 禁用pin_memory以节省内存
        drop_last=True  # 丢弃最后不完整的batch
    )

    # 加载基础模型 (内存优化配置)
    print(f"\nLoading base model to {device} with memory optimization...")

    # 内存优化配置
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,  # 使用半精度
        device_map=None,  # 禁用自动设备映射
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if hasattr(torch.nn, 'scaled_dot_product_attention') else None
    )

    # 确保基础模型在单一设备上
    base_model = base_model.to(device)

    # 冻结基础模型参数以节省内存
    for param in base_model.parameters():
        param.requires_grad = False

    # 启用梯度检查点以节省内存
    base_model.gradient_checkpointing_enable()

    # 验证模型设备
    model_device = next(base_model.parameters()).device

    # 检查内存使用
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"✅ Base model loaded on {model_device}")
        print(f"   Memory allocated: {memory_allocated:.2f} GB")
        print(f"   Memory reserved: {memory_reserved:.2f} GB")
    else:
        print(f"✅ Base model loaded on {model_device}")

    # 创建CultureMoE配置 (完整架构 - 与生产脚本一致)
    print("\nConfiguring Enhanced CultureMoE with FULL architecture...")
    model_args = ModelArgs(
        llama_model_path=args.base_model_path,
        num_experts=12,  # 恢复完整专家数量 (与生产一致)
        top_k=2,  # 保持top_k=2
        lora_rank=32,  # 恢复完整LoRA rank (与生产一致)
        lora_alpha=16,  # 恢复完整alpha (与生产一致)
        lora_dropout=args.lora_dropout,
        # 恢复完整架构参数 (与生产脚本一致)
        experts_hidden_dim=4096,  # 恢复完整维度 (与生产一致)
        router_hidden_dim=2048,   # 恢复完整维度 (与生产一致)
        shared_hidden_dim=4096,   # 恢复完整维度 (与生产一致)
        num_heads=8,              # 恢复完整头数 (与生产一致)
        classification_hidden_dim=256,  # 保持合理大小
        num_classes=2
    )

    print("🏭 完整生产架构配置:")
    print(f"  - 专家数量: {model_args.num_experts} (与生产一致)")
    print(f"  - LoRA rank: {model_args.lora_rank} (与生产一致)")
    print(f"  - 专家隐藏维度: {model_args.experts_hidden_dim} (与生产一致)")
    print(f"  - 路由器隐藏维度: {model_args.router_hidden_dim} (与生产一致)")
    print(f"  - 共享隐藏维度: {model_args.shared_hidden_dim} (与生产一致)")

    # 创建模型适配器 (完整架构 - 与生产一致)
    config = base_model.config  # 获取模型配置
    model_adapter = EnhancedCultureMoE(
        llama_model=base_model,
        config=config,
        args=model_args,
        culture_loss_lambda=args.culture_loss_weight,
        moe_fusion=0.4,  # 恢复生产融合权重 (与生产一致)
        num_cultures=6,  # 恢复完整文化数量 (与生产一致)
        culture_dim=256,  # 恢复完整文化维度 (与生产一致)
        use_gate=True    # 恢复门控机制 (与生产一致)
    )

    # 确保模型适配器在正确的设备上 (单卡运行)
    model_adapter = model_adapter.to(device)

    # 验证所有组件都在同一设备上
    adapter_device = next(model_adapter.parameters()).device
    llama_device = next(model_adapter.llama_model.parameters()).device

    print(f"✅ Enhanced CultureMoE configured:")
    print(f"  - Adapter device: {adapter_device}")
    print(f"  - LLaMA device: {llama_device}")

    if adapter_device != llama_device:
        print(f"⚠️  Warning: Device mismatch detected! Moving all to {device}")
        model_adapter = model_adapter.to(device)

    model_adapter.print_trainable_parameters()

    # 优化器
    optimizer = torch.optim.AdamW(
        model_adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 混合精度训练 - 关键的内存优化
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    print(f"✅ Mixed precision training: {'Enabled' if scaler else 'Disabled'}")

    # 创建诊断收集器
    collector = DiagnosticCollector()

    # 训练循环（诊断模式）
    print("\n" + "="*80)
    print("开始诊断训练...")
    print("="*80 + "\n")

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 训练 (完整架构 + 梯度累积 + 混合精度)
        train_metrics = train_epoch_diagnostic(
            model_adapter, train_loader, optimizer, device, collector,
            scaler=scaler, gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        print(f"  Train Loss: {train_metrics['loss']:.4f}")

        # 每个epoch后保存中间诊断结果
        print(f"\n📊 保存 Epoch {epoch + 1} 诊断结果...")
        save_epoch_diagnostic_results(collector, args.output_dir, epoch + 1)

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