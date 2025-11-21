#!/usr/bin/env python3
"""
Enhanced CultureMoE 路由优化实验脚本

目标：修复路由塌陷问题，改善专家利用率分布

关键优化策略：
1. 增强负载均衡损失权重
2. 增加路由熵正则化
3. 训练初期冻结shared专家
4. 提高路由器学习率
5. Gumbel-Softmax温度调度
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.enhanced_culturemoe_config import EnhancedCultureMoEConfig
from src.llamafactory.model.enhanced_culturemoe_adapter import create_enhanced_culturemoe_model

# 复用现有的数据集类和诊断工具
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    load_and_process_data,
    extract_answer_from_text,
    generate_answer
)

from ft_enhanced_culturemoe_diagnostic_ab import (
    DiagnosticCollector,
    analyze_expert_utilization,
    analyze_router_entropy,
    calculate_gini_coefficient
)


class RoutingOptimizer:
    """路由优化器"""

    def __init__(self, config):
        self.config = config
        self.routing_history = []
        self.temperature_schedule = []
        self.current_step = 0

    def get_gumbel_temperature(self, step):
        """获取当前步的Gumbel温度"""
        if step >= self.config.temperature_decay_steps:
            return self.config.gumbel_temperature_end

        # 线性衰减
        progress = step / self.config.temperature_decay_steps
        temp = self.config.gumbel_temperature_start * (1 - progress) + \
               self.config.gumbel_temperature_end * progress

        return temp

    def should_freeze_shared(self, epoch):
        """判断是否应该冻结shared专家"""
        return epoch < self.config.shared_freeze_epochs

    def get_learning_rates(self, base_lr, epoch):
        """获取不同组件的学习率"""
        if self.should_freeze_shared(epoch):
            return {
                'shared': 0.0,  # 冻结
                'router': self.config.router_learning_rate,
                'experts': base_lr,
                'other': base_lr
            }
        else:
            return {
                'shared': base_lr * 0.5,  # 解冻后较低学习率
                'router': self.config.router_learning_rate,
                'experts': base_lr,
                'other': base_lr
            }

    def record_routing_state(self, router_probs, step, epoch):
        """记录路由状态"""
        # 计算专家利用率
        util_stats = analyze_expert_utilization(router_probs)

        # 计算路由熵
        entropy = -(router_probs * (router_probs + 1e-12).log()).sum(dim=-1)
        entropy_stats = {
            'mean': float(entropy.mean()),
            'std': float(entropy.std()),
            'min': float(entropy.min()),
            'max': float(entropy.max())
        }

        # 记录状态
        state = {
            'step': step,
            'epoch': epoch,
            'gini_coefficient': util_stats['gini_coefficient'],
            'max_utilization': util_stats['max_utilization'],
            'min_utilization': util_stats['min_utilization'],
            'mean_entropy': entropy_stats['mean'],
            'entropy_std': entropy_stats['std'],
            'temperature': self.get_gumbel_temperature(step),
            'utilization_distribution': util_stats['mean_weights']
        }

        self.routing_history.append(state)
        self.current_step = step

    def get_optimization_summary(self):
        """获取优化总结"""
        if len(self.routing_history) < 2:
            return {'error': 'Insufficient routing history'}

        initial = self.routing_history[0]
        final = self.routing_history[-1]

        return {
            'gini_improvement': initial['gini_coefficient'] - final['gini_coefficient'],
            'entropy_improvement': final['mean_entropy'] - initial['mean_entropy'],
            'utilization_balance_improvement': initial['max_utilization'] - final['max_utilization'],
            'initial_gini': initial['gini_coefficient'],
            'final_gini': final['gini_coefficient'],
            'initial_entropy': initial['mean_entropy'],
            'final_entropy': final['mean_entropy'],
            'optimization_effective': (initial['gini_coefficient'] - final['gini_coefficient']) > 0.1
        }


def create_optimized_optimizer(model, config, epoch):
    """创建优化的优化器，支持不同组件的不同学习率"""
    optimizer_instance = RoutingOptimizer(config)
    learning_rates = optimizer_instance.get_learning_rates(config.learning_rate, epoch)

    param_groups = []

    # 分组参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'shared' in name.lower():
            lr = learning_rates['shared']
        elif 'router' in name.lower() or 'gate' in name.lower():
            lr = learning_rates['router']
        elif 'expert' in name.lower():
            lr = learning_rates['experts']
        else:
            lr = learning_rates['other']

        param_groups.append({
            'params': [param],
            'lr': lr,
            'weight_decay': config.weight_decay
        })

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer, optimizer_instance


def apply_gumbel_softmax_routing(router_logits, temperature):
    """应用Gumbel-Softmax路由"""
    # 添加Gumbel噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-12) + 1e-12)
    gumbel_logits = (router_logits + gumbel_noise) / temperature

    # Softmax
    router_probs = F.softmax(gumbel_logits, dim=-1)

    return router_probs


def train_epoch_routing_optimized(model_adapter, train_loader, optimizer, routing_optimizer, device, epoch):
    """路由优化的训练epoch"""
    model_adapter.base_model.train()

    # 设置冻结状态
    if routing_optimizer.should_freeze_shared(epoch):
        print(f"  🧊 冻结shared专家参数")
        for name, param in model_adapter.base_model.named_parameters():
            if 'shared' in name.lower():
                param.requires_grad = False
    else:
        print(f"  🔥 解冻所有参数")
        for param in model_adapter.base_model.parameters():
            param.requires_grad = True

    total_loss = 0
    total_main_loss = 0
    total_culture_loss = 0
    total_load_balance_loss = 0
    total_entropy_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Routing Opt Epoch {epoch+1}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model_adapter.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # 获取路由信息并应用优化
        if hasattr(outputs, 'router_info'):
            router_info = outputs.router_info
            router_logits = router_info.get('router_logits')

            if router_logits is not None:
                # 应用Gumbel-Softmax
                current_step = epoch * len(train_loader) + batch_idx
                temperature = routing_optimizer.get_gumbel_temperature(current_step)
                optimized_probs = apply_gumbel_softmax_routing(router_logits, temperature)

                # 记录路由状态
                if batch_idx % 50 == 0:  # 定期记录
                    routing_optimizer.record_routing_state(
                        optimized_probs.detach(),
                        current_step,
                        epoch
                    )

                # 计算增强的熵损失
                entropy = -(optimized_probs * (optimized_probs + 1e-12).log()).sum(dim=-1)
                entropy_loss = -entropy.mean() * routing_optimizer.config.entropy_weight

                # 添加到总损失
                loss = loss + entropy_loss

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if hasattr(outputs, 'loss_info'):
            loss_info = outputs.loss_info
            total_main_loss += loss_info.get('main_loss', 0)
            total_culture_loss += loss_info.get('culture_loss', 0)
            total_load_balance_loss += loss_info.get('load_balancing_loss', 0)

        num_batches += 1

        # 更新进度条
        postfix = {'loss': f"{loss.item():.4f}"}
        if routing_optimizer.routing_history:
            latest = routing_optimizer.routing_history[-1]
            postfix['gini'] = f"{latest['gini_coefficient']:.3f}"
            postfix['entropy'] = f"{latest['mean_entropy']:.3f}"
            postfix['temp'] = f"{latest['temperature']:.2f}"

        pbar.set_postfix(postfix)

    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'main_loss': total_main_loss / num_batches if num_batches > 0 else 0,
        'culture_loss': total_culture_loss / num_batches if num_batches > 0 else 0,
        'load_balance_loss': total_load_balance_loss / num_batches if num_batches > 0 else 0,
        'num_batches': num_batches
    }


def evaluate_ablation_effects(model_adapter, val_loader, device):
    """评估消融实验效果"""
    model_adapter.base_model.eval()

    results = {}

    with torch.no_grad():
        # 1. 正常模式
        normal_acc = evaluate_accuracy(model_adapter, val_loader, device, mode='normal')
        results['normal'] = normal_acc

        # 2. 无文化损失模式
        no_culture_acc = evaluate_accuracy(model_adapter, val_loader, device, mode='no_culture')
        results['no_culture'] = no_culture_acc

        # 3. 无load balance模式
        no_balance_acc = evaluate_accuracy(model_adapter, val_loader, device, mode='no_balance')
        results['no_balance'] = no_balance_acc

    # 计算差异
    results['culture_loss_effect'] = normal_acc - no_culture_acc
    results['load_balance_effect'] = normal_acc - no_balance_acc
    results['max_ablation_difference'] = max(
        abs(results['culture_loss_effect']),
        abs(results['load_balance_effect'])
    )

    return results


def evaluate_accuracy(model_adapter, val_loader, device, mode='normal'):
    """评估准确率"""
    correct = 0
    total = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 根据模式调整前向传播
        if mode == 'no_culture':
            # 临时禁用文化损失
            original_weight = model_adapter.config.culture_loss_weight
            model_adapter.config.culture_loss_weight = 0
            outputs = model_adapter.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            model_adapter.config.culture_loss_weight = original_weight
        elif mode == 'no_balance':
            # 临时禁用负载均衡
            original_weight = model_adapter.config.load_balance_weight
            model_adapter.config.load_balance_weight = 0
            outputs = model_adapter.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            model_adapter.config.load_balance_weight = original_weight
        else:
            outputs = model_adapter.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # 计算准确率（这里简化为示例）
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        # 注意：这里需要根据实际任务调整准确率计算方法
        correct += (predictions == labels).sum().item()
        total += labels.numel()

    return correct / total if total > 0 else 0


def create_routing_visualizations(routing_history, output_dir):
    """创建路由优化可视化"""
    viz_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)

    if not routing_history:
        print("⚠️  无路由历史数据，跳过可视化")
        return

    # 转换为DataFrame
    df = pd.DataFrame(routing_history)

    # 1. 路由演化图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Gini系数演化
    axes[0, 0].plot(df['step'], df['gini_coefficient'], 'b-', linewidth=2)
    axes[0, 0].set_title('Gini Coefficient Evolution')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Gini Coefficient')
    axes[0, 0].grid(True, alpha=0.3)

    # 熵演化
    axes[0, 1].plot(df['step'], df['mean_entropy'], 'g-', linewidth=2)
    axes[0, 1].set_title('Router Entropy Evolution')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Mean Entropy')
    axes[0, 1].grid(True, alpha=0.3)

    # 温度演化
    axes[1, 0].plot(df['step'], df['temperature'], 'r-', linewidth=2)
    axes[1, 0].set_title('Gumbel Temperature Schedule')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Temperature')
    axes[1, 0].grid(True, alpha=0.3)

    # 利用率平衡演化
    max_util = df['max_utilization']
    min_util = df['min_utilization']
    util_gap = max_util - min_util
    axes[1, 1].plot(df['step'], util_gap, 'm-', linewidth=2)
    axes[1, 1].set_title('Utilization Gap (Max - Min)')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Utilization Gap')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'routing_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 专家利用率分布演化
    if 'utilization_distribution' in df.columns and len(df) > 0:
        plt.figure(figsize=(12, 8))

        # 选择几个关键时间点
        n_snapshots = min(5, len(df))
        snapshot_indices = np.linspace(0, len(df)-1, n_snapshots, dtype=int)

        for i, idx in enumerate(snapshot_indices):
            utilization = df.iloc[idx]['utilization_distribution']
            step = df.iloc[idx]['step']
            plt.subplot(2, 3, i+1)
            plt.bar(range(len(utilization)), utilization)
            plt.title(f'Step {step}')
            plt.xlabel('Expert ID')
            plt.ylabel('Utilization')
            plt.ylim(0, max(0.5, max(utilization)))

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'expert_utilization_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ 路由优化可视化已保存到: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced CultureMoE Routing Fix Experiment")

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--router_learning_rate", type=float, default=4e-4)
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
    parser.add_argument("--load_balance_weight", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.1)
    parser.add_argument("--specialization_weight", type=float, default=0.1)
    parser.add_argument("--diversity_weight", type=float, default=0.05)

    # 路由优化参数
    parser.add_argument("--shared_freeze_epochs", type=int, default=1)
    parser.add_argument("--gumbel_temperature_start", type=float, default=2.0)
    parser.add_argument("--gumbel_temperature_end", type=float, default=0.5)
    parser.add_argument("--temperature_decay_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=2)

    # 实验控制参数
    parser.add_argument("--routing_fix_mode", action='store_true')
    parser.add_argument("--monitor_routing", action='store_true')
    parser.add_argument("--save_routing_evolution", action='store_true')

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print("Enhanced CultureMoE 路由优化实验")
    print("="*80)
    print(f"设备: {device}")
    print(f"基础模型: {args.base_model_path}")
    print(f"训练数据: {args.train_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"路由优化模式: {'开启' if args.routing_fix_mode else '关闭'}")
    print(f"Shared冻结轮数: {args.shared_freeze_epochs}")
    print(f"路由器学习率: {args.router_learning_rate}")
    print(f"负载均衡权重: {args.load_balance_weight}")
    print(f"熵正则权重: {args.entropy_weight}")
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
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

    # 创建优化的CultureMoE配置
    print("\nConfiguring Enhanced CultureMoE with routing optimization...")
    culturemoe_config = EnhancedCultureMoEConfig(
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_experts=args.num_experts,
        top_k=args.top_k,
        culture_loss_weight=args.culture_loss_weight,
        load_balance_weight=args.load_balance_weight,
        entropy_weight=args.entropy_weight,
        specialization_weight=args.specialization_weight,
        diversity_weight=args.diversity_weight,
        # 路由优化配置
        learning_rate=args.learning_rate,
        router_learning_rate=args.router_learning_rate,
        weight_decay=args.weight_decay,
        shared_freeze_epochs=args.shared_freeze_epochs,
        gumbel_temperature_start=args.gumbel_temperature_start,
        gumbel_temperature_end=args.gumbel_temperature_end,
        temperature_decay_steps=args.temperature_decay_steps
    )

    # 创建模型适配器
    model_adapter = create_enhanced_culturemoe_model(base_model, culturemoe_config)
    model_adapter.print_trainable_parameters()
    print("✅ Enhanced CultureMoE configured with routing optimization")

    # 训练循环
    print("\n" + "="*80)
    print("开始路由优化训练...")
    print("="*80 + "\n")

    best_gini_improvement = 0
    epoch_results = []
    routing_optimizer = None

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 创建优化器（每个epoch重新创建以应用冻结策略）
        optimizer, routing_optimizer = create_optimized_optimizer(
            model_adapter.base_model, culturemoe_config, epoch
        )

        # 训练
        train_metrics = train_epoch_routing_optimized(
            model_adapter, train_loader, optimizer, routing_optimizer, device, epoch
        )

        print(f"  Train Loss: {train_metrics['loss']:.4f}")

        # 定期评估
        if (epoch + 1) % args.eval_interval == 0:
            print("  进行消融实验评估...")
            ablation_results = evaluate_ablation_effects(model_adapter, val_loader, device)

            print(f"  消融实验结果:")
            print(f"    正常模式准确率: {ablation_results['normal']:.4f}")
            print(f"    无文化损失准确率: {ablation_results['no_culture']:.4f}")
            print(f"    无负载均衡准确率: {ablation_results['no_balance']:.4f}")
            print(f"    文化损失效果: {ablation_results['culture_loss_effect']:.4f}")
            print(f"    负载均衡效果: {ablation_results['load_balance_effect']:.4f}")

            # 记录结果
            epoch_result = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'ablation_results': ablation_results
            }

            if routing_optimizer.routing_history:
                optimization_summary = routing_optimizer.get_optimization_summary()
                epoch_result['optimization_summary'] = optimization_summary
                print(f"    Gini改善: {optimization_summary.get('gini_improvement', 0):.4f}")
                print(f"    熵提升: {optimization_summary.get('entropy_improvement', 0):.4f}")

            epoch_results.append(epoch_result)

    # 保存结果
    print("\n" + "="*80)
    print("保存路由优化结果...")
    print("="*80)

    # 1. 路由优化报告
    final_summary = routing_optimizer.get_optimization_summary() if routing_optimizer else {}
    routing_report = {
        'optimization_summary': final_summary,
        'routing_history': routing_optimizer.routing_history if routing_optimizer else [],
        'config': culturemoe_config.to_dict(),
        'final_performance': epoch_results[-1] if epoch_results else {}
    }

    with open(os.path.join(args.output_dir, 'routing_optimization_report.json'), 'w', encoding='utf-8') as f:
        json.dump(routing_report, f, indent=2, ensure_ascii=False)

    # 2. 路由演化数据
    if args.save_routing_evolution and routing_optimizer:
        with open(os.path.join(args.output_dir, 'routing_evolution.json'), 'w') as f:
            json.dump(routing_optimizer.routing_history, f, indent=2)

    # 3. 消融实验结果
    ablation_summary = {
        'epoch_results': epoch_results,
        'final_ablation': epoch_results[-1]['ablation_results'] if epoch_results else {},
        'improvement_achieved': final_summary.get('optimization_effective', False)
    }

    with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(ablation_summary, f, indent=2)

    # 4. 生成可视化
    if args.monitor_routing and routing_optimizer:
        print("生成路由优化可视化...")
        create_routing_visualizations(routing_optimizer.routing_history, args.output_dir)

    # 输出最终结果
    print("\n" + "="*80)
    print("✅ 路由优化实验完成！")
    print("="*80)

    if final_summary:
        print(f"\n📊 优化效果总结:")
        print(f"  Gini系数改善: {final_summary.get('gini_improvement', 0):.4f}")
        print(f"  路由熵提升: {final_summary.get('entropy_improvement', 0):.4f}")
        print(f"  优化是否有效: {'是' if final_summary.get('optimization_effective', False) else '否'}")

        if epoch_results:
            final_ablation = epoch_results[-1]['ablation_results']
            print(f"\n🎯 消融实验改善:")
            print(f"  最大消融差异: {final_ablation.get('max_ablation_difference', 0):.4f}")
            print(f"  文化损失效果: {final_ablation.get('culture_loss_effect', 0):.4f}")
            print(f"  负载均衡效果: {final_ablation.get('load_balance_effect', 0):.4f}")

    print(f"\n📁 详细结果查看: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()