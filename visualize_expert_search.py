#!/usr/bin/env python3
"""
可视化专家数搜索结果
"""

import json
import sys
import os
from pathlib import Path

def load_report(report_path):
    """加载搜索报告"""
    with open(report_path, 'r') as f:
        return json.load(f)

def print_summary(report):
    """打印结果摘要"""
    print("\n" + "="*70)
    print("CultureMoE Expert Number Search Results")
    print("="*70)
    print(f"Search Timestamp: {report['search_timestamp']}")
    print(f"Best Num Experts: {report['best_num_experts']}")
    print(f"Best Accuracy (5 epochs): {report['best_accuracy_5epochs']:.4f}")
    print("="*70)

    print("\n" + "-"*70)
    print("Search Results (All Tested Values)")
    print("-"*70)
    print(f"{'Num Experts':<15} {'Phase':<10} {'Best Accuracy':<15} {'Best Epoch':<12}")
    print("-"*70)

    for num_experts, result in sorted(report['search_results'].items(), key=lambda x: int(x[0])):
        print(f"{num_experts:<15} {result['phase']:<10} {result['best_accuracy']:<15.4f} {result['best_epoch']:<12}")

    print("-"*70)

    print("\n" + "-"*70)
    print("Final Model (10 epochs)")
    print("-"*70)
    final = report['final_model']
    print(f"Num Experts: {final['num_experts']}")
    print(f"Best Accuracy: {final['best_accuracy']:.4f}")
    print(f"Best Epoch: {final['best_epoch']}")
    print("-"*70)

    print("\n" + "="*70)
    print(f"Output Directory: {report['output_directory']}")
    print("="*70)

def plot_results(report, output_file='expert_search_plot.png'):
    """绘制准确率 vs 专家数曲线"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n⚠️  matplotlib not installed. Skipping plot generation.")
        print("   Install with: pip install matplotlib")
        return

    # 提取数据
    experts = []
    accuracies = []
    phases = []

    for num_experts, result in sorted(report['search_results'].items(), key=lambda x: int(x[0])):
        experts.append(int(num_experts))
        accuracies.append(result['best_accuracy'])
        phases.append(result['phase'])

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))

    # 分别绘制粗粒度和细粒度搜索的点
    coarse_experts = [e for e, p in zip(experts, phases) if p == 'coarse']
    coarse_acc = [a for a, p in zip(accuracies, phases) if p == 'coarse']
    fine_experts = [e for e, p in zip(experts, phases) if p == 'fine']
    fine_acc = [a for a, p in zip(accuracies, phases) if p == 'fine']

    # 绘制曲线
    ax.plot(experts, accuracies, 'b-', linewidth=2, alpha=0.6, label='Search Path')

    # 绘制点
    if coarse_experts:
        ax.scatter(coarse_experts, coarse_acc, s=150, c='blue', marker='o',
                  label='Coarse Search', zorder=5, edgecolors='black', linewidths=1.5)
    if fine_experts:
        ax.scatter(fine_experts, fine_acc, s=150, c='green', marker='s',
                  label='Fine Search', zorder=5, edgecolors='black', linewidths=1.5)

    # 标注最佳点
    best_idx = accuracies.index(max(accuracies))
    best_expert = experts[best_idx]
    best_acc = accuracies[best_idx]

    ax.scatter(best_expert, best_acc, s=300, c='red', marker='*',
              label=f'Best: {best_expert} experts', zorder=10,
              edgecolors='darkred', linewidths=2)

    # 添加最佳点标注
    ax.annotate(f'{best_acc:.4f}',
                xy=(best_expert, best_acc),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

    # 设置标签和标题
    ax.set_xlabel('Number of Experts', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Accuracy (5 epochs)', fontsize=14, fontweight='bold')
    ax.set_title('CultureMoE: Accuracy vs Number of Experts',
                fontsize=16, fontweight='bold', pad=20)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 设置 x 轴刻度
    ax.set_xticks(experts)

    # 设置 y 轴范围
    y_min = min(accuracies) - 0.02
    y_max = max(accuracies) + 0.02
    ax.set_ylim(y_min, y_max)

    # 添加图例
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")

    # 显示图表（如果在交互环境中）
    # plt.show()

def plot_epoch_curves(report, output_file='epoch_curves.png'):
    """绘制不同专家数的训练曲线"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir = report['output_directory']

    # 收集所有专家数的 epoch 数据
    epoch_data = {}

    for num_experts in sorted([int(k) for k in report['search_results'].keys()]):
        # 尝试读取 epoch_eval_results.json
        for phase in ['phase1', 'phase2']:
            epoch_file = os.path.join(output_dir, f"{phase}_experts_{num_experts}", "epoch_eval_results.json")
            if os.path.exists(epoch_file):
                with open(epoch_file, 'r') as f:
                    results = json.load(f)
                epoch_data[num_experts] = results
                break

    if not epoch_data:
        print("\n⚠️  No epoch data found for plotting curves.")
        return

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制准确率曲线
    for num_experts, results in sorted(epoch_data.items()):
        epochs = [r['epoch'] for r in results]
        accuracies = [r['eval_accuracy'] for r in results]
        ax1.plot(epochs, accuracies, marker='o', label=f'{num_experts} experts', linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Epoch (Different Expert Numbers)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 绘制损失曲线
    for num_experts, results in sorted(epoch_data.items()):
        epochs = [r['epoch'] for r in results]
        losses = [r['eval_loss'] for r in results]
        ax2.plot(epochs, losses, marker='o', label=f'{num_experts} experts', linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Loss vs Epoch (Different Expert Numbers)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Epoch curves saved to: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_expert_search.py <final_report.json>")
        print("\nExample:")
        print("  python visualize_expert_search.py /root/autodl-fs/output/expert_search_*/final_report.json")
        sys.exit(1)

    report_path = sys.argv[1]

    if not os.path.exists(report_path):
        print(f"❌ Error: Report file not found: {report_path}")
        sys.exit(1)

    # 加载报告
    report = load_report(report_path)

    # 打印摘要
    print_summary(report)

    # 生成图表
    output_dir = os.path.dirname(report_path)
    plot_file = os.path.join(output_dir, 'expert_search_plot.png')
    epoch_file = os.path.join(output_dir, 'epoch_curves.png')

    plot_results(report, plot_file)
    plot_epoch_curves(report, epoch_file)

    print("\n" + "="*70)
    print("Visualization completed!")
    print("="*70)

if __name__ == "__main__":
    main()

