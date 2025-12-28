#!/usr/bin/env python3
"""
CultureMoE训练脚本
基于新的CultureMoE模型架构进行训练，包含完整的验证评估功能
"""

import argparse
import json
import os
import sys
import pickle
import numpy as np
from typing import Dict, List, Optional
import logging
import re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from culture_moe_model import create_culture_moe_model, extract_answer_from_text
from culture_losses import compute_total_loss

# 复用现有的数据集类
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    dynamic_padding_collate_fn
)


def load_and_split_dataset_8_1_1(data_path: str, tokenizer, max_length: int = 512,
                                  output_dir: str = None, force_resplit: bool = False):
    """
    加载数据集并按8:1:1划分训练集/验证集/测试集

    Args:
        data_path: 数据文件路径
        tokenizer: Tokenizer
        max_length: 最大序列长度
        output_dir: 输出目录，用于保存划分索引
        force_resplit: 是否强制重新划分

    Returns:
        dict: 包含 'train', 'validation', 'test' 数据集和划分信息
    """
    # 检查是否已有划分文件
    split_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        split_file = os.path.join(output_dir, 'data_split_8_1_1.pkl')

    # 如果存在划分文件且不强制重新划分，则加载
    if split_file and os.path.exists(split_file) and not force_resplit:
        print(f"🔄 加载已有的数据划分: {split_file}")
        with open(split_file, 'rb') as f:
            split_info = pickle.load(f)

        # 验证划分文件是否匹配当前数据
        if split_info.get('data_path') == data_path and split_info.get('max_length') == max_length:
            print(f"✅ 使用已有划分: 训练{split_info['train_size']}, 验证{split_info['val_size']}, 测试{split_info['test_size']}")
        else:
            print(f"⚠️ 划分文件不匹配，重新划分...")
            force_resplit = True

    # 加载完整数据集
    full_dataset = CultureLLMNewFormatDataset(
        data_path, tokenizer, max_length,
        enable_mask=False, mask_prob=0.0
    )

    dataset_size = len(full_dataset)
    print(f"📊 数据集大小: {dataset_size} 样本")

    # 如果需要重新划分
    if force_resplit or not split_file or not os.path.exists(split_file):
        print(f"🔄 按8:1:1划分数据集...")

        # 计算划分大小
        train_size = int(dataset_size * 0.8)
        val_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size - val_size

        # 固定随机种子确保可重现
        np.random.seed(42)
        indices = np.random.permutation(dataset_size)

        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:train_size + val_size].tolist()
        test_indices = indices[train_size + val_size:].tolist()

        # 保存划分信息
        split_info = {
            'total_size': dataset_size,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'data_path': data_path,
            'max_length': max_length,
            'split_method': '8:1:1',
            'random_seed': 42
        }

        if split_file:
            with open(split_file, 'wb') as f:
                pickle.dump(split_info, f)
            print(f"✅ 保存数据划分: {split_file}")

        print(f"📊 数据划分完成: 训练{train_size}, 验证{val_size}, 测试{test_size}")

    # 创建子集
    train_dataset = Subset(full_dataset, split_info['train_indices'])
    val_dataset = Subset(full_dataset, split_info['val_indices'])
    test_dataset = Subset(full_dataset, split_info['test_indices'])

    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset,
        'full_dataset': full_dataset,
        'split_info': split_info
    }


def create_dynamic_collate_fn(tokenizer):
    """创建动态padding的collate函数"""
    def dynamic_collate_fn(batch):
        """动态padding的collate函数"""
        # 提取文化标签
        culture_labels = []
        for item in batch:
            if hasattr(item, 'get'):
                label = item.get('label', '0')
            else:
                # 处理Subset情况
                label = item.get('label', '0') if hasattr(item, 'get') else '0'

            # 将字符串标签转换为整数
            try:
                culture_labels.append(int(label))
            except:
                culture_labels.append(0)

        # 移除label字段避免冲突
        batch_data = []
        for item in batch:
            new_item = {k: v for k, v in item.items() if k != 'label'}
            batch_data.append(new_item)

        # 动态padding：找到batch中的最大长度
        max_length = 0
        for item in batch_data:
            if 'input_ids' in item:
                max_length = max(max_length, len(item['input_ids']))

        # 对每个样本进行padding或截断到batch内的最大长度
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        for item in batch_data:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']

            # 确保所有数据都是list类型
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()

            # 如果长度超过max_length，截断
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            # 如果长度小于max_length，padding
            elif len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                # 使用tokenizer的pad_token_id进行padding
                input_ids = input_ids + [pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
                labels = labels + [-100] * pad_length  # -100是忽略的标签

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        # 转换为tensor
        padded_batch = {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
            'labels': torch.tensor(labels_list, dtype=torch.long),
            'culture_labels': torch.tensor(culture_labels, dtype=torch.long)
        }

        return padded_batch

    return dynamic_collate_fn


def generate_answer(model, tokenizer, instruction: str, input_text: str, device, max_new_tokens=10):
    """生成答案"""
    # 构建输入
    if input_text.strip():
        prompt = f"{instruction}\n{input_text}\n"
    else:
        prompt = f"{instruction}\n"

    # 分词
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
    # 确保使用第一张GPU
    primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = inputs.input_ids.to(primary_device)
    attention_mask = inputs.attention_mask.to(primary_device)

    # 生成
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码生成的部分
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text.strip()


def evaluate_model(model, val_loader, device, tokenizer, config):
    """评估模型，计算损失和准确率"""
    model.eval()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # 移动数据到设备 - 确保使用第一张GPU
            primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input_ids = batch['input_ids'].to(primary_device)
            attention_mask = batch['attention_mask'].to(primary_device)
            labels = batch['labels'].to(primary_device)
            culture_labels = batch['culture_labels'].to(primary_device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 主任务损失
            main_loss = outputs.loss

            # 计算总损失
            total_loss_batch, loss_dict = compute_total_loss(
                main_loss=main_loss,
                moe_aux_info=getattr(outputs, 'moe_aux_info', []),
                culture_labels=culture_labels,
                lambda_weight=config['lambda'],
                alpha_weight=config['alpha'],
                beta_weight=config['beta'],
                use_culture_loss=config['use_culture_loss']
            )

            # 更新统计
            total_loss += total_loss_batch.item()
            total_main_loss += loss_dict['main_loss']
            total_aux_loss += loss_dict['aux_loss']
            num_batches += 1

    return {
        'eval_loss': total_loss / num_batches,
        'eval_main_loss': total_main_loss / num_batches,
        'eval_aux_loss': total_aux_loss / num_batches
    }


def generate_and_evaluate_answers(model, val_dataset, tokenizer, device, output_dir, epoch=None):
    """生成答案并评估准确率、精确率、召回率、F1分数"""
    model.eval()

    correct = 0
    total = 0
    generated_data = []
    predictions = []
    true_labels = []

    for idx in tqdm(range(len(val_dataset)), desc="Generating answers"):
        # 获取原始数据
        if hasattr(val_dataset, 'dataset'):
            original_idx = val_dataset.indices[idx]
            sample = val_dataset.dataset[original_idx]
        else:
            sample = val_dataset[idx]

        instruction = sample['instruction']
        input_text = sample['input']
        true_output = sample['output']
        label = sample['label']

        # 生成答案
        generated_text = generate_answer(
            model, tokenizer, instruction, input_text, device
        )

        # 提取答案
        predicted_answer = extract_answer_from_text(generated_text)

        # 比对答案
        is_correct = predicted_answer == true_output
        if is_correct:
            correct += 1
        total += 1

        # 收集用于计算指标的数据
        try:
            pred_int = int(predicted_answer) if predicted_answer.isdigit() else -1
            true_int = int(true_output) if true_output.isdigit() else -1
            predictions.append(pred_int)
            true_labels.append(true_int)
        except:
            predictions.append(-1)
            true_labels.append(-1)

        # 保存生成的数据
        generated_data.append({
            'instruction': instruction,
            'input': input_text,
            'true_output': true_output,
            'label': label,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': is_correct
        })

    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0

    # 计算精确率、召回率、F1分数
    # 过滤掉无效的预测和标签
    valid_indices = [i for i in range(len(predictions))
                     if predictions[i] != -1 and true_labels[i] != -1]

    if valid_indices:
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_labels = [true_labels[i] for i in valid_indices]

        # 计算指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_true_labels, valid_predictions, average='weighted', zero_division=0
        )

        # 重新计算基于有效样本的准确率
        valid_accuracy = accuracy_score(valid_true_labels, valid_predictions)
    else:
        precision = recall = f1 = valid_accuracy = 0.0

    # 保存生成的答案
    epoch_suffix = f"_epoch_{epoch}" if epoch is not None else ""
    generated_file = os.path.join(output_dir, f"generated_answers{epoch_suffix}.json")
    with open(generated_file, 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # 评估结果
    eval_results = {
        'accuracy': accuracy,
        'valid_accuracy': valid_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_samples': total,
        'correct_samples': correct,
        'valid_samples': len(valid_indices)
    }

    print(f"📊 评估结果 (epoch {epoch}):")
    print(f"  - 准确率: {accuracy:.4f} ({correct}/{total})")
    print(f"  - 有效准确率: {valid_accuracy:.4f}")
    print(f"  - 精确率: {precision:.4f}")
    print(f"  - 召回率: {recall:.4f}")
    print(f"  - F1分数: {f1:.4f}")

    return eval_results


def analyze_expert_activation_stats(moe_aux_info_list, num_layers, num_experts):
    """分析专家激活分布统计"""
    print("\n" + "="*60)
    print("📊 专家激活分布统计")
    print("="*60)

    # 收集所有层的激活统计
    layer_stats = []

    for layer_idx in range(num_layers):
        # 收集该层所有batch的激活信息
        layer_activations = []

        for batch_aux_info in moe_aux_info_list:
            if layer_idx < len(batch_aux_info):
                aux_info = batch_aux_info[layer_idx]
                if 'top_k_indices' in aux_info:
                    top_k_indices = aux_info['top_k_indices']  # [batch_size, seq_len, k]
                    # 展平所有激活的专家索引
                    activated_experts = top_k_indices.cpu().numpy().flatten()
                    layer_activations.extend(activated_experts)

        if layer_activations:
            # 统计每个专家的激活频率
            expert_counts = {}
            total_activations = len(layer_activations)

            for expert_id in range(num_experts):
                count = sum(1 for x in layer_activations if x == expert_id)
                expert_counts[expert_id] = count

            # 计算激活频率百分比
            expert_percentages = {}
            for expert_id, count in expert_counts.items():
                percentage = (count / total_activations) * 100 if total_activations > 0 else 0
                expert_percentages[expert_id] = percentage

            layer_stats.append(expert_percentages)

            # 输出该层的统计信息
            print(f"第{layer_idx+1:2d}层: ", end="")
            for expert_id in range(num_experts):
                percentage = expert_percentages[expert_id]
                print(f"专家{expert_id+1}: {percentage:5.1f}% ", end="")
            print()
        else:
            layer_stats.append({})
            print(f"第{layer_idx+1:2d}层: 无激活数据")

    # 计算全局统计
    print("\n" + "-"*60)
    print("🌍 全局专家激活统计")
    print("-"*60)

    global_expert_counts = {i: 0 for i in range(num_experts)}
    global_total = 0

    for layer_stats_dict in layer_stats:
        for expert_id, percentage in layer_stats_dict.items():
            # 这里用百分比重新计算总数（近似）
            if percentage > 0:
                global_expert_counts[expert_id] += percentage
                global_total += percentage

    if global_total > 0:
        print("平均激活频率: ", end="")
        for expert_id in range(num_experts):
            avg_percentage = (global_expert_counts[expert_id] / num_layers) if num_layers > 0 else 0
            print(f"专家{expert_id+1}: {avg_percentage:5.1f}% ", end="")
        print()

        # 找出最活跃和最不活跃的专家
        avg_percentages = [(global_expert_counts[i] / num_layers, i) for i in range(num_experts)]
        avg_percentages.sort(reverse=True)

        print(f"\n最活跃专家: 专家{avg_percentages[0][1]+1} ({avg_percentages[0][0]:.1f}%)")
        print(f"最不活跃专家: 专家{avg_percentages[-1][1]+1} ({avg_percentages[-1][0]:.1f}%)")

        # 计算负载均衡度（标准差）
        percentages = [p[0] for p in avg_percentages]
        import numpy as np
        std_dev = np.std(percentages)
        print(f"负载均衡度 (标准差): {std_dev:.2f}% (越小越均衡)")

    print("="*60)
    return layer_stats


def train_epoch(model, dataloader, optimizer, device, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0

    # 收集专家激活信息
    all_moe_aux_info = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # 移动数据到设备 - 确保使用第一张GPU
        primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_ids = batch['input_ids'].to(primary_device)
        attention_mask = batch['attention_mask'].to(primary_device)
        labels = batch['labels'].to(primary_device)
        culture_labels = batch['culture_labels'].to(primary_device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 主任务损失
        main_loss = outputs.loss

        # 收集MoE辅助信息
        batch_moe_aux_info = getattr(outputs, 'moe_aux_info', [])
        all_moe_aux_info.append(batch_moe_aux_info)

        # 计算总损失
        total_loss_batch, loss_dict = compute_total_loss(
            main_loss=main_loss,
            moe_aux_info=batch_moe_aux_info,
            culture_labels=culture_labels,
            lambda_weight=config['lambda'],
            alpha_weight=config['alpha'],
            beta_weight=config['beta'],
            use_culture_loss=config['use_culture_loss']
        )

        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 更新统计
        total_loss += total_loss_batch.item()
        total_main_loss += loss_dict['main_loss']
        total_aux_loss += loss_dict['aux_loss']
        num_batches += 1

        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f"{total_loss_batch.item():.4f}",
            'Main': f"{loss_dict['main_loss']:.4f}",
            'Aux': f"{loss_dict['aux_loss']:.4f}",
            'LB': f"{loss_dict['load_balance_loss']:.4f}",
            'Culture': f"{loss_dict['culture_loss']:.4f}"
        })

    return {
        'train_loss': total_loss / num_batches,
        'train_main_loss': total_main_loss / num_batches,
        'train_aux_loss': total_aux_loss / num_batches,
        'moe_aux_info': all_moe_aux_info
    }


def main():
    # 设置多GPU训练环境
    import torch
    import os

    # 检测GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 张GPU")

    # 设置CUDA优化
    torch.backends.cudnn.benchmark = True  # 优化cudnn性能
    if num_gpus > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使用前两张卡
        print("启用多GPU并行训练")

    parser = argparse.ArgumentParser(description="Train CultureMoE model")

    # 基本参数
    parser.add_argument("--backbone", type=str, default="llama", choices=["llama", "qwen"])
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # MoE参数
    parser.add_argument("--num_moe_experts", type=int, default=4)
    parser.add_argument("--num_activated_experts", type=int, default=2)
    parser.add_argument("--use_shared", type=str, default="true")
    parser.add_argument("--use_gate", type=str, default="true")

    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--apply_to_attention", type=str, default="false")

    # 损失函数参数
    parser.add_argument("--use_culture_loss", type=str, default="new")
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)

    # 其他参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=1)  # 每几个epoch评估一次

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"主设备: {device}")

    # 显示GPU内存信息
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({gpu_memory:.1f}GB)")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备配置
    config = {
        'backbone': args.backbone,
        'num_moe_experts': args.num_moe_experts,
        'num_activated_experts': args.num_activated_experts,
        'use_shared': args.use_shared.lower() == 'true',
        'use_gate': args.use_gate.lower() == 'true',
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'apply_to_attention': args.apply_to_attention.lower() == 'true',
        'use_culture_loss': args.use_culture_loss,
        'lambda': args.lambda_weight,
        'alpha': args.alpha,
        'beta': args.beta
    }

    print(f"Training configuration: {config}")

    # 创建模型
    print("Creating CultureMoE model...")
    model = create_culture_moe_model(args.base_model_path, config)
    # 模型已经通过device_map="auto"在GPU上，无需再次移动

    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # 加载和划分数据集
    print("Loading and splitting dataset...")
    datasets = load_and_split_dataset_8_1_1(
        args.train_file, tokenizer, args.max_length, args.output_dir
    )

    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    test_dataset = datasets['test']

    # 创建动态collate函数
    collate_fn = create_dynamic_collate_fn(tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # 保存配置信息
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 训练循环
    print(f"Starting training for {args.num_epochs} epochs...")

    best_accuracy = 0.0
    best_epoch = 0
    training_history = []
    all_eval_results = []  # 累积所有epoch的评估结果

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # 训练一个epoch
        train_stats = train_epoch(model, train_loader, optimizer, device, config)

        # 分析专家激活分布统计
        if 'moe_aux_info' in train_stats and train_stats['moe_aux_info']:
            print(f"\n🔍 Epoch {epoch + 1} 专家激活分布分析:")
            expert_stats = analyze_expert_activation_stats(
                train_stats['moe_aux_info'],
                num_layers=32 if config['backbone'] == 'llama' else 28,  # 根据backbone确定层数
                num_experts=config['num_moe_experts']
            )
            # 将专家统计信息保存到文件
            expert_stats_file = os.path.join(args.output_dir, f"expert_activation_stats_epoch_{epoch + 1}.json")
            with open(expert_stats_file, 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'layer_stats': expert_stats,
                    'config': {
                        'num_layers': 32 if config['backbone'] == 'llama' else 28,
                        'num_experts': config['num_moe_experts'],
                        'num_activated_experts': config['num_activated_experts']
                    }
                }, f, indent=2)

        # 验证评估
        eval_stats = evaluate_model(model, val_loader, device, tokenizer, config)

        # 生成答案评估（每eval_steps个epoch一次）
        if (epoch + 1) % args.eval_steps == 0 or epoch == args.num_epochs - 1:
            eval_results = generate_and_evaluate_answers(
                model, val_dataset, tokenizer, device, args.output_dir, epoch + 1
            )
            eval_stats.update(eval_results)

            # 将评估结果添加到累积列表
            eval_results_with_epoch = {'epoch': epoch + 1, **eval_results}
            all_eval_results.append(eval_results_with_epoch)

            # 检查是否是最佳模型
            if eval_results['accuracy'] > best_accuracy:
                best_accuracy = eval_results['accuracy']
                best_epoch = epoch + 1

                # 保存最佳模型（按用户要求的目录结构）
                best_model_dir = os.path.join(args.output_dir, "best_culturemoe")
                os.makedirs(best_model_dir, exist_ok=True)

                # 只保存MoE相关的可训练参数
                moe_state_dict = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and ('culture_moe_layers' in name or 'lora' in name.lower()):
                        moe_state_dict[name] = param

                # 保存MoE权重
                torch.save(moe_state_dict, os.path.join(best_model_dir, 'moe_weights.pt'))

                # 保存CultureMoE配置
                culturemoe_config = {
                    'epoch': epoch + 1,
                    'best_accuracy': best_accuracy,
                    'config': config,
                    'eval_results': eval_results,
                    'base_model_path': args.base_model_path,
                    'backbone': args.backbone
                }

                with open(os.path.join(best_model_dir, 'culturemoe_config.json'), 'w') as f:
                    json.dump(culturemoe_config, f, indent=2)

                # 保存tokenizer文件
                tokenizer.save_pretrained(best_model_dir)

                print(f"🏆 新的最佳模型! 准确率: {best_accuracy:.4f}")

        # 合并统计信息（移除moe_aux_info以节省存储空间）
        train_stats_clean = {k: v for k, v in train_stats.items() if k != 'moe_aux_info'}
        epoch_stats = {**train_stats_clean, **eval_stats, 'epoch': epoch + 1}
        training_history.append(epoch_stats)

        # 记录日志
        logging.info(f"Epoch {epoch + 1}: {epoch_stats}")



    # 保存累积的评估结果
    with open(os.path.join(args.output_dir, 'eval_results_epoch.json'), 'w') as f:
        json.dump(all_eval_results, f, indent=2)

    # 保存最终总结
    final_summary = {
        'best_epoch': best_epoch,
        'best_accuracy': best_accuracy,
        'total_epochs': args.num_epochs,
        'config': config,
        'dataset_info': {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset)
        }
    }

    with open(os.path.join(args.output_dir, 'final_summary.json'), 'w') as f:
        json.dump(final_summary, f, indent=2)

    print(f"\n🎉 训练完成!")
    print(f"📊 最佳模型: Epoch {best_epoch}, 准确率: {best_accuracy:.4f}")
    print(f"📁 输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()