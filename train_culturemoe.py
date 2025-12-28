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


def collate_fn(batch):
    """自定义collate函数"""
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

    # 使用动态padding
    padded_batch = dynamic_padding_collate_fn(batch_data)

    # 添加文化标签张量
    padded_batch['culture_labels'] = torch.tensor(culture_labels, dtype=torch.long)

    return padded_batch


def generate_answer(model, tokenizer, instruction: str, input_text: str, device, max_new_tokens=10):
    """生成答案"""
    # 构建输入
    if input_text.strip():
        prompt = f"{instruction}\n{input_text}\n"
    else:
        prompt = f"{instruction}\n"

    # 分词
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

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
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            culture_labels = batch['culture_labels'].to(device)

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

    # 保存评估结果
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

    eval_file = os.path.join(output_dir, f"eval_results{epoch_suffix}.json")
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"📊 评估结果 (epoch {epoch}):")
    print(f"  - 准确率: {accuracy:.4f} ({correct}/{total})")
    print(f"  - 有效准确率: {valid_accuracy:.4f}")
    print(f"  - 精确率: {precision:.4f}")
    print(f"  - 召回率: {recall:.4f}")
    print(f"  - F1分数: {f1:.4f}")

    return eval_results


def train_epoch(model, dataloader, optimizer, device, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        culture_labels = batch['culture_labels'].to(device)

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
        'train_aux_loss': total_aux_loss / num_batches
    }


def main():
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
    print(f"Using device: {device}")

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
    model.to(device)

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

    # 保存配置和数据划分信息
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.output_dir, 'dataset_info.json'), 'w') as f:
        dataset_info = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'split_ratio': '8:1:1',
            'data_path': args.train_file
        }
        json.dump(dataset_info, f, indent=2)

    # 训练循环
    print(f"Starting training for {args.num_epochs} epochs...")

    best_accuracy = 0.0
    best_epoch = 0
    training_history = []

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # 训练一个epoch
        train_stats = train_epoch(model, train_loader, optimizer, device, config)

        # 验证评估
        eval_stats = evaluate_model(model, val_loader, device, tokenizer, config)

        # 生成答案评估（每eval_steps个epoch一次）
        if (epoch + 1) % args.eval_steps == 0 or epoch == args.num_epochs - 1:
            eval_results = generate_and_evaluate_answers(
                model, val_dataset, tokenizer, device, args.output_dir, epoch + 1
            )
            eval_stats.update(eval_results)

            # 检查是否是最佳模型
            if eval_results['accuracy'] > best_accuracy:
                best_accuracy = eval_results['accuracy']
                best_epoch = epoch + 1

                # 保存最佳模型
                best_model_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'eval_results': eval_results,
                    'best_accuracy': best_accuracy
                }, os.path.join(best_model_dir, 'pytorch_model.bin'))

                print(f"🏆 新的最佳模型! 准确率: {best_accuracy:.4f}")

        # 合并统计信息
        epoch_stats = {**train_stats, **eval_stats, 'epoch': epoch + 1}
        training_history.append(epoch_stats)

        # 记录日志
        logging.info(f"Epoch {epoch + 1}: {epoch_stats}")

        # 保存检查点
        if (epoch + 1) % 2 == 0 or epoch == args.num_epochs - 1:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'training_history': training_history
            }, os.path.join(checkpoint_dir, 'pytorch_model.bin'))

            print(f"Checkpoint saved to {checkpoint_dir}")

    # 保存训练历史
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

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