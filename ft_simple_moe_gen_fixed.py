#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版本的简单 MoE 模型训练脚本
解决了NaN/Inf数值稳定性问题

修复内容：
1. 使用修复版本的SimpleMoEModel
2. 添加梯度裁剪
3. 增强错误处理和数值检查
4. 改进学习率调度
5. 添加更详细的日志记录

使用方法：
    python ft_simple_moe_gen_fixed.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --num_experts 12 \
        --top_k 2
"""

import argparse
import json
import os
import sys
import time
import logging
import random
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 使用修复版本的SimpleMoEModel
from llamafactory.model.simple_moe_fixed import SimpleMoEModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMoEDataset(Dataset):
    """简单的数据集类 - 支持训练和评估模式"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, mode: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.data = data

        print(f"Created {mode} dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建输入文本
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')

        if self.mode == "train":
            # 训练模式：包含完整的输入和输出
            if input_text:
                full_text = f"{instruction}\n{input_text}\n{output}"
            else:
                full_text = f"{instruction}\n{output}"
        else:
            # 评估模式：只包含输入，不包含输出
            if input_text:
                full_text = f"{instruction}\n{input_text}"
            else:
                full_text = instruction

        # 分词
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': full_text,
            'instruction': instruction,
            'input': input_text,
            'output': output
        }

        if self.mode == "train":
            # 训练模式需要labels
            result['labels'] = input_ids.clone()

        return result


def collate_fn(batch):
    """数据批处理函数，支持动态padding"""
    # 获取最大长度
    max_length = max([item['input_ids'].size(0) for item in batch])

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        # Padding
        pad_length = max_length - item['input_ids'].size(0)

        if pad_length > 0:
            # 右padding
            padded_input_ids = torch.cat([
                item['input_ids'],
                torch.full((pad_length,), fill_value=0)  # 使用0作为pad token
            ])
            padded_attention_mask = torch.cat([
                item['attention_mask'],
                torch.zeros(pad_length)
            ])
        else:
            padded_input_ids = item['input_ids']
            padded_attention_mask = item['attention_mask']

        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)

        if 'labels' in item:
            if pad_length > 0:
                padded_labels = torch.cat([
                    item['labels'],
                    torch.full((pad_length,), fill_value=-100)  # 使用-100作为ignore index
                ])
            else:
                padded_labels = item['labels']
            labels.append(padded_labels)

    result = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'texts': [item['text'] for item in batch],
        'instructions': [item['instruction'] for item in batch],
        'inputs': [item['input'] for item in batch],
        'outputs': [item['output'] for item in batch]
    }

    if labels:
        result['labels'] = torch.stack(labels)

    return result


def load_and_split_data(data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """加载数据并按8:1:1划分"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机打乱
    random.shuffle(data)

    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return train_data, val_data, test_data


def safe_train_step(model, batch, optimizer, device, max_grad_norm=1.0):
    """安全的训练步骤，包含数值稳定性检查"""
    try:
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 检查输入数据
        if torch.isnan(input_ids.float()).any() or torch.isinf(input_ids.float()).any():
            logger.warning("Invalid input_ids detected, skipping batch")
            return None, "invalid_input"

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs['loss']

        # 检查损失
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss detected: {loss}, skipping batch")
            return None, "invalid_loss"

        if loss.item() > 100.0:  # 损失过大
            logger.warning(f"Loss too large: {loss.item()}, skipping batch")
            return None, "loss_too_large"

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 检查梯度
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        if torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm)):
            logger.warning("Invalid gradients detected, skipping update")
            optimizer.zero_grad()
            return None, "invalid_gradients"

        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()

        return loss.item(), "success"

    except Exception as e:
        logger.warning(f"Training step failed: {e}")
        if optimizer is not None:
            optimizer.zero_grad()
        return None, f"exception: {e}"


def train_epoch(model, train_loader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    valid_batches = 0
    skipped_batches = 0
    error_stats = {}

    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        loss_value, status = safe_train_step(model, batch, optimizer, device)

        if loss_value is not None:
            total_loss += loss_value
            valid_batches += 1
        else:
            skipped_batches += 1
            error_stats[status] = error_stats.get(status, 0) + 1

        # 更新进度条
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'avg_loss': f"{avg_loss:.4f}",
                'valid': valid_batches,
                'skipped': skipped_batches
            })

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')

    logger.info(f"Epoch {epoch}: Valid batches: {valid_batches}, Skipped: {skipped_batches}")
    if error_stats:
        logger.info(f"Error statistics: {error_stats}")

    return {
        'avg_loss': avg_loss,
        'total_loss': total_loss,
        'valid_batches': valid_batches,
        'skipped_batches': skipped_batches,
        'error_stats': error_stats
    }


def validate_model(model, val_loader, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    valid_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs['loss']

                if not (torch.isnan(loss) or torch.isinf(loss)) and loss.item() < 100.0:
                    total_loss += loss.item()
                    valid_batches += 1

            except Exception as e:
                logger.warning(f"Validation step failed: {e}")
                continue

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Fixed Simple MoE Training")

    # 模型和数据参数
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # MoE参数
    parser.add_argument("--num_experts", type=int, default=12)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--expert_hidden_dim", type=int, default=4096)
    parser.add_argument("--router_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=1e-5)  # 更小的学习率
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)  # 更小的批大小
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)  # 更严格的梯度裁剪

    # 设备参数
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(42)

    logger.info("="*80)
    logger.info("Fixed Simple MoE Training")
    logger.info("="*80)
    logger.info(f"Base model: {args.base_model_path}")
    logger.info(f"LoRA weights: {args.lora_weights_path}")
    logger.info(f"Training data: {args.train_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"MoE config: {args.num_experts} experts, Top-{args.top_k}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max grad norm: {args.max_grad_norm}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载和划分数据
    logger.info("Loading and splitting data...")
    train_data, val_data, test_data = load_and_split_data(args.train_file)

    # 创建数据集
    train_dataset = SimpleMoEDataset(train_data, tokenizer, args.max_length, mode="train")
    val_dataset = SimpleMoEDataset(val_data, tokenizer, args.max_length, mode="train")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 加载基础模型
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载LoRA权重
    logger.info("Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, args.lora_weights_path)
    model = model.merge_and_unload()

    # 创建MoE模型
    logger.info("Creating MoE model...")
    moe_model = SimpleMoEModel(
        llama_model=model,
        config=model.config,
        num_experts=args.num_experts,
        top_k=args.top_k,
        expert_hidden_dim=args.expert_hidden_dim,
        router_hidden_dim=args.router_hidden_dim,
        dropout=args.dropout
    )

    # 冻结基础模型，只训练MoE部分
    for name, param in moe_model.named_parameters():
        if 'moe_layer' in name or 'moe_fusion_weight' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 打印可训练参数
    trainable_params = sum(p.numel() for p in moe_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in moe_model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # 优化器
    optimizer = torch.optim.AdamW(
        [p for p in moe_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.1
    )

    # 训练循环
    logger.info("Starting training...")
    best_val_loss = float('inf')
    training_results = []

    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_metrics = train_epoch(moe_model, train_loader, optimizer, args.device, epoch)

        # 验证
        val_loss = validate_model(moe_model, val_loader, args.device)

        # 更新学习率
        scheduler.step()

        # 记录结果
        result = {
            'epoch': epoch,
            'train_loss': train_metrics['avg_loss'],
            'val_loss': val_loss,
            'valid_batches': train_metrics['valid_batches'],
            'skipped_batches': train_metrics['skipped_batches'],
            'error_stats': train_metrics['error_stats'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_results.append(result)

        logger.info(f"Epoch {epoch}/{args.num_epochs}:")
        logger.info(f"  Train Loss: {train_metrics['avg_loss']:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Valid Batches: {train_metrics['valid_batches']}")
        logger.info(f"  Skipped Batches: {train_metrics['skipped_batches']}")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)

            # 保存MoE权重
            torch.save(moe_model.moe_layer.state_dict(),
                      os.path.join(best_model_path, "moe_weights.pt"))
            torch.save(moe_model.moe_fusion_weight,
                      os.path.join(best_model_path, "moe_fusion_weight.pt"))

            logger.info(f"  ✅ New best model saved (val_loss: {val_loss:.4f})")

        # 定期保存
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)

            torch.save(moe_model.moe_layer.state_dict(),
                      os.path.join(checkpoint_path, "moe_weights.pt"))
            torch.save(moe_model.moe_fusion_weight,
                      os.path.join(checkpoint_path, "moe_fusion_weight.pt"))

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)

    torch.save(moe_model.moe_layer.state_dict(),
              os.path.join(final_model_path, "moe_weights.pt"))
    torch.save(moe_model.moe_fusion_weight,
              os.path.join(final_model_path, "moe_fusion_weight.pt"))

    # 保存训练结果
    with open(os.path.join(args.output_dir, "training_results.json"), 'w') as f:
        json.dump(training_results, f, indent=2)

    # 保存配置
    config = {
        'base_model_path': args.base_model_path,
        'lora_weights_path': args.lora_weights_path,
        'num_experts': args.num_experts,
        'top_k': args.top_k,
        'expert_hidden_dim': args.expert_hidden_dim,
        'router_hidden_dim': args.router_hidden_dim,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'best_val_loss': best_val_loss,
        'trainable_params': trainable_params,
        'total_params': total_params
    }

    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("="*80)
    logger.info("✅ Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()