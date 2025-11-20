#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiLoRA (Mixture of LoRA) 训练脚本

MiLoRA 核心特性：
1. 每个 LoRA 模块视为一个专家（7个专家：Q, K, V, O, G, U, D）
2. 提示感知路由机制：只在生成第一个新词前计算一次路由
3. 负载均衡损失 + 可学习的理性激活函数
4. Top-k 专家选择（默认 k=3）

使用方法：
    python ft_milora_gen.py \
        --base_model_path /path/to/base_model \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --num_epochs 5 \
        --batch_size 4
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed
)
from tqdm import tqdm
import gc

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.milora import MiLoRAModel


# ===== 数据集类 =====
class MiLoRADataset(Dataset):
    """MiLoRA 训练数据集"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 获取数据字段
        instruction = item.get('instruction', '')
        output = item.get('output', '')

        # 构建输入文本
        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        full_text = input_text + output + "<|eot_id|>"

        # 分词
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 创建标签（用于语言建模损失）
        labels = input_ids.clone()

        # 计算输入部分长度（不计算输入部分的损失）
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_length = input_encoding['input_ids'].shape[1]

        # 掩码输入部分的标签
        labels[:input_length] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ===== 训练器类 =====
class MiLoRATrainer:
    """MiLoRA 训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 设置随机种子
        set_seed(args.seed)

        # 初始化模型和分词器
        self._initialize_model_and_tokenizer()

        # 初始化优化器和调度器
        self._initialize_optimizer_and_scheduler()

        # 训练统计
        self.global_step = 0
        self.best_eval_loss = float('inf')

    def _initialize_model_and_tokenizer(self):
        """初始化模型和分词器"""
        logging.info("Loading tokenizer and base model...")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.base_model_path,
            trust_remote_code=True,
            padding_side='right'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path,
            torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        # 获取模型配置
        config = self.base_model.config
        hidden_dim = config.hidden_size
        intermediate_dim = getattr(config, 'intermediate_size', hidden_dim * 4)
        num_attention_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers

        logging.info(f"Model config: hidden_dim={hidden_dim}, intermediate_dim={intermediate_dim}, "
                     f"num_heads={num_attention_heads}, num_layers={num_layers}")

        # 创建 MiLoRA 模型
        self.model = MiLoRAModel(
            base_model=self.base_model,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_attention_heads=num_attention_heads,
            lora_rank=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            top_k=self.args.top_k,
            pooling_type=self.args.pooling_type,
            dropout=self.args.dropout,
            load_balance_weight=self.args.load_balance_weight
        )

        self.model.to(self.device)

        # 统计可训练参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Trainable ratio: {trainable_params/total_params:.2%}")

    def _initialize_optimizer_and_scheduler(self):
        """初始化优化器和调度器"""
        # 只优化 MiLoRA 参数
        milora_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'milora' in name:
                milora_params.append(param)

        self.optimizer = optim.AdamW(
            milora_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 学习率调度器（在加载数据后初始化）
        self.scheduler = None

    def load_data(self):
        """加载和预处理数据"""
        logging.info(f"Loading data from {self.args.train_file}")

        with open(self.args.train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"Loaded {len(data)} samples")

        # 创建数据集
        full_dataset = MiLoRADataset(data, self.tokenizer, self.args.max_length)

        # 划分训练集和验证集
        if self.args.val_split > 0:
            val_size = int(len(full_dataset) * self.args.val_split)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        else:
            train_dataset = full_dataset
            val_dataset = None

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None

        # 初始化学习率调度器
        total_steps = len(self.train_loader) * self.args.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        logging.info(f"Train samples: {len(train_dataset)}")
        if val_dataset:
            logging.info(f"Validation samples: {len(val_dataset)}")
        logging.info(f"Total training steps: {total_steps}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0.0
        total_lm_loss = 0.0
        total_lb_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 计算语言建模损失
            logits = outputs['logits']
            lm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # 获取负载均衡损失
            lb_loss = outputs['load_balance_loss']

            # 总损失
            total_loss_batch = lm_loss + lb_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_loss_batch.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            # 优化器步骤
            self.optimizer.step()
            self.scheduler.step()

            # 统计
            total_loss += total_loss_batch.item()
            total_lm_loss += lm_loss.item()
            total_lb_loss += lb_loss.item()
            self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                'LM Loss': f'{lm_loss.item():.4f}',
                'LB Loss': f'{lb_loss.item():.4f}',
                'Total': f'{total_loss_batch.item():.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

            # 内存清理
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        return {
            'train_loss': total_loss / num_batches,
            'train_lm_loss': total_lm_loss / num_batches,
            'train_lb_loss': total_lb_loss / num_batches
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_lm_loss = 0.0
        total_lb_loss = 0.0
        num_batches = len(self.val_loader)

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 计算损失
            logits = outputs['logits']
            lm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            lb_loss = outputs['load_balance_loss']
            total_loss_batch = lm_loss + lb_loss

            # 统计
            total_loss += total_loss_batch.item()
            total_lm_loss += lm_loss.item()
            total_lb_loss += lb_loss.item()

        return {
            'eval_loss': total_loss / num_batches,
            'eval_lm_loss': total_lm_loss / num_batches,
            'eval_lb_loss': total_lb_loss / num_batches
        }

    def save_model(self, save_dir: str, is_best: bool = False):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存 MiLoRA 权重
        milora_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'milora' in name:
                milora_state_dict[name] = param.half().cpu()

        milora_weights_path = os.path.join(save_dir, 'milora_weights.pth')
        torch.save(milora_state_dict, milora_weights_path)

        # 保存配置
        config = {
            'model_config': {
                'base_model_path': self.args.base_model_path,
                'lora_rank': self.args.lora_rank,
                'lora_alpha': self.args.lora_alpha,
                'top_k': self.args.top_k,
                'pooling_type': self.args.pooling_type,
                'dropout': self.args.dropout,
                'load_balance_weight': self.args.load_balance_weight
            },
            'training_args': vars(self.args),
            'model_statistics': self.model.get_model_statistics()
        }

        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 计算文件大小
        file_size_mb = os.path.getsize(milora_weights_path) / (1024 * 1024)
        param_count = sum(p.numel() for p in milora_state_dict.values())

        logging.info(f"MiLoRA weights saved to {save_dir}")
        logging.info(f"  Parameters: {param_count:,}")
        logging.info(f"  File size: {file_size_mb:.1f} MB")

        if is_best:
            logging.info("  ✅ Best model saved")

    def train(self):
        """主训练循环"""
        logging.info("Starting MiLoRA training...")

        # 加载数据
        self.load_data()

        # 训练历史
        train_history = []

        for epoch in range(self.args.num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)

            # 评估
            eval_metrics = self.evaluate()

            # 合并指标
            epoch_metrics = {
                'epoch': epoch + 1,
                'global_step': self.global_step,
                **train_metrics,
                **eval_metrics
            }

            train_history.append(epoch_metrics)

            # 日志记录
            logging.info(f"Epoch {epoch+1}/{self.args.num_epochs} completed:")
            logging.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logging.info(f"  Train LM Loss: {train_metrics['train_lm_loss']:.4f}")
            logging.info(f"  Train LB Loss: {train_metrics['train_lb_loss']:.4f}")

            if eval_metrics:
                logging.info(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}")

            # 保存检查点
            if (epoch + 1) % self.args.save_interval == 0:
                checkpoint_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{epoch+1}')
                self.save_model(checkpoint_dir)

            # 保存最佳模型
            if eval_metrics and eval_metrics['eval_loss'] < self.best_eval_loss:
                self.best_eval_loss = eval_metrics['eval_loss']
                best_model_dir = os.path.join(self.args.output_dir, 'best_milora')
                self.save_model(best_model_dir, is_best=True)

        # 保存最终模型
        final_model_dir = os.path.join(self.args.output_dir, 'final_milora')
        self.save_model(final_model_dir)

        # 保存训练历史
        history_path = os.path.join(self.args.output_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(train_history, f, indent=2, ensure_ascii=False)

        logging.info(f"Training completed! Results saved to {self.args.output_dir}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MiLoRA Training Script")

    # 模型参数
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Path to the base model')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to the training data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for saving models')

    # MiLoRA 参数
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32.0,
                        help='LoRA alpha')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top experts to select')
    parser.add_argument('--pooling_type', type=str, default='self_attention',
                        choices=['last_token', 'average', 'max', 'self_attention'],
                        help='Pooling type for router')
    parser.add_argument('--load_balance_weight', type=float, default=0.01,
                        help='Load balance loss weight')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # 数据参数
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')

    # 系统参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save checkpoint every N epochs')

    return parser.parse_args()


def setup_logging(output_dir: str):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'training.log')

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
    logging.info("MiLoRA Training Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

    # 创建训练器并开始训练
    trainer = MiLoRATrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()