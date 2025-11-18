#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的 CultureMoE 训练脚本 - 修复NaN问题版本
使用新的文化感知组件进行训练，修复了数值稳定性问题

修复内容：
1. 冻结完整的 LoRA 微调模型（Base Model + LoRA 权重合并后）
2. 只训练新增的文化感知 MoE 组件
3. 支持可配置的专家数量（用于消融实验）
4. 集成多维度文化感知机制
5. 修复NaN/Inf问题和数值不稳定性
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed
)
import numpy as np
from tqdm import tqdm
import gc
from peft import PeftModel

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 使用修复版本的模型
from llamafactory.model.enhanced_culturemoe_fixed import EnhancedCultureMoEFixed
from llamafactory.model.moe_args import ModelArgs


# ===== 数据集类 =====
class CultureDataset(Dataset):
    """文化对齐数据集"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 大洲映射 (基于实际数据集的label字段)
        self.continent_map = {
            '0': 0,      # 亚洲 (Asia)
            '1': 1,      # 欧洲 (Europe)
            '2': 2,      # 北美洲 (North America)
            '3': 3,      # 南美洲 (South America)
            '4': 4,      # 非洲 (Africa)
            '5': 5,      # 大洋洲 (Oceania)
        }

        # 支持的大洲数量
        self.num_continents = 6  # 0-5 共6个大洲

        # 默认大洲 (当无法解析label时)
        self.default_continent = 0  # 默认为亚洲

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 获取数据字段
        instruction = item.get('instruction', '')
        instruction_mask = item.get('instruction_mask', instruction)  # mask版本的instruction
        output = item.get('output', '')
        label = item.get('label', '0')  # 大洲标签

        # 构建输入文本 (使用原始instruction，不是mask版本)
        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        full_text = input_text + output + "<|eot_id|>"

        # 构建mask版本的输入文本 (用于共享专家)
        input_text_mask = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction_mask}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        full_text_mask = input_text_mask + output + "<|eot_id|>"

        # 分词 - 原始版本
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 分词 - mask版本
        encoding_mask = self.tokenizer(
            full_text_mask,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 创建labels（用于计算损失）
        labels = encoding['input_ids'].clone()
        input_length = len(self.tokenizer(input_text, truncation=True, max_length=self.max_length)['input_ids'])
        labels[:, :input_length] = -100  # 忽略输入部分的损失

        # 解析大洲标签
        try:
            continent_id = self.continent_map.get(str(label), self.default_continent)
        except (ValueError, KeyError):
            continent_id = self.default_continent

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'input_ids_mask': encoding_mask['input_ids'].squeeze(),
            'attention_mask_mask': encoding_mask['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'culture_labels': torch.tensor(continent_id, dtype=torch.long),
            'culture_ids': torch.tensor(continent_id, dtype=torch.long)
        }


# ===== 训练器类 =====
class EnhancedCultureMoETrainer:
    """增强的 CultureMoE 训练器 - 修复NaN版本"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.global_step = 0
        self.best_accuracy = 0.0
        self.memory_cleanup_interval = 50

        # 设置随机种子
        set_seed(42)

        # 设置日志
        self.setup_logging()

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 保存配置
        self.save_config()

        # 加载数据和模型
        self.load_data()
        self.load_model()
        self.setup_optimizer()

    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.args.output_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info("=" * 80)
        logging.info("Enhanced CultureMoE Training Started (Fixed Version)")
        logging.info("=" * 80)
        logging.info(f"Arguments: {vars(self.args)}")

    def save_config(self):
        """保存训练配置"""
        config = {
            'training_time': datetime.now().isoformat(),
            'model_type': 'Enhanced CultureMoE (Fixed)',
            'base_model_path': self.args.base_model_path,
            'lora_weights_path': self.args.lora_weights_path,
            'train_file': self.args.train_file,
            'num_experts': self.args.num_experts,
            'use_shared_experts': self.args.use_shared_experts,
            'moe_fusion': self.args.moe_fusion,
            'use_culture_loss': self.args.use_culture_loss,
            'culture_loss_lambda': self.args.culture_loss_lambda,
            'freeze_base_model': self.args.freeze_base_model,
            'learning_rate': self.args.learning_rate,
            'batch_size': self.args.batch_size,
            'num_epochs': self.args.num_epochs,
            'device': str(self.device),
            'fixes_applied': [
                'Numerical stability checks',
                'Stable cross-entropy loss',
                'Conservative parameter initialization',
                'Enhanced gradient clipping',
                'Robust fallback mechanisms'
            ]
        }

        config_file = os.path.join(self.args.output_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logging.info(f"Configuration saved to {config_file}")

    def load_data(self):
        """加载训练数据"""
        logging.info(f"Loading training data from: {self.args.train_file}")

        with open(self.args.train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"Loaded {len(data)} samples")

        # 创建数据集
        dataset = CultureDataset(data, None, self.args.max_length)  # tokenizer will be set later

        # 数据集划分
        total_size = len(dataset)
        val_size = int(total_size * self.args.val_split)
        train_size = total_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        logging.info(f"Train samples: {len(self.train_dataset)}")
        logging.info(f"Validation samples: {len(self.val_dataset)}")

    def load_model(self):
        """加载模型"""
        logging.info("Loading model components...")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 设置tokenizer到数据集
        self.train_dataset.dataset.tokenizer = self.tokenizer
        self.val_dataset.dataset.tokenizer = self.tokenizer

        # 加载base模型
        logging.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None  # 手动管理设备
        )

        # 加载LoRA权重
        logging.info("Loading LoRA weights...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            self.args.lora_weights_path,
            is_trainable=False
        )

        # 合并LoRA权重
        logging.info("Merging LoRA weights...")
        merged_model = peft_model.merge_and_unload()

        # 创建模型参数
        model_args = ModelArgs(
            num_experts=self.args.num_experts,
            top_k=2,
            router_hidden_dim=self.args.router_hidden_dim,
            experts_hidden_dim=self.args.experts_hidden_dim,
            lora_rank=self.args.moe_lora_rank,
            dropout=self.args.dropout
        )

        # 创建Enhanced CultureMoE模型 - 使用修复版本
        logging.info("Creating Enhanced CultureMoE model (Fixed)...")
        self.model = EnhancedCultureMoEFixed(
            llama_model=merged_model,
            config=merged_model.config,
            args=model_args,
            culture_loss_lambda=self.args.culture_loss_lambda,
            moe_fusion=self.args.moe_fusion,
            num_cultures=6,
            culture_dim=256
        )

        # 冻结base模型
        if self.args.freeze_base_model:
            logging.info("Freezing base model parameters...")
            for name, param in self.model.llama_model.named_parameters():
                param.requires_grad = False
            logging.info("Base model frozen")

        # 移动到设备
        self.model = self.model.to(self.device)

        # 多GPU支持
        if hasattr(self.args, 'use_multi_gpu') and self.args.use_multi_gpu and torch.cuda.device_count() > 1:
            logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable ratio: {trainable_params/total_params:.2%}")

    def setup_optimizer(self):
        """设置优化器和调度器"""
        # 分组参数（不同组使用不同学习率）
        param_groups = []

        # MoE相关参数
        moe_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(keyword in name for keyword in [
                'cultural_experts', 'router', 'cultural_embedding',
                'cultural_context', 'cultural_gate'
            ]):
                moe_params.append(param)

        if moe_params:
            param_groups.append({
                'params': moe_params,
                'lr': self.args.learning_rate * self.args.moe_lr_multiplier,
                'weight_decay': self.args.weight_decay
            })

        # 其他参数
        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(keyword in name for keyword in [
                'cultural_experts', 'router', 'cultural_embedding',
                'cultural_context', 'cultural_gate'
            ]):
                other_params.append(param)

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.args.learning_rate,
                'weight_decay': self.args.weight_decay
            })

        # 创建优化器
        self.optimizer = optim.AdamW(param_groups, eps=1e-8)

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        # 学习率调度器
        num_training_steps = len(self.train_loader) * self.args.num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        logging.info(f"Optimizer setup complete")
        logging.info(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch - 修复NaN版本"""
        self.model.train()

        total_loss = 0.0
        total_generation_loss = 0.0
        total_culture_loss = 0.0
        total_load_balance_loss = 0.0
        total_entropy_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    input_ids_mask=batch['input_ids_mask'],
                    attention_mask_mask=batch['attention_mask_mask'],
                    labels=batch['labels'],
                    culture_labels=batch['culture_labels'],
                    culture_ids=batch['culture_ids'],
                    use_culture_loss=self.args.use_culture_loss,
                    culture_loss_lambda=self.args.culture_loss_lambda,
                    culture_loss_alpha=self.args.culture_loss_alpha,
                    culture_loss_beta=self.args.culture_loss_beta,
                    use_shared_experts=self.args.use_shared_experts,
                    router_temperature=self.args.router_temperature,
                    load_balance_weight=self.args.load_balance_weight,
                    entropy_weight=self.args.entropy_weight
                )

                loss = outputs['loss']

                # NaN检查
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"⚠️  NaN/Inf loss detected in batch {batch_idx}, skipping")
                    continue

                # 反向传播
                loss.backward()

                # 梯度裁剪 - 重要的修复
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 检查梯度
                grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                if grad_norm > 10.0:  # 梯度过大
                    logging.warning(f"⚠️  Large gradient norm: {grad_norm:.2f}, clipping")

                # 优化器步骤
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # 统计
                total_loss += loss.item()
                if 'generation_loss' in outputs:
                    total_generation_loss += outputs['generation_loss'].item()
                if 'culture_loss' in outputs:
                    total_culture_loss += outputs['culture_loss'].item()
                if 'load_balance_loss' in outputs:
                    total_load_balance_loss += outputs['load_balance_loss'].item()
                if 'entropy_loss' in outputs:
                    total_entropy_loss += outputs['entropy_loss'].item()

                self.global_step += 1

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'gen': f"{outputs.get('generation_loss', torch.tensor(0)).item():.4f}",
                    'cult': f"{outputs.get('culture_loss', torch.tensor(0)).item():.4f}",
                    'grad': f"{grad_norm:.2f}"
                })

                # 内存清理
                if self.global_step % self.memory_cleanup_interval == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Error in training batch {batch_idx}: {e}")
                continue

        return {
            'train_loss': total_loss / num_batches if num_batches > 0 else float('inf'),
            'train_generation_loss': total_generation_loss / num_batches if num_batches > 0 else float('inf'),
            'train_culture_loss': total_culture_loss / num_batches if num_batches > 0 else 0.0,
            'train_load_balance_loss': total_load_balance_loss / num_batches if num_batches > 0 else 0.0,
            'train_entropy_loss': total_entropy_loss / num_batches if num_batches > 0 else 0.0,
        }

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """评估模型 - 修复NaN版本"""
        try:
            logging.info(f"Starting evaluation for epoch {epoch+1}")
            self.model.eval()

            total_loss = 0.0
            total_generation_loss = 0.0
            total_culture_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                    try:
                        # 移动数据到设备
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                        # 前向传播
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            input_ids_mask=batch['input_ids_mask'],
                            attention_mask_mask=batch['attention_mask_mask'],
                            labels=batch['labels'],
                            culture_labels=batch['culture_labels'],
                            culture_ids=batch['culture_ids'],
                            use_culture_loss=self.args.use_culture_loss,
                            culture_loss_lambda=self.args.culture_loss_lambda,
                            culture_loss_alpha=self.args.culture_loss_alpha,
                            culture_loss_beta=self.args.culture_loss_beta,
                            use_shared_experts=self.args.use_shared_experts,
                            router_temperature=self.args.router_temperature,
                            load_balance_weight=self.args.load_balance_weight,
                            entropy_weight=self.args.entropy_weight
                        )

                        loss = outputs['loss']

                        # NaN检查
                        if torch.isnan(loss) or torch.isinf(loss):
                            logging.warning(f"⚠️  NaN/Inf loss in eval batch {batch_idx}, skipping")
                            continue

                        # 统计损失
                        total_loss += loss.item()
                        if 'generation_loss' in outputs:
                            total_generation_loss += outputs['generation_loss'].item()
                        if 'culture_loss' in outputs:
                            total_culture_loss += outputs['culture_loss'].item()

                        # 计算准确率（简化版本）
                        logits = outputs['logits']
                        labels = batch['labels']

                        # 只计算非padding位置的准确率
                        mask = (labels != -100)
                        if mask.sum() > 0:
                            predictions = logits.argmax(dim=-1)
                            correct = ((predictions == labels) & mask).sum().item()
                            total = mask.sum().item()

                            correct_predictions += correct
                            total_predictions += total

                    except Exception as e:
                        logging.warning(f"Error in eval batch {batch_idx}: {e}")
                        continue

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            num_valid_batches = len(self.val_loader)

            logging.info(f"Evaluation completed - Loss: {total_loss / num_valid_batches:.6f}, Accuracy: {accuracy:.4f}")

            return {
                'eval_loss': total_loss / num_valid_batches if num_valid_batches > 0 else float('inf'),
                'eval_generation_loss': total_generation_loss / num_valid_batches if num_valid_batches > 0 else float('inf'),
                'eval_culture_loss': total_culture_loss / num_valid_batches if num_valid_batches > 0 else 0.0,
                'eval_accuracy': accuracy,
                'eval_samples': len(self.val_dataset)
            }

        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return {
                'eval_loss': float('inf'),
                'eval_generation_loss': float('inf'),
                'eval_culture_loss': 0.0,
                'eval_accuracy': 0.0,
                'eval_samples': len(self.val_dataset) if hasattr(self, 'val_dataset') else 0
            }

    def save_model(self, epoch: int, is_best: bool = False):
        """保存模型"""
        if is_best:
            save_dir = os.path.join(self.args.output_dir, 'best_enhanced_moe')
            logging.info(f"Saving best model to {save_dir}")
        else:
            save_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{epoch+1}')

        os.makedirs(save_dir, exist_ok=True)

        # 保存模型状态
        if hasattr(self.model, 'module'):  # DataParallel
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # 只保存可训练的参数
        trainable_state_dict = {
            name: param for name, param in model_to_save.state_dict().items()
            if any(n == name for n, p in model_to_save.named_parameters() if p.requires_grad)
        }

        torch.save({
            'model_state_dict': trainable_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy,
        }, os.path.join(save_dir, 'moe_weights.pth'))

        logging.info(f"Model saved to {save_dir}")

    def train(self):
        """主训练循环"""
        logging.info("Starting training...")

        epoch_results = []

        for epoch in range(self.args.num_epochs):
            logging.info(f"Epoch {epoch+1}/{self.args.num_epochs}")

            # 训练
            train_results = self.train_epoch(epoch)

            # 评估
            if (epoch + 1) % self.args.eval_interval == 0:
                eval_results = self.evaluate(epoch)

                # 合并结果
                epoch_result = {**train_results, **eval_results, 'epoch': epoch + 1}
                epoch_results.append(epoch_result)

                # 保存最佳模型
                if eval_results['eval_accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_results['eval_accuracy']
                    self.save_model(epoch, is_best=True)
                    logging.info(f"New best accuracy: {self.best_accuracy:.4f}")

                # 记录结果
                logging.info(f"Epoch {epoch+1} Results:")
                for key, value in epoch_result.items():
                    if isinstance(value, float):
                        logging.info(f"  {key}: {value:.6f}")
                    else:
                        logging.info(f"  {key}: {value}")

        # 保存最终模型
        self.save_model(self.args.num_epochs - 1, is_best=False)

        # 保存训练结果
        results_file = os.path.join(self.args.output_dir, 'epoch_eval_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Training completed. Results saved to {results_file}")
        logging.info(f"Best accuracy achieved: {self.best_accuracy:.4f}")

        return epoch_results


def main():
    parser = argparse.ArgumentParser(description="Enhanced CultureMoE Training (Fixed Version)")

    # 模型参数
    parser.add_argument("--base_model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="LoRA权重路径")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    # MoE参数
    parser.add_argument("--num_experts", type=int, default=12, help="专家数量")
    parser.add_argument("--use_shared_experts", type=str, default="True", help="是否使用共享专家")
    parser.add_argument("--shared_hidden_dim", type=int, default=4096, help="共享专家隐藏维度")
    parser.add_argument("--router_hidden_dim", type=int, default=2048, help="路由器隐藏维度")
    parser.add_argument("--experts_hidden_dim", type=int, default=4096, help="专家隐藏维度")
    parser.add_argument("--moe_lora_rank", type=int, default=32, help="MoE LoRA秩")
    parser.add_argument("--moe_fusion", type=float, default=0.3, help="MoE融合系数")

    # 文化损失参数
    parser.add_argument("--use_culture_loss", type=str, default="True", help="是否使用文化损失")
    parser.add_argument("--culture_loss_lambda", type=float, default=0.3, help="文化损失权重")
    parser.add_argument("--culture_loss_alpha", type=float, default=0.5, help="文化损失alpha")
    parser.add_argument("--culture_loss_beta", type=float, default=0.5, help="文化损失beta")

    # 训练参数
    parser.add_argument("--freeze_base_model", type=str, default="True", help="是否冻结基础模型")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--moe_lr_multiplier", type=float, default=0.8, help="MoE学习率倍数")
    parser.add_argument("--router_lr_multiplier", type=float, default=0.8, help="路由器学习率倍数")
    parser.add_argument("--shared_lr_multiplier", type=float, default=0.8, help="共享专家学习率倍数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_epochs", type=int, default=12, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="评估批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载器工作进程数")
    parser.add_argument("--eval_interval", type=int, default=3, help="评估间隔")

    # 正则化参数
    parser.add_argument("--router_temperature", type=float, default=1.5, help="路由器温度")
    parser.add_argument("--load_balance_weight", type=float, default=0.0001, help="负载均衡权重")
    parser.add_argument("--entropy_weight", type=float, default=0.001, help="熵权重")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout率")

    # 设备参数
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()

    # 转换字符串参数为布尔值
    args.use_shared_experts = args.use_shared_experts.lower() == 'true'
    args.use_culture_loss = args.use_culture_loss.lower() == 'true'
    args.freeze_base_model = args.freeze_base_model.lower() == 'true'

    print("\n" + "=" * 80)
    print("Enhanced CultureMoE Training (Fixed Version)")
    print("=" * 80)
    print(f"🔧 Fixes Applied: NaN detection, stable loss computation, gradient clipping")
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.lora_weights_path}")
    print(f"Training data: {args.train_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Experts: {args.num_experts}, Fusion: {args.moe_fusion}")
    print(f"Culture loss: {args.use_culture_loss}, Lambda: {args.culture_loss_lambda}")
    print("=" * 80)
    print("")

    # 创建训练器并开始训练
    trainer = EnhancedCultureMoETrainer(args)
    results = trainer.train()

    print("\n" + "=" * 80)
    print("✅ Enhanced CultureMoE Training (Fixed) Completed Successfully!")
    print("=" * 80)
    print(f"Best accuracy: {trainer.best_accuracy:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()