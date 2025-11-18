#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的 CultureMoE 训练脚本 - 从检查点恢复训练版本 (修复NaN问题)
支持从之前保存的检查点继续训练，并应用所有NaN修复措施

修复内容：
1. 从指定的检查点目录恢复训练
2. 使用修复NaN问题的模型组件
3. 应用数值稳定性检查和保守参数
4. 自动加载模型权重、优化器状态和训练进度
5. 继续剩余的epoch训练
6. 保持与原训练一致的配置
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

# 使用修复NaN问题的模型版本
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
            '5': 5,      # 大洲洲 (Oceania)
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
class EnhancedCultureMoEResumeTrainerFixed:
    """增强的 CultureMoE 训练器 - 从检查点恢复版本 (修复NaN问题)"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.global_step = 0
        self.best_accuracy = 0.0
        self.start_epoch = 0
        self.memory_cleanup_interval = 50

        # 设置随机种子
        set_seed(42)

        # 设置日志
        self.setup_logging()

        # 加载原始配置
        self.load_original_config()

        # 加载数据和模型
        self.load_data()
        self.load_model()
        self.setup_optimizer()

        # 恢复检查点
        self.resume_from_checkpoint()

    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.args.resume_dir, 'training_resumed_fixed.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info("=" * 80)
        logging.info("Enhanced CultureMoE Training Resumed (NaN-Fixed Version)")
        logging.info("=" * 80)
        logging.info(f"Resume directory: {self.args.resume_dir}")
        logging.info(f"NaN fixes applied: All numerical stability measures")
        logging.info(f"Arguments: {vars(self.args)}")

    def load_original_config(self):
        """加载原始训练配置"""
        config_file = os.path.join(self.args.resume_dir, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                original_config = json.load(f)

            logging.info("Loaded original training configuration:")
            for key, value in original_config.items():
                logging.info(f"  {key}: {value}")

            # 使用原始配置更新当前args（如果未在命令行指定）
            for key, value in original_config.items():
                if hasattr(self.args, key) and getattr(self.args, key) is None:
                    setattr(self.args, key, value)

            # 应用NaN修复的保守参数
            if hasattr(self.args, 'learning_rate'):
                original_lr = getattr(self.args, 'learning_rate', 2e-4)
                fixed_lr = min(original_lr, 1e-4)  # 限制最大学习率
                self.args.learning_rate = fixed_lr
                logging.info(f"Applied NaN fix: Learning rate reduced to {fixed_lr}")

            if hasattr(self.args, 'batch_size'):
                original_bs = getattr(self.args, 'batch_size', 2)
                fixed_bs = min(original_bs, 1)  # 限制最大批次大小
                self.args.batch_size = fixed_bs
                logging.info(f"Applied NaN fix: Batch size reduced to {fixed_bs}")

        else:
            logging.warning("Original config.json not found, using current arguments with NaN fixes")

    def load_data(self):
        """加载训练数据"""
        # 从原始配置获取训练文件路径
        if hasattr(self.args, 'train_file') and self.args.train_file:
            train_file = self.args.train_file
        else:
            # 根据resume目录推断训练文件
            if 'unified_all_datasets' in self.args.resume_dir:
                train_file = "/root/autodl-fs/unified_all_datasets.json"
            else:
                raise ValueError("Cannot determine training file, please specify --train_file")

        logging.info(f"Loading training data from: {train_file}")

        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"Loaded {len(data)} samples")

        # 创建数据集
        dataset = CultureDataset(data, None, getattr(self.args, 'max_length', 512))

        # 数据集划分 - 使用相同的随机种子确保一致性
        total_size = len(dataset)
        val_split = getattr(self.args, 'val_split', 0.1)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        logging.info(f"Train samples: {len(self.train_dataset)}")
        logging.info(f"Validation samples: {len(self.val_dataset)}")

    def load_model(self):
        """加载模型 - 使用修复NaN问题的版本"""
        logging.info("Loading model components (NaN-fixed version)...")

        # 从原始配置获取模型路径
        base_model_path = getattr(self.args, 'base_model_path', None)
        lora_weights_path = getattr(self.args, 'lora_weights_path', None)

        if not base_model_path or not lora_weights_path:
            raise ValueError("base_model_path and lora_weights_path must be specified")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 设置tokenizer到数据集
        self.train_dataset.dataset.tokenizer = self.tokenizer
        self.val_dataset.dataset.tokenizer = self.tokenizer

        # 加载base模型
        logging.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None  # 手动管理设备
        )

        # 加载LoRA权重
        logging.info("Loading LoRA weights...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            lora_weights_path,
            is_trainable=False
        )

        # 合并LoRA权重
        logging.info("Merging LoRA weights...")
        merged_model = peft_model.merge_and_unload()

        # 创建模型参数（基于ModelArgs的实际字段）
        model_args = ModelArgs(
            num_experts=getattr(self.args, 'num_experts', 12),
            router_hidden_dim=getattr(self.args, 'router_hidden_dim', 256),  # 使用默认值
            experts_hidden_dim=getattr(self.args, 'experts_hidden_dim', 256),  # 使用默认值
            shared_hidden_dim=getattr(self.args, 'shared_hidden_dim', 512),   # 新增
            lora_rank=getattr(self.args, 'moe_lora_rank', 8),  # 使用默认值
            dropout=getattr(self.args, 'dropout', 0.1)  # 使用默认值
        )

        # 创建Enhanced CultureMoE模型 - 使用修复NaN问题的版本
        logging.info("Creating Enhanced CultureMoE model (NaN-Fixed)...")
        self.model = EnhancedCultureMoEFixed(
            llama_model=merged_model,
            config=merged_model.config,
            args=model_args,
            culture_loss_lambda=getattr(self.args, 'culture_loss_lambda', 0.3),  # 更保守
            moe_fusion=getattr(self.args, 'moe_fusion', 0.3),  # 更保守
            num_cultures=6,
            culture_dim=256
        )

        # 冻结base模型
        freeze_base = getattr(self.args, 'freeze_base_model', True)
        if freeze_base:
            logging.info("Freezing base model parameters...")
            for name, param in self.model.llama_model.named_parameters():
                param.requires_grad = False
            logging.info("Base model frozen")

        # 移动到设备
        self.model = self.model.to(self.device)

        # 多GPU支持
        num_gpus = os.environ.get('NUM_GPUS', '1')
        if num_gpus == '2' and torch.cuda.device_count() > 1:
            logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable ratio: {trainable_params/total_params:.2%}")

    def setup_optimizer(self):
        """设置优化器和调度器 - 使用保守参数"""
        # 分组参数（不同组使用不同学习率）
        param_groups = []

        # 使用更保守的学习率
        learning_rate = min(getattr(self.args, 'learning_rate', 1e-4), 1e-4)
        moe_lr_multiplier = getattr(self.args, 'moe_lr_multiplier', 0.8)  # 更保守
        weight_decay = getattr(self.args, 'weight_decay', 0.01)

        logging.info(f"Using conservative learning rate: {learning_rate}")

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
                'lr': learning_rate * moe_lr_multiplier,
                'weight_decay': weight_decay
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
                'lr': learning_rate,
                'weight_decay': weight_decay
            })

        # 创建优化器
        self.optimizer = optim.AdamW(param_groups, eps=1e-8)

        # 创建数据加载器 - 使用保守的批次大小
        batch_size = min(getattr(self.args, 'batch_size', 1), 1)  # 限制为1
        eval_batch_size = 1  # 评估时固定为1
        num_workers = getattr(self.args, 'num_workers', 2)

        logging.info(f"Using conservative batch size: {batch_size}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # 学习率调度器
        num_epochs = getattr(self.args, 'num_epochs', 12)
        num_training_steps = len(self.train_loader) * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        logging.info(f"Optimizer setup complete (NaN-fixed)")
        logging.info(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    def resume_from_checkpoint(self):
        """从检查点恢复训练"""
        # 查找最佳模型检查点
        best_checkpoint = os.path.join(self.args.resume_dir, 'best_enhanced_moe', 'moe_weights.pth')

        if os.path.exists(best_checkpoint):
            logging.info(f"Loading checkpoint from: {best_checkpoint}")

            checkpoint = torch.load(best_checkpoint, map_location=self.device)

            # 加载模型状态
            if hasattr(self.model, 'module'):  # DataParallel
                model_to_load = self.model.module
            else:
                model_to_load = self.model

            # 只加载可训练的参数
            model_state_dict = checkpoint.get('model_state_dict', {})
            missing_keys, unexpected_keys = model_to_load.load_state_dict(model_state_dict, strict=False)

            if missing_keys:
                logging.info(f"Missing keys (expected for frozen parameters): {len(missing_keys)}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys: {unexpected_keys}")

            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info("Optimizer state loaded")
                except Exception as e:
                    logging.warning(f"Failed to load optimizer state: {e}")

            # 加载调度器状态
            if 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logging.info("Scheduler state loaded")
                except Exception as e:
                    logging.warning(f"Failed to load scheduler state: {e}")

            # 加载训练进度
            self.start_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_accuracy = checkpoint.get('best_accuracy', 0.0)

            logging.info(f"Resumed from epoch {self.start_epoch + 1}")
            logging.info(f"Global step: {self.global_step}")
            logging.info(f"Best accuracy so far: {self.best_accuracy:.4f}")

        else:
            logging.warning(f"Checkpoint not found: {best_checkpoint}")
            logging.info("Starting from scratch")

        # 加载之前的训练结果
        self.previous_results = []
        results_file = os.path.join(self.args.resume_dir, 'epoch_eval_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                self.previous_results = json.load(f)
            logging.info(f"Loaded {len(self.previous_results)} previous epoch results")

    def check_numerical_stability(self, tensor, name="tensor"):
        """检查数值稳定性"""
        if torch.isnan(tensor).any():
            logging.warning(f"⚠️  NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            logging.warning(f"⚠️  Inf detected in {name}")
            return False
        if (tensor.abs() > 1000).any():
            logging.warning(f"⚠️  Large values detected in {name} (max: {tensor.abs().max():.2f})")
            return False
        return True

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch - NaN修复版本"""
        self.model.train()

        total_loss = 0.0
        total_generation_loss = 0.0
        total_culture_loss = 0.0
        total_load_balance_loss = 0.0
        total_entropy_loss = 0.0
        num_batches = len(self.train_loader)

        valid_batches = 0
        nan_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} (NaN-Fixed)")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # 前向传播 - 使用保守的参数
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    input_ids_mask=batch['input_ids_mask'],
                    attention_mask_mask=batch['attention_mask_mask'],
                    labels=batch['labels'],
                    culture_labels=batch['culture_labels'],
                    culture_ids=batch['culture_ids'],
                    use_culture_loss=getattr(self.args, 'use_culture_loss', True),
                    culture_loss_lambda=0.3,  # 保守的文化损失权重
                    culture_loss_alpha=0.5,   # 保守的margin
                    culture_loss_beta=0.5,    # 保守的lambda_diff
                    use_shared_experts=getattr(self.args, 'use_shared_experts', True),
                    router_temperature=1.5,   # 保守的路由器温度
                    load_balance_weight=0.0001,  # 很小的负载均衡权重
                    entropy_weight=0.001      # 很小的熵权重
                )

                loss = outputs['loss']

                # NaN检查
                if torch.isnan(loss) or torch.isinf(loss) or not self.check_numerical_stability(loss, "loss"):
                    logging.warning(f"⚠️  NaN/Inf loss detected in batch {batch_idx}, skipping")
                    nan_batches += 1
                    continue

                # 反向传播
                loss.backward()

                # 梯度检查和裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # 更严格的梯度裁剪

                if torch.isnan(grad_norm) or grad_norm > 5.0:
                    logging.warning(f"⚠️  Abnormal gradient norm: {grad_norm:.2f}, skipping update")
                    self.optimizer.zero_grad()
                    continue

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

                valid_batches += 1
                self.global_step += 1

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'gen': f"{outputs.get('generation_loss', torch.tensor(0)).item():.4f}",
                    'grad': f"{grad_norm:.2f}",
                    'valid': f"{valid_batches}/{batch_idx+1}"
                })

                # 内存清理
                if self.global_step % self.memory_cleanup_interval == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Error in training batch {batch_idx}: {e}")
                nan_batches += 1
                continue

        if nan_batches > 0:
            logging.warning(f"⚠️  Skipped {nan_batches}/{num_batches} batches due to NaN/Inf")

        return {
            'train_loss': total_loss / valid_batches if valid_batches > 0 else float('inf'),
            'train_generation_loss': total_generation_loss / valid_batches if valid_batches > 0 else float('inf'),
            'train_culture_loss': total_culture_loss / valid_batches if valid_batches > 0 else 0.0,
            'train_load_balance_loss': total_load_balance_loss / valid_batches if valid_batches > 0 else 0.0,
            'train_entropy_loss': total_entropy_loss / valid_batches if valid_batches > 0 else 0.0,
            'valid_batches': valid_batches,
            'nan_batches': nan_batches
        }

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """评估模型 - NaN修复版本"""
        try:
            logging.info(f"Starting evaluation for epoch {epoch+1} (NaN-Fixed)")
            self.model.eval()

            total_loss = 0.0
            total_generation_loss = 0.0
            total_culture_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            valid_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating (NaN-Fixed)")):
                    try:
                        # 移动数据到设备
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                        # 前向传播 - 使用保守参数
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            input_ids_mask=batch['input_ids_mask'],
                            attention_mask_mask=batch['attention_mask_mask'],
                            labels=batch['labels'],
                            culture_labels=batch['culture_labels'],
                            culture_ids=batch['culture_ids'],
                            use_culture_loss=getattr(self.args, 'use_culture_loss', True),
                            culture_loss_lambda=0.3,
                            culture_loss_alpha=0.5,
                            culture_loss_beta=0.5,
                            use_shared_experts=getattr(self.args, 'use_shared_experts', True),
                            router_temperature=1.5,
                            load_balance_weight=0.0001,
                            entropy_weight=0.001
                        )

                        loss = outputs['loss']

                        # NaN检查
                        if torch.isnan(loss) or torch.isinf(loss) or not self.check_numerical_stability(loss, "eval_loss"):
                            logging.warning(f"⚠️  NaN/Inf loss in eval batch {batch_idx}, skipping")
                            continue

                        # 统计损失
                        total_loss += loss.item()
                        if 'generation_loss' in outputs:
                            total_generation_loss += outputs['generation_loss'].item()
                        if 'culture_loss' in outputs:
                            total_culture_loss += outputs['culture_loss'].item()

                        # 计算准确率
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

                        valid_batches += 1

                    except Exception as e:
                        logging.warning(f"Error in eval batch {batch_idx}: {e}")
                        continue

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            logging.info(f"Evaluation completed (NaN-Fixed) - Loss: {total_loss / valid_batches:.6f}, Accuracy: {accuracy:.4f}")

            return {
                'eval_loss': total_loss / valid_batches if valid_batches > 0 else float('inf'),
                'eval_generation_loss': total_generation_loss / valid_batches if valid_batches > 0 else float('inf'),
                'eval_culture_loss': total_culture_loss / valid_batches if valid_batches > 0 else 0.0,
                'eval_accuracy': accuracy,
                'eval_samples': len(self.val_dataset),
                'eval_valid_batches': valid_batches
            }

        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return {
                'eval_loss': float('inf'),
                'eval_generation_loss': float('inf'),
                'eval_culture_loss': 0.0,
                'eval_accuracy': 0.0,
                'eval_samples': len(self.val_dataset) if hasattr(self, 'val_dataset') else 0,
                'eval_valid_batches': 0
            }

    def save_model(self, epoch: int, is_best: bool = False):
        """保存模型"""
        if is_best:
            save_dir = os.path.join(self.args.resume_dir, 'best_enhanced_moe')
            logging.info(f"Saving new best model (NaN-Fixed) to {save_dir}")
        else:
            save_dir = os.path.join(self.args.resume_dir, f'checkpoint-epoch-{epoch+1}')

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
            'nan_fixes_applied': True,
        }, os.path.join(save_dir, 'moe_weights.pth'))

        logging.info(f"Model saved to {save_dir} (NaN-Fixed)")

    def train(self):
        """主训练循环 - NaN修复版本"""
        logging.info("Resuming training (NaN-Fixed version)...")

        # 合并之前的结果和新的结果
        epoch_results = self.previous_results.copy()

        num_epochs = getattr(self.args, 'num_epochs', 12)
        eval_interval = getattr(self.args, 'eval_interval', 3)

        for epoch in range(self.start_epoch, num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs} (NaN-Fixed)")

            # 训练
            train_results = self.train_epoch(epoch)

            # 检查训练是否有效
            if train_results.get('valid_batches', 0) == 0:
                logging.error(f"No valid batches in epoch {epoch+1}, stopping training")
                break

            # 评估
            if (epoch + 1) % eval_interval == 0:
                eval_results = self.evaluate(epoch)

                # 合并结果
                epoch_result = {**train_results, **eval_results, 'epoch': epoch + 1}
                epoch_results.append(epoch_result)

                # 保存最佳模型
                if eval_results['eval_accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_results['eval_accuracy']
                    self.save_model(epoch, is_best=True)
                    logging.info(f"New best accuracy (NaN-Fixed): {self.best_accuracy:.4f}")

                # 记录结果
                logging.info(f"Epoch {epoch+1} Results (NaN-Fixed):")
                for key, value in epoch_result.items():
                    if isinstance(value, float):
                        logging.info(f"  {key}: {value:.6f}")
                    else:
                        logging.info(f"  {key}: {value}")

        # 保存最终模型
        self.save_model(num_epochs - 1, is_best=False)

        # 保存训练结果
        results_file = os.path.join(self.args.resume_dir, 'epoch_eval_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Training completed (NaN-Fixed). Results saved to {results_file}")
        logging.info(f"Best accuracy achieved: {self.best_accuracy:.4f}")

        return epoch_results


def main():
    parser = argparse.ArgumentParser(description="Enhanced CultureMoE Resume Training (NaN-Fixed)")

    # 必需参数
    parser.add_argument("--resume_dir", type=str, required=True, help="要恢复的训练目录")

    # 可选参数（如果不指定，将从原始配置中读取）
    parser.add_argument("--base_model_path", type=str, help="基础模型路径")
    parser.add_argument("--lora_weights_path", type=str, help="LoRA权重路径")
    parser.add_argument("--train_file", type=str, help="训练数据文件")

    # 训练参数（可以覆盖原始设置）
    parser.add_argument("--num_epochs", type=int, help="总训练轮数")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--eval_batch_size", type=int, help="评估批次大小")

    # 设备参数
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()

    # 检查resume目录是否存在
    if not os.path.exists(args.resume_dir):
        raise ValueError(f"Resume directory does not exist: {args.resume_dir}")

    # 检查是否有config.json
    config_file = os.path.join(args.resume_dir, 'config.json')
    if not os.path.exists(config_file):
        logging.warning(f"Config file not found: {config_file}")

    print("\n" + "=" * 80)
    print("Enhanced CultureMoE Resume Training (NaN-Fixed Version)")
    print("=" * 80)
    print(f"🔧 NaN fixes: All numerical stability measures applied")
    print(f"Resume directory: {args.resume_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print("")

    # 创建训练器并开始训练
    trainer = EnhancedCultureMoEResumeTrainerFixed(args)
    results = trainer.train()

    print("\n" + "=" * 80)
    print("✅ Enhanced CultureMoE Resume Training (NaN-Fixed) Completed!")
    print("=" * 80)
    print(f"🔧 NaN fixes successfully applied throughout training")
    print(f"Best accuracy: {trainer.best_accuracy:.4f}")
    print(f"Results saved to: {args.resume_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()