#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的 CultureMoE 训练脚本
使用新的文化感知组件进行训练

特点：
1. 冻结完整的 LoRA 微调模型（Base Model + LoRA 权重合并后）
2. 只训练新增的文化感知 MoE 组件
3. 支持可配置的专家数量（用于消融实验）
4. 集成多维度文化感知机制
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

from llamafactory.model.enhanced_culturemoe import EnhancedCultureMoE
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

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        input_ids_mask = encoding_mask['input_ids'].squeeze(0)
        attention_mask_mask = encoding_mask['attention_mask'].squeeze(0)

        # 创建标签
        labels = input_ids.clone()

        # 找到assistant响应开始位置
        assistant_start = input_text
        assistant_encoding = self.tokenizer(assistant_start, add_special_tokens=False)
        assistant_start_idx = len(assistant_encoding['input_ids'])

        # 掩盖instruction部分
        labels[:assistant_start_idx] = -100

        # 解析大洲标签
        continent_id, continent_ids_multi = self._parse_continent_label(label)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'input_ids_mask': input_ids_mask,
            'attention_mask_mask': attention_mask_mask,
            'labels': labels,
            'culture_ids': torch.tensor(continent_id, dtype=torch.long),
            'culture_ids_multi': continent_ids_multi  # 保留多标签大洲信息
        }

    def _parse_continent_label(self, label: str) -> tuple[int, List[int]]:
        """解析大洲标签，返回主要大洲ID和所有相关大洲ID列表"""
        # 清理标签
        label = str(label).strip()

        continent_ids = []

        # 处理多大洲情况 (例如 "0,1" 或 "1,0")
        if ',' in label:
            continents = [c.strip() for c in label.split(',')]
            for continent in continents:
                if continent in self.continent_map:
                    continent_ids.append(self.continent_map[continent])
        else:
            # 单个大洲
            if label in self.continent_map:
                continent_ids.append(self.continent_map[label])

        # 如果没有找到任何有效的大洲，使用默认值
        if not continent_ids:
            continent_ids.append(self.default_continent)

        # 主要大洲ID (用于模型输入，取第一个)
        primary_continent_id = continent_ids[0]

        return primary_continent_id, continent_ids


# ===== 分层优化器 =====
class LayeredOptimizer:
    """分层优化器：为不同组件使用不同学习率"""

    def __init__(self, model, base_lr: float = 2e-4,
                 moe_lr_multiplier: float = 1.0,
                 router_lr_multiplier: float = 1.0,
                 shared_lr_multiplier: float = 1.0,
                 weight_decay: float = 0.01):

        self.base_lr = base_lr
        self.moe_lr_multiplier = moe_lr_multiplier
        self.router_lr_multiplier = router_lr_multiplier
        self.shared_lr_multiplier = shared_lr_multiplier

        # 分组参数
        moe_params = []
        router_params = []
        shared_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # DataParallel兼容性：移除module.前缀
            clean_name = name.replace('module.', '')

            if 'cultural_experts' in clean_name or 'culture' in clean_name:
                moe_params.append(param)
            elif 'router' in clean_name:
                router_params.append(param)
            elif 'shared' in clean_name:
                shared_params.append(param)
            else:
                moe_params.append(param)  # 默认归入MoE组

        # 创建参数组
        param_groups = []
        if moe_params:
            param_groups.append({
                'params': moe_params,
                'lr': base_lr * moe_lr_multiplier,
                'weight_decay': weight_decay
            })
        if router_params:
            param_groups.append({
                'params': router_params,
                'lr': base_lr * router_lr_multiplier,
                'weight_decay': weight_decay
            })
        if shared_params:
            param_groups.append({
                'params': shared_params,
                'lr': base_lr * shared_lr_multiplier,
                'weight_decay': weight_decay
            })

        self.optimizer = optim.AdamW(param_groups)

        logging.info(f"LayeredOptimizer initialized:")
        logging.info(f"  MoE params: {len(moe_params)} (lr={base_lr * moe_lr_multiplier:.2e})")
        logging.info(f"  Router params: {len(router_params)} (lr={base_lr * router_lr_multiplier:.2e})")
        logging.info(f"  Shared params: {len(shared_params)} (lr={base_lr * shared_lr_multiplier:.2e})")

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# ===== 训练器 =====
class EnhancedCultureMoETrainer:
    """增强的 CultureMoE 训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # 设置随机种子
        set_seed(42)

        # 设置日志
        self.setup_logging()

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 保存配置
        self.save_config()

        # 初始化模型
        self.setup_model()

        # 加载数据
        self.setup_data()

        # 设置优化器和调度器
        self.setup_optimizer()

        # 训练状态
        self.global_step = 0
        self.best_accuracy = 0.0
        self.epoch_results = []

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

        logging.info("=" * 60)
        logging.info("Enhanced CultureMoE Training Started")
        logging.info("=" * 60)
        logging.info(f"Arguments: {vars(self.args)}")

    def save_config(self):
        """保存训练配置"""
        config = vars(self.args)
        config_file = os.path.join(self.args.output_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"Config saved to {config_file}")

    def setup_model(self):
        """设置模型"""
        logging.info("Loading base model...")

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True
        )

        # 加载LoRA权重
        if self.args.lora_weights_path:
            logging.info(f"Loading LoRA weights from {self.args.lora_weights_path}")
            base_model = PeftModel.from_pretrained(base_model, self.args.lora_weights_path)
            # 合并LoRA权重
            base_model = base_model.merge_and_unload()
            logging.info("LoRA weights merged successfully")

        # 创建MoE参数
        moe_args = ModelArgs(
            num_experts=self.args.num_experts,
            shared_hidden_dim=self.args.shared_hidden_dim,
            router_hidden_dim=self.args.router_hidden_dim,
            experts_hidden_dim=self.args.experts_hidden_dim,
            lora_rank=self.args.moe_lora_rank,
            dropout=self.args.dropout
        )

        # 创建增强的CultureMoE模型
        self.model = EnhancedCultureMoE(
            llama_model=base_model,
            config=base_model.config,
            args=moe_args,
            culture_loss_lambda=self.args.culture_loss_lambda,
            moe_fusion=self.args.moe_fusion,
            num_cultures=6,   # 支持6个大洲 (0-5: 亚洲、欧洲、北美、南美、非洲、大洋洲)
            culture_dim=256   # 文化嵌入维度
        )

        # 冻结基础模型参数
        if self.args.freeze_base_model:
            self.freeze_base_model()

        # 移动到设备
        self.model = self.model.to(self.device)

        # 多GPU支持
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            logging.info(f"Using {gpu_count} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
            self.use_dataparallel = True
        else:
            logging.info("Using single GPU for training")
            self.use_dataparallel = False

        # 计算参数统计
        self.log_model_info()

    def freeze_base_model(self):
        """冻结基础模型参数"""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            # 判断是否为MoE相关参数
            is_moe_param = any(keyword in name for keyword in [
                'cultural_experts', 'router', 'shared', 'cultural_gate',
                'cultural_embedding', 'cultural_context', 'culture_loss'
            ])

            if is_moe_param:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        logging.info(f"Model parameters frozen: {frozen_count:,} ({frozen_count/(frozen_count+trainable_count)*100:.1f}%)")
        logging.info(f"Model parameters trainable: {trainable_count:,} ({trainable_count/(frozen_count+trainable_count)*100:.1f}%)")

    def log_model_info(self):
        """记录模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logging.info(f"Model loaded successfully")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

        # 记录文化专家信息
        # DataParallel兼容性：访问实际模型
        actual_model = self.model.module if self.use_dataparallel else self.model
        expert_info = actual_model.get_culture_expert_info()
        logging.info("Culture Expert Assignments:")
        for info in expert_info:
            logging.info(f"  Expert {info['expert_id']}: {info['role']} ({info['param_count']:,} params)")

    def setup_data(self):
        """设置数据"""
        logging.info(f"Loading data from {self.args.train_file}")

        with open(self.args.train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"Loaded {len(data)} samples")

        # 创建数据集
        dataset = CultureDataset(data, self.tokenizer, self.args.max_length)

        # 划分训练和验证集
        val_size = int(len(dataset) * self.args.val_split)
        train_size = len(dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # 创建数据加载器 - 先用简单的随机采样
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

        logging.info(f"Train samples: {len(self.train_dataset)}")
        logging.info(f"Validation samples: {len(self.val_dataset)}")

    def setup_optimizer(self):
        """设置优化器和调度器"""
        # 使用分层优化器
        self.optimizer = LayeredOptimizer(
            self.model,
            base_lr=self.args.learning_rate,
            moe_lr_multiplier=self.args.moe_lr_multiplier,
            router_lr_multiplier=self.args.router_lr_multiplier,
            shared_lr_multiplier=self.args.shared_lr_multiplier,
            weight_decay=self.args.weight_decay
        )

        # 学习率调度器
        num_training_steps = len(self.train_loader) * self.args.num_epochs
        num_warmup_steps = int(num_training_steps * 0.1)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        logging.info(f"Optimizer setup completed")
        logging.info(f"Total training steps: {num_training_steps}")
        logging.info(f"Warmup steps: {num_warmup_steps}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        total_generation_loss = 0.0
        total_culture_loss = 0.0
        total_load_balance_loss = 0.0
        total_entropy_loss = 0.0
        total_specialization_loss = 0.0
        total_diversity_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            # 移动数据到设备，特殊处理culture_ids_multi
            batch_device = {}
            for k, v in batch.items():
                if k == 'culture_ids_multi':
                    # culture_ids_multi是列表，不能直接.to(device)
                    batch_device[k] = v
                else:
                    batch_device[k] = v.to(self.device)
            batch = batch_device

            # 前向传播
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                input_ids_mask=batch['input_ids_mask'],
                attention_mask_mask=batch['attention_mask_mask'],
                labels=batch['labels'],
                culture_labels=batch['culture_ids'],  # 传递给文化损失计算
                culture_ids=batch['culture_ids'],     # 传递给文化感知组件
                culture_ids_multi=batch['culture_ids_multi'],
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

            # 调试：检查文化损失计算（仅前3步）
            if self.global_step <= 2:
                unique_cultures = torch.unique(batch['culture_ids'])
                culture_loss_val = outputs.get('culture_loss', torch.tensor(0)).item() if 'culture_loss' in outputs else 0.0
                logging.info(f"Step {self.global_step}: cultures={unique_cultures.tolist()}, culture_loss={culture_loss_val:.6f}")

                if self.global_step == 0:
                    logging.info(f"use_culture_loss: {self.args.use_culture_loss}")

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
            if 'specialization_loss' in outputs:
                total_specialization_loss += outputs['specialization_loss'].item()
            if 'diversity_loss' in outputs:
                total_diversity_loss += outputs['diversity_loss'].item()

            num_batches += 1
            self.global_step += 1

            # 更新进度条 - 添加更多损失信息
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'gen': f"{outputs.get('generation_loss', torch.tensor(0)).item():.4f}",
                'cult': f"{outputs.get('culture_loss', torch.tensor(0)).item():.4f}",
                'load': f"{outputs.get('load_balance_loss', torch.tensor(0)).item():.4f}",
                'ent': f"{outputs.get('entropy_loss', torch.tensor(0)).item():.4f}"
            })

            # 清理内存
            if self.global_step % 50 == 0:
                torch.cuda.empty_cache()

        return {
            'train_loss': total_loss / num_batches,
            'train_generation_loss': total_generation_loss / num_batches,
            'train_culture_loss': total_culture_loss / num_batches,
            'train_load_balance_loss': total_load_balance_loss / num_batches,
            'train_entropy_loss': total_entropy_loss / num_batches,
            'train_specialization_loss': total_specialization_loss / num_batches,
            'train_diversity_loss': total_diversity_loss / num_batches,
        }

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()

        total_loss = 0.0
        total_generation_loss = 0.0
        total_culture_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        generated_answers = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # 移动数据到设备，特殊处理culture_ids_multi
                batch_device = {}
                for k, v in batch.items():
                    if k == 'culture_ids_multi':
                        # culture_ids_multi是列表，不能直接.to(device)
                        batch_device[k] = v
                    else:
                        batch_device[k] = v.to(self.device)
                batch = batch_device

                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    input_ids_mask=batch['input_ids_mask'],
                    attention_mask_mask=batch['attention_mask_mask'],
                    labels=batch['labels'],
                    culture_labels=batch['culture_ids'],  # 传递给文化损失计算
                    culture_ids=batch['culture_ids'],     # 传递给文化感知组件
                    culture_ids_multi=batch['culture_ids_multi'],
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
                logits = outputs['logits']

                # 统计损失
                total_loss += loss.item()
                if 'generation_loss' in outputs:
                    total_generation_loss += outputs['generation_loss'].item()
                if 'culture_loss' in outputs:
                    total_culture_loss += outputs['culture_loss'].item()

                # 计算准确率
                labels = batch['labels']
                predictions = torch.argmax(logits, dim=-1)

                # 只计算非-100位置的准确率
                mask = (labels != -100)
                correct = (predictions == labels) & mask
                correct_predictions += correct.sum().item()
                total_predictions += mask.sum().item()

                # 生成答案示例（前5个batch）
                if len(generated_answers) < 5:
                    for i in range(min(2, batch['input_ids'].size(0))):
                        input_text = self.tokenizer.decode(
                            batch['input_ids'][i],
                            skip_special_tokens=True
                        )
                        generated_answers.append({
                            'input': input_text,
                            'culture_id': batch['culture_ids'][i].item(),
                            'expert_weights': outputs['expert_weights'][i].cpu().tolist()
                        })

        # 保存生成的答案
        answers_file = os.path.join(self.args.output_dir, 'generated_answers.json')
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(generated_answers, f, indent=2, ensure_ascii=False)

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        return {
            'eval_loss': total_loss / len(self.val_loader),
            'eval_generation_loss': total_generation_loss / len(self.val_loader),
            'eval_culture_loss': total_culture_loss / len(self.val_loader),
            'eval_accuracy': accuracy,
            'eval_samples': len(self.val_dataset)
        }

    def save_model(self, epoch: int, is_best: bool = False):
        """保存模型"""
        if is_best:
            save_dir = os.path.join(self.args.output_dir, 'best_enhanced_moe')
        else:
            save_dir = os.path.join(self.args.output_dir, f'epoch_{epoch+1}')

        os.makedirs(save_dir, exist_ok=True)

        # 只保存MoE相关参数
        # DataParallel兼容性：获取实际模型
        actual_model = self.model.module if self.use_dataparallel else self.model
        moe_state_dict = {}
        moe_param_count = 0

        for name, param in actual_model.named_parameters():
            if param.requires_grad:  # 只保存可训练参数
                # 确保只保存MoE相关参数
                if any(keyword in name for keyword in [
                    'cultural_experts', 'router', 'shared', 'cultural_gate',
                    'cultural_embedding', 'cultural_context', 'culture_loss'
                ]):
                    # 保存为float16节省空间
                    moe_state_dict[name] = param.half().cpu()
                    moe_param_count += param.numel()
                else:
                    logging.warning(f"⚠️  Unexpected trainable parameter: {name}")

        # 保存MoE权重
        moe_weights_path = os.path.join(save_dir, 'moe_weights.pth')
        torch.save(moe_state_dict, moe_weights_path)

        # 计算文件大小
        file_size_mb = os.path.getsize(moe_weights_path) / (1024 * 1024)
        param_size_mb = moe_param_count * 2 / (1024 * 1024)  # float16 = 2 bytes per param

        logging.info(f"MoE weights saved to {save_dir}")
        logging.info(f"  Parameters: {moe_param_count:,}")
        logging.info(f"  File size: {file_size_mb:.1f} MB")
        logging.info(f"  Expected size: {param_size_mb:.1f} MB")

        if file_size_mb > 500:  # 如果超过500MB，发出警告
            logging.warning(f"⚠️  MoE weights file is unexpectedly large: {file_size_mb:.1f} MB")

        # 只在最佳模型时保存训练状态（可选）
        if is_best and hasattr(self.args, 'save_optimizer_state') and self.args.save_optimizer_state:
            torch.save(self.optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pth'))
            torch.save(self.scheduler.state_dict(), os.path.join(save_dir, 'scheduler.pth'))
            logging.info("Optimizer and scheduler states saved")

    def train(self):
        """主训练循环"""
        logging.info("Starting training...")

        for epoch in range(self.args.num_epochs):
            logging.info(f"\n{'='*60}")
            logging.info(f"Epoch {epoch+1}/{self.args.num_epochs}")
            logging.info(f"{'='*60}")

            # 训练
            train_metrics = self.train_epoch(epoch)

            # 评估
            if (epoch + 1) % self.args.eval_interval == 0:
                eval_metrics = self.evaluate(epoch)

                # 合并指标
                metrics = {**train_metrics, **eval_metrics, 'epoch': epoch + 1}
                self.epoch_results.append(metrics)

                # 记录详细结果
                logging.info(f"=== Epoch {epoch+1} Training Results ===")
                logging.info(f"Total Loss: {train_metrics['train_loss']:.6f}")
                logging.info(f"  ├─ Generation Loss: {train_metrics['train_generation_loss']:.6f}")
                logging.info(f"  ├─ Culture Loss: {train_metrics['train_culture_loss']:.6f}")
                logging.info(f"  ├─ Load Balance Loss: {train_metrics['train_load_balance_loss']:.6f}")
                logging.info(f"  ├─ Entropy Loss: {train_metrics['train_entropy_loss']:.6f}")
                logging.info(f"  ├─ Specialization Loss: {train_metrics['train_specialization_loss']:.6f}")
                logging.info(f"  └─ Diversity Loss: {train_metrics['train_diversity_loss']:.6f}")
                logging.info(f"")
                logging.info(f"=== Epoch {epoch+1} Evaluation Results ===")
                logging.info(f"Eval Loss: {eval_metrics['eval_loss']:.6f}")
                logging.info(f"Eval Accuracy: {eval_metrics['eval_accuracy']:.4f}")

                # 检查异常损失值
                if train_metrics['train_loss'] < 0:
                    logging.warning("⚠️  WARNING: Total loss is negative!")
                    logging.warning("This may indicate training instability or incorrect loss calculation.")
                if train_metrics['train_entropy_loss'] < -1.0:
                    logging.warning("⚠️  WARNING: Entropy loss is very negative, possible expert collapse!")
                if train_metrics['train_generation_loss'] < 0.001:
                    logging.warning("⚠️  WARNING: Generation loss is very low, possible overfitting!")

                # 保存最佳模型
                if eval_metrics['eval_accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_metrics['eval_accuracy']
                    self.save_model(epoch, is_best=True)
                    logging.info(f"New best accuracy: {self.best_accuracy:.4f}")

            # 保存epoch结果
            results_file = os.path.join(self.args.output_dir, 'epoch_eval_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.epoch_results, f, indent=2, ensure_ascii=False)

            # 内存清理
            torch.cuda.empty_cache()
            gc.collect()

        logging.info("\nTraining completed!")
        logging.info(f"Best accuracy: {self.best_accuracy:.4f}")

        # 检查输出目录总大小
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.args.output_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)

        total_size_mb = total_size / (1024 * 1024)
        total_size_gb = total_size_mb / 1024

        logging.info(f"\nOutput directory analysis:")
        logging.info(f"  Total size: {total_size_mb:.1f} MB ({total_size_gb:.2f} GB)")
        logging.info(f"  Location: {self.args.output_dir}")

        if total_size_gb > 1.0:
            logging.warning(f"⚠️  Output directory is large: {total_size_gb:.2f} GB")
            logging.warning("This may indicate unexpected files or optimizer states being saved")

            # 列出最大的文件
            file_sizes = []
            for dirpath, dirnames, filenames in os.walk(self.args.output_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    file_sizes.append((filepath, size_mb))

            file_sizes.sort(key=lambda x: x[1], reverse=True)
            logging.info("Largest files:")
            for filepath, size_mb in file_sizes[:5]:
                rel_path = os.path.relpath(filepath, self.args.output_dir)
                logging.info(f"  {rel_path}: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Enhanced CultureMoE Training")

    # 基础参数
    parser.add_argument('--base_model_path', type=str, required=True, help='基础模型路径')
    parser.add_argument('--lora_weights_path', type=str, help='LoRA权重路径')
    parser.add_argument('--train_file', type=str, required=True, help='训练数据文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')

    # 模型参数
    parser.add_argument('--num_experts', type=int, default=12, help='专家数量')
    parser.add_argument('--shared_hidden_dim', type=int, default=4096, help='共享层隐藏维度')
    parser.add_argument('--router_hidden_dim', type=int, default=2048, help='路由器隐藏维度')
    parser.add_argument('--experts_hidden_dim', type=int, default=4096, help='专家隐藏维度')
    parser.add_argument('--moe_lora_rank', type=int, default=32, help='MoE LoRA rank')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout率')

    # 文化损失参数
    parser.add_argument('--use_culture_loss', type=str, default='True', help='是否使用文化损失')
    parser.add_argument('--culture_loss_lambda', type=float, default=0.5, help='文化损失权重')
    parser.add_argument('--culture_loss_alpha', type=float, default=0.5, help='文化损失margin')
    parser.add_argument('--culture_loss_beta', type=float, default=1.0, help='文化损失lambda_diff')

    # MoE参数
    parser.add_argument('--moe_fusion', type=float, default=0.4, help='MoE融合系数')
    parser.add_argument('--use_shared_experts', type=str, default='True', help='是否使用共享专家')
    parser.add_argument('--router_temperature', type=float, default=2.0, help='路由器温度')
    parser.add_argument('--load_balance_weight', type=float, default=0.01, help='负载均衡权重')
    parser.add_argument('--entropy_weight', type=float, default=0.1, help='熵正则化权重')

    # 训练参数
    parser.add_argument('--freeze_base_model', type=str, default='True', help='是否冻结基础模型')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='评估批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--moe_lr_multiplier', type=float, default=1.0, help='MoE学习率倍数')
    parser.add_argument('--router_lr_multiplier', type=float, default=1.0, help='路由器学习率倍数')
    parser.add_argument('--shared_lr_multiplier', type=float, default=1.0, help='共享层学习率倍数')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--eval_interval', type=int, default=3, help='评估间隔')
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    args = parser.parse_args()

    # 转换字符串布尔值
    args.use_culture_loss = args.use_culture_loss.lower() == 'true'
    args.use_shared_experts = args.use_shared_experts.lower() == 'true'
    args.freeze_base_model = args.freeze_base_model.lower() == 'true'

    # 创建训练器并开始训练
    trainer = EnhancedCultureMoETrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()