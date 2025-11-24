#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: 带知识蒸馏的增强CultureMoE训练脚本
通过从LoRA teacher模型蒸馏知识来改善OOD性能

特点：
1. 使用稳健共享专家（通过蒸馏增强）
2. 知识蒸馏损失
3. 文化不变性学习
4. 自适应蒸馏权重调度
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

from llamafactory.model.enhanced_culturemoe_with_distill import EnhancedCultureMoEWithDistill
from llamafactory.model.moe_args import ModelArgs


# ===== 重用原有数据集类 =====
class CultureDataset(Dataset):
    """文化对齐数据集（复用原有实现）"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, use_mask: bool = True, effective_vocab_size: int = None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_mask = use_mask

        # 使用effective_vocab_size进行token验证
        if effective_vocab_size is None:
            tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', 128000)
            self.effective_vocab_size = max(tokenizer_vocab_size, 128300)
        else:
            self.effective_vocab_size = max(effective_vocab_size, 128300)

        # 大洲映射
        self.continent_map = {
            '0': 0,      # 亚洲 (Asia)
            '1': 1,      # 欧洲 (Europe)
            '2': 2,      # 北美洲 (North America)
            '3': 3,      # 南美洲 (South America)
            '4': 4,      # 非洲 (Africa)
            '5': 5,      # 大洋洲 (Oceania)
        }

        self.num_continents = 6
        self.default_continent = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 获取数据字段
        instruction = item.get('instruction', '')
        instruction_mask = item.get('instruction_mask', instruction)
        output = item.get('output', '')
        label = item.get('label', '0')

        # 简单清理：移除可能的special tokens
        clean_instruction = instruction
        special_tokens_to_remove = [
            '<|begin_of_text|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|eot_id|>'
        ]
        for token in special_tokens_to_remove:
            clean_instruction = clean_instruction.replace(token, '')
        clean_instruction = clean_instruction.strip()

        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{clean_instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        full_text = input_text + output + "<|eot_id|>"

        # 构建mask版本的输入文本
        if self.use_mask:
            clean_instruction_mask = instruction_mask
            for token in special_tokens_to_remove:
                clean_instruction_mask = clean_instruction_mask.replace(token, '')
            clean_instruction_mask = clean_instruction_mask.strip()
            input_text_mask = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{clean_instruction_mask}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            input_text_mask = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{clean_instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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

        # 验证token ID有效性
        vocab_size = self.effective_vocab_size

        # 检查原始版本的token
        invalid_tokens = input_ids[input_ids >= vocab_size]
        if len(invalid_tokens) > 0:
            unk_token_id = getattr(self.tokenizer, 'unk_token_id', None)
            if unk_token_id is None or unk_token_id >= vocab_size:
                unk_token_id = getattr(self.tokenizer, 'pad_token_id', None)
                if unk_token_id is None or unk_token_id >= vocab_size:
                    unk_token_id = 0
            input_ids[input_ids >= vocab_size] = unk_token_id

        # 检查mask版本的token
        invalid_tokens_mask = input_ids_mask[input_ids_mask >= vocab_size]
        if len(invalid_tokens_mask) > 0:
            unk_token_id = getattr(self.tokenizer, 'unk_token_id', None)
            if unk_token_id is None or unk_token_id >= vocab_size:
                unk_token_id = getattr(self.tokenizer, 'pad_token_id', None)
                if unk_token_id is None or unk_token_id >= vocab_size:
                    unk_token_id = 0
            input_ids_mask[input_ids_mask >= vocab_size] = unk_token_id

        # 创建标签
        labels = input_ids.clone()

        # 找到assistant响应开始位置
        assistant_start = input_text
        assistant_encoding = self.tokenizer(assistant_start, add_special_tokens=False)
        assistant_start_idx = len(assistant_encoding['input_ids'])

        # 掩盖instruction部分
        labels[:assistant_start_idx] = -100

        # 基本标签验证：确保有有效标签
        valid_labels = (labels != -100).sum()
        if valid_labels == 0 and len(labels) > 0:
            labels[-1] = input_ids[-1]

        # 解析大洲标签
        continent_id, continent_ids_multi = self._parse_continent_label(label)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'input_ids_mask': input_ids_mask,
            'attention_mask_mask': attention_mask_mask,
            'labels': labels,
            'culture_ids': torch.tensor(continent_id, dtype=torch.long),
            'culture_ids_multi': continent_ids_multi
        }

    def _parse_continent_label(self, label: str) -> tuple[int, List[int]]:
        """解析大洲标签，返回主要大洲ID和所有相关大洲ID列表"""
        label = str(label).strip()
        continent_ids = []

        # 处理多大洲情况
        if ',' in label:
            continents = [c.strip() for c in label.split(',')]
            for continent in continents:
                if continent in self.continent_map:
                    continent_ids.append(self.continent_map[continent])
        else:
            if label in self.continent_map:
                continent_ids.append(self.continent_map[label])

        # 如果没有找到任何有效的大洲，使用默认值
        if not continent_ids:
            continent_ids.append(self.default_continent)

        primary_continent_id = continent_ids[0]
        return primary_continent_id, continent_ids


# ===== 分层优化器（复用原有实现）=====
class LayeredOptimizer:
    """分层优化器：为不同组件使用不同学习率"""

    def __init__(self, model, base_lr: float = 2e-4,
                 moe_lr_multiplier: float = 1.0,
                 router_lr_multiplier: float = 1.0,
                 shared_lr_multiplier: float = 1.0,
                 distill_lr_multiplier: float = 0.5,  # 新增：蒸馏相关参数学习率
                 weight_decay: float = 0.01):

        self.base_lr = base_lr
        self.moe_lr_multiplier = moe_lr_multiplier
        self.router_lr_multiplier = router_lr_multiplier
        self.shared_lr_multiplier = shared_lr_multiplier
        self.distill_lr_multiplier = distill_lr_multiplier

        # 分组参数
        moe_params = []
        router_params = []
        shared_params = []
        distill_params = []  # 新增：蒸馏相关参数

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # DataParallel兼容性：移除module.前缀
            clean_name = name.replace('module.', '')

            # 分类参数
            if any(keyword in clean_name.lower() for keyword in [
                'robust_shared_expert', 'distill_loss_fn', 'invariant_loss_fn',
                'teacher_loader', 'distill_weight_scheduler'
            ]):
                distill_params.append(param)
            elif 'cultural_experts' in clean_name or 'culture' in clean_name:
                moe_params.append(param)
            elif 'router' in clean_name:
                router_params.append(param)
            elif 'shared' in clean_name:
                shared_params.append(param)
            else:
                moe_params.append(param)  # 默认归入MoE组

        # 创建参数组
        param_groups = []
        if distill_params:
            param_groups.append({
                'params': distill_params,
                'lr': base_lr * distill_lr_multiplier,
                'weight_decay': weight_decay * 0.1
            })
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
        logging.info(f"  Distillation params: {len(distill_params)} (lr={base_lr * distill_lr_multiplier:.2e})")
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
class EnhancedCultureMoEWithDistillTrainer:
    """带知识蒸馏的增强CultureMoE训练器"""

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

        # 验证tokenizer兼容性
        self.validate_tokenizer_model_compatibility()

        # 加载数据
        self.setup_data()

        # 设置优化器和调度器
        self.setup_optimizer()

        # 训练状态
        self.global_step = 0
        self.best_accuracy = 0.0
        self.epoch_results = []

        # 梯度累积步数
        self.gradient_accumulation_steps = 2

        # 混合精度训练
        self.use_amp = True
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            logging.info("AMP scaler initialized")

    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.args.output_dir, 'distill_training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info("=" * 60)
        logging.info("Enhanced CultureMoE with Distillation Training Started")
        logging.info("=" * 60)
        logging.info(f"Arguments: {vars(self.args)}")

    def save_config(self):
        """保存训练配置"""
        config = vars(self.args)
        config_file = os.path.join(self.args.output_dir, 'distill_config.json')
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

        # 创建带蒸馏的增强CultureMoE模型
        self.model = EnhancedCultureMoEWithDistill(
            llama_model=base_model,
            config=base_model.config,
            args=moe_args,
            culture_loss_lambda=self.args.culture_loss_lambda,
            moe_fusion=self.args.moe_fusion,
            num_cultures=6,
            culture_dim=256,
            use_gate=self.args.use_gate,
            teacher_model_path=self.args.teacher_model_path,
            teacher_lora_path=self.args.teacher_lora_path,
            distill_temperature=self.args.distill_temperature,
            distill_alpha=self.args.distill_alpha
        )

        # 冻结基础模型参数
        if self.args.freeze_base_model:
            self.freeze_base_model()

        # 移动到设备
        self.model = self.model.to(self.device)

        # 设置teacher模型
        self.model.setup_teacher_model()

        logging.info("Model setup completed")

    def freeze_base_model(self):
        """冻结基础模型参数"""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            # 判断是否为MoE或蒸馏相关参数
            is_trainable_param = any(keyword in name for keyword in [
                'cultural_experts', 'router', 'shared', 'cultural_gate',
                'cultural_embedding', 'cultural_context', 'culture_loss',
                'robust_shared_expert', 'distill_loss_fn', 'invariant_loss_fn'
            ])

            if is_trainable_param:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        logging.info(f"Model parameters frozen: {frozen_count:,} ({frozen_count/(frozen_count+trainable_count)*100:.1f}%)")
        logging.info(f"Model parameters trainable: {trainable_count:,} ({trainable_count/(frozen_count+trainable_count)*100:.1f}%)")

    def validate_tokenizer_model_compatibility(self):
        """验证tokenizer和模型embedding层的兼容性"""
        try:
            # 获取embedding层信息
            actual_model = self.model.module if hasattr(self.model, 'module') else self.model
            embed_layer = actual_model.llama_model.model.embed_tokens
            model_vocab_size = embed_layer.num_embeddings

            # 获取tokenizer信息
            tokenizer_vocab_size = getattr(self.tokenizer, 'vocab_size', 50000)
            special_token_ids = self.tokenizer.all_special_ids
            max_special_token_id = max(special_token_ids) if special_token_ids else 0

            logging.info(f"Tokenizer-Model Compatibility Check:")
            logging.info(f"   Tokenizer vocab_size: {tokenizer_vocab_size}")
            logging.info(f"   Model embedding size: {model_vocab_size}")
            logging.info(f"   Max special token ID: {max_special_token_id}")

            # 设置effective_vocab_size
            self._effective_vocab_size = max(model_vocab_size, 128300)
            logging.info(f"Using effective_vocab_size: {self._effective_vocab_size}")

        except Exception as e:
            logging.error(f"Failed to validate tokenizer-model compatibility: {e}")
            self._effective_vocab_size = 128300

    def setup_data(self):
        """设置数据"""
        logging.info(f"Loading data from {self.args.train_file}")

        with open(self.args.train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"Loaded {len(data)} samples")

        # 创建数据集
        use_mask = self.args.use_mask.lower() == 'true'
        logging.info(f"MASK mechanism: {'ENABLED' if use_mask else 'DISABLED'}")

        effective_vocab_size = getattr(self, '_effective_vocab_size', 128300)
        dataset = CultureDataset(data, self.tokenizer, self.args.max_length, use_mask, effective_vocab_size)

        # 划分训练、验证和测试集 (8:1:1)
        total_size = len(dataset)
        test_size = int(total_size * 0.1)
        val_size = int(total_size * 0.1)
        train_size = total_size - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

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

        logging.info(f"Train samples: {len(self.train_dataset)}")
        logging.info(f"Validation samples: {len(self.val_dataset)}")
        logging.info(f"Test samples: {len(self.test_dataset)}")

    def setup_optimizer(self):
        """设置优化器和调度器"""
        # 使用分层优化器（包含蒸馏参数）
        self.optimizer = LayeredOptimizer(
            self.model,
            base_lr=self.args.learning_rate,
            moe_lr_multiplier=self.args.moe_lr_multiplier,
            router_lr_multiplier=self.args.router_lr_multiplier,
            shared_lr_multiplier=self.args.shared_lr_multiplier,
            distill_lr_multiplier=0.5,  # 蒸馏参数使用较小学习率
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
        total_distill_loss = 0.0  # 新增：蒸馏损失
        total_invariant_loss = 0.0  # 新增：不变性损失
        num_batches = 0
        skipped_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Distill Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch_device = {}
            for k, v in batch.items():
                if k == 'culture_ids_multi':
                    batch_device[k] = v
                else:
                    batch_device[k] = v.to(self.device)
            batch = batch_device

            # 前向传播（使用AMP）
            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        input_ids_mask=batch['input_ids_mask'],
                        attention_mask_mask=batch['attention_mask_mask'],
                        labels=batch['labels'],
                        culture_labels=batch['culture_ids'],
                        culture_ids=batch['culture_ids'],
                        culture_ids_multi=batch['culture_ids_multi'],
                        use_culture_loss=self.args.use_culture_loss,
                        culture_loss_lambda=self.args.culture_loss_lambda,
                        culture_loss_alpha=self.args.culture_loss_alpha,
                        culture_loss_beta=self.args.culture_loss_beta,
                        use_shared_experts=self.args.use_shared_experts,
                        router_temperature=self.args.router_temperature,
                        load_balance_weight=self.args.load_balance_weight,
                        entropy_weight=self.args.entropy_weight,
                        current_epoch=epoch + 1,
                        total_epochs=self.args.num_epochs,
                        enable_distillation=True  # 启用知识蒸馏
                    )
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    input_ids_mask=batch['input_ids_mask'],
                    attention_mask_mask=batch['attention_mask_mask'],
                    labels=batch['labels'],
                    culture_labels=batch['culture_ids'],
                    culture_ids=batch['culture_ids'],
                    culture_ids_multi=batch['culture_ids_multi'],
                    use_culture_loss=self.args.use_culture_loss,
                    culture_loss_lambda=self.args.culture_loss_lambda,
                    culture_loss_alpha=self.args.culture_loss_alpha,
                    culture_loss_beta=self.args.culture_loss_beta,
                    use_shared_experts=self.args.use_shared_experts,
                    router_temperature=self.args.router_temperature,
                    load_balance_weight=self.args.load_balance_weight,
                    entropy_weight=self.args.entropy_weight,
                    current_epoch=epoch + 1,
                    total_epochs=self.args.num_epochs,
                    enable_distillation=True  # 启用知识蒸馏
                )

            # 计算总损失（包含蒸馏损失）
            loss = self.model.get_total_loss(outputs, distill_weight=0.3, invariant_weight=0.1)

            # 检查损失是否为NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Step {self.global_step}: Detected NaN/Inf loss, skipping batch")
                skipped_batches += 1
                self.optimizer.zero_grad()
                self.global_step += 1
                continue

            # 梯度累积反向传播
            original_loss = loss.clone()
            loss = loss / self.gradient_accumulation_steps

            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 更新参数
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp and self.scaler:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            # 统计
            total_loss += original_loss.item()
            if 'generation_loss' in outputs:
                total_generation_loss += outputs['generation_loss'].item()
            if 'culture_loss' in outputs:
                total_culture_loss += outputs['culture_loss'].item()
            if 'total_distill_loss' in outputs:
                total_distill_loss += outputs['total_distill_loss'].item()
            if 'cultural_invariant_loss' in outputs:
                total_invariant_loss += outputs['cultural_invariant_loss'].item()

            num_batches += 1
            self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{original_loss.item():.4f}",
                'gen': f"{outputs.get('generation_loss', torch.tensor(0)).item():.4f}",
                'dist': f"{outputs.get('total_distill_loss', torch.tensor(0)).item():.4f}",
                'inv': f"{outputs.get('cultural_invariant_loss', torch.tensor(0)).item():.4f}"
            })

            # 内存清理
            if self.global_step % 10 == 0:
                torch.cuda.empty_cache()

        # Epoch结束统计
        logging.info(f"\n=== Distillation Epoch {epoch+1} Summary ===")
        logging.info(f"Batch Processing: {num_batches} successful, {skipped_batches} skipped")

        if num_batches > 0:
            logging.info(f"Total Loss: {total_loss / num_batches:.6f}")
            logging.info(f"  ├─ Generation Loss: {total_generation_loss / num_batches:.6f}")
            logging.info(f"  ├─ Culture Loss: {total_culture_loss / num_batches:.6f}")
            logging.info(f"  ├─ Distillation Loss: {total_distill_loss / num_batches:.6f}")
            logging.info(f"  └─ Invariant Loss: {total_invariant_loss / num_batches:.6f}")

        safe_avg = lambda x: x / num_batches if num_batches > 0 else 0.0

        return {
            'train_loss': safe_avg(total_loss),
            'train_generation_loss': safe_avg(total_generation_loss),
            'train_culture_loss': safe_avg(total_culture_loss),
            'train_distill_loss': safe_avg(total_distill_loss),
            'train_invariant_loss': safe_avg(total_invariant_loss),
            'skipped_batches': skipped_batches
        }

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """评估模型"""
        logging.info(f"Starting evaluation for epoch {epoch+1}")
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        eval_skipped_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                try:
                    # 移动数据到设备
                    batch_device = {}
                    for k, v in batch.items():
                        if k == 'culture_ids_multi':
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
                        culture_labels=batch['culture_ids'],
                        culture_ids=batch['culture_ids'],
                        culture_ids_multi=batch['culture_ids_multi'],
                        use_culture_loss=self.args.use_culture_loss,
                        culture_loss_lambda=self.args.culture_loss_lambda,
                        culture_loss_alpha=self.args.culture_loss_alpha,
                        culture_loss_beta=self.args.culture_loss_beta,
                        use_shared_experts=self.args.use_shared_experts,
                        router_temperature=self.args.router_temperature,
                        load_balance_weight=self.args.load_balance_weight,
                        entropy_weight=self.args.entropy_weight,
                        current_epoch=epoch + 1,
                        total_epochs=self.args.num_epochs,
                        enable_distillation=False  # 评估时不使用蒸馏
                    )

                    loss = outputs['loss']
                    logits = outputs['logits']

                    # 检查评估损失
                    if torch.isnan(loss) or torch.isinf(loss):
                        eval_skipped_batches += 1
                        continue

                    # 统计损失
                    total_loss += loss.item()

                    # 计算准确率
                    labels = batch['labels']
                    predictions = torch.argmax(logits, dim=-1)

                    mask = (labels != -100)
                    correct = (predictions == labels) & mask
                    correct_predictions += correct.sum().item()
                    total_predictions += mask.sum().item()

                except Exception as e:
                    logging.error(f"Error processing evaluation batch {batch_idx}: {e}")
                    continue

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        successful_eval_batches = len(self.val_loader) - eval_skipped_batches
        if successful_eval_batches > 0:
            avg_loss = total_loss / successful_eval_batches
            logging.info(f"Evaluation completed - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
        else:
            avg_loss = float('inf')
            logging.error("All evaluation batches were skipped!")

        return {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'eval_samples': len(self.val_dataset),
            'eval_skipped_batches': eval_skipped_batches
        }

    def save_model(self, epoch: int, is_best: bool = False):
        """保存模型"""
        if is_best:
            save_dir = os.path.join(self.args.output_dir, 'best_distill_moe')
        else:
            save_dir = os.path.join(self.args.output_dir, f'distill_epoch_{epoch+1}')

        os.makedirs(save_dir, exist_ok=True)

        # 保存MoE和蒸馏相关参数
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        state_dict = {}
        param_count = 0

        # 定义要保存的参数模式
        save_patterns = [
            'cultural_experts.',
            'router.',
            'cultural_embedding.',
            'cultural_context.',
            'cultural_gate.',
            'shared.',
            'robust_shared_expert.',  # 新增：稳健共享专家
            'distill_loss_fn.',       # 新增：蒸馏损失函数
            'invariant_loss_fn.',     # 新增：不变性损失函数
            'moe_fusion_alpha',
            'culture_loss_lambda',
        ]

        for name, param in actual_model.named_parameters():
            if param.requires_grad and any(pattern in name for pattern in save_patterns):
                state_dict[name] = param.half().cpu()
                param_count += param.numel()

        # 保存权重
        weights_path = os.path.join(save_dir, 'distill_moe_weights.pth')
        torch.save(state_dict, weights_path)

        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        logging.info(f"Distillation MoE weights saved to {save_dir}")
        logging.info(f"  Parameters: {param_count:,}")
        logging.info(f"  File size: {file_size_mb:.1f} MB")

    def train(self):
        """主训练循环"""
        logging.info("Starting distillation training...")

        for epoch in range(self.args.num_epochs):
            logging.info(f"\n{'='*60}")
            logging.info(f"Distillation Epoch {epoch+1}/{self.args.num_epochs}")
            logging.info(f"{'='*60}")

            # 训练
            train_metrics = self.train_epoch(epoch)

            # 评估
            if (epoch + 1) % self.args.eval_interval == 0:
                try:
                    eval_metrics = self.evaluate(epoch)

                    # 合并指标
                    metrics = {**train_metrics, **eval_metrics, 'epoch': epoch + 1}
                    self.epoch_results.append(metrics)

                    # 保存最佳模型
                    current_accuracy = eval_metrics['eval_accuracy']
                    if current_accuracy > self.best_accuracy:
                        old_best = self.best_accuracy
                        self.best_accuracy = current_accuracy
                        self.save_model(epoch, is_best=True)
                        logging.info(f"🎉 NEW BEST DISTILLATION MODEL SAVED!")
                        logging.info(f"   Previous best accuracy: {old_best:.4f}")
                        logging.info(f"   New best accuracy: {self.best_accuracy:.4f}")

                except Exception as e:
                    logging.error(f"Evaluation failed for epoch {epoch+1}: {e}")

            # 保存epoch结果
            results_file = os.path.join(self.args.output_dir, 'distill_epoch_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.epoch_results, f, indent=2, ensure_ascii=False)

            # 内存清理
            torch.cuda.empty_cache()
            gc.collect()

        logging.info("\nDistillation training completed!")
        logging.info(f"Best accuracy achieved: {self.best_accuracy:.4f}")

        # 清理teacher模型资源
        self.model.cleanup_teacher()


def main():
    parser = argparse.ArgumentParser(description="Enhanced CultureMoE with Distillation Training")

    # 基础参数
    parser.add_argument('--base_model_path', type=str, required=True, help='基础模型路径')
    parser.add_argument('--lora_weights_path', type=str, help='LoRA权重路径')
    parser.add_argument('--train_file', type=str, required=True, help='训练数据文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')

    # 知识蒸馏参数
    parser.add_argument('--teacher_model_path', type=str, help='Teacher模型路径（通常与base_model_path相同）')
    parser.add_argument('--teacher_lora_path', type=str, help='Teacher LoRA权重路径')
    parser.add_argument('--distill_temperature', type=float, default=4.0, help='蒸馏温度')
    parser.add_argument('--distill_alpha', type=float, default=0.7, help='蒸馏权重')

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
    parser.add_argument('--use_mask', type=str, default='True', help='是否使用MASK机制')
    parser.add_argument('--use_gate', type=str, default='True', help='是否使用GATE机制')
    parser.add_argument('--router_temperature', type=float, default=3.0, help='路由器温度')
    parser.add_argument('--load_balance_weight', type=float, default=0.01, help='负载均衡权重')
    parser.add_argument('--entropy_weight', type=float, default=0.1, help='熵正则化权重')

    # 训练参数
    parser.add_argument('--freeze_base_model', type=str, default='True', help='是否冻结基础模型')
    parser.add_argument('--num_epochs', type=int, default=6, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='评估批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--moe_lr_multiplier', type=float, default=1.0, help='MoE学习率倍数')
    parser.add_argument('--router_lr_multiplier', type=float, default=0.1, help='路由器学习率倍数')
    parser.add_argument('--shared_lr_multiplier', type=float, default=1.0, help='共享层学习率倍数')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--max_length', type=int, default=1024, help='最大序列长度')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--eval_interval', type=int, default=2, help='评估间隔')
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    args = parser.parse_args()

    # 转换字符串布尔值
    args.use_culture_loss = args.use_culture_loss.lower() == 'true'
    args.use_shared_experts = args.use_shared_experts.lower() == 'true'
    args.freeze_base_model = args.freeze_base_model.lower() == 'true'
    args.use_gate = args.use_gate.lower() == 'true'

    # 设置默认teacher路径（如果未提供）
    if not args.teacher_model_path:
        args.teacher_model_path = args.base_model_path
    if not args.teacher_lora_path:
        args.teacher_lora_path = args.lora_weights_path

    # 创建训练器并开始训练
    trainer = EnhancedCultureMoEWithDistillTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()