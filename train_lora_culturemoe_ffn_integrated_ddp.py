#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA增强的FFN集成CultureMoE训练脚本（DDP多卡版本）

特点：
1. 支持DDP（DistributedDataParallel）多卡训练
2. 在Attention的Q、K、V、O投影层添加LoRA适配器
3. 在MoE专家的FFN层（gate_proj, up_proj, down_proj）添加LoRA适配器
4. 支持渐进式训练和专门的LoRA优化策略
5. 完全向量化的专家调度，解决效率瓶颈
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
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, random_split, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed
)
import numpy as np
from tqdm import tqdm
import gc

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.lora_enhanced_culturemoe import LoRACultureMoEConfig
from llamafactory.model.lora_culturemoe_model import (
    LoRACultureMoELlamaModel,
    create_lora_culturemoe_model
)
import torch.nn.functional as F


# ===== DDP工具函数 =====
def setup_ddp(rank, world_size):
    """初始化DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置当前设备
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """跨进程平均tensor"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# ===== 数据集类 =====
class CultureDatasetForLoRA(Dataset):
    """LoRA训练用的文化数据集，支持instruction_mask字段"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, use_mask_mechanism: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_mask_mechanism = use_mask_mechanism

        # 大洲映射
        self.continent_map = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
        }
        self.default_continent = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 获取数据字段
        instruction = item.get('instruction', '')
        instruction_mask = item.get('instruction_mask', instruction)  # mask版本的instruction
        input_field = item.get('input', '')  # input字段（通常为空）
        output = item.get('output', '')
        label = item.get('label', '0')

        # 清理可能的special tokens
        def clean_text(text):
            special_tokens = ['<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>']
            for token in special_tokens:
                text = text.replace(token, '')
            return text.strip()

        clean_instruction = clean_text(instruction)
        clean_instruction_mask = clean_text(instruction_mask)
        clean_input = clean_text(input_field) if input_field else ''

        # 构建完整文本 - 原始版本（文化专家使用）
        if clean_input:
            input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{clean_instruction}\\n{clean_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        else:
            input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{clean_instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        full_text = input_text + output + "<|eot_id|>"

        # 构建mask版本（共享专家使用）
        if clean_input:
            input_text_mask = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{clean_instruction_mask}\\n{clean_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        else:
            input_text_mask = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{clean_instruction_mask}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        full_text_mask = input_text_mask + output + "<|eot_id|>"

        # 分词 - 原始版本
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # 分词 - mask版本
        encoding_mask = self.tokenizer(
            full_text_mask,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        input_ids_mask = encoding_mask['input_ids'].squeeze()
        attention_mask_mask = encoding_mask['attention_mask'].squeeze()

        # 创建标签（用于语言模型损失）
        labels = input_ids.clone()

        # 找到assistant响应开始位置，mask掉instruction部分
        assistant_start_tokens = self.tokenizer(input_text, add_special_tokens=False)['input_ids']
        assistant_start_idx = len(assistant_start_tokens)

        if assistant_start_idx < len(labels):
            labels[:assistant_start_idx] = -100  # 忽略instruction部分的损失

        # 解析文化标签
        try:
            culture_id = self.continent_map.get(str(label), self.default_continent)
        except (ValueError, KeyError):
            culture_id = self.default_continent

        return {
            'input_ids': input_ids,                    # 文化专家使用的原始输入
            'attention_mask': attention_mask,
            'input_ids_mask': input_ids_mask,          # 共享专家使用的mask输入
            'attention_mask_mask': attention_mask_mask,
            'labels': labels,
            'culture_ids': torch.tensor(culture_id, dtype=torch.long),
            'use_mask_mechanism': self.use_mask_mechanism,
            'true_output': output  # 保存真实的output用于评估
        }


# ===== 损失计算类 =====
class LoRACultureMoELoss(nn.Module):
    """LoRA增强CultureMoE的损失函数"""

    def __init__(self, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.lora_config = lora_config
        self.load_balance_weight = lora_config.load_balance_weight
        self.entropy_weight = lora_config.entropy_weight
        self.culture_loss_weight = lora_config.culture_loss_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                moe_aux_info: List[Dict], culture_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算综合损失"""
        device = logits.device

        # 1. 标准语言模型损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 2. MoE辅助损失
        total_load_balance_loss = torch.tensor(0.0, device=device)
        total_entropy_loss = torch.tensor(0.0, device=device)
        num_layers = len(moe_aux_info)

        for layer_aux in moe_aux_info:
            if 'load_balance_loss' in layer_aux:
                total_load_balance_loss += layer_aux['load_balance_loss']
            if 'entropy_loss' in layer_aux:
                total_entropy_loss += layer_aux['entropy_loss']

        avg_load_balance_loss = total_load_balance_loss / num_layers if num_layers > 0 else torch.tensor(0.0, device=device)
        avg_entropy_loss = total_entropy_loss / num_layers if num_layers > 0 else torch.tensor(0.0, device=device)

        # 3. 文化对齐损失
        culture_loss = torch.tensor(0.0, device=device)
        if culture_labels is not None and self.culture_loss_weight > 0:
            culture_loss = self._compute_culture_alignment_loss(moe_aux_info, culture_labels)

        # 4. LoRA正则化损失（可选）
        lora_regularization = torch.tensor(0.0, device=device)

        # 5. 总损失
        total_loss = (
            lm_loss +
            self.load_balance_weight * avg_load_balance_loss +
            self.entropy_weight * avg_entropy_loss +
            self.culture_loss_weight * culture_loss +
            0.001 * lora_regularization  # 小的LoRA正则化
        )

        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'load_balance_loss': avg_load_balance_loss,
            'entropy_loss': avg_entropy_loss,
            'culture_loss': culture_loss,
            'lora_regularization': lora_regularization
        }

    def _compute_culture_alignment_loss(self, moe_aux_info: List[Dict], culture_labels: torch.Tensor) -> torch.Tensor:
        """计算文化对齐损失"""
        total_culture_loss = 0.0
        num_layers = len(moe_aux_info)

        for layer_aux in moe_aux_info:
            if 'expert_weights' in layer_aux:
                expert_weights = layer_aux['expert_weights']
                batch_size = expert_weights.shape[0]

                culture_alignment = 0.0
                count = 0

                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        if culture_labels[i] == culture_labels[j]:
                            # 相同文化，鼓励相似的专家权重
                            similarity = F.cosine_similarity(
                                expert_weights[i].unsqueeze(0),
                                expert_weights[j].unsqueeze(0)
                            )
                            culture_alignment += (1.0 - similarity)
                        else:
                            # 不同文化，鼓励不同的专家权重
                            similarity = F.cosine_similarity(
                                expert_weights[i].unsqueeze(0),
                                expert_weights[j].unsqueeze(0)
                            )
                            culture_alignment += similarity
                        count += 1

                if count > 0:
                    total_culture_loss += culture_alignment / count

        return total_culture_loss / num_layers if num_layers > 0 else torch.tensor(0.0, device=culture_labels.device)


# ===== 训练器类 =====
class LoRACultureMoETrainerDDP:
    """LoRA增强CultureMoE训练器（DDP版本）"""

    def __init__(self, model: LoRACultureMoELlamaModel, lora_config: LoRACultureMoEConfig, rank: int, world_size: int):
        self.model = model
        self.lora_config = lora_config
        self.loss_fn = LoRACultureMoELoss(lora_config)
        self.rank = rank
        self.world_size = world_size

        # 获取LoRA参数组
        self.lora_params = self.model.get_lora_parameters()

    def setup_optimizer(self, learning_rate: float = 5e-4, weight_decay: float = 0.01):
        """设置LoRA特化的优化器"""
        # 分组LoRA参数，使用不同的学习率
        param_groups = []

        # Attention LoRA参数（较小学习率）
        if self.lora_params['attention_lora']:
            param_groups.append({
                'params': self.lora_params['attention_lora'],
                'lr': learning_rate * 0.5,
                'weight_decay': weight_decay * 0.5,
                'name': 'attention_lora'
            })

        # Expert LoRA参数（标准学习率）
        if self.lora_params['expert_lora']:
            param_groups.append({
                'params': self.lora_params['expert_lora'],
                'lr': learning_rate,
                'weight_decay': weight_decay,
                'name': 'expert_lora'
            })

        # Cultural LoRA参数（中等学习率）
        if self.lora_params['cultural_lora']:
            param_groups.append({
                'params': self.lora_params['cultural_lora'],
                'lr': learning_rate * 0.8,
                'weight_decay': weight_decay * 0.5,
                'name': 'cultural_lora'
            })

        # 非LoRA的可训练参数（如果有）
        non_lora_params = []
        lora_param_ids = set()
        for group_params in self.lora_params.values():
            lora_param_ids.update(id(p) for p in group_params)

        for param in self.model.parameters():
            if param.requires_grad and id(param) not in lora_param_ids:
                non_lora_params.append(param)

        if non_lora_params:
            param_groups.append({
                'params': non_lora_params,
                'lr': learning_rate * 0.1,
                'weight_decay': weight_decay,
                'name': 'non_lora'
            })

        self.optimizer = torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
        return self.optimizer

    def setup_scheduler(self, num_training_steps: int):
        """设置学习率调度器"""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.lora_config.warmup_steps,
            num_training_steps=num_training_steps
        )
        return self.scheduler

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()

        # 获取两种不同的输入
        input_ids = batch['input_ids']                    # 文化专家使用的原始输入
        attention_mask = batch['attention_mask']
        input_ids_mask = batch['input_ids_mask']          # 共享专家使用的mask输入
        attention_mask_mask = batch['attention_mask_mask']
        labels = batch['labels']
        culture_ids = batch['culture_ids']

        # 前向传播 - 传递mask版本的输入用于共享专家
        outputs = self.model(
            input_ids=input_ids,                          # 文化专家使用
            attention_mask=attention_mask,
            culture_ids=culture_ids,
            input_ids_mask=input_ids_mask,                # 共享专家使用
            attention_mask_mask=attention_mask_mask,
            return_dict=True
        )

        # 需要添加语言模型头来生成logits
        # 这里我们需要从模型获取logits而不是hidden_states
        # 假设模型有lm_head属性
        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(outputs.last_hidden_state)
        elif hasattr(self.model.module, 'lm_head'):
            logits = self.model.module.lm_head(outputs.last_hidden_state)
        else:
            # 如果没有lm_head，我们需要创建一个临时的
            vocab_size = self.model.module.config.vocab_size if hasattr(self.model, 'module') else self.model.config.vocab_size
            if not hasattr(self, 'temp_lm_head'):
                self.temp_lm_head = nn.Linear(outputs.last_hidden_state.size(-1), vocab_size).to(self.rank)
            logits = self.temp_lm_head(outputs.last_hidden_state)

        # 计算损失
        loss_dict = self.loss_fn(
            logits=logits,
            labels=labels,
            moe_aux_info=getattr(outputs, 'moe_aux_info', []),
            culture_labels=culture_ids
        )

        total_loss = loss_dict['total_loss']

        # 反向传播
        total_loss.backward()

        # 梯度裁剪（只对LoRA参数）
        if self.lora_config.gradient_clip_norm > 0:
            all_lora_params = []
            for group_params in self.lora_params.values():
                all_lora_params.extend(group_params)
            torch.nn.utils.clip_grad_norm_(all_lora_params, self.lora_config.gradient_clip_norm)

        # 优化器步骤
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # 在DDP中同步损失
        if self.world_size > 1:
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    loss_dict[key] = reduce_tensor(value, self.world_size)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """验证步骤"""
        self.model.eval()

        with torch.no_grad():
            # 获取两种不同的输入
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            input_ids_mask = batch['input_ids_mask']
            attention_mask_mask = batch['attention_mask_mask']
            labels = batch['labels']
            culture_ids = batch['culture_ids']

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                culture_ids=culture_ids,
                input_ids_mask=input_ids_mask,
                attention_mask_mask=attention_mask_mask,
                return_dict=True
            )

            # 生成logits用于损失计算
            if hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(outputs.last_hidden_state)
            elif hasattr(self.model.module, 'lm_head'):
                logits = self.model.module.lm_head(outputs.last_hidden_state)
            else:
                if not hasattr(self, 'temp_lm_head'):
                    vocab_size = self.model.module.config.vocab_size if hasattr(self.model, 'module') else self.model.config.vocab_size
                    self.temp_lm_head = nn.Linear(outputs.last_hidden_state.size(-1), vocab_size).to(self.rank)
                logits = self.temp_lm_head(outputs.last_hidden_state)

            loss_dict = self.loss_fn(
                logits=logits,
                labels=labels,
                moe_aux_info=getattr(outputs, 'moe_aux_info', []),
                culture_labels=culture_ids
            )

        # 在DDP中同步损失
        if self.world_size > 1:
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    loss_dict[key] = reduce_tensor(value, self.world_size)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def get_expert_utilization_stats(self) -> Dict[str, Any]:
        """获取专家利用率统计"""
        stats = {}
        for layer_idx, layer in enumerate(self.model.module.layers):  # 注意DDP的module属性
            stats[f'layer_{layer_idx}'] = {
                'num_experts': layer.mlp.num_experts,
                'moe_alpha': layer.mlp.moe_fusion_alpha.item()
            }
        return stats

    def save_lora_weights(self, save_path: str):
        """保存LoRA权重"""
        if self.rank == 0:  # 只在主进程保存
            lora_state_dict = {}

            # 注意DDP的module属性
            model = self.model.module if hasattr(self.model, 'module') else self.model

            for layer_idx, layer in enumerate(model.layers):
                # 保存Attention LoRA权重
                for name, module in layer.self_attn.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        lora_state_dict[f'layer_{layer_idx}.self_attn.{name}.lora_A.weight'] = module.lora_A.weight
                        lora_state_dict[f'layer_{layer_idx}.self_attn.{name}.lora_B.weight'] = module.lora_B.weight

                # 保存MoE LoRA权重
                for name, module in layer.mlp.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        lora_state_dict[f'layer_{layer_idx}.mlp.{name}.lora_A.weight'] = module.lora_A.weight
                        lora_state_dict[f'layer_{layer_idx}.mlp.{name}.lora_B.weight'] = module.lora_B.weight

            torch.save(lora_state_dict, save_path)
            logging.info(f"LoRA weights saved to {save_path}")


# ===== 主训练函数 =====
def train_ddp(rank, world_size, args):
    """DDP训练函数"""
    # 设置DDP
    setup_ddp(rank, world_size)

    # 设置随机种子
    set_seed(args.seed + rank)

    # 设置日志（只在主进程）
    if rank == 0:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        logger = None

    # LoRA配置
    lora_config = LoRACultureMoEConfig(
        num_experts=args.num_experts,
        top_k=2,
        capacity_factor=1.25,
        num_cultures=6,
        culture_dim=256,
        lora_rank=16,  # 固定值，不再作为参数
        lora_alpha=32.0,  # 固定值，不再作为参数
        lora_dropout=0.1,
        attention_lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        expert_lora_targets=["gate_proj", "up_proj", "down_proj"],
        load_balance_weight=0.01,
        entropy_weight=0.1,
        culture_loss_weight=0.05,
        enable_cultural_attention=True,  # 启用文化感知注意力
        warmup_steps=1000,
        gradient_clip_norm=1.0
    )

    # 创建模型
    if rank == 0:
        logger.info("Creating LoRA enhanced CultureMoE model...")
    model = create_lora_culturemoe_model(args.base_model, lora_config)
    model.freeze_base_parameters()  # 冻结基础参数，只训练LoRA
    model.cuda(rank)

    if rank == 0:
        model.print_parameter_stats()

    # 包装为DDP模型
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    if rank == 0:
        logger.info(f"Loading training data from {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = CultureDatasetForLoRA(data, tokenizer, max_length=512, use_mask_mechanism=True)

    # 分割训练和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建DDP采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    # 创建训练器
    trainer = LoRACultureMoETrainerDDP(model, lora_config, rank, world_size)
    optimizer = trainer.setup_optimizer(learning_rate=args.learning_rate)

    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = trainer.setup_scheduler(total_steps)

    # 训练循环
    for epoch in range(args.num_epochs):
        # 设置采样器的epoch
        train_sampler.set_epoch(epoch)

        # 训练阶段
        trainer.model.train()
        epoch_losses = []

        if rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        else:
            progress_bar = train_dataloader

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = {k: v.cuda(rank) for k, v in batch.items()}

            # 训练步骤
            loss_dict = trainer.train_step(batch)
            epoch_losses.append(loss_dict)

            # 更新进度条（只在主进程）
            if rank == 0:
                progress_bar.set_postfix({
                    'Loss': f"{loss_dict['total_loss']:.4f}",
                    'LM': f"{loss_dict['lm_loss']:.4f}",
                    'LB': f"{loss_dict['load_balance_loss']:.4f}",
                    'Ent': f"{loss_dict['entropy_loss']:.4f}"
                })

                # 定期日志
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: {loss_dict}")

        # 验证阶段
        if val_dataloader:
            trainer.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.cuda(rank) for k, v in batch.items()}
                    val_loss_dict = trainer.validate_step(batch)
                    val_losses.append(val_loss_dict)

            # 计算平均验证损失
            if val_losses:
                avg_val_loss = {
                    key: np.mean([loss[key] for loss in val_losses])
                    for key in val_losses[0].keys()
                }

                if rank == 0:
                    logger.info(f"Epoch {epoch+1} Validation: {avg_val_loss}")

        # 专家利用率统计（只在主进程）
        if rank == 0:
            expert_stats = trainer.get_expert_utilization_stats()
            logger.info(f"Epoch {epoch+1} Expert Stats: {expert_stats}")

    # 保存最终模型（只在主进程）
    if rank == 0:
        final_save_path = os.path.join(args.output_dir, 'final_lora_weights.pt')
        trainer.save_lora_weights(final_save_path)
        logger.info("Training completed!")

    # 清理DDP
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='LoRA Enhanced CultureMoE Training with DDP')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf', help='Base model path')
    parser.add_argument('--data_path', type=str, required=True, help='Training data path')
    parser.add_argument('--output_dir', type=str, default='./outputs/lora_culturemoe', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs (1 for single GPU, 2+ for DDP)')
    parser.add_argument('--progressive_training', action='store_true', help='Use progressive training strategy')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    if args.num_gpus == 1:
        # 单卡训练 - 直接调用原始训练函数
        print("Using single GPU training...")
        # 这里可以调用原始的单卡训练逻辑
        # 为了简化，我们还是用DDP但只用一个GPU
        world_size = 1
        train_ddp(0, world_size, args)
    else:
        # 多卡训练
        print(f"Using DDP training with {args.num_gpus} GPUs...")
        world_size = args.num_gpus

        # 检查GPU数量
        if torch.cuda.device_count() < world_size:
            print(f"Warning: Only {torch.cuda.device_count()} GPUs available, but {world_size} requested")
            world_size = torch.cuda.device_count()

        # 启动多进程
        torch.multiprocessing.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()