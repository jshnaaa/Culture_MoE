#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA增强的FFN集成CultureMoE训练脚本

特点：
1. 在Attention的Q、K、V、O投影层添加LoRA适配器
2. 在MoE专家的FFN层（gate_proj, up_proj, down_proj）添加LoRA适配器
3. 支持渐进式训练和专门的LoRA优化策略
4. 完全向量化的专家调度，解决效率瓶颈
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

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.lora_enhanced_culturemoe import LoRACultureMoEConfig
from llamafactory.model.lora_culturemoe_model import (
    LoRACultureMoELlamaModel,
    create_lora_culturemoe_model
)
import torch.nn.functional as F


# ===== 数据集类 =====
class CultureDatasetForLoRA(Dataset):
    """LoRA训练用的文化数据集"""

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

        instruction = item.get('instruction', '')
        output = item.get('output', '')
        label = item.get('label', '0')

        # 构建完整的对话文本
        full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        # 分词
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 创建标签（用于语言模型损失）
        labels = input_ids.clone()

        # 找到response开始位置，mask掉instruction部分
        instruction_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
        instruction_tokens = self.tokenizer(instruction_text, add_special_tokens=False)['input_ids']
        instruction_length = len(instruction_tokens)

        if instruction_length < len(labels):
            labels[:instruction_length] = -100  # 忽略instruction部分的损失

        # 解析文化标签
        try:
            culture_id = self.continent_map.get(str(label), self.default_continent)
        except (ValueError, KeyError):
            culture_id = self.default_continent

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'culture_ids': torch.tensor(culture_id, dtype=torch.long)
        }

        # 如果启用mask机制，生成masked版本的hidden states
        if self.use_mask_mechanism:
            # 这里我们为共享专家生成一个mask标记
            # 在实际训练中，这个mask会在forward过程中应用到hidden states
            result['use_mask_mechanism'] = True
        else:
            result['use_mask_mechanism'] = False

        return result


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


# ===== Mask生成函数 =====
def generate_cultural_mask(hidden_states: torch.Tensor, culture_ids: torch.Tensor,
                          mask_ratio: float = 0.3) -> torch.Tensor:
    """
    为共享专家生成文化敏感的mask

    Args:
        hidden_states: [B, L, H] 输入的hidden states
        culture_ids: [B] 文化标识
        mask_ratio: mask的比例

    Returns:
        masked_hidden_states: [B, L, H] mask后的hidden states
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device

    # 创建mask后的hidden states副本
    masked_hidden_states = hidden_states.clone()

    # 为每个样本生成不同的mask模式
    for i in range(batch_size):
        culture_id = culture_ids[i].item()

        # 基于文化ID生成确定性的mask模式
        torch.manual_seed(42 + culture_id + i)  # 确保可重现性

        # 生成随机mask
        mask_indices = torch.rand(seq_len, hidden_dim, device=device) < mask_ratio

        # 应用mask（将被mask的位置设为0或小的随机值）
        masked_hidden_states[i][mask_indices] *= 0.1  # 保留10%的信息而不是完全置零

    return masked_hidden_states


# ===== 训练器类 =====
class LoRACultureMoETrainer:
    """LoRA增强CultureMoE训练器"""

    def __init__(self, model: LoRACultureMoELlamaModel, lora_config: LoRACultureMoEConfig):
        self.model = model
        self.lora_config = lora_config
        self.loss_fn = LoRACultureMoELoss(lora_config)

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

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        culture_ids = batch['culture_ids']
        use_mask = batch.get('use_mask_mechanism', True)

        # 生成mask（如果启用）
        hidden_states_mask = None
        if use_mask:
            # 首先进行一次前向传播获取hidden states
            with torch.no_grad():
                temp_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    culture_ids=culture_ids,
                    return_dict=True
                )
                # 使用embedding后的hidden states生成mask
                temp_hidden_states = self.model.embed_tokens(input_ids)
                hidden_states_mask = generate_cultural_mask(temp_hidden_states, culture_ids)

        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            culture_ids=culture_ids,
            hidden_states_mask=hidden_states_mask,
            return_dict=True
        )

        # 计算损失
        loss_dict = self.loss_fn(
            logits=outputs.last_hidden_state,
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

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """验证步骤"""
        self.model.eval()

        with torch.no_grad():
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            culture_ids = batch['culture_ids']
            use_mask = batch.get('use_mask_mechanism', True)

            # 生成mask（如果启用）
            hidden_states_mask = None
            if use_mask:
                temp_hidden_states = self.model.embed_tokens(input_ids)
                hidden_states_mask = generate_cultural_mask(temp_hidden_states, culture_ids)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                culture_ids=culture_ids,
                hidden_states_mask=hidden_states_mask,
                return_dict=True
            )

            loss_dict = self.loss_fn(
                logits=outputs.last_hidden_state,
                labels=labels,
                moe_aux_info=getattr(outputs, 'moe_aux_info', []),
                culture_labels=culture_ids
            )

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def get_expert_utilization_stats(self) -> Dict[str, Any]:
        """获取专家利用率统计"""
        stats = {}
        for layer_idx, layer in enumerate(self.model.layers):
            stats[f'layer_{layer_idx}'] = {
                'num_experts': layer.mlp.num_experts,
                'moe_alpha': layer.mlp.moe_fusion_alpha.item()
            }
        return stats

    def save_lora_weights(self, save_path: str):
        """保存LoRA权重"""
        lora_state_dict = {}

        for layer_idx, layer in enumerate(self.model.layers):
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


# ===== 渐进式训练策略 =====
class ProgressiveLoRATrainingStrategy:
    """渐进式LoRA训练策略"""

    def __init__(self, model: LoRACultureMoELlamaModel, lora_config: LoRACultureMoEConfig):
        self.model = model
        self.lora_config = lora_config
        self.total_layers = len(model.layers)

    def stage_1_attention_only(self):
        """阶段1：只训练Attention LoRA"""
        print("Stage 1: Training Attention LoRA only")

        # 冻结所有MoE参数
        for layer in self.model.layers:
            for param in layer.mlp.parameters():
                param.requires_grad = False

        # 解冻Attention LoRA参数
        for layer in self.model.layers:
            for name, module in layer.self_attn.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    module.lora_A.weight.requires_grad = True
                    module.lora_B.weight.requires_grad = True

        return "Stage 1 setup complete. Training Attention LoRA only."

    def stage_2_experts_only(self):
        """阶段2：只训练Expert LoRA"""
        print("Stage 2: Training Expert LoRA only")

        # 冻结Attention LoRA参数
        for layer in self.model.layers:
            for name, module in layer.self_attn.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    module.lora_A.weight.requires_grad = False
                    module.lora_B.weight.requires_grad = False

        # 解冻Expert LoRA参数
        for layer in self.model.layers:
            for name, module in layer.mlp.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    if 'expert' in name:
                        module.lora_A.weight.requires_grad = True
                        module.lora_B.weight.requires_grad = True

        return "Stage 2 setup complete. Training Expert LoRA only."

    def stage_3_all_lora(self):
        """阶段3：训练所有LoRA参数"""
        print("Stage 3: Training all LoRA parameters")

        # 解冻所有LoRA参数
        for layer in self.model.layers:
            for name, module in layer.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    module.lora_A.weight.requires_grad = True
                    module.lora_B.weight.requires_grad = True

        return "Stage 3 setup complete. Training all LoRA parameters."

    def get_trainable_params_count(self) -> Dict[str, int]:
        """获取当前可训练参数数量"""
        attention_lora_params = 0
        expert_lora_params = 0
        cultural_lora_params = 0
        other_params = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'self_attn' in name and ('lora_A' in name or 'lora_B' in name):
                    attention_lora_params += param.numel()
                elif 'mlp' in name and 'expert' in name and ('lora_A' in name or 'lora_B' in name):
                    expert_lora_params += param.numel()
                elif 'mlp' in name and ('lora_A' in name or 'lora_B' in name):
                    cultural_lora_params += param.numel()
                else:
                    other_params += param.numel()

        return {
            'attention_lora': attention_lora_params,
            'expert_lora': expert_lora_params,
            'cultural_lora': cultural_lora_params,
            'other': other_params,
            'total': attention_lora_params + expert_lora_params + cultural_lora_params + other_params
        }


# ===== 主训练函数 =====
def main():
    parser = argparse.ArgumentParser(description='LoRA Enhanced CultureMoE Training')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf', help='Base model path')
    parser.add_argument('--data_path', type=str, required=True, help='Training data path')
    parser.add_argument('--output_dir', type=str, default='./outputs/lora_culturemoe', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32.0, help='LoRA alpha')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--progressive_training', action='store_true', help='Use progressive training strategy')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # LoRA配置
    lora_config = LoRACultureMoEConfig(
        num_experts=args.num_experts,
        top_k=2,
        capacity_factor=1.25,
        num_cultures=6,
        culture_dim=256,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        attention_lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        expert_lora_targets=["gate_proj", "up_proj", "down_proj"],
        load_balance_weight=0.01,
        entropy_weight=0.1,
        culture_loss_weight=0.05,
        warmup_steps=1000,
        gradient_clip_norm=1.0
    )

    # 创建模型
    logger.info("Creating LoRA enhanced CultureMoE model...")
    model = create_lora_culturemoe_model(args.base_model, lora_config)
    model.freeze_base_parameters()  # 冻结基础参数，只训练LoRA
    model.print_parameter_stats()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    logger.info(f"Loading training data from {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = CultureDatasetForLoRA(data, tokenizer, max_length=512, use_mask_mechanism=True)

    # 分割训练和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建训练器
    trainer = LoRACultureMoETrainer(model, lora_config)
    optimizer = trainer.setup_optimizer(learning_rate=args.learning_rate)

    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = trainer.setup_scheduler(total_steps)

    # 渐进式训练策略
    if args.progressive_training:
        progressive_strategy = ProgressiveLoRATrainingStrategy(model, lora_config)

        # 阶段1：Attention LoRA
        logger.info("=== Stage 1: Attention LoRA Training ===")
        progressive_strategy.stage_1_attention_only()
        logger.info(f"Trainable params: {progressive_strategy.get_trainable_params_count()}")
        train_epochs(trainer, train_dataloader, val_dataloader, 2, logger, "Stage1")

        # 阶段2：Expert LoRA
        logger.info("=== Stage 2: Expert LoRA Training ===")
        progressive_strategy.stage_2_experts_only()
        logger.info(f"Trainable params: {progressive_strategy.get_trainable_params_count()}")
        train_epochs(trainer, train_dataloader, val_dataloader, 3, logger, "Stage2")

        # 阶段3：All LoRA
        logger.info("=== Stage 3: All LoRA Training ===")
        progressive_strategy.stage_3_all_lora()
        logger.info(f"Trainable params: {progressive_strategy.get_trainable_params_count()}")
        train_epochs(trainer, train_dataloader, val_dataloader, 3, logger, "Stage3")
    else:
        # 直接训练所有LoRA参数
        logger.info("=== Direct LoRA Training ===")
        train_epochs(trainer, train_dataloader, val_dataloader, args.num_epochs, logger, "Direct")

    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, 'final_lora_weights.pt')
    trainer.save_lora_weights(final_save_path)

    logger.info("Training completed!")


def train_epochs(trainer, train_dataloader, val_dataloader, num_epochs, logger, stage_name):
    """训练指定数量的epochs"""
    device = next(trainer.model.parameters()).device

    for epoch in range(num_epochs):
        # 训练阶段
        trainer.model.train()
        epoch_losses = []

        progress_bar = tqdm(train_dataloader, desc=f'{stage_name} Epoch {epoch+1}/{num_epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = {k: v.to(device) for k, v in batch.items()}

            # 训练步骤
            loss_dict = trainer.train_step(batch)
            epoch_losses.append(loss_dict)

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss']:.4f}",
                'LM': f"{loss_dict['lm_loss']:.4f}",
                'LB': f"{loss_dict['load_balance_loss']:.4f}",
                'Ent': f"{loss_dict['entropy_loss']:.4f}"
            })

            # 定期日志
            if batch_idx % 100 == 0:
                logger.info(f"{stage_name} Epoch {epoch+1}, Batch {batch_idx}: {loss_dict}")

        # 验证阶段
        if val_dataloader:
            trainer.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    val_loss_dict = trainer.validate_step(batch)
                    val_losses.append(val_loss_dict)

            # 计算平均验证损失
            avg_val_loss = {
                key: np.mean([loss[key] for loss in val_losses])
                for key in val_losses[0].keys()
            }

            logger.info(f"{stage_name} Epoch {epoch+1} Validation: {avg_val_loss}")

        # 专家利用率统计
        expert_stats = trainer.get_expert_utilization_stats()
        logger.info(f"{stage_name} Epoch {epoch+1} Expert Stats: {expert_stats}")


if __name__ == "__main__":
    main()