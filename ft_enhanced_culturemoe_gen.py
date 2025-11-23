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

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, use_mask: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_mask = use_mask

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
        # 检查instruction是否包含special tokens
        if '<|begin_of_text|>' in instruction:
            logging.warning(f"Found <|begin_of_text|> in instruction - possible double-wrapping")
        if '<|start_header_id|>' in instruction:
            logging.warning(f"Found <|start_header_id|> in instruction - double-wrapping detected")

        # 🔧 智能清理：处理截断导致的重复包装问题
        clean_instruction = instruction

        # 检测是否是被截断的对话格式
        is_truncated_chat = False
        if '<|begin_of_text|>' in instruction and '<|start_header_id|>' in instruction:
            is_truncated_chat = True

            # 尝试提取用户内容部分
            try:
                # 找到用户内容的开始和结束
                user_start = instruction.find('<|start_header_id|>user<|end_header_id|>')
                if user_start != -1:
                    content_start = user_start + len('<|start_header_id|>user<|end_header_id|>')

                    # 查找内容结束位置
                    eot_pos = instruction.find('<|eot_id|>', content_start)
                    assistant_pos = instruction.find('<|start_header_id|>assistant', content_start)

                    if eot_pos != -1:
                        clean_instruction = instruction[content_start:eot_pos].strip()
                    elif assistant_pos != -1:
                        clean_instruction = instruction[content_start:assistant_pos].strip()
                    else:
                        # 截断在用户内容中间，取剩余部分
                        clean_instruction = instruction[content_start:].strip()

            except Exception as e:
                logging.warning(f"Failed to extract user content: {e}")
                is_truncated_chat = False

        if not is_truncated_chat:
            # 简单的token清理（原来的方法）
            special_tokens_to_remove = [
                '<|begin_of_text|>',
                '<|start_header_id|>',
                '<|end_header_id|>',
                '<|eot_id|>'
            ]
            for token in special_tokens_to_remove:
                clean_instruction = clean_instruction.replace(token, '')
            clean_instruction = clean_instruction.strip()

        # 最终验证：确保没有残留的special tokens
        if any(token in clean_instruction for token in ['<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>']):
            logging.warning("Still found special tokens after cleaning, applying aggressive cleaning")
            for token in ['<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>']:
                clean_instruction = clean_instruction.replace(token, '')
            clean_instruction = clean_instruction.strip()

        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{clean_instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        full_text = input_text + output + "<|eot_id|>"

        # 构建mask版本的输入文本 (用于共享专家)
        # 如果使用MASK机制，共享专家使用instruction_mask；否则使用原始instruction
        if self.use_mask:
            # 检查instruction_mask是否包含special tokens
            if '<|begin_of_text|>' in instruction_mask:
                logging.warning(f"Found <|begin_of_text|> in instruction_mask - possible double-wrapping")

            # 清理instruction_mask中可能重复的special tokens
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

        # ✅ 验证token ID范围，防止超大token ID导致NaN
        vocab_size = getattr(self.tokenizer, 'vocab_size', 128256)  # LLaMA默认是128256


        # 获取安全的替换token ID
        def get_safe_replacement_token_id():
            # 尝试获取UNK token
            unk_id = getattr(self.tokenizer, 'unk_token_id', None)
            if unk_id is not None and 0 <= unk_id < vocab_size:
                return unk_id

            # 尝试获取PAD token
            pad_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_id is not None and 0 <= pad_id < vocab_size:
                return pad_id

            # 尝试获取EOS token
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_id is not None and 0 <= eos_id < vocab_size:
                return eos_id

            # 最后使用0（通常是安全的）
            return 0

        safe_token_id = get_safe_replacement_token_id()

        # 检查并修复超出范围的token ID
        invalid_mask = input_ids >= vocab_size
        if invalid_mask.any():
            logging.warning(f"Found {invalid_mask.sum().item()} invalid token IDs >= {vocab_size}, replacing with token {safe_token_id}")
            input_ids = torch.where(invalid_mask, safe_token_id, input_ids)

        invalid_mask_mask = input_ids_mask >= vocab_size
        if invalid_mask_mask.any():
            logging.warning(f"Found {invalid_mask_mask.sum().item()} invalid token IDs in mask >= {vocab_size}, replacing with token {safe_token_id}")
            input_ids_mask = torch.where(invalid_mask_mask, safe_token_id, input_ids_mask)

        # 创建标签
        labels = input_ids.clone()

        # 找到assistant响应开始位置
        assistant_start = input_text
        assistant_encoding = self.tokenizer(assistant_start, add_special_tokens=False)
        assistant_start_idx = len(assistant_encoding['input_ids'])

        # 掩盖instruction部分
        labels[:assistant_start_idx] = -100

        # ✅ 验证标签有效性，防止全部被掩码导致损失异常
        valid_labels = (labels != -100).sum()
        if valid_labels == 0:
            logging.warning("All labels are masked (-100), this may cause training instability")
            # 至少保留最后一个token作为有效标签
            if len(labels) > 0:
                labels[-1] = input_ids[-1]
                logging.warning("Added last token as valid label to prevent empty target")

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

        # 根据模型类型调整梯度累积
        if hasattr(args, 'base_model_path') and 'llama' in args.base_model_path.lower():
            self.gradient_accumulation_steps = 2  # LLaMA需要更多梯度累积
            logging.info("Detected LLaMA model, using gradient accumulation steps: 2")
        else:
            self.gradient_accumulation_steps = 1
            logging.info("Using gradient accumulation steps: 1")

        # 设置随机种子
        set_seed(42)

        # 设置日志
        self.setup_logging()

        # 创建路由汇总保存目录
        self.routing_logs_dir = os.path.join(args.output_dir, "routing_logs")
        os.makedirs(self.routing_logs_dir, exist_ok=True)

        # 用于收集epoch级别路由信息的变量
        self.epoch_expert_weights = []
        self.epoch_gate_values = []
        self.epoch_culture_ids = []

        # ✅ 路由恶化检测系统
        self.routing_history = []  # 存储历史路由指标
        self.routing_alert_thresholds = {
            'gini_coefficient': 0.5,        # Gini系数超过0.5认为严重不均衡
            'entropy_drop': 0.3,            # 熵下降超过0.3认为塌陷风险
            'expert_collapse': 0.05,        # 专家使用率低于0.05认为塌陷
            'consecutive_degradation': 2     # 连续2个epoch恶化触发警报
        }
        self.routing_alerts = []  # 存储警报历史

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

        # 🔍 NaN问题诊断
        self.nan_batch_positions = []  # 记录出现NaN的batch位置

    def collect_routing_info_for_summary(self, outputs: Dict, batch: Dict):
        """收集路由信息用于epoch汇总"""
        try:
            # 收集专家权重
            if 'expert_weights' in outputs:
                expert_weights = outputs['expert_weights'].detach().cpu().numpy()
                self.epoch_expert_weights.append(expert_weights)

            # 收集gate值（如果存在）
            if hasattr(self.model, 'cultural_gate') and self.model.cultural_gate is not None:
                if 'cultural_analysis' in outputs:
                    cultural_analysis = outputs['cultural_analysis']
                    gate_values = cultural_analysis.get('gate_values', None)
                    if gate_values is not None:
                        self.epoch_gate_values.append(gate_values.detach().cpu().numpy())

            # 收集文化ID
            if 'culture_ids' in batch:
                culture_ids = batch['culture_ids'].cpu().numpy()
                self.epoch_culture_ids.extend(culture_ids.tolist())

        except Exception as e:
            # 静默处理错误，不影响训练
            pass


    def analyze_nan_patterns(self, epoch: int):
        """分析NaN出现的模式"""
        if not self.nan_batch_positions:
            return

        logging.error("\n" + "=" * 80)
        logging.error("🔍 NaN PATTERN ANALYSIS")
        logging.error("=" * 80)

        # 当前epoch的NaN位置
        current_epoch_nans = [pos for pos in self.nan_batch_positions if pos['epoch'] == epoch + 1]

        if current_epoch_nans:
            logging.error(f"Epoch {epoch + 1} NaN positions:")
            for pos in current_epoch_nans:
                logging.error(f"  Batch {pos['batch_idx']} ({pos['progress_percent']:.1f}%)")

        # 检查是否有重复的batch位置
        all_batch_indices = [pos['batch_idx'] for pos in self.nan_batch_positions]
        unique_indices = set(all_batch_indices)
        repeated_positions = {}

        if len(unique_indices) < len(all_batch_indices):
            logging.error("⚠️  REPEATED NaN POSITIONS DETECTED!")

            # 统计每个位置出现的次数
            from collections import Counter
            position_counts = Counter(all_batch_indices)
            repeated_positions = {idx: count for idx, count in position_counts.items() if count > 1}

            if repeated_positions:
                logging.error("Repeated problematic batch positions:")
                for batch_idx, count in repeated_positions.items():
                    progress = (batch_idx + 1) / len(self.train_loader) * 100
                    logging.error(f"  Batch {batch_idx} ({progress:.1f}%): appeared {count} times")

                logging.error("")
                logging.error("💡 DIAGNOSIS: Specific data samples are causing NaN!")
                logging.error("RECOMMENDED ACTIONS:")
                logging.error("1. Examine the problematic batch data files saved in output directory")
                logging.error("2. Check for extremely long sequences, unusual characters, or corrupted data")
                logging.error("3. Consider filtering or fixing these specific data samples")
                logging.error("4. Alternatively, add data validation in the dataset preprocessing")

        # 保存完整的NaN分析报告
        nan_analysis_file = os.path.join(self.args.output_dir, f'nan_analysis_epoch_{epoch + 1}.json')
        analysis_data = {
            'epoch': epoch + 1,
            'total_nan_occurrences': len(self.nan_batch_positions),
            'current_epoch_nans': current_epoch_nans,
            'all_nan_positions': self.nan_batch_positions,
            'repeated_positions': repeated_positions,
            'dataset_size': len(self.train_loader),
            'recommendations': [
                "Check problematic batch files in output directory",
                "Validate data quality and preprocessing",
                "Consider filtering problematic samples",
                "Monitor if pattern persists across epochs"
            ]
        }

        with open(nan_analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        logging.error(f"Detailed NaN analysis saved to: {nan_analysis_file}")
        logging.error("=" * 80)

    def save_epoch_routing_summary(self, epoch: int):
        """保存每个epoch的路由汇总信息"""
        try:
            # 检查是否有收集到的数据
            if not self.epoch_expert_weights:
                logging.warning(f"Epoch {epoch + 1}: 没有收集到专家权重数据")
                return

            # 合并所有batch的数据
            all_expert_weights = np.concatenate(self.epoch_expert_weights, axis=0)  # [total_samples, num_experts]

            # 计算专家使用统计
            expert_usage = np.mean(all_expert_weights, axis=0)  # [num_experts]

            # 计算路由熵
            all_entropies = []
            for batch_weights in all_expert_weights:
                probs_safe = np.clip(batch_weights, 1e-8, 1.0)
                entropy = -np.sum(probs_safe * np.log(probs_safe))
                all_entropies.append(entropy)

            # 处理gate信息
            gate_statistics = {}
            if self.epoch_gate_values:
                all_gate_values = np.concatenate(self.epoch_gate_values, axis=0)
                gate_statistics = {
                    'mean': float(np.mean(all_gate_values)),
                    'std': float(np.std(all_gate_values)),
                    'min': float(np.min(all_gate_values)),
                    'max': float(np.max(all_gate_values))
                }

            # 按文化分组的专家使用率
            culture_expert_usage = {}
            if self.epoch_culture_ids:
                unique_cultures = np.unique(self.epoch_culture_ids)
                for culture in unique_cultures:
                    culture_mask = np.array(self.epoch_culture_ids) == culture
                    if np.any(culture_mask):
                        culture_weights = all_expert_weights[culture_mask[:len(all_expert_weights)]]
                        culture_mean_usage = np.mean(culture_weights, axis=0)
                        culture_expert_usage[f'culture_{int(culture)}'] = {
                            'mean_usage_per_expert': culture_mean_usage.tolist(),
                            'sample_count': int(np.sum(culture_mask)),
                            'dominant_experts': np.argsort(culture_mean_usage)[-3:].tolist()
                        }

            # 计算汇总统计
            summary = {
                'epoch': epoch + 1,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_samples': len(all_expert_weights),
                'expert_utilization': {
                    'mean_usage_per_expert': expert_usage.tolist(),
                    'usage_std_per_expert': np.std(all_expert_weights, axis=0).tolist(),
                    'gini_coefficient': float(self.compute_gini_coefficient(expert_usage)),
                    'max_usage': float(np.max(expert_usage)),
                    'min_usage': float(np.min(expert_usage)),
                    'usage_balance_ratio': float(np.min(expert_usage) / np.max(expert_usage)) if np.max(expert_usage) > 0 else 0.0
                },
                'routing_entropy': {
                    'mean': float(np.mean(all_entropies)),
                    'std': float(np.std(all_entropies)),
                    'min': float(np.min(all_entropies)),
                    'max': float(np.max(all_entropies))
                },
                'gate_distribution': gate_statistics,
                'culture_specific_routing': culture_expert_usage
            }

            # 保存汇总到routing_logs目录
            summary_file = os.path.join(self.routing_logs_dir, f'epoch_{epoch + 1}_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logging.info(f"✅ Epoch {epoch + 1} 路由汇总已保存: {summary_file}")
            logging.info(f"   - 样本总数: {len(all_expert_weights)}")
            logging.info(f"   - Gini系数: {summary['expert_utilization']['gini_coefficient']:.4f}")
            logging.info(f"   - 平均路由熵: {summary['routing_entropy']['mean']:.4f}")

            # ✅ 路由恶化检测
            routing_metrics = {
                'gini_coefficient': summary['expert_utilization']['gini_coefficient'],
                'mean_entropy': summary['routing_entropy']['mean'],
                'min_expert_usage': summary['expert_utilization']['min_usage'],
                'max_expert_usage': summary['expert_utilization']['max_usage']
            }

            alerts = self.detect_routing_degradation(epoch, routing_metrics)
            if alerts:
                self.log_routing_alerts(epoch, alerts)
            else:
                logging.info(f"✅ Epoch {epoch + 1} 路由健康状况良好，无警报")

            # 清理当前epoch的数据，为下一个epoch准备
            self.epoch_expert_weights = []
            self.epoch_gate_values = []
            self.epoch_culture_ids = []

        except Exception as e:
            logging.warning(f"保存Epoch {epoch + 1}路由汇总失败: {e}")
            # 即使失败也要清理数据
            self.epoch_expert_weights = []
            self.epoch_gate_values = []
            self.epoch_culture_ids = []

    def compute_gini_coefficient(self, values):
        """计算基尼系数"""
        values = np.array(values)
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0

    def detect_routing_degradation(self, epoch: int, current_metrics: Dict) -> List[str]:
        """
        检测路由恶化情况

        Args:
            epoch: 当前epoch
            current_metrics: 当前epoch的路由指标

        Returns:
            alerts: 检测到的警报列表
        """
        alerts = []

        # 添加当前指标到历史记录
        self.routing_history.append({
            'epoch': epoch + 1,
            'gini_coefficient': current_metrics.get('gini_coefficient', 0.0),
            'mean_entropy': current_metrics.get('mean_entropy', 0.0),
            'min_expert_usage': current_metrics.get('min_expert_usage', 0.0),
            'max_expert_usage': current_metrics.get('max_expert_usage', 0.0)
        })

        # 1. 检查Gini系数是否过高（专家利用严重不均衡）
        gini = current_metrics.get('gini_coefficient', 0.0)
        if gini > self.routing_alert_thresholds['gini_coefficient']:
            alerts.append(f"🚨 专家利用严重不均衡! Gini系数={gini:.4f} > {self.routing_alert_thresholds['gini_coefficient']}")

        # 2. 检查是否有专家塌陷（使用率过低）
        min_usage = current_metrics.get('min_expert_usage', 0.0)
        if min_usage < self.routing_alert_thresholds['expert_collapse']:
            alerts.append(f"🚨 专家塌陷风险! 最低使用率={min_usage:.4f} < {self.routing_alert_thresholds['expert_collapse']}")

        # 3. 检查路由熵是否急剧下降（路由过于确定）
        if len(self.routing_history) >= 2:
            prev_entropy = self.routing_history[-2]['mean_entropy']
            curr_entropy = current_metrics.get('mean_entropy', 0.0)
            entropy_drop = prev_entropy - curr_entropy

            if entropy_drop > self.routing_alert_thresholds['entropy_drop']:
                alerts.append(f"⚠️  路由熵急剧下降! 下降幅度={entropy_drop:.4f} > {self.routing_alert_thresholds['entropy_drop']}")

        # 4. 检查连续恶化趋势
        if len(self.routing_history) >= self.routing_alert_thresholds['consecutive_degradation']:
            recent_ginis = [h['gini_coefficient'] for h in self.routing_history[-self.routing_alert_thresholds['consecutive_degradation']:]]
            if all(recent_ginis[i] < recent_ginis[i+1] for i in range(len(recent_ginis)-1)):
                alerts.append(f"⚠️  连续{self.routing_alert_thresholds['consecutive_degradation']}个epoch Gini系数上升，可能存在持续恶化趋势")

        # 5. 检查专家使用率极化（一个专家过于占优）
        max_usage = current_metrics.get('max_expert_usage', 0.0)
        if max_usage > 0.7:  # 单个专家使用率超过70%
            alerts.append(f"⚠️  专家使用极化! 最高使用率={max_usage:.4f} > 0.7，单个专家过于占优")

        return alerts

    def log_routing_alerts(self, epoch: int, alerts: List[str]):
        """记录路由警报"""
        if not alerts:
            return

        # 记录到警报历史
        alert_record = {
            'epoch': epoch + 1,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'alerts': alerts
        }
        self.routing_alerts.append(alert_record)

        # 输出警报日志
        logging.warning("=" * 80)
        logging.warning(f"🚨 EPOCH {epoch + 1} 路由健康检查警报")
        logging.warning("=" * 80)
        for alert in alerts:
            logging.warning(f"   {alert}")
        logging.warning("")

        # 提供建议
        gini = self.routing_history[-1]['gini_coefficient'] if self.routing_history else 0.0
        entropy = self.routing_history[-1]['mean_entropy'] if self.routing_history else 0.0

        logging.warning("💡 建议的解决方案:")
        if gini > 0.5:
            logging.warning("   - 增加负载均衡权重: --load_balance_weight 0.05")
            logging.warning("   - 增加熵正则化权重: --entropy_weight 0.2")
        if entropy < 0.5:
            logging.warning("   - 增加路由器温度: --router_temperature 5.0")
            logging.warning("   - 减少路由器学习率: --router_lr_multiplier 0.05")

        logging.warning("   - 考虑调整专家分配策略")
        logging.warning("   - 监控后续epoch是否持续恶化")
        logging.warning("=" * 80)

        # 保存警报到文件
        alerts_file = os.path.join(self.args.output_dir, 'routing_alerts.json')
        with open(alerts_file, 'w', encoding='utf-8') as f:
            json.dump(self.routing_alerts, f, indent=2, ensure_ascii=False)

    def generate_routing_health_summary(self):
        """生成训练完成后的路由健康总结报告"""
        if not self.routing_history:
            logging.info("无路由历史数据，跳过健康总结")
            return

        logging.info("\n" + "=" * 80)
        logging.info("🏥 路由健康总结报告")
        logging.info("=" * 80)

        # 基础统计
        total_epochs = len(self.routing_history)
        total_alerts = len(self.routing_alerts)

        logging.info(f"📊 训练概况:")
        logging.info(f"   - 训练轮数: {total_epochs}")
        logging.info(f"   - 触发警报次数: {total_alerts}")

        # 路由指标趋势分析
        if total_epochs >= 2:
            first_epoch = self.routing_history[0]
            last_epoch = self.routing_history[-1]

            gini_change = last_epoch['gini_coefficient'] - first_epoch['gini_coefficient']
            entropy_change = last_epoch['mean_entropy'] - first_epoch['mean_entropy']

            logging.info(f"\n📈 路由指标变化:")
            logging.info(f"   - Gini系数: {first_epoch['gini_coefficient']:.4f} → {last_epoch['gini_coefficient']:.4f} ({gini_change:+.4f})")
            logging.info(f"   - 平均熵: {first_epoch['mean_entropy']:.4f} → {last_epoch['mean_entropy']:.4f} ({entropy_change:+.4f})")

            # 趋势判断
            if gini_change > 0.1:
                logging.warning(f"   ⚠️  Gini系数显著上升，专家利用不均衡加剧")
            elif gini_change < -0.05:
                logging.info(f"   ✅ Gini系数下降，专家利用更加均衡")
            else:
                logging.info(f"   ➡️  Gini系数变化平稳")

            if entropy_change < -0.5:
                logging.warning(f"   ⚠️  路由熵显著下降，可能存在专家塌陷风险")
            elif entropy_change > 0.2:
                logging.info(f"   ✅ 路由熵上升，路由多样性增加")
            else:
                logging.info(f"   ➡️  路由熵变化平稳")

        # 警报分析
        if total_alerts > 0:
            logging.info(f"\n🚨 警报分析:")
            alert_types = {}
            for alert_record in self.routing_alerts:
                for alert in alert_record['alerts']:
                    if '专家利用严重不均衡' in alert:
                        alert_types['gini_high'] = alert_types.get('gini_high', 0) + 1
                    elif '专家塌陷风险' in alert:
                        alert_types['expert_collapse'] = alert_types.get('expert_collapse', 0) + 1
                    elif '路由熵急剧下降' in alert:
                        alert_types['entropy_drop'] = alert_types.get('entropy_drop', 0) + 1
                    elif '连续' in alert and 'epoch' in alert:
                        alert_types['consecutive_degradation'] = alert_types.get('consecutive_degradation', 0) + 1
                    elif '专家使用极化' in alert:
                        alert_types['expert_polarization'] = alert_types.get('expert_polarization', 0) + 1

            for alert_type, count in alert_types.items():
                alert_name = {
                    'gini_high': '专家利用不均衡',
                    'expert_collapse': '专家塌陷风险',
                    'entropy_drop': '路由熵下降',
                    'consecutive_degradation': '连续恶化',
                    'expert_polarization': '专家极化'
                }.get(alert_type, alert_type)
                logging.info(f"   - {alert_name}: {count}次")
        else:
            logging.info(f"\n✅ 整个训练过程无路由健康警报")

        # 最终健康评级
        final_gini = self.routing_history[-1]['gini_coefficient']
        final_entropy = self.routing_history[-1]['mean_entropy']
        final_min_usage = self.routing_history[-1]['min_expert_usage']

        health_score = 0
        health_issues = []

        # 评分系统
        if final_gini < 0.2:
            health_score += 30
        elif final_gini < 0.4:
            health_score += 20
        else:
            health_issues.append(f"Gini系数过高({final_gini:.3f})")

        if final_entropy > 1.0:
            health_score += 30
        elif final_entropy > 0.5:
            health_score += 20
        else:
            health_issues.append(f"路由熵过低({final_entropy:.3f})")

        if final_min_usage > 0.05:
            health_score += 25
        elif final_min_usage > 0.02:
            health_score += 15
        else:
            health_issues.append(f"最低使用率过低({final_min_usage:.3f})")

        if total_alerts == 0:
            health_score += 15
        elif total_alerts <= 2:
            health_score += 10
        else:
            health_issues.append(f"警报次数过多({total_alerts}次)")

        # 健康等级
        if health_score >= 90:
            health_grade = "优秀 ✅"
        elif health_score >= 75:
            health_grade = "良好 ✅"
        elif health_score >= 60:
            health_grade = "一般 ⚠️"
        elif health_score >= 40:
            health_grade = "较差 ⚠️"
        else:
            health_grade = "严重 🚨"

        logging.info(f"\n🏆 最终路由健康评级: {health_grade} (评分: {health_score}/100)")

        if health_issues:
            logging.info(f"   主要问题:")
            for issue in health_issues:
                logging.info(f"   - {issue}")

        # 保存完整的路由健康报告
        health_report = {
            'summary': {
                'total_epochs': total_epochs,
                'total_alerts': total_alerts,
                'health_score': health_score,
                'health_grade': health_grade,
                'health_issues': health_issues
            },
            'final_metrics': self.routing_history[-1] if self.routing_history else {},
            'routing_history': self.routing_history,
            'alert_history': self.routing_alerts
        }

        health_report_file = os.path.join(self.args.output_dir, 'routing_health_report.json')
        with open(health_report_file, 'w', encoding='utf-8') as f:
            json.dump(health_report, f, indent=2, ensure_ascii=False)

        logging.info(f"\n📄 完整路由健康报告已保存: {health_report_file}")
        logging.info("=" * 80)

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
            culture_dim=256,  # 文化嵌入维度
            use_gate=self.args.use_gate  # 是否使用门控机制
        )

        # 冻结基础模型参数
        if self.args.freeze_base_model:
            self.freeze_base_model()

        # 移动到设备
        self.model = self.model.to(self.device)

        # LLaMA特殊内存优化
        if 'llama' in self.args.base_model_path.lower():
            logging.info("Applying LLaMA-specific memory optimizations")
            # 启用混合精度训练
            self.use_amp = True
            # 更频繁的内存清理
            self.memory_cleanup_interval = 10
        else:
            self.use_amp = False
            self.memory_cleanup_interval = 50

        # 初始化AMP scaler（在use_amp设置后）
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            logging.info("AMP scaler initialized for LLaMA model")
        else:
            self.scaler = None

        # 多GPU支持 - 基于NUM_GPUS参数决定
        import os
        total_gpu_count = torch.cuda.device_count()
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        num_gpus_requested = int(os.environ.get('NUM_GPUS', '1'))  # 从环境变量获取NUM_GPUS

        logging.info(f"System total GPUs: {total_gpu_count}")
        logging.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        logging.info(f"Requested NUM_GPUS: {num_gpus_requested}")

        # 计算实际可用GPU数量
        if cuda_visible:
            visible_gpus = [x.strip() for x in cuda_visible.split(',') if x.strip()]
            actual_gpu_count = len(visible_gpus)
        else:
            actual_gpu_count = total_gpu_count

        # 核心逻辑：只有当NUM_GPUS=2且实际可用GPU>=2时才启用DataParallel
        if num_gpus_requested == 2 and actual_gpu_count >= 2:
            logging.info(f"NUM_GPUS=2 requested and {actual_gpu_count} GPUs available, enabling DataParallel")
            try:
                self.model = nn.DataParallel(self.model)
                self.use_dataparallel = True
                logging.info("✅ DataParallel enabled successfully")
            except Exception as e:
                logging.warning(f"❌ DataParallel initialization failed: {e}")
                logging.info("Falling back to single GPU training")
                self.use_dataparallel = False
        else:
            # 所有其他情况都使用单GPU训练
            if num_gpus_requested == 2 and actual_gpu_count < 2:
                logging.warning(f"⚠️  NUM_GPUS=2 requested but only {actual_gpu_count} GPUs available")
                logging.info("Falling back to single GPU training")
            else:
                logging.info(f"Using single GPU training (NUM_GPUS={num_gpus_requested})")
            self.use_dataparallel = False

        # 最终确认DataParallel状态
        if hasattr(self.model, 'module'):
            logging.info("Model is wrapped with DataParallel")
        else:
            logging.info("Model is NOT wrapped with DataParallel")

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

        logging.info("=" * 60)
        logging.info("📋 DETAILED CULTURE EXPERT ASSIGNMENTS")
        logging.info("=" * 60)

        # 统计各类专家数量
        culture_specific_experts = 0
        generalist_experts = 0
        conflict_resolution_experts = 0

        for info in expert_info:
            expert_id = info['expert_id']
            role = info['role']
            param_count = info['param_count']
            primary_cultures = info.get('primary_cultures', [])

            if len(primary_cultures) == 0:
                conflict_resolution_experts += 1
                logging.info(f"  🔧 Expert {expert_id}: {role} ({param_count:,} params)")
                logging.info(f"      └─ Specializes in: Cultural conflict resolution")
            elif len(primary_cultures) == 6:  # 假设总共6个文化
                generalist_experts += 1
                logging.info(f"  🌍 Expert {expert_id}: {role} ({param_count:,} params)")
                logging.info(f"      └─ Covers: All cultures (cross-cultural specialist)")
            else:
                culture_specific_experts += 1
                culture_names = [f"Culture-{c}" for c in primary_cultures]
                logging.info(f"  🎯 Expert {expert_id}: {role} ({param_count:,} params)")
                logging.info(f"      └─ Specialized cultures: {', '.join(culture_names)}")

        logging.info("-" * 60)
        logging.info(f"📊 Expert Distribution Summary:")
        logging.info(f"  Culture-specific experts: {culture_specific_experts}")
        logging.info(f"  Cross-cultural generalists: {generalist_experts}")
        logging.info(f"  Conflict resolution specialists: {conflict_resolution_experts}")
        logging.info(f"  Total experts: {len(expert_info)}")

        # 检查文化覆盖情况
        culture_coverage = {i: [] for i in range(6)}  # 假设6个文化
        for info in expert_info:
            primary_cultures = info.get('primary_cultures', [])
            expert_id = info['expert_id']
            for culture_id in primary_cultures:
                if culture_id < 6:  # 确保文化ID有效
                    culture_coverage[culture_id].append(expert_id)

        logging.info("-" * 60)
        logging.info(f"🎨 Culture Coverage Analysis:")
        for culture_id, expert_ids in culture_coverage.items():
            if expert_ids:
                logging.info(f"  Culture-{culture_id}: Covered by experts {expert_ids} ({len(expert_ids)} experts)")
            else:
                logging.warning(f"  ⚠️  Culture-{culture_id}: NO DEDICATED EXPERTS!")

        logging.info("=" * 60)

    def setup_data(self):
        """设置数据"""
        logging.info(f"Loading data from {self.args.train_file}")

        with open(self.args.train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"Loaded {len(data)} samples")

        # 创建数据集
        use_mask = self.args.use_mask.lower() == 'true'
        logging.info(f"MASK mechanism: {'ENABLED' if use_mask else 'DISABLED (ABLATION STUDY)'}")
        if use_mask:
            logging.info("  - Shared experts will use instruction_mask field")
            logging.info("  - Culture experts will use instruction field")
        else:
            logging.info("  - Both shared and culture experts will use instruction field")
        dataset = CultureDataset(data, self.tokenizer, self.args.max_length, use_mask)

        # 划分训练、验证和测试集 (8:1:1)
        total_size = len(dataset)
        test_size = int(total_size * 0.1)  # 10% 测试集
        val_size = int(total_size * 0.1)   # 10% 验证集
        train_size = total_size - val_size - test_size  # 80% 训练集

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
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

        # 创建测试集数据加载器
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        logging.info(f"Train samples: {len(self.train_dataset)}")
        logging.info(f"Validation samples: {len(self.val_dataset)}")
        logging.info(f"Test samples: {len(self.test_dataset)}")
        logging.info(f"Data split ratio: {len(self.train_dataset)}:{len(self.val_dataset)}:{len(self.test_dataset)} = {len(self.train_dataset)/total_size:.1%}:{len(self.val_dataset)/total_size:.1%}:{len(self.test_dataset)/total_size:.1%}")

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
        skipped_batches = 0  # 跳过的batch数量

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备，特殊处理culture_ids_multi
            batch_device = {}
            for k, v in batch.items():
                if k == 'culture_ids_multi':
                    # culture_ids_multi是列表，不能直接.to(device)
                    batch_device[k] = v
                else:
                    batch_device[k] = v.to(self.device)
            batch = batch_device

            # 前向传播（使用AMP如果启用）
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
                        entropy_weight=self.args.entropy_weight
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
                    entropy_weight=self.args.entropy_weight
                )

            loss = outputs['loss']

            # ✅ 检查损失是否为NaN/Inf，如果是则跳过当前batch
            if torch.isnan(loss) or torch.isinf(loss):
                try:
                    loss_val = loss.item()
                except:
                    loss_val = "NaN/Inf"

                # 🔍 详细诊断有问题的batch
                # 记录NaN出现的位置
                batch_position_info = {
                    'epoch': epoch + 1,
                    'batch_idx': batch_idx,
                    'global_step': self.global_step,
                    'progress_percent': (batch_idx+1)/len(self.train_loader)*100,
                    'total_batches': len(self.train_loader)
                }
                self.nan_batch_positions.append(batch_position_info)

                logging.error("=" * 80)
                logging.error(f"🚨 PROBLEMATIC BATCH DETECTED - Step {self.global_step}, Batch {batch_idx}")
                logging.error("=" * 80)
                logging.error(f"Loss value: {loss_val}")
                logging.error(f"Batch progress: {batch_idx+1}/{len(self.train_loader)} ({(batch_idx+1)/len(self.train_loader)*100:.1f}%)")

                # 检查是否是重复位置
                if len(self.nan_batch_positions) > 1:
                    prev_positions = [pos['batch_idx'] for pos in self.nan_batch_positions[:-1]]
                    if batch_idx in prev_positions:
                        logging.error(f"⚠️  WARNING: This batch position has caused NaN before!")
                        logging.error(f"Previous NaN positions: {prev_positions}")
                        logging.error(f"This suggests a problematic data sample in the dataset!")

                # 分析batch内容
                try:
                    culture_ids = batch['culture_ids'].cpu().numpy()
                    input_ids_shape = batch['input_ids'].shape
                    labels_shape = batch['labels'].shape

                    logging.error(f"Batch size: {input_ids_shape[0]}")
                    logging.error(f"Sequence length: {input_ids_shape[1]}")
                    logging.error(f"Culture IDs in batch: {culture_ids.tolist()}")
                    logging.error(f"Unique cultures: {np.unique(culture_ids).tolist()}")

                    # 检查输入是否有异常值
                    input_ids = batch['input_ids']
                    labels = batch['labels']

                    logging.error(f"Input IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")
                    logging.error(f"Labels range: [{labels.min().item()}, {labels.max().item()}]")
                    logging.error(f"Labels unique values: {torch.unique(labels).cpu().numpy()[:10].tolist()}...")  # 只显示前10个

                    # 检查attention mask
                    attention_mask = batch['attention_mask']
                    logging.error(f"Attention mask sum per sample: {attention_mask.sum(dim=1).cpu().numpy().tolist()}")

                    # 🔍 打印batch内的具体数据内容
                    logging.error("\n" + "-" * 60)
                    logging.error("📋 DETAILED BATCH CONTENT:")
                    logging.error("-" * 60)

                    for sample_idx in range(input_ids_shape[0]):
                        logging.error(f"\n🔸 Sample {sample_idx + 1}/{input_ids_shape[0]}:")

                        # 获取当前样本的数据
                        sample_input_ids = input_ids[sample_idx].cpu().numpy()
                        sample_labels = labels[sample_idx].cpu().numpy()
                        sample_attention_mask = attention_mask[sample_idx].cpu().numpy()
                        sample_culture_id = culture_ids[sample_idx].item()

                        # 解码输入文本
                        try:
                            # 只取有效的token（attention_mask=1的部分）
                            valid_length = sample_attention_mask.sum()
                            valid_input_ids = sample_input_ids[:valid_length]
                            decoded_input = self.tokenizer.decode(valid_input_ids, skip_special_tokens=False)

                            logging.error(f"Culture ID: {sample_culture_id}")
                            logging.error(f"Valid sequence length: {valid_length}")
                            logging.error(f"Input text (first 500 chars):")
                            logging.error(f"'{decoded_input[:500]}{'...' if len(decoded_input) > 500 else ''}'")

                            # 检查labels中的有效部分（非-100的部分）
                            valid_label_mask = sample_labels != -100
                            if valid_label_mask.any():
                                valid_labels = sample_labels[valid_label_mask]
                                try:
                                    decoded_labels = self.tokenizer.decode(valid_labels, skip_special_tokens=False)
                                    logging.error(f"Target text (first 300 chars):")
                                    logging.error(f"'{decoded_labels[:300]}{'...' if len(decoded_labels) > 300 else ''}'")
                                except Exception as e:
                                    logging.error(f"Failed to decode labels: {e}")
                                    logging.error(f"Raw label tokens: {valid_labels[:20].tolist()}...")
                            else:
                                logging.error("No valid labels found (all -100)")

                            # 检查是否有异常的token
                            max_vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 50000
                            invalid_tokens = valid_input_ids[valid_input_ids >= max_vocab_size]
                            if len(invalid_tokens) > 0:
                                logging.error(f"⚠️  Found {len(invalid_tokens)} invalid tokens >= {max_vocab_size}")
                                logging.error(f"Invalid tokens: {invalid_tokens[:10].tolist()}")

                        except Exception as e:
                            logging.error(f"Failed to decode sample {sample_idx}: {e}")
                            logging.error(f"Raw input IDs (first 20): {sample_input_ids[:20].tolist()}")
                            logging.error(f"Raw labels (first 20): {sample_labels[:20].tolist()}")

                    logging.error("-" * 60)

                    # 🔍 数据质量检查摘要
                    logging.error("\n📊 DATA QUALITY SUMMARY:")
                    total_samples = input_ids_shape[0]
                    unique_culture_count = len(np.unique(culture_ids))
                    avg_seq_length = attention_mask.sum(dim=1).float().mean().item()
                    min_seq_length = attention_mask.sum(dim=1).min().item()
                    max_seq_length = attention_mask.sum(dim=1).max().item()

                    logging.error(f"Total samples in batch: {total_samples}")
                    logging.error(f"Unique cultures: {unique_culture_count}")
                    logging.error(f"Sequence lengths - Avg: {avg_seq_length:.1f}, Min: {min_seq_length}, Max: {max_seq_length}")

                    # 检查是否有极端长度的序列
                    if max_seq_length > 400:
                        logging.error(f"⚠️  Very long sequence detected: {max_seq_length} tokens")
                    if min_seq_length < 50:
                        logging.error(f"⚠️  Very short sequence detected: {min_seq_length} tokens")

                    # 检查token范围
                    input_min, input_max = input_ids.min().item(), input_ids.max().item()
                    if input_max > 200000:  # 对于大多数tokenizer来说这是异常大的
                        logging.error(f"⚠️  Unusually large token ID detected: {input_max}")

                    logging.error("-" * 60)

                    # 保存有问题的batch数据用于分析，包含具体文本内容
                    problematic_batch_file = os.path.join(self.args.output_dir, f'problematic_batch_step_{self.global_step}.json')

                    # 收集每个样本的详细信息
                    samples_data = []
                    for sample_idx in range(input_ids_shape[0]):
                        sample_input_ids = input_ids[sample_idx].cpu().numpy()
                        sample_labels = labels[sample_idx].cpu().numpy()
                        sample_attention_mask = attention_mask[sample_idx].cpu().numpy()
                        sample_culture_id = culture_ids[sample_idx].item()

                        sample_info = {
                            'sample_index': sample_idx,
                            'culture_id': sample_culture_id,
                            'valid_length': int(sample_attention_mask.sum()),
                            'input_ids': sample_input_ids.tolist(),
                            'labels': sample_labels.tolist(),
                            'attention_mask': sample_attention_mask.tolist()
                        }

                        # 尝试解码文本
                        try:
                            valid_length = sample_attention_mask.sum()
                            valid_input_ids = sample_input_ids[:valid_length]
                            decoded_input = self.tokenizer.decode(valid_input_ids, skip_special_tokens=False)
                            sample_info['decoded_input'] = decoded_input

                            # 解码labels
                            valid_label_mask = sample_labels != -100
                            if valid_label_mask.any():
                                valid_labels = sample_labels[valid_label_mask]
                                try:
                                    decoded_labels = self.tokenizer.decode(valid_labels, skip_special_tokens=False)
                                    sample_info['decoded_labels'] = decoded_labels
                                except:
                                    sample_info['decoded_labels'] = "DECODE_ERROR"
                            else:
                                sample_info['decoded_labels'] = "NO_VALID_LABELS"

                            # 检查异常token
                            max_vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 50000
                            invalid_tokens = valid_input_ids[valid_input_ids >= max_vocab_size]
                            sample_info['invalid_tokens'] = invalid_tokens.tolist() if len(invalid_tokens) > 0 else []

                        except Exception as e:
                            sample_info['decode_error'] = str(e)
                            sample_info['decoded_input'] = "DECODE_ERROR"
                            sample_info['decoded_labels'] = "DECODE_ERROR"

                        samples_data.append(sample_info)

                    batch_info = {
                        'step': self.global_step,
                        'batch_idx': batch_idx,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'batch_summary': {
                            'culture_ids': culture_ids.tolist(),
                            'input_ids_shape': list(input_ids_shape),
                            'labels_shape': list(labels_shape),
                            'input_ids_range': [input_ids.min().item(), input_ids.max().item()],
                            'labels_range': [labels.min().item(), labels.max().item()],
                            'attention_mask_sums': attention_mask.sum(dim=1).cpu().numpy().tolist(),
                        },
                        'loss_components': {
                            'generation_loss': outputs.get('generation_loss', torch.tensor(0)).item() if 'generation_loss' in outputs else "N/A",
                            'culture_loss': outputs.get('culture_loss', torch.tensor(0)).item() if 'culture_loss' in outputs else "N/A",
                            'load_balance_loss': outputs.get('load_balance_loss', torch.tensor(0)).item() if 'load_balance_loss' in outputs else "N/A",
                            'entropy_loss': outputs.get('entropy_loss', torch.tensor(0)).item() if 'entropy_loss' in outputs else "N/A"
                        },
                        'samples': samples_data
                    }

                    with open(problematic_batch_file, 'w', encoding='utf-8') as f:
                        json.dump(batch_info, f, indent=2, ensure_ascii=False)

                    logging.error(f"Problematic batch info saved to: {problematic_batch_file}")

                except Exception as e:
                    logging.error(f"Failed to analyze problematic batch: {e}")

                logging.error("=" * 80)

                skipped_batches += 1
                # 清零梯度以防止累积异常值
                self.optimizer.zero_grad()
                # 跳过当前batch，继续下一个
                self.global_step += 1
                continue

            # 调试：检查损失计算（仅前5步）
            if self.global_step <= 4:
                unique_cultures = torch.unique(batch['culture_ids'])

                # 详细损失分析
                gen_loss = outputs.get('generation_loss', torch.tensor(0)).item()
                culture_loss_val = outputs.get('culture_loss', torch.tensor(0)).item()
                load_loss = outputs.get('load_balance_loss', torch.tensor(0)).item()
                entropy_loss_val = outputs.get('entropy_loss', torch.tensor(0)).item()

                logging.info(f"🔍 Step {self.global_step} 详细损失分析:")
                logging.info(f"   Generation Loss: {gen_loss:.6f}")
                logging.info(f"   Culture Loss: {culture_loss_val:.6f}")
                logging.info(f"   Load Balance Loss: {load_loss:.6f}")
                logging.info(f"   Entropy Loss: {entropy_loss_val:.6f}")
                logging.info(f"   Total Loss: {loss.item():.6f}")
                logging.info(f"   Cultures: {unique_cultures.tolist()}")

                # 检查expert_weights
                if 'expert_weights' in outputs:
                    expert_weights = outputs['expert_weights']
                    logging.info(f"   Expert weights shape: {expert_weights.shape}")
                    logging.info(f"   Expert weights sum: {expert_weights.sum(dim=1).mean().item():.6f}")
                    logging.info(f"   Expert weights mean: {expert_weights.mean(dim=0).tolist()}")

                if self.global_step == 0:
                    logging.info(f"   📋 参数确认:")
                    logging.info(f"      use_culture_loss: {self.args.use_culture_loss}")
                    logging.info(f"      router_temperature: {self.args.router_temperature}")
                    logging.info(f"      load_balance_weight: {self.args.load_balance_weight}")
                    logging.info(f"      entropy_weight: {self.args.entropy_weight}")

            # 梯度累积反向传播
            original_loss = loss.clone()  # 保存原始损失用于统计
            loss = loss / self.gradient_accumulation_steps

            if self.use_amp and self.scaler:
                # 使用AMP的反向传播
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 每隔gradient_accumulation_steps步更新一次
            should_skip_batch = False  # 标记是否需要跳过当前batch
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp and self.scaler:
                    # AMP优化器步骤
                    self.scaler.unscale_(self.optimizer.optimizer)

                    # 对路由器参数进行更严格的梯度裁剪
                    router_params = []
                    other_params = []
                    for name, param in self.model.named_parameters():
                        if 'router' in name.lower():
                            router_params.append(param)
                        else:
                            other_params.append(param)

                    # ✅ 检查梯度中的NaN/Inf (AMP版本) - 跳过策略
                    has_nan_grad = False
                    for param in router_params + other_params:
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break  # 发现异常梯度就停止检查

                    if has_nan_grad:
                        logging.warning(f"⚠️  Step {self.global_step}: Detected NaN/Inf gradients, skipping optimizer step")
                        skipped_batches += 1
                        # 清零梯度但跳过优化器步骤
                        self.optimizer.zero_grad()
                        # 标记需要跳过当前batch
                        should_skip_batch = True
                    else:
                        # 只有在没有NaN梯度时才执行正常的优化步骤
                        if router_params:
                            torch.nn.utils.clip_grad_norm_(router_params, max_norm=0.5)  # 路由器更严格
                        if other_params:
                            torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)   # 其他参数正常

                        self.scaler.step(self.optimizer.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                else:
                    # 标准优化器步骤
                    # 对路由器参数进行更严格的梯度裁剪
                    router_params = []
                    other_params = []
                    for name, param in self.model.named_parameters():
                        if 'router' in name.lower():
                            router_params.append(param)
                        else:
                            other_params.append(param)

                    # ✅ 检查梯度中的NaN/Inf (标准版本) - 跳过策略
                    has_nan_grad = False
                    for param in router_params + other_params:
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break  # 发现异常梯度就停止检查

                    if has_nan_grad:
                        logging.warning(f"⚠️  Step {self.global_step}: Detected NaN/Inf gradients, skipping optimizer step")
                        skipped_batches += 1
                        # 清零梯度但跳过优化器步骤
                        self.optimizer.zero_grad()
                        # 标记需要跳过当前batch
                        should_skip_batch = True
                    else:
                        # 只有在没有NaN梯度时才执行正常的优化步骤
                        if router_params:
                            torch.nn.utils.clip_grad_norm_(router_params, max_norm=0.5)  # 路由器更严格
                        if other_params:
                            torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)   # 其他参数正常

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

            # 检查是否需要跳过当前batch
            if should_skip_batch:
                self.global_step += 1
                continue

            # 统计
            total_loss += original_loss.item()
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

            # 收集路由信息用于epoch汇总（不保存单个batch文件）
            self.collect_routing_info_for_summary(outputs, batch)

            # 更新进度条 - 添加更多损失信息
            progress_bar.set_postfix({
                'loss': f"{original_loss.item():.4f}",
                'gen': f"{outputs.get('generation_loss', torch.tensor(0)).item():.4f}",
                'cult': f"{outputs.get('culture_loss', torch.tensor(0)).item():.4f}",
                'load': f"{outputs.get('load_balance_loss', torch.tensor(0)).item():.4f}",
                'ent': f"{outputs.get('entropy_loss', torch.tensor(0)).item():.4f}"
            })

            # 专家使用情况监控（每50步记录一次）
            if self.global_step % 50 == 0 and 'expert_weights' in outputs:
                try:
                    expert_weights = outputs['expert_weights']  # [batch_size, num_experts]
                    if expert_weights is not None:
                        # 计算每个专家的平均使用权重
                        expert_usage = expert_weights.mean(dim=0).cpu().numpy()  # [num_experts]
                        usage_str = ', '.join([f"E{i}:{usage:.3f}" for i, usage in enumerate(expert_usage)])
                        logging.info(f"Step {self.global_step} - Expert Usage: {usage_str}")

                        # 检查专家使用是否均衡
                        usage_std = expert_usage.std()
                        usage_max = expert_usage.max()
                        usage_min = expert_usage.min()
                        if usage_std > 0.3:  # 标准差过大表示不均衡
                            logging.warning(f"Expert usage imbalance detected: std={usage_std:.3f}, max={usage_max:.3f}, min={usage_min:.3f}")
                except Exception as e:
                    logging.debug(f"Failed to monitor expert usage: {e}")

            # 清理内存（使用动态间隔）
            if self.global_step % self.memory_cleanup_interval == 0:
                torch.cuda.empty_cache()

        # Epoch结束时的专家使用情况汇总
        logging.info(f"\n=== Epoch {epoch+1} Training Summary ===")

        # 批次处理统计
        total_attempted_batches = num_batches + skipped_batches
        success_rate = (num_batches / total_attempted_batches * 100) if total_attempted_batches > 0 else 0
        logging.info(f"Batch Processing: {num_batches} successful, {skipped_batches} skipped ({success_rate:.1f}% success rate)")

        if num_batches > 0:
            logging.info(f"Total Loss: {total_loss / num_batches:.6f}")
            logging.info(f"  ├─ Generation Loss: {total_generation_loss / num_batches:.6f}")
            logging.info(f"  ├─ Culture Loss: {total_culture_loss / num_batches:.6f}")
            logging.info(f"  ├─ Load Balance Loss: {total_load_balance_loss / num_batches:.6f}")
            logging.info(f"  ├─ Entropy Loss: {total_entropy_loss / num_batches:.6f}")
            logging.info(f"  ├─ Specialization Loss: {total_specialization_loss / num_batches:.6f}")
            logging.info(f"  └─ Diversity Loss: {total_diversity_loss / num_batches:.6f}")
        else:
            logging.error("⚠️  All batches were skipped due to NaN/Inf issues!")

        # 负载均衡和熵损失的健康检查
        if num_batches > 0:
            avg_load_loss = total_load_balance_loss / num_batches
            avg_entropy_loss = total_entropy_loss / num_batches
        else:
            avg_load_loss = 0.0
            avg_entropy_loss = 0.0

        if avg_load_loss > 0.5:
            logging.warning(f"⚠️  High load balance loss ({avg_load_loss:.4f}) - Expert usage may be imbalanced")
        elif avg_load_loss < 0.001:
            logging.info(f"✅ Good load balance loss ({avg_load_loss:.4f}) - Experts are well balanced")

        if avg_entropy_loss > 0.3:
            logging.warning(f"⚠️  High entropy loss ({avg_entropy_loss:.4f}) - Router decisions may be uncertain")
        elif avg_entropy_loss < 0.1:
            logging.info(f"✅ Good entropy loss ({avg_entropy_loss:.4f}) - Router decisions are confident")

        # 保存epoch路由汇总信息
        self.save_epoch_routing_summary(epoch)

        # 🔍 NaN位置分析报告
        if skipped_batches > 0:
            self.analyze_nan_patterns(epoch)

        # 安全的平均值计算
        safe_avg = lambda x: x / num_batches if num_batches > 0 else 0.0

        return {
            'train_loss': safe_avg(total_loss),
            'train_generation_loss': safe_avg(total_generation_loss),
            'train_culture_loss': safe_avg(total_culture_loss),
            'train_load_balance_loss': safe_avg(total_load_balance_loss),
            'train_entropy_loss': safe_avg(total_entropy_loss),
            'train_specialization_loss': safe_avg(total_specialization_loss),
            'train_diversity_loss': safe_avg(total_diversity_loss),
            'skipped_batches': skipped_batches,  # 添加跳过的batch统计
            'success_rate': (num_batches / (num_batches + skipped_batches) * 100) if (num_batches + skipped_batches) > 0 else 0.0
        }

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """评估模型"""
        try:
            logging.info(f"Starting evaluation for epoch {epoch+1}")
            self.model.eval()

            total_loss = 0.0
            total_generation_loss = 0.0
            total_culture_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            eval_skipped_batches = 0  # 评估时跳过的batch数量

            generated_answers = []

            # 检查验证集大小
            logging.info(f"Validation dataset size: {len(self.val_dataset)}")
            logging.info(f"Validation dataloader batches: {len(self.val_loader)}")

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                    try:
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

                        # ✅ 检查评估损失是否为NaN/Inf，如果是则跳过当前batch
                        if torch.isnan(loss) or torch.isinf(loss):
                            try:
                                loss_val = loss.item()
                            except:
                                loss_val = "NaN/Inf"
                            logging.warning(f"⚠️  Evaluation batch {batch_idx}: Loss is NaN/Inf ({loss_val}), skipping batch")
                            eval_skipped_batches += 1
                            continue

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

                        # 生成答案示例（前5个batch，但每批减少样本数）
                        if len(generated_answers) < 5 and batch_idx < 5:
                            try:
                                for i in range(min(1, batch['input_ids'].size(0))):  # 每批只取1个样本
                                    input_text = self.tokenizer.decode(
                                        batch['input_ids'][i],
                                        skip_special_tokens=True
                                    )

                                    # 安全获取expert_weights
                                    if 'expert_weights' in outputs:
                                        expert_weights = outputs['expert_weights'][i].cpu().tolist()
                                    else:
                                        expert_weights = []

                                    generated_answers.append({
                                        'input': input_text,
                                        'culture_id': batch['culture_ids'][i].item(),
                                        'expert_weights': expert_weights
                                    })
                            except Exception as e:
                                logging.warning(f"Failed to collect sample for batch {batch_idx}: {e}")

                        # LLaMA内存优化：每5个batch清理一次内存
                        if 'llama' in str(self.args.base_model_path).lower() and batch_idx % 5 == 0:
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logging.error(f"Error processing evaluation batch {batch_idx}: {e}")
                        # 继续处理下一个batch，不中断评估
                        continue

            # 保存生成的答案
            try:
                answers_file = os.path.join(self.args.output_dir, 'generated_answers.json')
                with open(answers_file, 'w', encoding='utf-8') as f:
                    json.dump(generated_answers, f, indent=2, ensure_ascii=False)
                logging.info(f"Saved {len(generated_answers)} generated answer samples")
            except Exception as e:
                logging.warning(f"Failed to save generated answers: {e}")

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            # 评估批次统计
            total_eval_batches = len(self.val_loader)
            successful_eval_batches = total_eval_batches - eval_skipped_batches
            eval_success_rate = (successful_eval_batches / total_eval_batches * 100) if total_eval_batches > 0 else 0.0

            if eval_skipped_batches > 0:
                logging.warning(f"⚠️  Evaluation: {eval_skipped_batches} batches skipped due to NaN/Inf losses ({eval_success_rate:.1f}% success rate)")

            if successful_eval_batches > 0:
                avg_loss = total_loss / successful_eval_batches
                avg_gen_loss = total_generation_loss / successful_eval_batches
                avg_culture_loss = total_culture_loss / successful_eval_batches
                logging.info(f"Evaluation completed - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
            else:
                avg_loss = avg_gen_loss = avg_culture_loss = float('inf')
                logging.error("⚠️  All evaluation batches were skipped due to NaN/Inf issues!")

            return {
                'eval_loss': avg_loss,
                'eval_generation_loss': avg_gen_loss,
                'eval_culture_loss': avg_culture_loss,
                'eval_accuracy': accuracy,
                'eval_samples': len(self.val_dataset),
                'eval_skipped_batches': eval_skipped_batches,
                'eval_success_rate': eval_success_rate
            }

        except Exception as e:
            logging.error(f"Critical error during evaluation: {e}")
            import traceback
            traceback.print_exc()

            # 返回默认值以避免训练中断
            return {
                'eval_loss': float('inf'),
                'eval_generation_loss': float('inf'),
                'eval_culture_loss': 0.0,
                'eval_accuracy': 0.0,
                'eval_samples': len(self.val_dataset) if hasattr(self, 'val_dataset') else 0
            }

    def test_model_on_test_set(self) -> Dict[str, float]:
        """在测试集上评估最佳模型"""
        try:
            logging.info("=" * 60)
            logging.info("🧪 TESTING BEST MODEL ON TEST SET")
            logging.info("=" * 60)

            # 清理当前训练模型释放GPU内存
            if hasattr(self, 'model'):
                del self.model
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("Cleared training model from GPU memory")

            # 1. 重新加载基础模型
            logging.info("Loading base model for testing...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model_path,
                torch_dtype=torch.float16,
                device_map=None,
                trust_remote_code=True
            )

            # 2. 加载LoRA权重（如果有）
            if self.args.lora_weights_path:
                logging.info(f"Loading LoRA weights from {self.args.lora_weights_path}")
                from peft import PeftModel
                base_model = PeftModel.from_pretrained(base_model, self.args.lora_weights_path)
                base_model = base_model.merge_and_unload()
                logging.info("LoRA weights merged successfully")

            # 3. 创建MoE参数
            moe_args = ModelArgs(
                num_experts=self.args.num_experts,
                shared_hidden_dim=self.args.shared_hidden_dim,
                router_hidden_dim=self.args.router_hidden_dim,
                experts_hidden_dim=self.args.experts_hidden_dim,
                lora_rank=self.args.moe_lora_rank,
                dropout=self.args.dropout
            )

            # 4. 创建新的增强CultureMoE模型
            test_model = EnhancedCultureMoE(
                llama_model=base_model,
                config=base_model.config,
                args=moe_args,
                culture_loss_lambda=self.args.culture_loss_lambda,
                moe_fusion=self.args.moe_fusion,
                num_cultures=6,
                culture_dim=256,
                use_gate=self.args.use_gate
            )

            # 5. 加载最佳MoE权重
            best_moe_path = os.path.join(self.args.output_dir, 'best_enhanced_moe', 'moe_weights.pth')
            if not os.path.exists(best_moe_path):
                logging.error(f"Best MoE weights not found at {best_moe_path}")
                return {
                    'test_loss': float('inf'),
                    'test_generation_loss': float('inf'),
                    'test_culture_loss': 0.0,
                    'test_accuracy': 0.0,
                    'test_samples': len(self.test_dataset),
                    'error': 'Best MoE weights not found'
                }

            logging.info(f"Loading best MoE weights from {best_moe_path}")
            moe_state_dict = torch.load(best_moe_path, map_location='cpu')

            # 6. 加载MoE权重到模型
            try:
                # 使用 load_state_dict 方法加载权重（更安全）
                missing_keys, unexpected_keys = test_model.load_state_dict(moe_state_dict, strict=False)

                if missing_keys:
                    logging.warning(f"Missing keys when loading MoE weights: {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Unexpected keys in MoE weights: {unexpected_keys}")

                logging.info(f"Successfully loaded MoE weights. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

            except Exception as e:
                logging.error(f"Failed to load MoE weights using load_state_dict: {e}")
                # 回退到手动加载方法
                missing_keys = []
                unexpected_keys = []
                for name, param in moe_state_dict.items():
                    try:
                        # 获取模型中对应的参数
                        model_param = test_model
                        for attr in name.split('.'):
                            model_param = getattr(model_param, attr)

                        # 加载权重 - 确保设备和数据类型匹配
                        if model_param.device != self.device:
                            model_param.data = param.to(self.device, dtype=model_param.dtype)
                        else:
                            model_param.data = param.to(model_param.device, dtype=model_param.dtype)
                        logging.debug(f"Manually loaded parameter: {name}")
                    except Exception as param_e:
                        missing_keys.append(name)
                        logging.warning(f"Failed to load parameter {name}: {param_e}")

                if missing_keys:
                    logging.warning(f"Missing keys in manual loading: {missing_keys}")
                logging.info(f"Fallback manual loading completed. Missing: {len(missing_keys)}")

            # 7. 移动模型到设备并设置为评估模式
            test_model = test_model.to(self.device)

            # 测试时不使用DataParallel，确保简单性和兼容性
            if hasattr(test_model, 'module'):
                test_model = test_model.module

            test_model.eval()

            logging.info("Successfully loaded best MoE weights for testing")

            # 8. 在测试集上进行评估
            logging.info(f"Testing on {len(self.test_dataset)} samples...")

            total_loss = 0.0
            total_generation_loss = 0.0
            total_culture_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            test_expert_weights = []  # 收集专家权重用于分析
            test_culture_ids = []     # 收集文化ID

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
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
                        outputs = test_model(
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
                            entropy_weight=self.args.entropy_weight
                        )

                        loss = outputs['loss']
                        logits = outputs['logits']

                        # ✅ 检查测试损失是否为NaN/Inf，如果是则跳过当前batch
                        if torch.isnan(loss) or torch.isinf(loss):
                            try:
                                loss_val = loss.item()
                            except:
                                loss_val = "NaN/Inf"
                            logging.warning(f"⚠️  Test batch {batch_idx}: Loss is NaN/Inf ({loss_val}), skipping batch")
                            continue

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

                        # 收集专家权重和文化ID用于分析
                        if 'expert_weights' in outputs:
                            test_expert_weights.append(outputs['expert_weights'].cpu().numpy())
                        test_culture_ids.extend(batch['culture_ids'].cpu().numpy().tolist())

                    except Exception as e:
                        logging.error(f"Error processing test batch {batch_idx}: {e}")
                        continue

            # 9. 计算最终指标
            test_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            test_results = {
                'test_loss': total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0,
                'test_generation_loss': total_generation_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0,
                'test_culture_loss': total_culture_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0,
                'test_accuracy': test_accuracy,
                'test_samples': len(self.test_dataset)
            }

            # 10. 分析测试集上的专家使用情况
            if test_expert_weights:
                all_test_expert_weights = np.concatenate(test_expert_weights, axis=0)
                test_expert_usage = np.mean(all_test_expert_weights, axis=0)

                test_routing_analysis = {
                    'expert_usage_on_test': test_expert_usage.tolist(),
                    'gini_coefficient_test': float(self.compute_gini_coefficient(test_expert_usage)),
                    'usage_balance_ratio_test': float(np.min(test_expert_usage) / np.max(test_expert_usage)) if np.max(test_expert_usage) > 0 else 0.0,
                    'culture_distribution_test': {f'culture_{c}': test_culture_ids.count(c) for c in set(test_culture_ids)}
                }

                # 保存测试集路由分析
                test_routing_file = os.path.join(self.args.output_dir, 'test_set_routing_analysis.json')
                with open(test_routing_file, 'w', encoding='utf-8') as f:
                    json.dump(test_routing_analysis, f, indent=2, ensure_ascii=False)

                logging.info(f"Test set routing analysis saved to {test_routing_file}")
                logging.info(f"Test set expert usage: {[f'E{i}:{usage:.3f}' for i, usage in enumerate(test_expert_usage)]}")
                logging.info(f"Test set Gini coefficient: {test_routing_analysis['gini_coefficient_test']:.4f}")

            logging.info("=" * 60)
            logging.info("🎯 TEST RESULTS")
            logging.info("=" * 60)
            logging.info(f"Test Loss: {test_results['test_loss']:.6f}")
            logging.info(f"Test Generation Loss: {test_results['test_generation_loss']:.6f}")
            logging.info(f"Test Culture Loss: {test_results['test_culture_loss']:.6f}")
            logging.info(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
            logging.info(f"Test Samples: {test_results['test_samples']}")
            logging.info("=" * 60)

            # 清理测试模型内存
            del test_model
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("Cleared test model from GPU memory")

            return test_results

        except Exception as e:
            logging.error(f"Critical error during testing: {e}")
            import traceback
            traceback.print_exc()

            # 错误情况下也要清理内存
            try:
                if 'test_model' in locals():
                    del test_model
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass

            return {
                'test_loss': float('inf'),
                'test_generation_loss': float('inf'),
                'test_culture_loss': 0.0,
                'test_accuracy': 0.0,
                'test_samples': len(self.test_dataset) if hasattr(self, 'test_dataset') else 0,
                'error': str(e)
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
                logging.info(f"Starting evaluation for epoch {epoch+1}")
                try:
                    eval_metrics = self.evaluate(epoch)
                    logging.info(f"Evaluation completed successfully for epoch {epoch+1}")

                    # 合并指标
                    metrics = {**train_metrics, **eval_metrics, 'epoch': epoch + 1}
                    self.epoch_results.append(metrics)
                except Exception as e:
                    logging.error(f"Evaluation failed for epoch {epoch+1}: {e}")
                    # 创建默认评估指标以继续训练
                    eval_metrics = {
                        'eval_loss': float('inf'),
                        'eval_generation_loss': float('inf'),
                        'eval_culture_loss': 0.0,
                        'eval_accuracy': 0.0,
                        'eval_samples': 0
                    }
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

                # 保存最佳模型（只在评估成功时）
                if 'eval_accuracy' in eval_metrics and eval_metrics['eval_accuracy'] != float('inf'):
                    current_accuracy = eval_metrics['eval_accuracy']
                    if current_accuracy > self.best_accuracy:
                        old_best = self.best_accuracy
                        self.best_accuracy = current_accuracy
                        try:
                            self.save_model(epoch, is_best=True)
                            logging.info(f"🎉 NEW BEST MODEL SAVED!")
                            logging.info(f"   Previous best accuracy: {old_best:.4f}")
                            logging.info(f"   New best accuracy: {self.best_accuracy:.4f}")
                            logging.info(f"   Improvement: +{self.best_accuracy - old_best:.4f}")
                        except Exception as e:
                            logging.error(f"Failed to save best model: {e}")
                    else:
                        logging.info(f"Current accuracy: {current_accuracy:.4f} (Best: {self.best_accuracy:.4f})")
                else:
                    logging.warning("Evaluation failed, skipping best model check")

            # 保存epoch结果
            results_file = os.path.join(self.args.output_dir, 'epoch_eval_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.epoch_results, f, indent=2, ensure_ascii=False)

            # 不再保存最后一个epoch的模型，只保留最佳模型
            if epoch == self.args.num_epochs - 1:  # 最后一个epoch
                logging.info(f"Training completed. Best model already saved during training.")

            # 内存清理
            torch.cuda.empty_cache()
            gc.collect()

        logging.info("\nTraining completed!")
        logging.info(f"Best accuracy achieved: {self.best_accuracy:.4f}")

        # 训练完成总结
        if len(self.epoch_results) > 0:
            final_results = self.epoch_results[-1]
            logging.info(f"Final epoch results:")
            logging.info(f"  Final accuracy: {final_results.get('eval_accuracy', 'N/A'):.4f}")
            logging.info(f"  Final loss: {final_results.get('eval_loss', 'N/A'):.6f}")

            # 显示所有评估轮次的准确率
            eval_accuracies = [r.get('eval_accuracy', 0) for r in self.epoch_results if 'eval_accuracy' in r]
            if eval_accuracies:
                logging.info(f"Accuracy progression: {[f'{acc:.4f}' for acc in eval_accuracies]}")
        else:
            logging.warning("No evaluation results recorded during training")

        # ✅ 路由健康总结报告
        self.generate_routing_health_summary()

        # 🧪 在测试集上评估最佳模型
        logging.info("\n" + "="*80)
        logging.info("🧪 FINAL TESTING PHASE")
        logging.info("="*80)

        test_results = self.test_model_on_test_set()

        # 保存完整的结果（包含验证集和测试集结果）
        final_results = {
            'training_summary': {
                'num_epochs': self.args.num_epochs,
                'best_validation_accuracy': self.best_accuracy,
                'final_epoch_results': self.epoch_results[-1] if self.epoch_results else {},
                'total_training_samples': len(self.train_dataset),
                'total_validation_samples': len(self.val_dataset),
                'total_test_samples': len(self.test_dataset)
            },
            'validation_results': self.epoch_results,
            'test_results': test_results
        }

        # 保存最终结果
        final_results_file = os.path.join(self.args.output_dir, 'final_results.json')
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        logging.info(f"📄 Final results saved to {final_results_file}")

        # 对比验证集和测试集性能
        if self.epoch_results and 'test_accuracy' in test_results:
            best_val_acc = max([r.get('eval_accuracy', 0) for r in self.epoch_results])
            test_acc = test_results.get('test_accuracy', 0)

            logging.info("\n" + "="*60)
            logging.info("📊 PERFORMANCE COMPARISON")
            logging.info("="*60)
            logging.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
            logging.info(f"Test Set Accuracy: {test_acc:.4f}")
            logging.info(f"Generalization Gap: {best_val_acc - test_acc:+.4f}")

            if abs(best_val_acc - test_acc) < 0.01:
                logging.info("✅ Excellent generalization! Very small gap between validation and test.")
            elif abs(best_val_acc - test_acc) < 0.05:
                logging.info("✅ Good generalization. Reasonable gap between validation and test.")
            else:
                logging.warning("⚠️  Large generalization gap. Model may be overfitting to validation set.")

            logging.info("="*60)

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
    parser.add_argument('--use_mask', type=str, default='True', help='是否使用MASK机制 (True=共享专家使用instruction_mask, False=共享专家使用instruction)')
    parser.add_argument('--use_gate', type=str, default='True', help='是否使用GATE机制 (True=使用文化感知门控, False=不使用门控)')
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

    # 创建训练器并开始训练
    trainer = EnhancedCultureMoETrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()