#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单 MoE 模型训练脚本 - 支持8:1:1数据划分和完整评估
在 LoRA 微调后的完整模型基础上添加简单的 MoE 结构

功能：
1. 8:1:1划分训练集、验证集、测试集
2. 在训练集上训练，验证集上评估选择最佳模型
3. 在测试集上最终评估，输出详细结果
4. 保存最佳MoE权重和生成答案

使用方法：
    python ft_simple_moe_gen.py \
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

from llamafactory.model.simple_moe import SimpleMoEModel


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
            'original_item': item  # 保存原始数据用于评估
        }

        if self.mode == "train":
            # 创建正确的标签：将输入部分设为-100，只计算输出部分的损失
            labels = input_ids.clone()

            # 如果有明确的输出分割，只对输出部分计算损失
            if output:
                # 找到输出开始的位置
                instruction_text = instruction + "\n" + input_text if input_text else instruction
                instruction_encoding = self.tokenizer(instruction_text, add_special_tokens=False)
                instruction_len = len(instruction_encoding['input_ids'])

                # 将指令部分的标签设为-100（不计算损失）
                if instruction_len < len(labels):
                    labels[:instruction_len] = -100

            result['labels'] = labels

        return result


def split_dataset(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1,
                  test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按照8:1:1的比例划分数据集

    Args:
        data: 原始数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        train_data, val_data, test_data
    """
    # 设置随机种子确保可复现
    random.seed(seed)
    np.random.seed(seed)

    # 打乱数据
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)

    total_size = len(data_shuffled)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = data_shuffled[:train_size]
    val_data = data_shuffled[train_size:train_size + val_size]
    test_data = data_shuffled[train_size + val_size:]

    print(f"Dataset split:")
    print(f"  Total: {total_size}")
    print(f"  Train: {len(train_data)} ({len(train_data)/total_size*100:.1f}%)")
    print(f"  Val: {len(val_data)} ({len(val_data)/total_size*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/total_size*100:.1f}%)")

    return train_data, val_data, test_data


# 旧的collate_fn已移动到SimpleMoETrainer类中


class SimpleMoETrainer:
    """简单 MoE 训练器 - 支持8:1:1数据划分和完整评估"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 设置随机种子
        set_seed(42)

        # 设置日志
        self.setup_logging()

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 保存配置
        self.save_config()

        # 初始化最佳模型跟踪
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.best_moe_state_dict = None

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
        logging.info("Simple MoE Training Started")
        logging.info("=" * 80)
        logging.info(f"Arguments: {vars(self.args)}")

    def save_config(self):
        """保存训练配置"""
        config = {
            'training_time': datetime.now().isoformat(),
            'base_model_path': self.args.base_model_path,
            'lora_weights_path': self.args.lora_weights_path,
            'train_file': self.args.train_file,
            'num_experts': self.args.num_experts,
            'top_k': self.args.top_k,
            'expert_hidden_dim': self.args.expert_hidden_dim,
            'router_hidden_dim': self.args.router_hidden_dim,
            'learning_rate': self.args.learning_rate,
            'num_epochs': self.args.num_epochs,
            'batch_size': self.args.batch_size,
            'max_length': self.args.max_length,
            'device': str(self.device)
        }

        config_file = os.path.join(self.args.output_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logging.info(f"Configuration saved to {config_file}")

    def load_model_from_components(self):
        """从 Base 模型和 LoRA 权重加载完整模型"""
        logging.info("=" * 80)
        logging.info("Loading Model from Components")
        logging.info("=" * 80)
        logging.info(f"Base model: {self.args.base_model_path}")
        logging.info(f"LoRA weights: {self.args.lora_weights_path}")

        # 1. 加载 Tokenizer
        logging.info("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.args.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logging.info("   ✅ Tokenizer loaded")

        # 2. 加载 Base 模型
        logging.info("2. Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None  # 手动管理设备
        )
        logging.info("   ✅ Base model loaded")

        # 3. 加载 LoRA 权重并合并
        logging.info("3. Loading and merging LoRA weights...")
        model_with_lora = PeftModel.from_pretrained(
            base_model,
            self.args.lora_weights_path,
            is_trainable=False
        )
        logging.info("   ✅ LoRA weights loaded")

        # 合并 LoRA 权重到 base 模型
        logging.info("   Merging LoRA weights into base model...")
        merged_model = model_with_lora.merge_and_unload()
        logging.info("   ✅ LoRA weights merged")

        # 4. 创建简单 MoE 模型
        logging.info("4. Creating Simple MoE model...")
        simple_moe_model = SimpleMoEModel(
            llama_model=merged_model,
            config=merged_model.config,
            num_experts=self.args.num_experts,
            top_k=self.args.top_k,
            expert_hidden_dim=self.args.expert_hidden_dim,
            router_hidden_dim=self.args.router_hidden_dim,
            dropout=self.args.dropout
        )
        logging.info("   ✅ Simple MoE model created")

        # 5. 冻结 LLaMA 模型，只训练 MoE 组件
        logging.info("5. Freezing LLaMA model parameters...")

        # 找到最后一层的索引
        model_config = simple_moe_model.config
        num_layers = getattr(model_config, 'num_hidden_layers', 32)
        last_layer_idx = num_layers - 1

        logging.info(f"   Model has {num_layers} layers, keeping layer {last_layer_idx} trainable")

        for name, param in simple_moe_model.named_parameters():
            if 'moe_layer' in name or 'moe_fusion_weight' in name:
                param.requires_grad = True
            elif f'model.layers.{last_layer_idx}' in name:
                # 保持最后一层的梯度，确保hidden_states有梯度连接
                param.requires_grad = True
                logging.info(f"   Keeping last layer parameter trainable: {name}")
            else:
                param.requires_grad = False

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in simple_moe_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in simple_moe_model.parameters())

        logging.info(f"   Total parameters: {total_params:,}")
        logging.info(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

        # 6. 移动到设备
        logging.info(f"6. Moving model to {self.device}...")
        simple_moe_model = simple_moe_model.to(self.device)
        simple_moe_model.train()
        logging.info("   ✅ Model ready for training")

        logging.info("=" * 80)
        logging.info("✅ Model Loading Completed Successfully!")
        logging.info("=" * 80)

        return simple_moe_model, tokenizer

    def load_and_split_data(self, tokenizer):
        """加载数据并按8:1:1划分"""
        logging.info("=" * 80)
        logging.info("Loading and Splitting Data")
        logging.info("=" * 80)
        logging.info(f"Data file: {self.args.train_file}")

        # 加载原始数据
        with open(self.args.train_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        logging.info(f"Loaded {len(raw_data)} total samples")

        # 8:1:1划分数据
        train_data, val_data, test_data = split_dataset(raw_data)

        # 创建数据集
        train_dataset = SimpleMoEDataset(train_data, tokenizer, self.args.max_length, mode="train")
        val_dataset = SimpleMoEDataset(val_data, tokenizer, self.args.max_length, mode="eval")
        test_dataset = SimpleMoEDataset(test_data, tokenizer, self.args.max_length, mode="eval")

        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn_train,
            num_workers=self.args.num_workers
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_eval,
            num_workers=self.args.num_workers
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_eval,
            num_workers=self.args.num_workers
        )

        logging.info(f"DataLoaders created:")
        logging.info(f"  Train: {len(train_dataloader)} batches")
        logging.info(f"  Val: {len(val_dataloader)} batches")
        logging.info(f"  Test: {len(test_dataloader)} batches")
        logging.info("=" * 80)

        return train_dataloader, val_dataloader, test_dataloader, val_data, test_data

    def collate_fn_train(self, batch):
        """训练模式的批处理函数"""
        max_len = max([item['input_ids'].size(0) for item in batch])

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            input_id = item['input_ids']
            attn_mask = item['attention_mask']
            label = item['labels']

            # 右侧填充
            pad_len = max_len - input_id.size(0)
            if pad_len > 0:
                input_id = torch.cat([input_id, torch.full((pad_len,), 0)])  # pad_token_id = 0
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len)])
                label = torch.cat([label, torch.full((pad_len,), -100)])  # ignore_index = -100

            input_ids.append(input_id)
            attention_mask.append(attn_mask)
            labels.append(label)

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels)
        }

    def collate_fn_eval(self, batch):
        """评估模式的批处理函数"""
        max_len = max([item['input_ids'].size(0) for item in batch])

        input_ids = []
        attention_mask = []
        original_items = []

        for item in batch:
            input_id = item['input_ids']
            attn_mask = item['attention_mask']

            # 右侧填充
            pad_len = max_len - input_id.size(0)
            if pad_len > 0:
                input_id = torch.cat([input_id, torch.full((pad_len,), 0)])  # pad_token_id = 0
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len)])

            input_ids.append(input_id)
            attention_mask.append(attn_mask)
            original_items.append(item['original_item'])

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'original_items': original_items
        }

    def _extract_answer(self, text: str) -> str:
        """从生成的文本中提取答案"""
        text = text.strip()

        # 查找常见的答案模式
        # 优先查找数字答案（1-10）
        number_patterns = [
            r'答案[是：:]\s*(\d+)',
            r'选择\s*(\d+)',
            r'应该选择\s*(\d+)',
            r'正确答案[是：:]\s*(\d+)',
            r'^(\d+)[\.。]',
            r'\b(\d+)\b'
        ]

        for pattern in number_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(1)
                # 确保数字在合理范围内（1-10）
                if number.isdigit() and 1 <= int(number) <= 10:
                    return number

        # 查找字母答案（A-D）作为备用
        letter_patterns = [
            r'答案[是：:]\s*([A-D])',
            r'选择\s*([A-D])',
            r'应该选择\s*([A-D])',
            r'正确答案[是：:]\s*([A-D])',
            r'^([A-D])[\.。]',
            r'\b([A-D])\b'
        ]

        for pattern in letter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 如果没有找到明确的答案，优先查找单独的数字
        for char in text:
            if char.isdigit() and 1 <= int(char) <= 10:
                return char

        # 查找单独的字母作为备用
        for char in text:
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()

        # 默认返回1（如果完全没有找到）
        return '1'

    def _is_answer_correct(self, predicted_answer: str, true_answer: str) -> bool:
        """检查预测答案是否正确，支持逗号分隔的多个正确答案"""
        predicted = predicted_answer.strip()
        true_answer = true_answer.strip()

        # 如果true_answer包含逗号，说明有多个正确答案
        if ',' in true_answer:
            valid_answers = [ans.strip() for ans in true_answer.split(',')]
            return predicted in valid_answers
        else:
            # 单一答案的情况
            return predicted == true_answer

    def evaluate_model(self, model, dataloader, tokenizer, data_samples, mode="val"):
        """评估模型"""
        logging.info(f"Starting {mode} evaluation...")

        model.eval()
        predictions = []
        true_labels = []
        generated_answers = []

        # 评估参数 - 优化为简洁回答
        max_length = 512
        max_new_tokens = 5  # 只够回答数字

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{mode.upper()} Eval")):
                try:
                    # 移动数据到设备
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    original_items = batch['original_items']

                    # 生成回答
                    with torch.amp.autocast('cuda'):
                        generate_kwargs = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'max_new_tokens': max_new_tokens,
                            'do_sample': False,  # 使用贪婪解码，更确定性
                            'temperature': 1.0,
                            'pad_token_id': tokenizer.eos_token_id,
                            'eos_token_id': tokenizer.eos_token_id,
                            'repetition_penalty': 1.0,
                            'length_penalty': 0.0,
                            'early_stopping': True,
                        }

                        outputs = model.generate(**generate_kwargs)

                    # 处理每个样本
                    for i in range(len(original_items)):
                        original_item = original_items[i]

                        # 解码生成的文本
                        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)

                        # 提取原始prompt长度
                        prompt = batch['input_ids'][i]
                        prompt_text = tokenizer.decode(prompt, skip_special_tokens=True)

                        # 提取生成的答案部分
                        if len(generated_text) > len(prompt_text):
                            new_content = generated_text[len(prompt_text):].strip()
                        else:
                            new_content = ""

                        # 提取答案
                        pred_answer = self._extract_answer(new_content)
                        true_answer = original_item['output'].strip()

                        # 检查正确性
                        is_correct = self._is_answer_correct(pred_answer, true_answer)

                        predictions.append(pred_answer)
                        true_labels.append(true_answer)

                        # 保存详细结果
                        answer_data = {
                            'index': batch_idx * self.args.batch_size + i,
                            'question': original_item.get('instruction', ''),
                            'input_text': original_item.get('input', ''),
                            'true_answer': true_answer,
                            'generated_text': new_content,
                            'predicted_answer': pred_answer,
                            'is_correct': is_correct
                        }
                        generated_answers.append(answer_data)

                except Exception as e:
                    logging.warning(f"Error in batch {batch_idx}: {e}")
                    # 添加默认答案
                    for i in range(len(batch['original_items'])):
                        predictions.append("1")
                        true_labels.append(batch['original_items'][i]['output'].strip())
                        generated_answers.append({
                            'index': batch_idx * self.args.batch_size + i,
                            'question': batch['original_items'][i].get('instruction', ''),
                            'input_text': batch['original_items'][i].get('input', ''),
                            'true_answer': batch['original_items'][i]['output'].strip(),
                            'generated_text': f"Error: {str(e)}",
                            'predicted_answer': "1",
                            'is_correct': False
                        })

        # 计算评估指标
        correct_count = sum(1 for answer in generated_answers if answer['is_correct'])
        total_count = len(generated_answers)
        accuracy = correct_count / total_count if total_count > 0 else 0

        # 计算其他指标
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='weighted', zero_division=0
            )
        except Exception:
            precision = recall = f1 = 0.0

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'total_samples': total_count,
            'correct_predictions': int(correct_count),
            'evaluation_time': datetime.now().isoformat()
        }

        logging.info(f"{mode.upper()} evaluation completed:")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  Total samples: {total_count}")
        logging.info(f"  Correct predictions: {correct_count}")

        model.train()  # 恢复训练模式
        return results, generated_answers

    def train_with_validation(self, model, train_dataloader, val_dataloader, val_data, tokenizer):
        """训练模型 - 支持验证集评估和最佳模型选择"""
        logging.info("=" * 80)
        logging.info("Starting Training with Validation")
        logging.info("=" * 80)

        # 优化器
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        # 训练循环
        model.train()
        global_step = 0
        epoch_results = []

        for epoch in range(self.args.num_epochs):
            # 训练阶段
            epoch_loss = 0
            epoch_steps = 0

            logging.info(f"\\nEpoch {epoch+1}/{self.args.num_epochs} - Training")
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

            model.train()
            for batch in progress_bar:
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )

                loss = outputs['loss']

                # 检查损失是否为NaN或无穷大
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
                    # 清理优化器状态，防止NaN传播
                    optimizer.zero_grad()
                    # 强制垃圾回收
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪 - 更严格的裁剪防止数值爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)

                optimizer.step()

                # 记录损失
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # 定期健康检查
                if global_step % 50 == 0 and hasattr(model, 'check_model_health'):
                    health = model.check_model_health()
                    if health['has_nan_params'] or health['has_inf_params']:
                        logging.warning(f"Model health issues detected at step {global_step}")

                # 更新进度条
                current_avg_loss = epoch_loss / epoch_steps
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{current_avg_loss:.4f}'
                })

                # 检查平均损失是否异常
                if epoch_steps > 10 and (torch.isnan(torch.tensor(current_avg_loss)) or current_avg_loss > 100):
                    logging.warning(f"Abnormal average loss detected: {current_avg_loss}, at step {global_step}")
                    # 记录梯度信息
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    logging.info(f"Total gradient norm: {total_norm}")

                # 定期清理内存
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()

            # 计算 epoch 平均损失
            avg_epoch_loss = epoch_loss / epoch_steps

            # 验证阶段
            logging.info(f"Epoch {epoch+1}/{self.args.num_epochs} - Validation")
            val_results, val_answers = self.evaluate_model(model, val_dataloader, tokenizer, val_data, mode="val")
            val_accuracy = val_results['accuracy']

            # 检查是否是最佳模型
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch + 1

                # 保存最佳MoE权重
                self.best_moe_state_dict = {}
                for name, param in model.named_parameters():
                    if 'moe_layer' in name or 'moe_fusion_weight' in name:
                        self.best_moe_state_dict[name] = param.cpu().clone().detach().to(torch.float16)

                logging.info(f"  🎉 New best model! Val accuracy: {val_accuracy:.4f}")

                # 保存最佳模型检查点
                self.save_best_model_checkpoint(model, tokenizer, epoch + 1, val_accuracy, val_answers)
            else:
                logging.info(f"  Val accuracy: {val_accuracy:.4f} (Best: {self.best_val_accuracy:.4f} at epoch {self.best_epoch})")

            # 记录 epoch 结果
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'val_accuracy': val_accuracy,
                'val_precision': val_results['precision'],
                'val_recall': val_results['recall'],
                'val_f1': val_results['f1_score'],
                'is_best': val_accuracy > self.best_val_accuracy,
                'timestamp': datetime.now().isoformat()
            }
            epoch_results.append(epoch_result)

            logging.info(f"Epoch {epoch+1} Summary:")
            logging.info(f"  Train Loss: {avg_epoch_loss:.4f}")
            logging.info(f"  Val Accuracy: {val_accuracy:.4f}")
            logging.info(f"  Best Val Accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")

        # 保存完整训练结果
        results_file = os.path.join(self.args.output_dir, 'training_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)

        logging.info("=" * 80)
        logging.info("✅ Training Completed!")
        logging.info(f"Best validation accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}")
        logging.info(f"Training results saved to: {results_file}")
        logging.info("=" * 80)

        return epoch_results

    def save_best_model_checkpoint(self, model, tokenizer, epoch, accuracy, val_answers):
        """保存最佳模型检查点"""
        best_dir = os.path.join(self.args.output_dir, 'best_model')
        os.makedirs(best_dir, exist_ok=True)

        # 保存最佳MoE权重
        moe_weights_file = os.path.join(best_dir, 'moe_weights.pth')
        torch.save(self.best_moe_state_dict, moe_weights_file)

        # 保存最佳模型信息
        best_info = {
            'epoch': epoch,
            'val_accuracy': accuracy,
            'num_experts': self.args.num_experts,
            'top_k': self.args.top_k,
            'expert_hidden_dim': self.args.expert_hidden_dim,
            'router_hidden_dim': self.args.router_hidden_dim,
            'dropout': self.args.dropout,
            'timestamp': datetime.now().isoformat()
        }

        info_file = os.path.join(best_dir, 'model_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(best_info, f, indent=2, ensure_ascii=False)

        # 保存验证集生成答案
        val_answers_file = os.path.join(best_dir, 'val_generated_answers.json')
        with open(val_answers_file, 'w', encoding='utf-8') as f:
            json.dump(val_answers, f, indent=2, ensure_ascii=False)

        logging.info(f"  Best model saved to: {best_dir}")

    def final_test_evaluation(self, model, test_dataloader, test_data, tokenizer):
        """最终测试集评估"""
        logging.info("=" * 80)
        logging.info("Final Test Evaluation")
        logging.info("=" * 80)

        # 加载最佳MoE权重
        if self.best_moe_state_dict is not None:
            logging.info("Loading best MoE weights for final evaluation...")

            # 创建当前MoE权重的备份
            current_moe_state = {}
            for name, param in model.named_parameters():
                if 'moe_layer' in name or 'moe_fusion_weight' in name:
                    current_moe_state[name] = param.cpu().clone().detach()

            # 加载最佳权重
            for name, param in model.named_parameters():
                if name in self.best_moe_state_dict:
                    param.data = self.best_moe_state_dict[name].to(param.device).to(param.dtype)

            logging.info(f"✅ Best MoE weights loaded (from epoch {self.best_epoch})")

        # 在测试集上评估
        test_results, test_answers = self.evaluate_model(model, test_dataloader, tokenizer, test_data, mode="test")

        # 保存测试结果
        test_results_file = os.path.join(self.args.output_dir, 'test_results.json')
        with open(test_results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)

        # 保存测试集生成答案
        test_answers_file = os.path.join(self.args.output_dir, 'generated_answers.json')
        with open(test_answers_file, 'w', encoding='utf-8') as f:
            json.dump(test_answers, f, indent=2, ensure_ascii=False)

        # 保存最佳MoE权重（最终版本）
        final_moe_dir = os.path.join(self.args.output_dir, 'final_moe_weights')
        os.makedirs(final_moe_dir, exist_ok=True)

        final_moe_file = os.path.join(final_moe_dir, 'moe_weights.pth')
        torch.save(self.best_moe_state_dict, final_moe_file)

        # 保存最终模型配置
        final_config = {
            'model_type': 'Simple MoE',
            'base_model_path': self.args.base_model_path,
            'lora_weights_path': self.args.lora_weights_path,
            'num_experts': self.args.num_experts,
            'top_k': self.args.top_k,
            'expert_hidden_dim': self.args.expert_hidden_dim,
            'router_hidden_dim': self.args.router_hidden_dim,
            'dropout': self.args.dropout,
            'best_epoch': self.best_epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'test_accuracy': test_results['accuracy'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'test_f1': test_results['f1_score'],
            'timestamp': datetime.now().isoformat()
        }

        config_file = os.path.join(final_moe_dir, 'model_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(final_config, f, indent=2, ensure_ascii=False)

        logging.info("=" * 80)
        logging.info("✅ Final Test Evaluation Completed!")
        logging.info("=" * 80)
        logging.info(f"Best model from epoch {self.best_epoch}:")
        logging.info(f"  Validation accuracy: {self.best_val_accuracy:.4f}")
        logging.info(f"  Test accuracy: {test_results['accuracy']:.4f}")
        logging.info(f"  Test precision: {test_results['precision']:.4f}")
        logging.info(f"  Test recall: {test_results['recall']:.4f}")
        logging.info(f"  Test F1: {test_results['f1_score']:.4f}")
        logging.info("")
        logging.info("Files saved:")
        logging.info(f"  Test results: {test_results_file}")
        logging.info(f"  Generated answers: {test_answers_file}")
        logging.info(f"  Final MoE weights: {final_moe_file}")
        logging.info(f"  Model config: {config_file}")
        logging.info("=" * 80)

        return test_results, test_answers

    def run_training_with_evaluation(self):
        """运行完整的训练和评估流程 - 8:1:1数据划分"""
        start_time = time.time()

        try:
            # 1. 加载模型
            model, tokenizer = self.load_model_from_components()

            # 2. 加载和划分数据
            train_dataloader, val_dataloader, test_dataloader, val_data, test_data = self.load_and_split_data(tokenizer)

            # 3. 训练模型（带验证集评估和最佳模型选择）
            training_results = self.train_with_validation(model, train_dataloader, val_dataloader, val_data, tokenizer)

            # 4. 最终测试集评估
            test_results, test_answers = self.final_test_evaluation(model, test_dataloader, test_data, tokenizer)

            end_time = time.time()
            training_time = end_time - start_time

            logging.info("\\n" + "=" * 80)
            logging.info("🎉 Simple MoE Training and Evaluation Completed Successfully!")
            logging.info("=" * 80)
            logging.info(f"Total time: {training_time:.2f} seconds")
            logging.info("")
            logging.info("📊 Final Results Summary:")
            logging.info(f"  Best validation accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")
            logging.info(f"  Final test accuracy: {test_results['accuracy']:.4f}")
            logging.info(f"  Final test precision: {test_results['precision']:.4f}")
            logging.info(f"  Final test recall: {test_results['recall']:.4f}")
            logging.info(f"  Final test F1: {test_results['f1_score']:.4f}")
            logging.info("")
            logging.info("📁 Output Files:")
            logging.info(f"  Training results: {self.args.output_dir}/training_results.json")
            logging.info(f"  Test results: {self.args.output_dir}/test_results.json")
            logging.info(f"  Generated answers: {self.args.output_dir}/generated_answers.json")
            logging.info(f"  Best MoE weights: {self.args.output_dir}/final_moe_weights/moe_weights.pth")
            logging.info(f"  Model config: {self.args.output_dir}/final_moe_weights/model_config.json")
            logging.info("=" * 80)

            return True

        except Exception as e:
            logging.error(f"Training and evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Simple MoE Training")

    # 必需参数
    parser.add_argument('--base_model_path', type=str, required=True, help='基础模型路径')
    parser.add_argument('--lora_weights_path', type=str, required=True, help='LoRA权重路径')
    parser.add_argument('--train_file', type=str, required=True, help='训练数据文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')

    # MoE 参数
    parser.add_argument('--num_experts', type=int, default=12, help='专家数量')
    parser.add_argument('--top_k', type=int, default=2, help='Top-K 路由')
    parser.add_argument('--expert_hidden_dim', type=int, default=None, help='专家隐藏维度')
    parser.add_argument('--router_hidden_dim', type=int, default=512, help='路由器隐藏维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')

    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载器工作进程数')
    parser.add_argument('--save_interval', type=int, default=2, help='保存间隔（轮数）')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    args = parser.parse_args()

    # 创建训练器并运行
    trainer = SimpleMoETrainer(args)
    success = trainer.run_training_with_evaluation()

    if success:
        print("\n🎉 Simple MoE training and evaluation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Simple MoE training and evaluation failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()