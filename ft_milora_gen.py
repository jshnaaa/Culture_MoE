#!/usr/bin/env python3
"""
使用 MiLoRA (Matrix-informed Low-Rank Adaptation) 微调模型（新数据格式）

MiLoRA 是一种改进的参数高效微调方法，通过 SVD 分解原始权重矩阵来初始化
低秩适应矩阵，而不是使用随机初始化。

使用方法：
    python ft_milora_gen.py \
        --base_model_path /path/to/base_model \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --num_epochs 12 \
        --milora_r 64

数据格式（新格式）：
    {
        "instruction": "Give me the answer from 1 to 4: Do you agree with ...",
        "instruction_mask": "Give me the answer from 1 to 4: Do you agree with ... [MASK]",
        "input": "This question is for a country or language that is Arabic.",
        "output": "2",
        "label": "0"
    }

关键特性：
    1. 使用 SVD 分解初始化低秩矩阵（MiLoRA 的核心创新）
    2. 冻结主矩阵 W_p，只训练次矩阵相关的 A_m 和 B_m
    3. 使用标准语言建模损失
    4. 支持 post eval（生成答案并评估准确率）
    5. 与 LoRA 相比减少了超参数调优需求
"""

import argparse
import json
import os
import re
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sklearn.metrics import accuracy_score
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.milora import apply_milora_to_model, get_milora_state_dict


class CultureLLMNewFormatDataset(Dataset):
    """
    CultureLLM 新格式数据集

    数据格式：
    {
        "instruction": "Give me the answer from 1 to 4: ...",
        "instruction_mask": "Give me the answer from 1 to 4: ... [MASK]",
        "input": "This question is for a country or language that is Arabic.",
        "output": "2",
        "label": "0"
    }
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        Args:
            data_path: 数据文件路径
            tokenizer: Tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 新格式：instruction + input + output
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        label = item.get('label', '')

        # 构建完整的输入和输出
        # 格式：instruction + input → output
        if input_text:
            full_input = f"{instruction}\n{input_text}"
        else:
            full_input = instruction

        # 完整的文本（用于语言建模）
        # 这样模型会学习：给定 instruction + input，生成 output
        full_text = f"{full_input}\n{output_text}"

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # 创建标签（语言建模任务）
        labels = input_ids.clone()

        # 计算 input 部分的长度，将其标签设为 -100（不计算损失）
        input_encoded = self.tokenizer(
            full_input,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False
        )

        input_length = len(input_encoded['input_ids'])
        if input_length < len(labels):
            labels[:input_length] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'full_input': full_input,
            'expected_output': output_text,
            'original_item': item  # 保存原始数据用于评估
        }


def collate_fn(batch):
    """批处理函数"""
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
            input_id = torch.cat([input_id, torch.zeros(pad_len, dtype=input_id.dtype)])
            attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=attn_mask.dtype)])
            label = torch.cat([label, torch.full((pad_len,), -100, dtype=label.dtype)])

        input_ids.append(input_id)
        attention_mask.append(attn_mask)
        labels.append(label)

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
        'batch_data': batch  # 保存原始批次数据用于评估
    }


class MiLoRATrainer:
    """MiLoRA 训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 设置日志
        self.setup_logging()

        # 设置随机种子
        set_seed(42)

        # 加载tokenizer和模型
        self.tokenizer, self.model = self.load_model_and_tokenizer()

        # 应用MiLoRA
        self.apply_milora()

        # 加载数据
        self.train_dataloader, self.val_dataloader, self.val_data = self.load_data()

        # 设置优化器和调度器
        self.setup_optimizer()

        # 训练状态
        self.global_step = 0
        self.best_accuracy = 0.0
        self.epoch_results = []

    def setup_logging(self):
        """设置日志"""
        os.makedirs(self.args.output_dir, exist_ok=True)
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
        logging.info("MiLoRA Training Started")
        logging.info("=" * 60)
        logging.info(f"Arguments: {vars(self.args)}")

    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        logging.info("Loading model and tokenizer...")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True
        )

        # 移动到设备
        model = model.to(self.device)

        logging.info(f"Model loaded: {self.args.base_model_path}")
        logging.info(f"Model device: {next(model.parameters()).device}")
        logging.info(f"Model dtype: {next(model.parameters()).dtype}")

        return tokenizer, model

    def apply_milora(self):
        """应用MiLoRA到模型"""
        logging.info("Applying MiLoRA to model...")

        # 确定目标模块（与LoRA相同的模块）
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # 应用MiLoRA
        milora_info = apply_milora_to_model(
            model=self.model,
            rank=self.args.milora_r,
            target_modules=target_modules,
            dropout=self.args.milora_dropout
        )

        # 冻结非MiLoRA参数
        for name, param in self.model.named_parameters():
            if 'A_m' in name or 'B_m' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logging.info(f"MiLoRA applied successfully:")
        logging.info(f"  Converted modules: {len(milora_info['converted_modules'])}")
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logging.info(f"  MiLoRA rank: {self.args.milora_r}")

    def load_data(self):
        """加载和划分数据"""
        logging.info("Loading data...")

        # 创建数据集
        dataset = CultureLLMNewFormatDataset(
            self.args.train_file,
            self.tokenizer,
            self.args.max_length
        )

        # 划分训练和验证集
        val_size = int(len(dataset) * self.args.val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        logging.info(f"Train samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")

        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers
        )

        # 保存验证数据用于生成评估
        val_data = [dataset.data[i] for i in val_dataset.indices]

        return train_dataloader, val_dataloader, val_data

    def setup_optimizer(self):
        """设置优化器和调度器"""
        # 只优化MiLoRA参数
        milora_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            milora_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        # 学习率调度器
        num_training_steps = len(self.train_dataloader) * self.args.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps
        )

        logging.info(f"Optimizer: AdamW with {len(milora_params)} parameter groups")
        logging.info(f"Learning rate: {self.args.learning_rate}")
        logging.info(f"Weight decay: {self.args.weight_decay}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
                continue

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

            # 定期清理内存
            if self.global_step % 50 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def evaluate(self):
        """评估模型"""
        logging.info("Evaluating model...")

        self.model.eval()
        predictions = []
        ground_truths = []

        # 生成参数
        generation_config = {
            'max_new_tokens': 10,
            'do_sample': False,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                batch_data = batch['batch_data']

                for item in batch_data:
                    # 准备输入
                    full_input = item['full_input']
                    expected_output = item['expected_output']

                    # Tokenize输入
                    input_encoding = self.tokenizer(
                        full_input,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.args.max_length
                    )

                    input_ids = input_encoding['input_ids'].to(self.device)
                    attention_mask = input_encoding['attention_mask'].to(self.device)

                    # 生成回答
                    try:
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **generation_config
                        )

                        # 解码生成的文本
                        generated_ids = outputs[0][input_ids.shape[1]:]
                        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                        # 提取数字答案
                        predicted_answer = self.extract_answer(generated_text)
                        expected_answer = self.extract_answer(expected_output)

                        predictions.append(predicted_answer)
                        ground_truths.append(expected_answer)

                    except Exception as e:
                        logging.warning(f"Generation failed: {e}")
                        predictions.append("")
                        ground_truths.append(expected_output)

        # 计算准确率
        accuracy = accuracy_score(ground_truths, predictions)

        # 计算详细统计
        correct_count = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        total_count = len(predictions)

        logging.info(f"Evaluation Results:")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  Correct: {correct_count}/{total_count}")

        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'predictions': predictions,
            'ground_truths': ground_truths
        }

    def extract_answer(self, text: str) -> str:
        """从生成的文本中提取数字答案"""
        # 查找第一个数字
        import re
        numbers = re.findall(r'\d+', text.strip())
        if numbers:
            return numbers[0]
        return ""

    def save_model(self, epoch: int, accuracy: float, is_best: bool = False):
        """保存模型"""
        # 保存MiLoRA权重
        milora_state_dict = get_milora_state_dict(self.model)

        save_dir = os.path.join(self.args.output_dir, "best_milora" if is_best else f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(milora_state_dict, os.path.join(save_dir, "milora_weights.pth"))

        # 保存配置
        config = {
            'base_model_path': self.args.base_model_path,
            'milora_r': self.args.milora_r,
            'milora_dropout': self.args.milora_dropout,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'accuracy': accuracy,
            'epoch': epoch
        }

        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        logging.info(f"Model saved to {save_dir} (accuracy: {accuracy:.4f})")

    def train(self):
        """主训练循环"""
        logging.info("Starting training...")

        for epoch in range(self.args.num_epochs):
            # 训练
            avg_loss = self.train_epoch(epoch)

            # 评估
            if (epoch + 1) % self.args.eval_interval == 0:
                eval_results = self.evaluate()
                accuracy = eval_results['accuracy']

                # 记录结果
                epoch_result = {
                    'epoch': epoch + 1,
                    'avg_loss': avg_loss,
                    'eval_accuracy': accuracy,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                }
                self.epoch_results.append(epoch_result)

                # 保存最佳模型
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.save_model(epoch + 1, accuracy, is_best=True)

                logging.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, Best={self.best_accuracy:.4f}")

        # 保存训练结果
        results_file = os.path.join(self.args.output_dir, "epoch_eval_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.epoch_results, f, indent=2)

        logging.info("Training completed!")
        logging.info(f"Best accuracy: {self.best_accuracy:.4f}")

        return self.best_accuracy


def main():
    parser = argparse.ArgumentParser(description="MiLoRA Fine-tuning")

    # 基础参数
    parser.add_argument('--base_model_path', type=str, required=True, help='基础模型路径')
    parser.add_argument('--train_file', type=str, required=True, help='训练数据文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=12, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='评估批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='权重衰减')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--eval_interval', type=int, default=3, help='评估间隔')
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    # MiLoRA参数
    parser.add_argument('--milora_r', type=int, default=64, help='MiLoRA rank')
    parser.add_argument('--milora_dropout', type=float, default=0.05, help='MiLoRA dropout')

    args = parser.parse_args()

    # 创建训练器并开始训练
    trainer = MiLoRATrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()