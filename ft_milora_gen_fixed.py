#!/usr/bin/env python3
"""
使用 MiLoRA (Matrix-informed Low-Rank Adaptation) 微调模型（修复版本）

修复内容：
1. 使用修复版本的MiLoRA实现（milora_fixed.py）
2. 调整了初始化验证阈值，避免误报
3. 改进了SVD分解的数值稳定性
4. 增强了错误处理和日志记录

使用方法：
    python ft_milora_gen_fixed.py \
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

# 使用修复版本的MiLoRA
from src.llamafactory.model.milora_fixed import apply_milora_to_model, get_milora_state_dict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

        logger.info(f"Loading data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} samples")

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
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # 对于生成式模型，labels = input_ids（用于语言建模损失）
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'instruction': instruction,
            'input': input_text,
            'output': output_text,
            'label': label
        }


def load_and_process_data(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    val_split: float = 0.1
):
    """
    加载并处理数据，按 9:1 比例划分训练集和验证集

    Args:
        data_path: 数据文件路径
        tokenizer: Tokenizer
        max_length: 最大序列长度
        val_split: 验证集比例

    Returns:
        dict: 包含 'train' 和 'validation' 的字典
    """
    dataset = CultureLLMNewFormatDataset(data_path, tokenizer, max_length)

    # 按 9:1 比例划分
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    return {
        'train': train_dataset,
        'validation': val_dataset
    }


def extract_answer_from_text(text: str) -> str:
    """
    从生成的文本中提取答案

    使用正则表达式查找数字

    Args:
        text: 生成的文本

    Returns:
        提取的答案（数字字符串）
    """
    # 查找数字（1-10 或 1-4）
    match = re.search(r'\d+', text)
    if match:
        return match.group(0)

    return ""


def generate_answer(model, tokenizer, instruction: str, input_text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    """
    使用模型生成答案

    关键改进：
    - 只输入 instruction + input，不输入 output
    - 让模型生成 output

    Args:
        model: 模型
        tokenizer: tokenizer
        instruction: 指令
        input_text: 输入文本
        device: 设备
        max_new_tokens: 最大生成 token 数

    Returns:
        生成的文本
    """
    # 构建输入
    if input_text:
        full_input = f"{instruction}\n{input_text}"
    else:
        full_input = instruction

    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # 贪婪解码
            num_beams=1,      # 禁用 beam search
            repetition_penalty=1.0
        )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text


def train_epoch(model, train_loader, optimizer, device, num_accumulation_steps=1):
    """
    训练一个 epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        num_accumulation_steps: 梯度累积步数

    Returns:
        dict: 包含训练指标的字典
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", disable=False, mininterval=1.0)

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # ✅ 检查 NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"❌ NaN or Inf loss detected at batch {batch_idx}")
            continue

        # 梯度累积
        loss = loss / num_accumulation_steps
        loss.backward()

        total_loss += loss.item() * num_accumulation_steps
        num_batches += 1

        # 梯度更新
        if (batch_idx + 1) % num_accumulation_steps == 0:
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix({'loss': f"{loss.item() * num_accumulation_steps:.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'num_batches': num_batches
    }


def evaluate(model, val_loader, device):
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备

    Returns:
        dict: 包含评估指标的字典
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Evaluating", disable=False, mininterval=1.0)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'num_batches': num_batches
    }


def generate_and_evaluate_answers(model, val_dataset, tokenizer, device, output_dir, epoch=None):
    """
    在验证集上生成答案并评估准确率（Post Eval）

    Args:
        model: 模型
        val_dataset: 验证数据集
        tokenizer: tokenizer
        device: 设备
        output_dir: 输出目录
        epoch: 当前 epoch 数（用于保存文件名）

    Returns:
        dict: 包含准确率等指标的字典
    """
    model.eval()

    correct = 0
    total = 0
    generated_data = []

    for idx in tqdm(range(len(val_dataset)), desc="Generating", disable=False, mininterval=1.0):
        # 获取原始数据集（处理 Subset 对象）
        if hasattr(val_dataset, 'dataset'):
            # val_dataset 是 Subset 对象
            original_idx = val_dataset.indices[idx]
            sample = val_dataset.dataset[original_idx]
        else:
            # val_dataset 是普通 Dataset 对象
            sample = val_dataset[idx]

        instruction = sample['instruction']
        input_text = sample['input']
        true_output = sample['output']
        label = sample['label']

        # 生成答案（只输入 instruction + input）
        generated_text = generate_answer(model, tokenizer, instruction, input_text, device)

        # 提取答案
        predicted_answer = extract_answer_from_text(generated_text)

        # 比对答案
        if predicted_answer == true_output:
            correct += 1
        total += 1

        # 保存生成的数据
        generated_data.append({
            'instruction': instruction,
            'input': input_text,
            'true_output': true_output,
            'label': label,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': predicted_answer == true_output
        })

    accuracy = correct / total if total > 0 else 0

    # 保存生成的答案（最新的）
    with open(os.path.join(output_dir, 'generated_answers.json'), 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # 保存每个 epoch 的生成答案（不覆盖）
    if epoch is not None:
        epoch_answers_file = os.path.join(output_dir, f'generated_answers_epoch_{epoch}.json')
        with open(epoch_answers_file, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # 打印前五条生成的答案
    logger.info("\n📋 前五条生成的答案:")
    logger.info("-" * 100)
    for idx in range(min(5, len(generated_data))):
        item = generated_data[idx]
        logger.info(f"\n样本 {idx + 1}:")
        logger.info(f"  Instruction: {item['instruction'][:80]}...")
        logger.info(f"  Input: {item['input']}")
        logger.info(f"  True Output: {item['true_output']}")
        logger.info(f"  Generated Text: {item['generated_text']}")
        logger.info(f"  Predicted Answer: {item['predicted_answer']}")
        logger.info(f"  Correct: {'✅' if item['correct'] else '❌'}")
    logger.info("\n" + "-" * 100)

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with Fixed MiLoRA")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (new format JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=6,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval (every N epochs)")

    # MiLoRA 参数
    parser.add_argument("--milora_r", type=int, default=64,
                        help="MiLoRA rank")
    parser.add_argument("--milora_dropout", type=float, default=0.05,
                        help="MiLoRA dropout")

    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(42)

    logger.info("\n" + "="*80)
    logger.info("Fine-tuning with Fixed MiLoRA")
    logger.info("="*80)
    logger.info(f"Base model: {args.base_model_path}")
    logger.info(f"Training data: {args.train_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"MiLoRA rank: {args.milora_r}")
    logger.info(f"MiLoRA dropout: {args.milora_dropout}")
    logger.info("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志文件
    log_file = os.path.join(args.output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 加载 tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("✅ Tokenizer loaded")

    # 加载数据
    logger.info("\nLoading and processing data...")
    datasets = load_and_process_data(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        val_split=args.val_split
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    logger.info("✅ Data loaded")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 加载模型
    logger.info("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    logger.info("✅ Base model loaded")

    # 应用 MiLoRA
    logger.info("\nApplying Fixed MiLoRA...")
    milora_info = apply_milora_to_model(
        model,
        rank=args.milora_r,
        dropout=args.milora_dropout
    )
    logger.info("✅ Fixed MiLoRA applied")

    # 打印参数统计
    logger.info(f"MiLoRA Parameters:")
    logger.info(f"  Total parameters: {milora_info['total_params']:,}")
    logger.info(f"  Trainable parameters: {milora_info['trainable_params']:,}")
    logger.info(f"  Converted modules: {len(milora_info['converted_modules'])}")

    if milora_info.get('failed_conversions'):
        logger.warning(f"  Failed conversions: {len(milora_info['failed_conversions'])}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 训练循环
    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80 + "\n")

    best_eval_accuracy = 0.0
    best_model_dir = os.path.join(args.output_dir, 'best_milora')

    epoch_results = []

    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            num_accumulation_steps=args.gradient_accumulation_steps
        )

        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")

        # 每 eval_interval 个 epoch 进行一次验证
        if (epoch + 1) % args.eval_interval == 0:
            # 验证
            val_metrics = evaluate(model, val_loader, args.device)

            # 生成答案并评估准确率（Post Eval）
            gen_metrics = generate_and_evaluate_answers(
                model, val_dataset, tokenizer, args.device, args.output_dir, epoch=epoch+1
            )

            logger.info(f"  Eval Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Eval Accuracy: {gen_metrics['accuracy']:.4f} ({gen_metrics['correct']}/{gen_metrics['total']})")

            # 根据 accuracy 保存最好的模型
            if gen_metrics['accuracy'] > best_eval_accuracy:
                best_eval_accuracy = gen_metrics['accuracy']

                # 删除旧的最好模型
                if os.path.exists(best_model_dir):
                    import shutil
                    shutil.rmtree(best_model_dir)

                # 保存新的最好模型
                os.makedirs(best_model_dir, exist_ok=True)

                # 保存MiLoRA权重
                milora_state_dict = get_milora_state_dict(model)
                torch.save(milora_state_dict, os.path.join(best_model_dir, 'milora_weights.pt'))

                # 保存tokenizer
                tokenizer.save_pretrained(best_model_dir)

                # 保存配置
                milora_config = {
                    'rank': args.milora_r,
                    'dropout': args.milora_dropout,
                    'target_modules': milora_info['target_modules'],
                    'base_model_path': args.base_model_path
                }
                with open(os.path.join(best_model_dir, 'milora_config.json'), 'w') as f:
                    json.dump(milora_config, f, indent=2)

                logger.info(f"  ✅ Best model saved (accuracy: {best_eval_accuracy:.4f})")

            # 记录结果
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'eval_loss': val_metrics['loss'],
                'eval_accuracy': gen_metrics['accuracy'],
                'correct': gen_metrics['correct'],
                'total': gen_metrics['total'],
                'is_best': gen_metrics['accuracy'] == best_eval_accuracy
            })
        else:
            # 不评估的 epoch，只记录训练损失
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'eval_loss': None,
                'eval_accuracy': None,
                'correct': None,
                'total': None,
                'is_best': False
            })

    # 保存训练结果
    with open(os.path.join(args.output_dir, 'epoch_eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(epoch_results, f, indent=2, ensure_ascii=False)

    # 保存最终配置
    config = {
        'base_model': args.base_model_path,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'milora_r': args.milora_r,
        'milora_dropout': args.milora_dropout,
        'eval_interval': args.eval_interval,
        'best_eval_accuracy': best_eval_accuracy,
        'data_format': 'new_format (instruction + input + output)',
        'milora_info': milora_info
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "="*80)
    logger.info("✅ Fixed MiLoRA Training completed!")
    logger.info("="*80)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"\nFiles generated:")
    logger.info(f"  - best_milora/ (Best MiLoRA weights and config)")
    logger.info(f"  - epoch_eval_results.json (Epoch-by-epoch results)")
    logger.info(f"  - generated_answers.json (Generated answers on validation set)")
    logger.info(f"  - config.json (Training configuration)")
    logger.info(f"  - training.log (Detailed training logs)")
    logger.info(f"\nBest validation accuracy: {best_eval_accuracy:.4f}")
    logger.info("="*80)


if __name__ == "__main__":
    main()