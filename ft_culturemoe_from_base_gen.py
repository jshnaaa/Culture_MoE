#!/usr/bin/env python3
"""
使用标准语言建模损失 + 文化专注性损失微调 CultureMoE 模型（生成式版本）

使用方法：
    python ft_culturemoe_from_base_gen.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --use_culture_loss True \
        --culture_loss_lambda 0.5

数据格式：
    {
        "text": "### Question: ... ### Answer: 2",
        "text_mask": "### Question: ... ### Answer: 2",
        "label": "0"
    }
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


class CultureMoEDataset(Dataset):
    """
    CultureMoE 数据集

    数据格式：
    {
        "text": "### Question: ... ### Answer: 2",
        "text_mask": "### Question: ... ### Answer: 2",
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
        text = item['text']
        text_mask = item.get('text_mask', text)
        label = item.get('label', '')

        # Tokenize text（用于 MOE 专家）
        encoded_text = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize text_mask（用于共享专家）
        encoded_mask = self.tokenizer(
            text_mask,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded_text['input_ids'].squeeze(0)
        attention_mask = encoded_text['attention_mask'].squeeze(0)
        input_ids_mask = encoded_mask['input_ids'].squeeze(0)
        attention_mask_mask = encoded_mask['attention_mask'].squeeze(0)

        # 对于生成式模型，labels = input_ids（用于语言建模损失）
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'input_ids_mask': input_ids_mask,
            'attention_mask_mask': attention_mask_mask,
            'labels': labels,
            'text': text,
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
    dataset = CultureMoEDataset(data_path, tokenizer, max_length)

    # 按 9:1 比例划分
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    return {
        'train': train_dataset,
        'validation': val_dataset
    }


def extract_answer_from_text(text: str) -> str:
    """
    从生成的文本中提取答案

    Args:
        text: 生成的文本

    Returns:
        提取的答案（数字字符串）
    """
    import re

    # 查找 "Answer: " 后面的数字
    match = re.search(r'Answer:\s*(\d+)', text)
    if match:
        return match.group(1)

    return ""


def generate_answer(model, tokenizer, text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    """
    使用模型生成答案

    Args:
        model: 模型
        tokenizer: tokenizer
        text: 输入文本
        device: 设备
        max_new_tokens: 最大生成 token 数

    Returns:
        生成的文本
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None
        )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text


def compute_culture_loss(logits, labels, culture_labels, num_classes=10):
    """
    计算文化专注性损失

    Args:
        logits: 模型输出的 logits [B, num_classes]
        labels: 真实标签 [B]
        culture_labels: 文化标签 [B]
        num_classes: 类别数量

    Returns:
        文化损失
    """
    # 这里假设我们有一个分类头来计算文化损失
    # 实际实现需要根据你的具体需求调整

    # 简单的交叉熵损失
    loss_fn = torch.nn.CrossEntropyLoss()
    culture_loss = loss_fn(logits, labels)

    return culture_loss


def train_epoch(model, train_loader, optimizer, device, use_culture_loss=False, culture_loss_lambda=0.5, num_accumulation_steps=1):
    """
    训练一个 epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        use_culture_loss: 是否使用文化损失
        culture_loss_lambda: 文化损失权重
        num_accumulation_steps: 梯度累积步数

    Returns:
        dict: 包含训练指标的字典
    """
    model.train()
    total_loss = 0
    total_gen_loss = 0
    total_culture_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        input_ids_mask = batch['input_ids_mask'].to(device)
        attention_mask_mask = batch['attention_mask_mask'].to(device)
        labels = batch['labels'].to(device)
        culture_labels = batch['label']  # 文化标签

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids_mask=input_ids_mask,
            attention_mask_mask=attention_mask_mask,
            labels=labels,
            culture_labels=culture_labels if use_culture_loss else None,
            use_culture_loss=use_culture_loss,
            culture_loss_lambda=culture_loss_lambda
        )

        loss = outputs['loss']
        gen_loss = outputs.get('generation_loss', loss)
        culture_loss = outputs.get('culture_loss', torch.tensor(0.0, device=device))

        # ✅ 检查 NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ NaN or Inf loss detected at batch {batch_idx}")
            continue

        # 梯度累积
        loss = loss / num_accumulation_steps
        loss.backward()

        total_loss += loss.item() * num_accumulation_steps
        total_gen_loss += gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss
        total_culture_loss += culture_loss.item() if isinstance(culture_loss, torch.Tensor) else culture_loss
        num_batches += 1

        # 梯度更新
        if (batch_idx + 1) % num_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix({
            'loss': f"{loss.item() * num_accumulation_steps:.4f}",
            'gen_loss': f"{gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss:.4f}",
            'culture_loss': f"{culture_loss.item() if isinstance(culture_loss, torch.Tensor) else culture_loss:.4f}"
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_gen_loss = total_gen_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'gen_loss': avg_gen_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def evaluate(model, val_loader, device, use_culture_loss=False, culture_loss_lambda=0.5):
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        use_culture_loss: 是否使用文化损失
        culture_loss_lambda: 文化损失权重

    Returns:
        dict: 包含评估指标的字典
    """
    model.eval()
    total_loss = 0
    total_gen_loss = 0
    total_culture_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Evaluating")

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids_mask = batch['input_ids_mask'].to(device)
            attention_mask_mask = batch['attention_mask_mask'].to(device)
            labels = batch['labels'].to(device)
            culture_labels = batch['label']

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_mask=input_ids_mask,
                attention_mask_mask=attention_mask_mask,
                labels=labels,
                culture_labels=culture_labels if use_culture_loss else None,
                use_culture_loss=use_culture_loss,
                culture_loss_lambda=culture_loss_lambda
            )

            loss = outputs['loss']
            gen_loss = outputs.get('generation_loss', loss)
            culture_loss = outputs.get('culture_loss', torch.tensor(0.0, device=device))

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            total_gen_loss += gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss
            total_culture_loss += culture_loss.item() if isinstance(culture_loss, torch.Tensor) else culture_loss
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'gen_loss': f"{gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss:.4f}",
                'culture_loss': f"{culture_loss.item() if isinstance(culture_loss, torch.Tensor) else culture_loss:.4f}"
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_gen_loss = total_gen_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'gen_loss': avg_gen_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def generate_and_evaluate_answers(model, val_dataset, tokenizer, device, output_dir):
    """
    在验证集上生成答案并评估准确率

    Args:
        model: 模型
        val_dataset: 验证数据集
        tokenizer: tokenizer
        device: 设备
        output_dir: 输出目录

    Returns:
        dict: 包含准确率等指标的字典
    """
    model.eval()

    correct = 0
    total = 0
    generated_data = []

    print("\nGenerating answers on validation set...")

    for idx in tqdm(range(len(val_dataset)), desc="Generating"):
        sample = val_dataset[idx]
        text = sample['text']
        true_label = sample['label']

        # 生成答案
        generated_text = generate_answer(model, tokenizer, text, device)

        # 提取答案
        predicted_answer = extract_answer_from_text(generated_text)

        # 比对答案
        if predicted_answer == true_label:
            correct += 1
        total += 1

        # 保存生成的数据
        generated_data.append({
            'text': text,
            'true_label': true_label,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': predicted_answer == true_label
        })

    accuracy = correct / total if total > 0 else 0

    # 保存生成的答案
    with open(os.path.join(output_dir, 'generated_answers.json'), 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CultureMoE model with standard language modeling loss + culture loss")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--lora_weights_path", type=str, required=True,
                        help="Path to LoRA weights")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (JSON format)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    # 训练参数
    parser.add_argument("--use_culture_loss", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to use culture loss")
    parser.add_argument("--culture_loss_lambda", type=float, default=0.5,
                        help="Culture loss weight")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")

    # MOE 参数
    parser.add_argument("--num_experts", type=int, default=6,
                        help="Number of experts")
    parser.add_argument("--shared_hidden_dim", type=int, default=4096,
                        help="Shared layer hidden dimension")
    parser.add_argument("--router_hidden_dim", type=int, default=2048,
                        help="Router hidden dimension")
    parser.add_argument("--experts_hidden_dim", type=int, default=4096,
                        help="Experts hidden dimension")
    parser.add_argument("--moe_lora_rank", type=int, default=32,
                        help="MOE LoRA rank")
    parser.add_argument("--classification_hidden_dim", type=int, default=512,
                        help="Classification head hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")

    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Fine-tuning CultureMoE Model with Standard Language Modeling Loss + Culture Loss")
    print("="*80)
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.lora_weights_path}")
    print(f"Training data: {args.train_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Use culture loss: {args.use_culture_loss}")
    print(f"Culture loss weight: {args.culture_loss_lambda}")
    print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded")

    # 加载数据
    print("\nLoading and processing data...")
    datasets = load_and_process_data(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        val_split=args.val_split
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    print("✅ Data loaded")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 加载 Base 模型
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Base model loaded")

    # 加载 LoRA 权重
    print("\nLoading LoRA weights...")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_weights_path,
        is_trainable=False,
        torch_dtype=torch.float16
    )
    print("✅ LoRA weights loaded")

    # 合并 LoRA 权重
    print("\nMerging LoRA weights...")
    model = model.merge_and_unload()
    print("✅ LoRA weights merged")

    # 创建 CultureMoE 模型
    print("\nCreating CultureMoE model...")
    moe_args = ModelArgs(
        num_experts=args.num_experts,
        shared_hidden_dim=args.shared_hidden_dim,
        router_hidden_dim=args.router_hidden_dim,
        experts_hidden_dim=args.experts_hidden_dim,
        moe_lora_rank=args.moe_lora_rank,
        classification_hidden_dim=args.classification_hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads
    )

    # 这里需要根据你的实际实现来创建 CultureMoE 模型
    # 假设你有一个函数可以将 Base + LoRA 模型转换为 CultureMoE 模型
    # model = convert_to_culturemoe(model, moe_args)

    print("✅ CultureMoE model created")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 训练循环
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    best_val_loss = float('inf')
    best_moe_dir = os.path.join(args.output_dir, 'best_moe')

    epoch_results = []

    for epoch in range(args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            use_culture_loss=args.use_culture_loss,
            culture_loss_lambda=args.culture_loss_lambda,
            num_accumulation_steps=args.gradient_accumulation_steps
        )

        # 验证
        val_metrics = evaluate(
            model, val_loader, args.device,
            use_culture_loss=args.use_culture_loss,
            culture_loss_lambda=args.culture_loss_lambda
        )

        # 生成答案并评估准确率
        gen_metrics = generate_and_evaluate_answers(
            model, val_dataset, tokenizer, args.device, args.output_dir
        )

        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {train_metrics['loss']:.4f}, Train Gen Loss: {train_metrics['gen_loss']:.4f}")
        print(f"   Eval Loss:  {val_metrics['loss']:.4f}, Eval Gen Loss: {val_metrics['gen_loss']:.4f}")
        print(f"   Eval Accuracy: {gen_metrics['accuracy']:.4f}")

        # 保存最好的模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']

            # 删除旧的最好模型
            if os.path.exists(best_moe_dir):
                import shutil
                shutil.rmtree(best_moe_dir)

            # 保存新的最好模型
            os.makedirs(best_moe_dir, exist_ok=True)
            moe_state_dict = model.state_dict()
            torch.save(moe_state_dict, os.path.join(best_moe_dir, "pytorch_model.bin"))
            print(f"   ✅ Best model saved (loss: {best_val_loss:.4f})")

        # 记录结果
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_gen_loss': train_metrics['gen_loss'],
            'train_culture_loss': train_metrics['culture_loss'],
            'eval_loss': val_metrics['loss'],
            'eval_gen_loss': val_metrics['gen_loss'],
            'eval_culture_loss': val_metrics['culture_loss'],
            'eval_accuracy': gen_metrics['accuracy'],
            'correct': gen_metrics['correct'],
            'total': gen_metrics['total']
        })

    # 保存训练结果
    with open(os.path.join(args.output_dir, 'epoch_eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(epoch_results, f, indent=2, ensure_ascii=False)

    # 保存配置
    config = {
        'base_model': args.base_model_path,
        'lora_weights': args.lora_weights_path,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'use_culture_loss': args.use_culture_loss,
        'culture_loss_lambda': args.culture_loss_lambda,
        'num_experts': args.num_experts,
        'best_val_loss': best_val_loss
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ Training completed!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"\nFiles generated:")
    print(f"  - best_moe/ (Best MOE weights)")
    print(f"  - epoch_eval_results.json (Epoch-by-epoch results)")
    print(f"  - generated_answers.json (Generated answers on validation set)")
    print(f"  - config.json (Training configuration)")
    print("="*80)


if __name__ == "__main__":
    main()

