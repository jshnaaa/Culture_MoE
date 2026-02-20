#!/usr/bin/env python3
"""
MixLoRA Fine-tuning Script

使用MixLoRA方法进行模型微调，结合LoRA的参数高效性与MoE的强大性能。

核心特性：
1. MixLoRA架构：在FFN层使用多个LoRA专家 + 注意力层使用普通LoRA
2. Top-K路由：每次只激活K个专家，提高计算效率
3. 负载均衡：使用辅助损失确保专家使用的均衡分布
4. 计算优化：共享FFN计算，减少重复计算

数据格式：与现有格式保持一致
    {
        "instruction": "Give me the answer from 1 to 4: Do you agree with ...",
        "instruction_mask": "Give me the answer from 1 to 4: Do you agree with ... [MASK]",
        "input": "This question is for a country or language that is Arabic.",
        "output": "2",
        "label": "0"
    }
"""

import argparse
import json
import os
import sys
from functools import partial
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.mixlora import MixLoRAConfig
from src.llamafactory.model.mixlora_adapter import create_mixlora_model

# 复用现有的数据集类
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    load_and_process_data,
    load_and_process_merged_data_8_1_1,
    extract_answer_from_text,
    generate_answer
)


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        print(f"Initializing distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")

        # 初始化进程组
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        # 设置当前进程的GPU
        torch.cuda.set_device(local_rank)

        # 多GPU模式下的内存分配器设置
        torch.cuda.empty_cache()
        # 同步所有进程
        dist.barrier()

        return rank, world_size, local_rank
    else:
        # 单GPU模式
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """检查是否为主进程"""
    return rank == 0


def mixlora_collate_fn(batch, pad_token_id=0):
    """
    MixLoRA的collate函数 - 处理不同长度的序列

    Args:
        batch: 数据集返回的样本列表
        pad_token_id: padding token的ID

    Returns:
        批次数据字典
    """
    # 找到batch内最长的序列长度（包括原始版本和mask版本）
    max_length = max(len(item['input_ids']) for item in batch)
    max_length_mask = max(len(item.get('input_ids_mask', item['input_ids'])) for item in batch)

    # 准备batch数据
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_input_ids_mask = []
    batch_attention_mask_mask = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        # 兼容两种可能的labels键名
        labels = item.get('labels', item.get('culture_labels'))
        input_ids_mask = item.get('input_ids_mask', input_ids)
        attention_mask_mask = item.get('attention_mask_mask', attention_mask)

        # 计算需要padding的长度
        pad_length = max_length - len(input_ids)
        pad_length_mask = max_length_mask - len(input_ids_mask)

        if pad_length > 0:
            # 右侧padding - 原始版本
            padded_input_ids = torch.cat([input_ids, torch.full((pad_length,), pad_token_id, dtype=torch.long)])
            padded_attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
            padded_labels = torch.cat([labels, torch.full((pad_length,), -100, dtype=torch.long)])
        else:
            padded_input_ids = input_ids
            padded_attention_mask = attention_mask
            padded_labels = labels

        if pad_length_mask > 0:
            # 右侧padding - mask版本
            padded_input_ids_mask = torch.cat([input_ids_mask, torch.full((pad_length_mask,), pad_token_id, dtype=torch.long)])
            padded_attention_mask_mask = torch.cat([attention_mask_mask, torch.zeros(pad_length_mask, dtype=torch.long)])
        else:
            padded_input_ids_mask = input_ids_mask
            padded_attention_mask_mask = attention_mask_mask

        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)
        batch_labels.append(padded_labels)
        batch_input_ids_mask.append(padded_input_ids_mask)
        batch_attention_mask_mask.append(padded_attention_mask_mask)

    return {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'labels': torch.stack(batch_labels),
        'input_ids_mask': torch.stack(batch_input_ids_mask),
        'attention_mask_mask': torch.stack(batch_attention_mask_mask),
    }


def train_epoch_mixlora(model_adapter, train_loader, optimizer, device, num_accumulation_steps=1, rank=0):
    """
    MixLoRA训练一个epoch

    Args:
        model_adapter: MixLoRA模型适配器
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        num_accumulation_steps: 梯度累积步数

    Returns:
        dict: 包含训练指标的字典
    """
    model_adapter.base_model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", disable=(rank != 0), mininterval=1.0)

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model_adapter.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # ✅ 检查 NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ NaN or Inf loss detected at batch {batch_idx}")
            continue

        # 梯度累积
        loss = loss / num_accumulation_steps
        loss.backward()

        total_loss += loss.item() * num_accumulation_steps

        # 如果有损失信息，记录详细损失
        if hasattr(outputs, 'loss_info'):
            loss_info = outputs.loss_info
            total_main_loss += loss_info['main_loss'].item()
            total_aux_loss += loss_info['load_balancing_loss'].item()

        num_batches += 1

        # 梯度更新
        if (batch_idx + 1) % num_accumulation_steps == 0:
            # 在多GPU模式下确保梯度同步完成
            if hasattr(model_adapter.base_model, 'module'):  # DDP wrapped
                # 确保所有梯度都已同步
                torch.distributed.barrier()

            optimizer.step()
            optimizer.zero_grad()

        # 定期清理GPU缓存（多GPU模式下更频繁）
        cache_clear_interval = (num_accumulation_steps * 5) if hasattr(model_adapter.base_model, 'module') else (num_accumulation_steps * 10)
        if (batch_idx + 1) % cache_clear_interval == 0:
            torch.cuda.empty_cache()
            # 多GPU模式下同步清理
            if hasattr(model_adapter.base_model, 'module'):
                torch.distributed.barrier()

        # 更新进度条
        postfix = {'loss': f"{loss.item() * num_accumulation_steps:.4f}"}
        if hasattr(outputs, 'loss_info'):
            loss_info = outputs.loss_info
            postfix['main'] = f"{loss_info['main_loss'].item():.4f}"
            postfix['aux'] = f"{loss_info['load_balancing_loss'].item():.4f}"

        pbar.set_postfix(postfix)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_main_loss = total_main_loss / num_batches if num_batches > 0 else 0
    avg_aux_loss = total_aux_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'aux_loss': avg_aux_loss,
        'num_batches': num_batches
    }


def evaluate_mixlora(model_adapter, val_loader, device, rank=0):
    """
    MixLoRA验证

    Args:
        model_adapter: MixLoRA模型适配器
        val_loader: 验证数据加载器
        device: 设备

    Returns:
        dict: 包含评估指标的字典
    """
    model_adapter.base_model.eval()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Evaluating", disable=(rank != 0), mininterval=1.0)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model_adapter.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()

            # 记录详细损失
            if hasattr(outputs, 'loss_info'):
                loss_info = outputs.loss_info
                total_main_loss += loss_info['main_loss'].item()
                total_aux_loss += loss_info['load_balancing_loss'].item()

            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_main_loss = total_main_loss / num_batches if num_batches > 0 else 0
    avg_aux_loss = total_aux_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'aux_loss': avg_aux_loss,
        'num_batches': num_batches
    }


def generate_and_evaluate_answers_mixlora(
    model_adapter, val_dataset, tokenizer, device, output_dir, epoch=None, rank=0,
    individual_val_datasets=None
):
    """
    MixLoRA生成答案并评估准确率

    Args:
        model_adapter: MixLoRA模型适配器
        val_dataset: 验证数据集（可能是合并的）
        tokenizer: tokenizer
        device: 设备
        output_dir: 输出目录
        epoch: 当前epoch数
        individual_val_datasets: 各个数据集的验证集列表（用于分离统计）

    Returns:
        dict: 包含准确率等指标的字典
    """
    model_adapter.base_model.eval()

    # 合并数据集的统计
    correct = 0
    total = 0
    generated_data = []

    # 分离数据集的统计（如果有）
    individual_stats = {}

    # 1. 评估合并数据集（用于模型选择）
    for idx in tqdm(range(len(val_dataset)), desc="Evaluating merged dataset", disable=(rank != 0), mininterval=1.0):
        # 获取原始数据集（处理 Subset 对象）
        if hasattr(val_dataset, 'dataset'):
            original_idx = val_dataset.indices[idx]
            sample = val_dataset.dataset[original_idx]
        else:
            sample = val_dataset[idx]

        instruction = sample['instruction']
        input_text = sample['input']
        true_output = sample['output']
        label = sample['label']

        # 生成答案（使用基础模型进行推理）
        generated_text = generate_answer(
            model_adapter.base_model, tokenizer, instruction, input_text, device
        )

        # 提取答案
        predicted_answer = extract_answer_from_text(generated_text)

        # 比对答案
        is_correct = predicted_answer == true_output
        if is_correct:
            correct += 1
        total += 1

        # 保存生成的数据
        item = {
            'instruction': instruction,
            'input': input_text,
            'true_output': true_output,
            'label': label,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': is_correct
        }

        # 添加数据集来源信息（如果有）
        if 'source_dataset' in sample:
            item['source_dataset'] = sample['source_dataset']

        generated_data.append(item)

    merged_accuracy = correct / total if total > 0 else 0

    # 2. 分离评估各个数据集（如果有individual_val_datasets）
    if individual_val_datasets is not None:
        if is_main_process(rank):
            print(f"\n🔧 开始分离评估各个数据集...")

        for dataset_idx, individual_dataset in enumerate(individual_val_datasets):
            dataset_name = f"Dataset_{dataset_idx}"

            # 尝试从数据集中获取名称
            if hasattr(individual_dataset, 'dataset') and hasattr(individual_dataset.dataset, 'dataset_names'):
                if dataset_idx < len(individual_dataset.dataset.dataset_names):
                    dataset_name = individual_dataset.dataset.dataset_names[dataset_idx]

            if is_main_process(rank):
                print(f"  评估 {dataset_name}...")

            dataset_correct = 0
            dataset_total = 0
            dataset_generated_data = []

            for idx in tqdm(range(len(individual_dataset)), desc=f"Evaluating {dataset_name}", disable=(rank != 0), mininterval=1.0, leave=False):
                # 获取样本
                if hasattr(individual_dataset, 'dataset'):
                    original_idx = individual_dataset.indices[idx]
                    sample = individual_dataset.dataset[original_idx]
                else:
                    sample = individual_dataset[idx]

                instruction = sample['instruction']
                input_text = sample['input']
                true_output = sample['output']
                label = sample['label']

                # 生成答案
                generated_text = generate_answer(
                    model_adapter.base_model, tokenizer, instruction, input_text, device
                )
                predicted_answer = extract_answer_from_text(generated_text)

                # 比对答案
                is_correct = predicted_answer == true_output
                if is_correct:
                    dataset_correct += 1
                dataset_total += 1

                dataset_generated_data.append({
                    'instruction': instruction,
                    'input': input_text,
                    'true_output': true_output,
                    'label': label,
                    'generated_text': generated_text,
                    'predicted_answer': predicted_answer,
                    'correct': is_correct
                })

            dataset_accuracy = dataset_correct / dataset_total if dataset_total > 0 else 0
            individual_stats[dataset_name] = {
                'accuracy': dataset_accuracy,
                'correct': dataset_correct,
                'total': dataset_total,
                'generated_data': dataset_generated_data
            }

            if is_main_process(rank):
                print(f"    {dataset_name} - Accuracy: {dataset_accuracy:.4f} ({dataset_correct}/{dataset_total})")

    accuracy = merged_accuracy

    # 保存生成的答案（只在主进程执行）
    if is_main_process(rank):
        with open(os.path.join(output_dir, 'generated_answers.json'), 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=2, ensure_ascii=False)

        if epoch is not None:
            epoch_answers_file = os.path.join(output_dir, f'generated_answers_epoch_{epoch}.json')
            with open(epoch_answers_file, 'w', encoding='utf-8') as f:
                json.dump(generated_data, f, indent=2, ensure_ascii=False)

        # 🔧 新增：保存分离数据集的结果（如果有）
        if individual_stats:
            # 保存分离统计结果
            individual_results_file = os.path.join(output_dir, 'individual_dataset_results.json')
            with open(individual_results_file, 'w', encoding='utf-8') as f:
                # 只保存统计信息，不保存详细的generated_data（避免文件过大）
                stats_only = {}
                for dataset_name, stats in individual_stats.items():
                    stats_only[dataset_name] = {
                        'accuracy': stats['accuracy'],
                        'correct': stats['correct'],
                        'total': stats['total']
                    }
                json.dump(stats_only, f, indent=2, ensure_ascii=False)

            # 保存每个epoch的分离统计结果
            if epoch is not None:
                epoch_individual_file = os.path.join(output_dir, f'individual_dataset_results_epoch_{epoch}.json')
                with open(epoch_individual_file, 'w', encoding='utf-8') as f:
                    stats_only = {}
                    for dataset_name, stats in individual_stats.items():
                        stats_only[dataset_name] = {
                            'accuracy': stats['accuracy'],
                            'correct': stats['correct'],
                            'total': stats['total']
                        }
                    json.dump(stats_only, f, indent=2, ensure_ascii=False)

        # 打印前五条生成的答案
        print("\n📋 前五条生成的答案:")
        print("-" * 100)
        for idx in range(min(5, len(generated_data))):
            item = generated_data[idx]
            print(f"\n样本 {idx + 1}:")
            print(f"  Instruction: {item['instruction'][:80]}...")
            print(f"  Input: {item['input']}")
            print(f"  True Output: {item['true_output']}")
            print(f"  Generated Text: {item['generated_text']}")
            print(f"  Predicted Answer: {item['predicted_answer']}")
            print(f"  Correct: {'✅' if item['correct'] else '❌'}")
        print("\n" + "-" * 100)

        # 打印分离统计信息
        if individual_stats:
            print(f"\n  📊 分离数据集统计（用于观察）:")
            for dataset_name, stats in individual_stats.items():
                print(f"    {dataset_name}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    result = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

    # 添加分离统计信息（如果有）
    if individual_stats:
        result['individual_stats'] = {}
        for dataset_name, stats in individual_stats.items():
            result['individual_stats'][dataset_name] = {
                'accuracy': stats['accuracy'],
                'correct': stats['correct'],
                'total': stats['total']
            }

    return result


def main():
    # 初始化分布式训练
    rank, world_size, local_rank = setup_distributed()

    parser = argparse.ArgumentParser(description="Fine-tune model with MixLoRA")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (JSON)")
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

    # MixLoRA参数
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--num_experts", type=int, default=6,
                        help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Top-K routing")
    parser.add_argument("--aux_loss_coef", type=float, default=0.01,
                        help="Auxiliary loss coefficient")

    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--memory_efficient", action='store_true',
                        help="Enable memory efficient training")

    args = parser.parse_args()

    # 设置内存优化
    if world_size > 1:  # 多GPU训练时自动启用内存优化
        args.memory_efficient = True

    if args.memory_efficient:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # 多GPU内存优化设置
        if world_size > 1:
            # 禁用可能导致内存分配器问题的功能
            torch.cuda.empty_cache()
            # 设置更保守的内存分配策略
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)  # 限制每个进程使用80%显存

    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        print("\n" + "="*80)
        print("Fine-tuning Model with MixLoRA (Multi-GPU)")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Base model: {args.base_model_path}")
        print(f"Training data: {args.train_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size} (per GPU)")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"MixLoRA config:")
        print(f"  - LoRA rank: {args.lora_r}")
        print(f"  - LoRA alpha: {args.lora_alpha}")
        print(f"  - Number of experts: {args.num_experts}")
        print(f"  - Top-K: {args.top_k}")
        print(f"  - Aux loss coef: {args.aux_loss_coef}")
        print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded")

    # 加载数据
    print("\nLoading and processing data...")

    # 🔧 新增：检查是否为合并数据集
    individual_val_datasets = None
    if args.train_file.startswith("MERGED:"):
        # 处理合并数据集
        merge_spec = args.train_file.replace("MERGED:", "")
        print(f"🔧 检测到合并数据集: {merge_spec}")

        if merge_spec == "CulturalBench+CultureLLM":
            # CulturalBench + CultureLLM 合并
            data_paths = [
                "/root/autodl-fs/CulturalBench_merge_gen.json",
                "/root/autodl-fs/cultureLLM_merge_gen.json"
            ]
            dataset_names = ["CulturalBench", "CultureLLM"]

            print(f"🔧 使用合并数据集: {dataset_names}")
            datasets = load_and_process_merged_data_8_1_1(
                data_paths=data_paths,
                dataset_names=dataset_names,
                tokenizer=tokenizer,
                max_length=args.max_length,
                output_dir=args.output_dir,
                seed=42
            )

            # 合并数据集的训练和验证集
            train_dataset = datasets['train']
            val_dataset = datasets['validation']

            # 保存个体数据集信息，用于分离统计
            individual_val_datasets = datasets['individual_datasets']['validation']
        else:
            raise ValueError(f"Unsupported merged dataset: {merge_spec}")

        print("✅ Merged data loaded")
    else:
        # 处理单个数据集
        datasets = load_and_process_data(
            args.train_file,
            tokenizer,
            max_length=args.max_length,
            val_split=args.val_split
        )
        train_dataset = datasets['train']
        val_dataset = datasets['validation']
        individual_val_datasets = None  # 单个数据集不需要分离统计

        print("✅ Data loaded")

    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    # 创建collate_fn
    collate_fn = partial(mixlora_collate_fn, pad_token_id=tokenizer.pad_token_id)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 加载基础模型
    if is_main_process(rank):
        print("\nLoading base model...")

    # 使用更节省内存的方式加载模型
    # 在多GPU模式下使用更保守的加载策略
    load_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': None,  # 先不分配设备，后面手动分配
        'trust_remote_code': True,
        'low_cpu_mem_usage': True
    }

    # 多GPU模式下的额外内存优化
    if world_size > 1:
        load_kwargs.update({
            'max_memory': {i: "6GiB" for i in range(world_size)},  # 限制每个GPU的最大内存使用
            'offload_folder': None,  # 禁用磁盘offload避免内存分配器问题
        })

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **load_kwargs)

    # 在移动模型前清理内存
    torch.cuda.empty_cache()

    # 将模型移动到对应的GPU
    base_model = base_model.to(device)

    # 移动后再次清理
    torch.cuda.empty_cache()

    # 启用梯度检查点以节省内存（仅在单GPU模式下）
    if hasattr(base_model, 'gradient_checkpointing_enable') and world_size == 1:
        base_model.gradient_checkpointing_enable()
        if is_main_process(rank):
            print("✅ Gradient checkpointing enabled")
    elif world_size > 1:
        if is_main_process(rank):
            print("🚫 Gradient checkpointing disabled for multi-GPU DDP compatibility")

    if is_main_process(rank):
        print("✅ Base model loaded")

    # 创建MixLoRA配置
    if is_main_process(rank):
        print("\nConfiguring MixLoRA...")

    mixlora_config = MixLoRAConfig(
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_experts=args.num_experts,
        top_k=args.top_k,
        aux_loss_coef=args.aux_loss_coef,
        ffn_target_modules=['gate_proj', 'up_proj', 'down_proj'],
        attention_target_modules=['q_proj', 'v_proj'],  # Attention层使用普通LoRA适配器
        apply_mixlora_to_attention=False  # Attention层不使用MixLoRA专家路由，仅使用普通LoRA
    )

    # 创建MixLoRA模型
    model_adapter = create_mixlora_model(base_model, mixlora_config)

    if is_main_process(rank):
        model_adapter.print_trainable_parameters()
        print("✅ MixLoRA configured")

    # 使用DDP包装模型（仅在多GPU时）
    if world_size > 1:
        model_adapter.base_model = DDP(
            model_adapter.base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # 设为False避免参数重复标记
            broadcast_buffers=False,       # 禁用buffer广播
            gradient_as_bucket_view=True   # 优化内存使用
        )

        # 设置静态图以避免DDP参数重复标记问题
        try:
            model_adapter.base_model._set_static_graph()
            if is_main_process(rank):
                print("✅ DDP static graph enabled")
        except Exception as e:
            if is_main_process(rank):
                print(f"⚠️  Warning: Could not set static graph: {e}")

        if is_main_process(rank):
            print("✅ Model wrapped with DDP")

    # 优化器
    optimizer = torch.optim.AdamW(
        model_adapter.base_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 训练循环
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80 + "\n")

    best_eval_accuracy = 0.0
    best_model_dir = os.path.join(args.output_dir, 'best_mixlora')

    epoch_results = []

    for epoch in range(args.num_epochs):
        # 设置分布式采样器的epoch（用于随机化）
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process(rank):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_metrics = train_epoch_mixlora(
            model_adapter, train_loader, optimizer, device,
            num_accumulation_steps=args.gradient_accumulation_steps,
            rank=rank
        )

        if is_main_process(rank):
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if train_metrics['main_loss'] > 0:
                print(f"    Main Loss: {train_metrics['main_loss']:.4f}")
                print(f"    Aux Loss: {train_metrics['aux_loss']:.4f}")

        # 每eval_interval个epoch进行一次验证
        if (epoch + 1) % args.eval_interval == 0:
            # 验证
            val_metrics = evaluate_mixlora(model_adapter, val_loader, device, rank=rank)

            # 生成答案并评估准确率（只在主进程执行）
            if is_main_process(rank):
                gen_metrics = generate_and_evaluate_answers_mixlora(
                    model_adapter, val_dataset, tokenizer, device, args.output_dir, epoch=epoch+1, rank=rank,
                    individual_val_datasets=individual_val_datasets
                )
            else:
                gen_metrics = {'accuracy': 0.0, 'correct': 0, 'total': 0}

            # 同步所有进程
            if world_size > 1:
                dist.barrier()

            if is_main_process(rank):
                print(f"  Eval Loss: {val_metrics['loss']:.4f}")
                if val_metrics['main_loss'] > 0:
                    print(f"    Main Loss: {val_metrics['main_loss']:.4f}")
                    print(f"    Aux Loss: {val_metrics['aux_loss']:.4f}")
                print(f"  Eval Accuracy: {gen_metrics['accuracy']:.4f} ({gen_metrics['correct']}/{gen_metrics['total']})")

                # 根据accuracy保存最好的模型
                if gen_metrics['accuracy'] > best_eval_accuracy:
                    best_eval_accuracy = gen_metrics['accuracy']

                    # 删除旧的最好模型
                    if os.path.exists(best_model_dir):
                        import shutil
                        shutil.rmtree(best_model_dir)

                    # 保存新的最好模型
                    os.makedirs(best_model_dir, exist_ok=True)

                    # 保存MixLoRA权重
                    model_adapter.save_mixlora(best_model_dir)

                    # 保存tokenizer
                    tokenizer.save_pretrained(best_model_dir)

                    print(f"  ✅ Best model saved (accuracy: {best_eval_accuracy:.4f})")

            # 记录结果
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_main_loss': train_metrics.get('main_loss', 0),
                'train_aux_loss': train_metrics.get('aux_loss', 0),
                'eval_loss': val_metrics['loss'],
                'eval_main_loss': val_metrics.get('main_loss', 0),
                'eval_aux_loss': val_metrics.get('aux_loss', 0),
                'eval_accuracy': gen_metrics['accuracy'],
                'correct': gen_metrics['correct'],
                'total': gen_metrics['total'],
                'is_best': gen_metrics['accuracy'] == best_eval_accuracy
            }

            # 🔧 新增：添加分离统计信息
            if 'individual_stats' in gen_metrics:
                epoch_result['individual_stats'] = gen_metrics['individual_stats']

            epoch_results.append(epoch_result)
        else:
            # 不评估的epoch，只记录训练损失
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_main_loss': train_metrics.get('main_loss', 0),
                'train_aux_loss': train_metrics.get('aux_loss', 0),
                'eval_loss': None,
                'eval_main_loss': None,
                'eval_aux_loss': None,
                'eval_accuracy': None,
                'correct': None,
                'total': None,
                'is_best': False
            })

    # 保存训练结果（只在主进程执行）
    if is_main_process(rank):
        with open(os.path.join(args.output_dir, 'epoch_eval_results.json'), 'w', encoding='utf-8') as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)

        # 保存配置
        config = {
            'base_model': args.base_model_path,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'effective_batch_size': args.batch_size * world_size,
            'world_size': world_size,
            'learning_rate': args.learning_rate,
            'mixlora_config': mixlora_config.to_dict(),
            'eval_interval': args.eval_interval,
            'best_eval_accuracy': best_eval_accuracy,
            'data_format': 'new_format (instruction + input + output)'
        }

        with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print("\n" + "="*80)
        print("✅ Training completed!")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")
        print(f"\nFiles generated:")
        print(f"  - best_mixlora/ (Best MixLoRA weights - 仅保存训练参数)")
        print(f"    ├── mixlora_config.json")
        print(f"    └── mixlora_weights.pt")
        print(f"  - epoch_eval_results.json (Epoch-by-epoch results, 包含合并和分离评估)")
        if individual_val_datasets:
            print(f"  - individual_dataset_results.json (分离数据集评估结果)")
        print(f"  - generated_answers.json (Generated answers on merged validation set)")
        print(f"  - data_split_8_1_1_*.pkl (各数据集划分信息)")
        print(f"  - config.json (Training configuration)")
        print(f"\nBest validation accuracy: {best_eval_accuracy:.4f}")
        print("="*80)

    # 清理分布式训练
    cleanup_distributed()


if __name__ == "__main__":
    main()