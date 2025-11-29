#!/usr/bin/env python3
"""
只训练最后两层MoE的训练脚本
将最后两层FFN替换为MoE专家（LoRA适配器），冻结其他所有参数
针对48GB×2卡优化
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.only_moe_model import OnlyMoEModel, OnlyMoEConfig

# 复用现有的数据集类和函数
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    load_and_process_data,
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


def train_epoch_only_moe(model, train_loader, optimizer, device, tokenizer,
                        num_accumulation_steps=1, rank=0):
    """
    只训练MoE的一个epoch
    """
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_moe_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Only MoE Training", disable=(rank != 0), mininterval=1.0)

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 正确处理labels masking - 只计算output部分的loss
        for i in range(labels.shape[0]):
            instruction = batch['instruction'][i] if isinstance(batch['instruction'], list) else batch['instruction']
            input_text = batch['input'][i] if isinstance(batch['input'], list) else batch['input']
            output_text = batch['output'][i] if isinstance(batch['output'], list) else batch['output']

            # 构建input部分（需要mask的部分）- 与数据集格式保持一致
            if input_text:
                input_part = f"{instruction}\n{input_text}\n"  # 保持与数据集一致的格式
            else:
                input_part = f"{instruction}\n"

            # 计算input部分的token长度 - 使用与数据集相同的tokenization方式
            input_tokens = tokenizer(input_part, add_special_tokens=False, truncation=False)['input_ids']
            input_length = len(input_tokens)

            # 更安全的masking策略
            if input_length < labels.shape[1]:
                # 确保至少保留一些output tokens用于训练
                max_mask_length = min(input_length, labels.shape[1] - 5)  # 至少保留5个token用于output
                labels[i, :max_mask_length] = -100
            else:
                # 如果input太长，保留最后10个token用于训练
                labels[i, :-10] = -100

            # 调试信息：检查labels masking
            if batch_idx < 3 and i == 0:  # 前3个batch的第一个样本
                valid_labels = (labels[i] != -100).sum().item()
                total_labels = labels.shape[1]
                print(f"🔍 Batch {batch_idx}, Sample {i}:")
                print(f"  input_length={input_length}, seq_len={labels.shape[1]}")
                print(f"  valid_labels={valid_labels}/{total_labels}")
                if valid_labels == 0:
                    print(f"  ⚠️ WARNING: No valid labels for training!")

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # 获取各种损失
        lm_loss = outputs.loss  # 语言模型损失
        moe_aux_loss = getattr(outputs, 'moe_aux_loss', torch.tensor(0.0, device=device, dtype=torch.float16))

        # 将主损失转换为float16以保持一致性
        lm_loss = lm_loss.to(dtype=torch.float16)

        # 总损失
        total_batch_loss = lm_loss + moe_aux_loss

        # 检查 NaN/Inf loss
        if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
            print(f"❌ NaN or Inf total loss detected at batch {batch_idx}")
            print(f"  LM loss: {lm_loss.item()}, MoE loss: {moe_aux_loss.item()}")
            continue

        # 调试信息：显示实际损失值
        if batch_idx < 5 or batch_idx % 50 == 0:  # 前5个batch和每50个batch
            print(f"🔍 Batch {batch_idx} - Actual loss values:")
            print(f"  Total: {total_batch_loss.item():.8f}, LM: {lm_loss.item():.8f}, MoE: {moe_aux_loss.item():.8f}")

        # 梯度累积
        total_batch_loss = total_batch_loss / num_accumulation_steps
        total_batch_loss.backward()

        total_loss += total_batch_loss.item() * num_accumulation_steps
        total_lm_loss += lm_loss.item()
        total_moe_loss += moe_aux_loss.item()
        num_batches += 1

        # 梯度更新
        if (batch_idx + 1) % num_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

        # 内存清理
        if (batch_idx + 1) % 2 == 0:  # 每2个batch清理一次
            torch.cuda.empty_cache()

        # 检查内存使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            if memory_allocated > 40:  # 如果超过40GB，强制清理
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                if rank == 0:
                    print(f"⚠️ High memory usage ({memory_allocated:.1f}GB), forced cleanup")

        # 更新进度条
        postfix = {
            'loss': f"{total_batch_loss.item() * num_accumulation_steps:.6f}",
            'lm': f"{lm_loss.item():.6f}",
            'moe': f"{moe_aux_loss.item():.6f}"
        }
        pbar.set_postfix(postfix)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_lm_loss = total_lm_loss / num_batches if num_batches > 0 else 0
    avg_moe_loss = total_moe_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'moe_loss': avg_moe_loss,
        'num_batches': num_batches
    }


def evaluate_only_moe(model, val_loader, device, tokenizer, rank=0):
    """
    只训练MoE的验证
    """
    model.eval()
    total_loss = 0
    total_lm_loss = 0
    total_moe_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Evaluating", disable=(rank != 0), mininterval=1.0)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 正确处理labels masking
            for i in range(labels.shape[0]):
                instruction = batch['instruction'][i] if isinstance(batch['instruction'], list) else batch['instruction']
                input_text = batch['input'][i] if isinstance(batch['input'], list) else batch['input']

                # 构建input部分（需要mask的部分）- 与数据集格式保持一致
                if input_text:
                    input_part = f"{instruction}\n{input_text}\n"  # 保持与数据集一致的格式
                else:
                    input_part = f"{instruction}\n"

                # 计算input部分的token长度
                input_tokens = tokenizer(input_part, add_special_tokens=False, truncation=False)['input_ids']
                input_length = len(input_tokens)

                # 更安全的masking策略
                if input_length < labels.shape[1]:
                    max_mask_length = min(input_length, labels.shape[1] - 5)  # 至少保留5个token
                    labels[i, :max_mask_length] = -100
                else:
                    labels[i, :-10] = -100  # 保留最后10个token

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

            lm_loss = outputs.loss
            moe_aux_loss = getattr(outputs, 'moe_aux_loss', torch.tensor(0.0, device=device, dtype=torch.float16))

            lm_loss = lm_loss.to(dtype=torch.float16)
            total_batch_loss = lm_loss + moe_aux_loss

            # 检查总损失是否为NaN/Inf
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                continue

            total_loss += total_batch_loss.item()
            total_lm_loss += lm_loss.item()
            total_moe_loss += moe_aux_loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{total_batch_loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_lm_loss = total_lm_loss / num_batches if num_batches > 0 else 0
    avg_moe_loss = total_moe_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'moe_loss': avg_moe_loss,
        'num_batches': num_batches
    }


def generate_and_evaluate_answers_only_moe(
    model, val_dataset, tokenizer, device, output_dir, epoch=None, rank=0
):
    """
    只训练MoE模型生成答案并评估准确率
    """
    model.eval()

    correct = 0
    total = 0
    generated_data = []

    for idx in tqdm(range(len(val_dataset)), desc="Generating", disable=(rank != 0), mininterval=1.0):
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
        # 处理DDP包装的模型
        base_model = model.module if hasattr(model, 'module') else model
        generated_text = generate_answer(
            base_model, tokenizer, instruction, input_text, device
        )

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

    # 保存生成的答案
    with open(os.path.join(output_dir, 'generated_answers.json'), 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    if epoch is not None:
        epoch_answers_file = os.path.join(output_dir, f'generated_answers_epoch_{epoch}.json')
        with open(epoch_answers_file, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # 打印前五条生成的答案
    if rank == 0:
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

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    # 初始化分布式训练
    rank, world_size, local_rank = setup_distributed()

    parser = argparse.ArgumentParser(description="Only MoE Training")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval (every N epochs)")

    # 模型参数
    parser.add_argument("--backbone", type=str, default="qwen", choices=["llama", "qwen"],
                        help="Model backbone type")
    parser.add_argument("--num_moe_experts", type=int, default=4,
                        help="Number of MoE experts per layer")

    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    parser.add_argument("--memory_efficient", action='store_true',
                        help="Enable memory efficient training")

    args = parser.parse_args()

    # 设置内存优化
    if world_size > 1:
        args.memory_efficient = True

    if args.memory_efficient:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if world_size > 1:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)

    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 确定MoE层
    if args.backbone == "llama":
        total_layers = 32
        moe_layers = [30, 31]  # 最后两层 (0-indexed)
    elif args.backbone == "qwen":
        total_layers = 28
        moe_layers = [26, 27]  # 最后两层 (0-indexed)
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    if is_main_process(rank):
        print("\n" + "="*80)
        print("只训练最后两层MoE")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Base model: {args.base_model_path}")
        print(f"Backbone: {args.backbone}")
        print(f"Total layers: {total_layers}")
        print(f"MoE layers: {moe_layers}")
        print(f"Training data: {args.train_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size} (per GPU)")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max length: {args.max_length}")
        print(f"MoE experts per layer: {args.num_moe_experts}")
        print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")
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
    datasets = load_and_process_data(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        val_split=args.val_split
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    print("✅ Data loaded")

    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )

    # 加载基础模型
    if is_main_process(rank):
        print("\nLoading base model...")

    load_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': None,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True
    }

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **load_kwargs)
    torch.cuda.empty_cache()
    base_model = base_model.to(device)
    torch.cuda.empty_cache()

    if is_main_process(rank):
        print("✅ Base model loaded")

    # 创建OnlyMoE配置
    if is_main_process(rank):
        print(f"\nConfiguring Only MoE...")

    only_moe_config = OnlyMoEConfig(
        # LoRA配置
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        # MoE配置
        num_moe_experts=args.num_moe_experts,
        top_k=2,
        moe_layers=moe_layers,

        # 辅助损失配置
        aux_loss_coef=0.01,

        # 其他配置
        dropout=0.1
    )

    # 创建OnlyMoE模型
    model = OnlyMoEModel(base_model, only_moe_config)

    if is_main_process(rank):
        model.print_trainable_parameters()
        print("✅ Only MoE configured")

    # 确保所有参数在正确设备上（在DDP包装前）
    torch.cuda.empty_cache()

    # 使用DDP包装模型（仅在多GPU时）
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )

        if is_main_process(rank):
            print("✅ Model wrapped with DDP")

    # 设置优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # 训练循环
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Starting only MoE training...")
        print("="*80 + "\n")

    best_eval_accuracy = 0.0
    best_model_dir = os.path.join(args.output_dir, 'best_only_moe_model')
    epoch_results = []

    for epoch in range(args.num_epochs):
        # 设置分布式采样器的epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process(rank):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_metrics = train_epoch_only_moe(
            model, train_loader, optimizer, device, tokenizer,
            num_accumulation_steps=args.gradient_accumulation_steps,
            rank=rank
        )

        if is_main_process(rank):
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    LM Loss: {train_metrics['lm_loss']:.4f}")
            print(f"    MoE Loss: {train_metrics['moe_loss']:.4f}")

        # 每eval_interval个epoch进行一次验证
        if (epoch + 1) % args.eval_interval == 0:
            # 验证
            val_metrics = evaluate_only_moe(
                model, val_loader, device, tokenizer, rank=rank
            )

            # 生成答案并评估准确率（只在主进程执行）
            if is_main_process(rank):
                gen_metrics = generate_and_evaluate_answers_only_moe(
                    model, val_dataset, tokenizer, device, args.output_dir, epoch=epoch+1, rank=rank
                )
            else:
                gen_metrics = {'accuracy': 0.0, 'correct': 0, 'total': 0}

            # 同步所有进程
            if world_size > 1:
                dist.barrier()

            if is_main_process(rank):
                print(f"  Eval Loss: {val_metrics['loss']:.4f}")
                print(f"    LM Loss: {val_metrics['lm_loss']:.4f}")
                print(f"    MoE Loss: {val_metrics['moe_loss']:.4f}")
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

                    # 保存模型权重
                    if hasattr(model, 'module'):
                        model.module.save_model(best_model_dir)
                    else:
                        model.save_model(best_model_dir)

                    # 保存tokenizer
                    tokenizer.save_pretrained(best_model_dir)

                    print(f"  ✅ Best model saved (accuracy: {best_eval_accuracy:.4f})")

            # 记录结果
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_lm_loss': train_metrics['lm_loss'],
                'train_moe_loss': train_metrics['moe_loss'],
                'eval_loss': val_metrics['loss'],
                'eval_lm_loss': val_metrics['lm_loss'],
                'eval_moe_loss': val_metrics['moe_loss'],
                'eval_accuracy': gen_metrics['accuracy'],
                'correct': gen_metrics['correct'],
                'total': gen_metrics['total'],
                'is_best': gen_metrics['accuracy'] == best_eval_accuracy
            })
        else:
            # 不评估的epoch，只记录训练损失
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_lm_loss': train_metrics['lm_loss'],
                'train_moe_loss': train_metrics['moe_loss'],
                'eval_loss': None,
                'eval_lm_loss': None,
                'eval_moe_loss': None,
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
            'backbone': args.backbone,
            'training_mode': 'only_moe',
            'total_layers': total_layers,
            'moe_layers': moe_layers,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'effective_batch_size': args.batch_size * world_size * args.gradient_accumulation_steps,
            'world_size': world_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'num_moe_experts': args.num_moe_experts,
            'lora_config': {
                'rank': args.lora_rank,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout
            },
            'eval_interval': args.eval_interval,
            'best_eval_accuracy': best_eval_accuracy,
            'architecture': 'only_moe_last_two_layers'
        }

        with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print("\n" + "="*80)
        print("✅ Only MoE training completed!")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")
        print(f"\nFiles generated:")
        print(f"  - best_only_moe_model/ (Best model weights)")
        print(f"  - epoch_eval_results.json (Epoch-by-epoch results)")
        print(f"  - generated_answers.json (Generated answers on validation set)")
        print(f"  - config.json (Training configuration)")
        print(f"\nBest validation accuracy: {best_eval_accuracy:.4f}")
        print(f"Architecture: Only MoE (last {len(moe_layers)} layers)")
        print(f"MoE experts per layer: {args.num_moe_experts}")
        print("="*80)

    # 清理分布式训练
    cleanup_distributed()


if __name__ == "__main__":
    main()