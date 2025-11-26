#!/usr/bin/env python3
"""
简化版FFN CultureMoE训练脚本
基于MixLoRA实现，只在最后2层使用MoE，添加文化损失
针对48GB×2卡优化

特点：
1. 基于成功的MixLoRA架构
2. 只在最后2层(26-27 for Qwen, 30-31 for LLaMA)替换为MoE
3. 添加文化感知损失
4. 极简内存优化配置
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

from src.llamafactory.model.simplified_culturemoe import SimplifiedCultureMoEConfig
from src.llamafactory.model.simplified_culturemoe_adapter import create_simplified_culturemoe_model

# 复用现有的数据集类
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


def compute_culture_loss(model_outputs, culture_labels, loss_weight=0.01):
    """
    计算文化感知损失

    Args:
        model_outputs: 模型输出，应该包含expert_weights等信息
        culture_labels: 文化标签 [B]
        loss_weight: 损失权重

    Returns:
        culture_loss: 文化损失
    """
    if not hasattr(model_outputs, 'expert_weights') or model_outputs.expert_weights is None:
        return torch.tensor(0.0, device=culture_labels.device, dtype=torch.float32)

    expert_weights = model_outputs.expert_weights  # [B, num_experts]
    batch_size = expert_weights.shape[0]

    if batch_size < 2:
        return torch.tensor(0.0, device=culture_labels.device, dtype=expert_weights.dtype)

    culture_loss = torch.tensor(0.0, device=culture_labels.device, dtype=expert_weights.dtype)
    count = 0

    # 计算同文化样本间的相似性和不同文化样本间的差异性
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if culture_labels[i] == culture_labels[j]:
                # 相同文化，鼓励相似的专家权重
                similarity = F.cosine_similarity(
                    expert_weights[i].unsqueeze(0),
                    expert_weights[j].unsqueeze(0)
                )
                culture_loss += (1.0 - similarity)
            else:
                # 不同文化，鼓励不同的专家权重
                similarity = F.cosine_similarity(
                    expert_weights[i].unsqueeze(0),
                    expert_weights[j].unsqueeze(0)
                )
                culture_loss += similarity
            count += 1

    if count > 0:
        culture_loss = culture_loss / count * loss_weight

    # 确保返回的是标量张量
    if not isinstance(culture_loss, torch.Tensor):
        culture_loss = torch.tensor(culture_loss, device=culture_labels.device)

    return culture_loss


def train_epoch_simplified(model_adapter, train_loader, optimizer, device,
                         num_accumulation_steps=1, rank=0, use_culture_loss=True, culture_loss_weight=0.01):
    """
    简化版CultureMoE训练一个epoch
    """
    model_adapter.base_model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    total_culture_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", disable=(rank != 0), mininterval=1.0)

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 获取文化标签（从label字段提取）
        culture_labels = None
        if 'culture_labels' in batch:
            culture_labels = batch['culture_labels'].to(device)
        elif 'label' in batch:
            # label字段是字符串列表，需要转换为张量
            if isinstance(batch['label'], list):
                # 将字符串标签转换为整数张量
                label_ints = [int(label) if label.isdigit() else 0 for label in batch['label']]
                culture_labels = torch.tensor(label_ints, dtype=torch.long, device=device)
            else:
                culture_labels = batch['label'].to(device)

        # 前向传播
        outputs = model_adapter.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # 检查 NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ NaN or Inf loss detected at batch {batch_idx}")
            continue

        # 计算文化损失
        culture_loss = torch.tensor(0.0, device=device, dtype=outputs.loss.dtype)
        if use_culture_loss and culture_labels is not None:
            culture_loss = compute_culture_loss(outputs, culture_labels, culture_loss_weight)

        # 总损失
        total_batch_loss = loss + culture_loss

        # 梯度累积
        total_batch_loss = total_batch_loss / num_accumulation_steps
        total_batch_loss.backward()

        total_loss += total_batch_loss.item() * num_accumulation_steps

        # 如果有损失信息，记录详细损失
        if hasattr(outputs, 'loss_info'):
            loss_info = outputs.loss_info
            total_main_loss += loss_info['main_loss'].item()
            total_aux_loss += loss_info['load_balancing_loss'].item()

        total_culture_loss += culture_loss.item()
        num_batches += 1

        # 梯度更新
        if (batch_idx + 1) % num_accumulation_steps == 0:
            # 在多GPU模式下确保梯度同步完成
            if hasattr(model_adapter.base_model, 'module'):  # DDP wrapped
                torch.distributed.barrier()

            optimizer.step()
            optimizer.zero_grad()

        # 定期清理GPU缓存
        cache_clear_interval = (num_accumulation_steps * 5) if hasattr(model_adapter.base_model, 'module') else (num_accumulation_steps * 10)
        if (batch_idx + 1) % cache_clear_interval == 0:
            torch.cuda.empty_cache()
            if hasattr(model_adapter.base_model, 'module'):
                torch.distributed.barrier()

        # 更新进度条
        postfix = {'loss': f"{total_batch_loss.item() * num_accumulation_steps:.4f}"}
        if hasattr(outputs, 'loss_info'):
            loss_info = outputs.loss_info
            postfix['main'] = f"{loss_info['main_loss'].item():.4f}"
            postfix['aux'] = f"{loss_info['load_balancing_loss'].item():.4f}"
        if use_culture_loss:
            postfix['culture'] = f"{culture_loss.item():.4f}"

        pbar.set_postfix(postfix)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_main_loss = total_main_loss / num_batches if num_batches > 0 else 0
    avg_aux_loss = total_aux_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'aux_loss': avg_aux_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def evaluate_simplified(model_adapter, val_loader, device, rank=0, use_culture_loss=True, culture_loss_weight=0.01):
    """
    简化版CultureMoE验证
    """
    model_adapter.base_model.eval()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    total_culture_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Evaluating", disable=(rank != 0), mininterval=1.0)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 获取文化标签
            culture_labels = None
            if 'culture_labels' in batch:
                culture_labels = batch['culture_labels'].to(device)
            elif 'label' in batch:
                # label字段是字符串列表，需要转换为张量
                if isinstance(batch['label'], list):
                    # 将字符串标签转换为整数张量
                    label_ints = [int(label) if label.isdigit() else 0 for label in batch['label']]
                    culture_labels = torch.tensor(label_ints, dtype=torch.long, device=device)
                else:
                    culture_labels = batch['label'].to(device)

            # 前向传播
            outputs = model_adapter.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # 计算文化损失
            culture_loss = torch.tensor(0.0, device=device, dtype=outputs.loss.dtype)
            if use_culture_loss and culture_labels is not None:
                culture_loss = compute_culture_loss(outputs, culture_labels, culture_loss_weight)

            total_batch_loss = loss + culture_loss
            total_loss += total_batch_loss.item()

            # 记录详细损失
            if hasattr(outputs, 'loss_info'):
                loss_info = outputs.loss_info
                total_main_loss += loss_info['main_loss'].item()
                total_aux_loss += loss_info['load_balancing_loss'].item()

            total_culture_loss += culture_loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{total_batch_loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_main_loss = total_main_loss / num_batches if num_batches > 0 else 0
    avg_aux_loss = total_aux_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'aux_loss': avg_aux_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def generate_and_evaluate_answers_simplified(
    model_adapter, val_dataset, tokenizer, device, output_dir, epoch=None, rank=0
):
    """
    简化版CultureMoE生成答案并评估准确率
    """
    model_adapter.base_model.eval()

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
        base_model = model_adapter.base_model.module if hasattr(model_adapter.base_model, 'module') else model_adapter.base_model
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

    parser = argparse.ArgumentParser(description="Train simplified FFN CultureMoE")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=6,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=2,
                        help="Evaluation interval (every N epochs)")

    # 模型参数
    parser.add_argument("--backbone", type=str, default="qwen", choices=["llama", "qwen"],
                        help="Model backbone type")
    parser.add_argument("--num_routing_experts", type=int, default=2,
                        help="Number of routing experts")
    parser.add_argument("--use_culture_loss", type=str, default="true",
                        help="Whether to use culture loss")
    parser.add_argument("--culture_loss_weight", type=float, default=0.01,
                        help="Culture loss weight")

    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=8,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    parser.add_argument("--memory_efficient", action='store_true',
                        help="Enable memory efficient training")

    args = parser.parse_args()

    # 转换字符串参数
    use_culture_loss = args.use_culture_loss.lower() == 'true'

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

    if is_main_process(rank):
        print("\n" + "="*80)
        print("训练简化版FFN CultureMoE (基于MixLoRA)")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Base model: {args.base_model_path}")
        print(f"Backbone: {args.backbone}")
        print(f"Training data: {args.train_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size} (per GPU)")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max length: {args.max_length}")
        print(f"Routing experts: {args.num_routing_experts}")
        print(f"Use culture loss: {use_culture_loss}")
        if use_culture_loss:
            print(f"Culture loss weight: {args.culture_loss_weight}")
        print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
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
        num_workers=0,  # 设为0避免多进程问题
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

    # 确定MoE层
    if args.backbone == "qwen":
        total_layers = 28
        moe_layers = [26, 27]  # 最后2层 (0-indexed)
    else:  # llama
        total_layers = 32
        moe_layers = [30, 31]  # 最后2层 (0-indexed)

    # 创建简化版CultureMoE配置
    if is_main_process(rank):
        print(f"\nConfiguring Simplified CultureMoE...")
        print(f"Total layers: {total_layers}")
        print(f"MoE layers: {moe_layers}")

    culturemoe_config = SimplifiedCultureMoEConfig(
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_routing_experts=args.num_routing_experts,
        moe_layers=moe_layers,
        use_culture_loss=use_culture_loss,
        culture_loss_weight=args.culture_loss_weight,
        aux_loss_coef=0.001  # 小的辅助损失
    )

    # 创建简化版CultureMoE模型
    model_adapter = create_simplified_culturemoe_model(base_model, culturemoe_config)

    if is_main_process(rank):
        model_adapter.print_trainable_parameters()
        print("✅ Simplified CultureMoE configured")

    # 确保所有参数在正确设备上（在DDP包装前）
    torch.cuda.empty_cache()

    # 验证设备一致性
    device_check_passed = True
    devices = set()
    for name, param in model_adapter.base_model.named_parameters():
        devices.add(param.device)

    if len(devices) > 1:
        if is_main_process(rank):
            print(f"❌ Device inconsistency detected: {devices}")
        device_check_passed = False
    else:
        if is_main_process(rank):
            print(f"✅ All parameters on device: {list(devices)[0]}")

    # 如果设备不一致，强制移动到正确设备
    if not device_check_passed:
        target_device = torch.device(f"cuda:{local_rank}")
        model_adapter.base_model = model_adapter.base_model.to(target_device)
        torch.cuda.empty_cache()
        if is_main_process(rank):
            print(f"✅ Forced all parameters to device: {target_device}")

    # 使用DDP包装模型（仅在多GPU时）
    if world_size > 1:
        model_adapter.base_model = DDP(
            model_adapter.base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )

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
    best_model_dir = os.path.join(args.output_dir, 'best_simplified_culturemoe')
    epoch_results = []

    for epoch in range(args.num_epochs):
        # 设置分布式采样器的epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process(rank):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_metrics = train_epoch_simplified(
            model_adapter, train_loader, optimizer, device,
            num_accumulation_steps=args.gradient_accumulation_steps,
            rank=rank,
            use_culture_loss=use_culture_loss,
            culture_loss_weight=args.culture_loss_weight
        )

        if is_main_process(rank):
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if train_metrics['main_loss'] > 0:
                print(f"    Main Loss: {train_metrics['main_loss']:.4f}")
                print(f"    Aux Loss: {train_metrics['aux_loss']:.4f}")
            if use_culture_loss:
                print(f"    Culture Loss: {train_metrics['culture_loss']:.4f}")

        # 每eval_interval个epoch进行一次验证
        if (epoch + 1) % args.eval_interval == 0:
            # 验证
            val_metrics = evaluate_simplified(
                model_adapter, val_loader, device, rank=rank,
                use_culture_loss=use_culture_loss,
                culture_loss_weight=args.culture_loss_weight
            )

            # 生成答案并评估准确率（只在主进程执行）
            if is_main_process(rank):
                gen_metrics = generate_and_evaluate_answers_simplified(
                    model_adapter, val_dataset, tokenizer, device, args.output_dir, epoch=epoch+1, rank=rank
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
                if use_culture_loss:
                    print(f"    Culture Loss: {val_metrics['culture_loss']:.4f}")
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
                    model_adapter.save_model(best_model_dir)

                    # 保存tokenizer
                    tokenizer.save_pretrained(best_model_dir)

                    print(f"  ✅ Best model saved (accuracy: {best_eval_accuracy:.4f})")

            # 记录结果
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_main_loss': train_metrics.get('main_loss', 0),
                'train_aux_loss': train_metrics.get('aux_loss', 0),
                'train_culture_loss': train_metrics.get('culture_loss', 0),
                'eval_loss': val_metrics['loss'],
                'eval_main_loss': val_metrics.get('main_loss', 0),
                'eval_aux_loss': val_metrics.get('aux_loss', 0),
                'eval_culture_loss': val_metrics.get('culture_loss', 0),
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
                'train_main_loss': train_metrics.get('main_loss', 0),
                'train_aux_loss': train_metrics.get('aux_loss', 0),
                'train_culture_loss': train_metrics.get('culture_loss', 0),
                'eval_loss': None,
                'eval_main_loss': None,
                'eval_aux_loss': None,
                'eval_culture_loss': None,
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
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'effective_batch_size': args.batch_size * world_size * args.gradient_accumulation_steps,
            'world_size': world_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'num_routing_experts': args.num_routing_experts,
            'moe_layers': moe_layers,
            'use_culture_loss': use_culture_loss,
            'culture_loss_weight': args.culture_loss_weight,
            'lora_config': {
                'rank': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout
            },
            'eval_interval': args.eval_interval,
            'best_eval_accuracy': best_eval_accuracy,
            'architecture': 'simplified_culturemoe_based_on_mixlora'
        }

        with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print("\n" + "="*80)
        print("✅ Training completed!")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")
        print(f"\nFiles generated:")
        print(f"  - best_simplified_culturemoe/ (Best model weights)")
        print(f"  - epoch_eval_results.json (Epoch-by-epoch results)")
        print(f"  - generated_answers.json (Generated answers on validation set)")
        print(f"  - config.json (Training configuration)")
        print(f"\nBest validation accuracy: {best_eval_accuracy:.4f}")
        print(f"Architecture: Simplified CultureMoE based on MixLoRA")
        print(f"MoE layers: {moe_layers} (last 2 layers only)")
        print(f"Routing experts: {args.num_routing_experts}")
        print(f"Culture loss: {'enabled' if use_culture_loss else 'disabled'}")
        print("="*80)

    # 清理分布式训练
    cleanup_distributed()


if __name__ == "__main__":
    main()