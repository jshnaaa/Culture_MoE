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
    generate_answer,
    dynamic_padding_collate_fn
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
        return torch.tensor(0.0, device=culture_labels.device, dtype=torch.float16)

    expert_weights = model_outputs.expert_weights  # [B, num_experts]
    batch_size = expert_weights.shape[0]

    # 检查维度匹配
    if expert_weights.shape[0] != culture_labels.shape[0]:
        print(f"⚠️ Dimension mismatch: expert_weights.shape={expert_weights.shape}, culture_labels.shape={culture_labels.shape}")
        return torch.tensor(0.0, device=culture_labels.device, dtype=torch.float16)

    if batch_size < 2:
        return torch.tensor(0.0, device=culture_labels.device, dtype=torch.float16)

    # 统一使用float16以节省显存
    culture_loss = torch.tensor(0.0, device=culture_labels.device, dtype=torch.float16)
    count = 0

    # 计算同文化样本间的相似性和不同文化样本间的差异性
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if culture_labels[i] == culture_labels[j]:
                # 相同文化，鼓励相似的专家权重
                vec1 = expert_weights[i].unsqueeze(0)
                vec2 = expert_weights[j].unsqueeze(0)

                # 检查向量是否为零向量，避免cosine_similarity中的NaN
                norm1 = torch.norm(vec1)
                norm2 = torch.norm(vec2)
                if norm1 < 1e-8 or norm2 < 1e-8:
                    # 如果任一向量接近零向量，跳过这对比较
                    continue

                similarity = F.cosine_similarity(vec1, vec2)
                # 确保similarity使用float16并检查数值稳定性
                similarity = similarity.to(dtype=torch.float16)
                if torch.isnan(similarity) or torch.isinf(similarity):
                    continue  # 跳过无效的相似度计算

                # 确保similarity是标量值，避免广播问题
                similarity_scalar = similarity.item() if similarity.numel() == 1 else similarity.mean().item()
                culture_loss += (1.0 - similarity_scalar)
            else:
                # 不同文化，鼓励不同的专家权重
                vec1 = expert_weights[i].unsqueeze(0)
                vec2 = expert_weights[j].unsqueeze(0)

                # 检查向量是否为零向量，避免cosine_similarity中的NaN
                norm1 = torch.norm(vec1)
                norm2 = torch.norm(vec2)
                if norm1 < 1e-8 or norm2 < 1e-8:
                    # 如果任一向量接近零向量，跳过这对比较
                    continue

                similarity = F.cosine_similarity(vec1, vec2)
                # 确保similarity使用float16并检查数值稳定性
                similarity = similarity.to(dtype=torch.float16)
                if torch.isnan(similarity) or torch.isinf(similarity):
                    continue  # 跳过无效的相似度计算

                # 确保similarity是标量值，避免广播问题
                similarity_scalar = similarity.item() if similarity.numel() == 1 else similarity.mean().item()
                culture_loss += similarity_scalar
            count += 1

    if count > 0:
        culture_loss = culture_loss / count * loss_weight

    # 确保返回的是标量张量
    if not isinstance(culture_loss, torch.Tensor):
        culture_loss = torch.tensor(culture_loss, device=culture_labels.device, dtype=torch.float16)

    # 检查文化损失是否为NaN/Inf，如果是则返回零损失
    if torch.isnan(culture_loss) or torch.isinf(culture_loss):
        culture_loss = torch.tensor(0.0, device=culture_labels.device, dtype=torch.float16)

    return culture_loss


def train_epoch_simplified(model_adapter, train_loader, optimizer, device, tokenizer,
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

        # 正确处理labels masking - 只计算output部分的loss
        # 获取instruction和input的文本长度来确定mask位置
        for i in range(labels.shape[0]):
            instruction = batch['instruction'][i] if isinstance(batch['instruction'], list) else batch['instruction']
            input_text = batch['input'][i] if isinstance(batch['input'], list) else batch['input']

            # 构建input部分（需要mask的部分）
            if input_text:
                input_part = f"{instruction}\n{input_text}\n"
            else:
                input_part = f"{instruction}\n"

            # 计算input部分的token长度
            input_tokens = tokenizer(input_part, add_special_tokens=False)['input_ids']
            input_length = len(input_tokens)

            # Mask掉input部分，只保留output部分用于loss计算
            if input_length < labels.shape[1]:
                labels[i, :input_length] = -100

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

        # 计算文化损失 - 统一使用float16节省显存
        culture_loss = torch.tensor(0.0, device=device, dtype=torch.float16)
        if use_culture_loss != 'false' and culture_labels is not None:
            culture_loss = compute_culture_loss(outputs, culture_labels, culture_loss_weight)

        # 获取MoE的z-loss用于稳定router
        z_loss = model_adapter.get_accumulated_z_loss()

        # 将主损失转换为float16以保持一致性和节省显存
        loss = loss.to(dtype=torch.float16)

        # 总损失
        total_batch_loss = loss + culture_loss + z_loss

        # 检查 NaN/Inf loss - 在所有损失计算完成后检查
        if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
            print(f"❌ NaN or Inf total loss detected at batch {batch_idx}")
            print(f"  Main loss: {loss.item()}, Culture loss: {culture_loss.item()}, Z loss: {z_loss.item()}")
            continue

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

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model_adapter.base_model.parameters(), max_norm=1.0)

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
        if use_culture_loss != 'false':
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


def evaluate_simplified(model_adapter, val_loader, device, tokenizer, rank=0, use_culture_loss=True, culture_loss_weight=0.01):
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

            # 正确处理labels masking - 只计算output部分的loss
            for i in range(labels.shape[0]):
                instruction = batch['instruction'][i] if isinstance(batch['instruction'], list) else batch['instruction']
                input_text = batch['input'][i] if isinstance(batch['input'], list) else batch['input']

                # 构建input部分（需要mask的部分）
                if input_text:
                    input_part = f"{instruction}\n{input_text}\n"
                else:
                    input_part = f"{instruction}\n"

                # 计算input部分的token长度
                input_tokens = tokenizer(input_part, add_special_tokens=False)['input_ids']
                input_length = len(input_tokens)

                # Mask掉input部分，只保留output部分用于loss计算
                if input_length < labels.shape[1]:
                    labels[i, :input_length] = -100

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

            # 计算文化损失 - 统一使用float16节省显存
            culture_loss = torch.tensor(0.0, device=device, dtype=torch.float16)
            if use_culture_loss != 'false' and culture_labels is not None:
                culture_loss = compute_culture_loss(outputs, culture_labels, culture_loss_weight)

            # 获取MoE的z-loss用于稳定router
            z_loss = model_adapter.get_accumulated_z_loss()

            # 将主损失转换为float16以保持一致性和节省显存
            loss = loss.to(dtype=torch.float16)

            total_batch_loss = loss + culture_loss + z_loss

            # 检查总损失是否为NaN/Inf
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                continue
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

    # 模型参数 - 与joint版本保持一致
    parser.add_argument("--backbone", type=str, default="llama", choices=["llama", "qwen"],
                        help="Model backbone type")
    parser.add_argument("--use_shared", type=str, default="false",
                        help="Whether to use shared expert (placeholder)")
    parser.add_argument("--use_gate", type=str, default="false",
                        help="Whether to use MoE gate (placeholder)")
    parser.add_argument("--num_moe_experts", type=int, default=4,
                        help="Number of MoE experts")
    parser.add_argument("--use_culture_loss", type=str, default="new",
                        help="Culture loss mode: ori/new/kl/false")
    parser.add_argument("--num_activated_experts", type=int, default=2,
                        help="Number of activated experts (top-k), if equal to num_moe_experts then dense mode")
    parser.add_argument("--use_lora", type=str, default="true",
                        help="Whether to enable LoRA fine-tuning")
    parser.add_argument("--culture_loss_weight", type=float, default=0.01,
                        help="Culture loss weight")

    # LoRA参数 - 与joint版本保持一致
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    parser.add_argument("--memory_efficient", action='store_true',
                        help="Enable memory efficient training")

    args = parser.parse_args()

    # 转换字符串参数 - 与joint版本保持一致
    use_culture_loss = args.use_culture_loss.lower() if args.use_culture_loss.lower() in ['ori', 'new', 'kl', 'false'] else 'new'
    use_shared = args.use_shared.lower() == 'true'
    use_gate = args.use_gate.lower() == 'true'
    use_lora = args.use_lora.lower() == 'true'

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
        print(f"Learning rate: {args.learning_rate} (简化版使用单一学习率)")
        print(f"Max length: {args.max_length}")
        print(f"MoE experts: {args.num_moe_experts}")
        print(f"Activated experts: {args.num_activated_experts} ({'dense mode' if args.num_activated_experts == args.num_moe_experts else f'top-{args.num_activated_experts}'})")
        print(f"Use shared expert: {use_shared} (占位符)")
        print(f"Use MoE gate: {use_gate} (占位符)")
        print(f"Use culture loss: {use_culture_loss}")
        if use_culture_loss != 'false':
            print(f"Culture loss weight: {args.culture_loss_weight}")
        print(f"Use LoRA: {use_lora}")
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

    # 创建动态padding的collate函数
    def collate_fn(batch):
        return dynamic_padding_collate_fn(batch, tokenizer)

    # 创建数据加载器 - 使用动态padding
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,  # 设为0避免多进程问题
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
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

    # 获取模型层数
    if args.backbone == "qwen":
        total_layers = 28
    else:  # llama
        total_layers = 32

    # 创建简化版CultureMoE配置 - 与MixLoRA保持一致，所有层都使用MoE
    if is_main_process(rank):
        print(f"\nConfiguring Simplified CultureMoE (like MixLoRA)...")
        print(f"Total layers: {total_layers}")
        print(f"MoE layers: ALL layers (like MixLoRA)")

    culturemoe_config = SimplifiedCultureMoEConfig(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_moe_experts=args.num_moe_experts,
        num_activated_experts=args.num_activated_experts,
        moe_layers=None,  # None表示最后两层替换为MoE
        use_shared=use_shared,  # 占位符
        use_gate=use_gate,      # 占位符
        use_culture_loss=use_culture_loss,
        culture_loss_weight=args.culture_loss_weight,
        use_lora=use_lora,
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

        # DDP包装后确保dtype一致性
        model_adapter._ensure_device_consistency()
        if is_main_process(rank):
            print("✅ Ensured dtype consistency after DDP wrapping")

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
            model_adapter, train_loader, optimizer, device, tokenizer,
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
            if use_culture_loss != 'false':
                print(f"    Culture Loss: {train_metrics['culture_loss']:.4f}")

        # 每eval_interval个epoch进行一次验证
        if (epoch + 1) % args.eval_interval == 0:
            # 验证
            val_metrics = evaluate_simplified(
                model_adapter, val_loader, device, tokenizer, rank=rank,
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
                if use_culture_loss != 'false':
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
            'training_mode': 'simplified_last_two_layers_moe',
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'effective_batch_size': args.batch_size * world_size * args.gradient_accumulation_steps,
            'world_size': world_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'use_shared_expert': use_shared,
            'use_moe_gate': use_gate,
            'num_moe_experts': args.num_moe_experts,
            'num_activated_experts': args.num_activated_experts,
            'moe_layers': 'Last 2 layers FFN replaced with MoE',
            'use_culture_loss': use_culture_loss,
            'culture_loss_weight': args.culture_loss_weight,
            'use_lora': use_lora,
            'lora_config': {
                'rank': args.lora_rank,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout
            },
            'eval_interval': args.eval_interval,
            'best_eval_accuracy': best_eval_accuracy,
            'architecture': 'simplified_last_two_layers_moe'
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
        print(f"Architecture: Simplified Last Two Layers MoE")
        print(f"MoE layers: Last 2 layers FFN replaced with MoE")
        print(f"MoE experts: {args.num_moe_experts}")
        print(f"Activated experts: {args.num_activated_experts} ({'dense mode' if args.num_activated_experts == args.num_moe_experts else f'top-{args.num_activated_experts}'})")
        print(f"Use LoRA: {use_lora}")
        print(f"Culture loss: {use_culture_loss}")
        print("="*80)

    # 清理分布式训练
    cleanup_distributed()


if __name__ == "__main__":
    main()