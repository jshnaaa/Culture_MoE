#!/usr/bin/env python3
"""
联合训练脚本：同时训练预训练LoRA适配器 + 新增MoE处理层
实现端到端的LoRA+MoE联合优化训练
针对48GB×2卡优化

特点：
1. 同时训练基础模型的LoRA适配器和新增MoE层
2. 分层学习率：基础LoRA使用较小学习率，MoE使用较大学习率
3. 端到端优化，避免预训练LoRA权重冻结
4. 内存优化和DDP支持
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

from src.llamafactory.model.joint_lora_moe_model import JointLoRAMoEModel, JointLoRAMoEConfig

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


def compute_culture_loss(expert_weights, culture_labels, loss_weight=0.01):
    """
    计算文化感知损失

    Args:
        expert_weights: 专家权重 [B, num_experts]
        culture_labels: 文化标签 [B]
        loss_weight: 损失权重

    Returns:
        culture_loss: 文化损失
    """
    if expert_weights is None or culture_labels is None:
        return torch.tensor(0.0, device=culture_labels.device if culture_labels is not None else torch.device('cuda'), dtype=torch.float16)

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
                    continue

                similarity = F.cosine_similarity(vec1, vec2)
                similarity = similarity.to(dtype=torch.float16)
                if torch.isnan(similarity) or torch.isinf(similarity):
                    continue

                culture_loss += (1.0 - similarity)
            else:
                # 不同文化，鼓励不同的专家权重
                vec1 = expert_weights[i].unsqueeze(0)
                vec2 = expert_weights[j].unsqueeze(0)

                norm1 = torch.norm(vec1)
                norm2 = torch.norm(vec2)
                if norm1 < 1e-8 or norm2 < 1e-8:
                    continue

                similarity = F.cosine_similarity(vec1, vec2)
                similarity = similarity.to(dtype=torch.float16)
                if torch.isnan(similarity) or torch.isinf(similarity):
                    continue

                culture_loss += similarity
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


def train_epoch_joint(model, train_loader, optimizer, device, tokenizer,
                     num_accumulation_steps=1, rank=0, use_culture_loss=True, culture_loss_weight=0.01):
    """
    联合训练一个epoch：同时训练LoRA和MoE
    """
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_moe_loss = 0
    total_culture_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Joint Training", disable=(rank != 0), mininterval=1.0)

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # ✅ 标签掩码已在数据集级别正确处理，不需要在训练时重复处理
        # 🔍 只添加调试信息来验证数据集的标签掩码是否正确
        if batch_idx < 1:  # 只在第一个batch显示
            for i in range(min(1, labels.shape[0])):  # 只检查第一个样本
                valid_labels = (labels[i] != -100).sum().item()
                total_labels = labels.shape[1]
                non_pad_labels = (labels[i] != tokenizer.pad_token_id).sum().item()

                print(f"🔍 数据集标签掩码验证 - Batch {batch_idx}, Sample {i}:")
                print(f"  总标签数: {total_labels}")
                print(f"  有效训练标签数: {valid_labels}")
                print(f"  非pad标签数: {non_pad_labels}")
                print(f"  有效标签比例: {valid_labels/non_pad_labels:.1%}")

                if valid_labels == 0:
                    print(f"  ❌ 警告: 没有有效的训练标签!")
                elif valid_labels < 3:
                    print(f"  ⚠️ 警告: 有效训练标签太少 ({valid_labels})")
                else:
                    print(f"  ✅ 有效训练标签数量合理")

        # 获取文化标签
        culture_labels = None
        if 'culture_labels' in batch:
            culture_labels = batch['culture_labels'].to(device)
        elif 'label' in batch:
            if isinstance(batch['label'], list):
                label_ints = [int(label) if label.isdigit() else 0 for label in batch['label']]
                culture_labels = torch.tensor(label_ints, dtype=torch.long, device=device)
            else:
                culture_labels = batch['label'].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # 📋 Labels调试信息（前3个batch）- 注释掉，专注tokenizer问题
        # if batch_idx < 3:
        #     first_sample_labels = labels[0]
        #     non_mask_positions = (first_sample_labels != -100).nonzero(as_tuple=True)[0]
        #     valid_count = len(non_mask_positions)

        #     print(f"📋 Batch {batch_idx} Labels: 有效标签数={valid_count}")

        #     if valid_count > 0 and valid_count <= 10:  # 只有当标签数合理时才显示详情
        #         for i, pos in enumerate(non_mask_positions[:3]):  # 只显示前3个
        #             pos_idx = pos.item()
        #             label_val = first_sample_labels[pos_idx].item()
        #             try:
        #                 token_text = tokenizer.decode([label_val], skip_special_tokens=True)
        #                 print(f"  位置{pos_idx}: {label_val}='{token_text}'")
        #             except:
        #                 print(f"  位置{pos_idx}: {label_val}=(解码失败)")
        #     elif valid_count > 10:
        #         print(f"  ⚠️ 标签数过多，可能仍有padding问题")
        #     else:
        #         print(f"  ❌ 没有有效训练标签")

        # 简化的logits检查（仅在前3个batch）- 注释掉，专注tokenizer问题
        # if batch_idx < 3:
        #     logits = outputs.logits
        #     if torch.isnan(logits).any() or torch.isinf(logits).any():
        #         print(f"⚠️ Batch {batch_idx} - Invalid logits detected")

        # 获取各种损失
        lm_loss = outputs.loss  # 语言模型损失
        expert_weights = getattr(outputs, 'expert_weights', None)  # 专家权重

        # 确保moe_aux_loss有梯度连接
        moe_aux_loss = getattr(outputs, 'moe_aux_loss', None)
        if moe_aux_loss is None:
            # 使用requires_grad=True的零张量确保梯度连接
            moe_aux_loss = torch.tensor(0.0, device=device, dtype=lm_loss.dtype, requires_grad=True)

        # 简化的MoE检查（仅在前3个batch）- 注释掉，专注tokenizer问题
        # if batch_idx < 3:
        #     if torch.isnan(moe_aux_loss).any() or torch.isinf(moe_aux_loss).any():
        #         print(f"⚠️ Batch {batch_idx} - Invalid MoE aux loss")
        #     if expert_weights is not None and (torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any()):
        #         print(f"⚠️ Batch {batch_idx} - Invalid expert weights")

        # 计算文化损失 - 确保有梯度连接
        if use_culture_loss and culture_labels is not None and expert_weights is not None:
            culture_loss = compute_culture_loss(expert_weights, culture_labels, culture_loss_weight)
        else:
            # 使用requires_grad=True的零张量确保梯度连接
            culture_loss = torch.tensor(0.0, device=device, dtype=lm_loss.dtype, requires_grad=True)

        # 注意：不要转换lm_loss的dtype，这会断开梯度连接
        # lm_loss = lm_loss.to(dtype=torch.float16)  # 这行代码会断开梯度！

        # 总损失
        total_batch_loss = lm_loss + moe_aux_loss + culture_loss

        # 简化的梯度检查（仅在前3个batch）- 注释掉，专注tokenizer问题
        # if batch_idx < 3:
        #     if not total_batch_loss.requires_grad:
        #         print(f"⚠️ Batch {batch_idx} - Total loss missing gradients")

        # 检查 NaN/Inf loss
        if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
            print(f"❌ NaN or Inf total loss detected at batch {batch_idx}")
            print(f"  LM loss: {lm_loss.item()}, MoE loss: {moe_aux_loss.item()}, Culture loss: {culture_loss.item()}")
            continue

        # 显示关键训练信息 - 注释掉，专注tokenizer问题
        # if batch_idx < 5 or batch_idx % 50 == 0:  # 前5个batch和每50个batch
        #     print(f"📊 Batch {batch_idx} - Loss: Total={total_batch_loss.item():.4f}, LM={lm_loss.item():.4f}, MoE={moe_aux_loss.item():.6f}")
        #     if expert_weights is not None:
        #         expert_avg = expert_weights.mean(dim=0).detach().cpu().numpy()
        #         expert_str = ", ".join([f"E{i}={w:.3f}" for i, w in enumerate(expert_avg)])
        #         print(f"📊 Expert weights: [{expert_str}]")

        # 梯度累积
        total_batch_loss = total_batch_loss / num_accumulation_steps
        total_batch_loss.backward()

        total_loss += total_batch_loss.item() * num_accumulation_steps
        total_lm_loss += lm_loss.item()
        total_moe_loss += moe_aux_loss.item()
        total_culture_loss += culture_loss.item()
        num_batches += 1

        # 梯度更新
        if (batch_idx + 1) % num_accumulation_steps == 0:
            # 简化的梯度处理（Float32路由器不需要特殊处理）
            # 检查和清理任何NaN/Inf梯度
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"⚠️ Cleaning NaN/Inf gradient in {name}")
                        param.grad.zero_()

            # 统一的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            # 不需要手动清理梯度，zero_grad()已经处理了
            # for param in model.parameters():
            #     if param.grad is not None:
            #         param.grad = None

        # 积极的内存清理
        if (batch_idx + 1) % 1 == 0:  # 每个batch都清理
            torch.cuda.empty_cache()

        # 检查内存使用并提前清理
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            if memory_allocated > 35:  # 降低阈值，提前清理
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()  # 再次清理
                if rank == 0:
                    print(f"⚠️ High memory usage ({memory_allocated:.1f}GB), forced cleanup")

        # 如果内存仍然过高，暂停一下
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            if memory_allocated > 42:  # 接近限制时暂停
                if rank == 0:
                    print(f"⚠️ Critical memory usage ({memory_allocated:.1f}GB), pausing...")
                import time
                time.sleep(1)
                torch.cuda.empty_cache()
                import gc
                gc.collect()

        # 更新进度条 - 增加显示精度
        postfix = {
            'loss': f"{total_batch_loss.item() * num_accumulation_steps:.6f}",
            'lm': f"{lm_loss.item():.6f}",
            'moe': f"{moe_aux_loss.item():.6f}"
        }
        if use_culture_loss:
            postfix['culture'] = f"{culture_loss.item():.4f}"

        pbar.set_postfix(postfix)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_lm_loss = total_lm_loss / num_batches if num_batches > 0 else 0
    avg_moe_loss = total_moe_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'moe_loss': avg_moe_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def evaluate_joint(model, val_loader, device, tokenizer, rank=0, use_culture_loss=True, culture_loss_weight=0.01):
    """
    联合模型验证
    """
    model.eval()
    total_loss = 0
    total_lm_loss = 0
    total_moe_loss = 0
    total_culture_loss = 0
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

            # 获取文化标签
            culture_labels = None
            if 'culture_labels' in batch:
                culture_labels = batch['culture_labels'].to(device)
            elif 'label' in batch:
                if isinstance(batch['label'], list):
                    label_ints = [int(label) if label.isdigit() else 0 for label in batch['label']]
                    culture_labels = torch.tensor(label_ints, dtype=torch.long, device=device)
                else:
                    culture_labels = batch['label'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

            lm_loss = outputs.loss
            expert_weights = getattr(outputs, 'expert_weights', None)
            moe_aux_loss = getattr(outputs, 'moe_aux_loss', torch.tensor(0.0, device=device, dtype=torch.float16))

            # 计算文化损失
            culture_loss = torch.tensor(0.0, device=device, dtype=torch.float16)
            if use_culture_loss and culture_labels is not None and expert_weights is not None:
                culture_loss = compute_culture_loss(expert_weights, culture_labels, culture_loss_weight)

            lm_loss = lm_loss.to(dtype=torch.float16)
            total_batch_loss = lm_loss + moe_aux_loss + culture_loss

            # 检查总损失是否为NaN/Inf
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                continue

            total_loss += total_batch_loss.item()
            total_lm_loss += lm_loss.item()
            total_moe_loss += moe_aux_loss.item()
            total_culture_loss += culture_loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{total_batch_loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_lm_loss = total_lm_loss / num_batches if num_batches > 0 else 0
    avg_moe_loss = total_moe_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'moe_loss': avg_moe_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def generate_and_evaluate_answers_joint(
    model, val_dataset, tokenizer, device, output_dir, epoch=None, rank=0
):
    """
    联合模型生成答案并评估准确率
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

        # 生成答案（使用完整的联合模型，包含MoE层）
        # 处理DDP包装的模型
        full_model = model.module if hasattr(model, 'module') else model
        generated_text = generate_answer(
            full_model, tokenizer, instruction, input_text, device
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

    # 打印前三条生成的答案（简化版）
    if rank == 0:
        print("\n📋 生成答案样例:")
        for idx in range(min(3, len(generated_data))):
            item = generated_data[idx]
            correct_mark = '✅' if item['correct'] else '❌'
            print(f"  样本{idx+1}: 真实={item['true_output']}, 预测={item['predicted_answer']}, 生成='{item['generated_text'][:20]}...' {correct_mark}")
        print()

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    # 初始化分布式训练
    rank, world_size, local_rank = setup_distributed()

    parser = argparse.ArgumentParser(description="Joint LoRA + MoE Training")

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
    parser.add_argument("--learning_rate_base", type=float, default=5e-5,
                        help="Learning rate for base LoRA")
    parser.add_argument("--learning_rate_moe", type=float, default=1e-4,
                        help="Learning rate for MoE components")
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
                        help="Number of MoE experts")
    parser.add_argument("--use_culture_loss", type=str, default="true",
                        help="Whether to use culture loss")
    parser.add_argument("--culture_loss_weight", type=float, default=0.01,
                        help="Culture loss weight")

    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
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
        print("联合训练 LoRA + MoE")
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
        print(f"Base LoRA learning rate: {args.learning_rate_base}")
        print(f"MoE learning rate: {args.learning_rate_moe}")
        print(f"Max length: {args.max_length}")
        print(f"MoE experts: {args.num_moe_experts}")
        print(f"Use culture loss: {use_culture_loss}")
        if use_culture_loss:
            print(f"Culture loss weight: {args.culture_loss_weight}")
        print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")
        print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    # 🔧 修复Llama 3.1 tokenizer配置问题
    if tokenizer.pad_token is None:
        # 检查是否是Llama 3.1 (eos_token_id是128009)
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128009:
            # Llama 3.1: 查找真正的<unk> token
            if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
                # 使用真正的<unk> token
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
                print(f"🔧 Llama 3.1: 使用真正的<unk> token: '{tokenizer.unk_token}' (id={tokenizer.pad_token_id})")
            else:
                # 如果没有<unk>，使用eos_token作为padding（避免添加新token）
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"🔧 Llama 3.1: 使用eos_token作为padding: '{tokenizer.eos_token}' (id={tokenizer.pad_token_id})")
        else:
            # 其他模型：使用标准配置
            tokenizer.pad_token = tokenizer.eos_token
            print(f"🔧 标准配置: pad_token = eos_token")

    tokenizer.padding_side = "right"

    # 验证tokenizer配置
    print(f"✅ Tokenizer配置验证:")
    print(f"  pad_token: {repr(tokenizer.pad_token)}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  eos_token: {repr(tokenizer.eos_token)}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  unk_token: {repr(tokenizer.unk_token)}")
    if hasattr(tokenizer, 'unk_token_id'):
        print(f"  unk_token_id: {tokenizer.unk_token_id}")

    # 🚨 强制验证和修复
    if tokenizer.pad_token_id == 128009:
        print(f"🚨 严重错误: pad_token_id仍然是128009 (<|eot_id|>)!")
        print(f"   强制修复tokenizer配置...")

        # 使用eos_token作为padding（避免添加新token）
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"   修复后pad_token_id: {tokenizer.pad_token_id}")
        print(f"   修复后pad_token: {repr(tokenizer.pad_token)}")

    elif tokenizer.pad_token_id is None:
        print(f"🚨 错误: pad_token_id is None!")
        print(f"   强制设置eos_token作为padding...")

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"   设置后pad_token_id: {tokenizer.pad_token_id}")
        print(f"   设置后pad_token: {repr(tokenizer.pad_token)}")

    elif tokenizer.pad_token_id == 0:
        # 检查token_id=0对应的实际字符
        token_0_text = tokenizer.decode([0], skip_special_tokens=True)
        print(f"⚠️ pad_token_id = 0，对应字符: '{token_0_text}'")

        if token_0_text.strip() not in ['<unk>', '']:  # 如果不是真正的<unk>
            print(f"🚨 问题: token_id=0 不是真正的<unk>，而是'{token_0_text}'!")
            print(f"   这会导致padding区域填充'{token_0_text}'字符")
            print(f"   强制创建专用padding token...")

            # 使用一个现有的低频token作为padding，避免添加新token
            # 找一个不常用的标点符号token
            candidate_tokens = ['~', '`', '|', '^']
            chosen_pad_token = None

            for token in candidate_tokens:
                try:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if token_id != tokenizer.unk_token_id:  # 确保不是unk
                        chosen_pad_token = token
                        tokenizer.pad_token = token
                        tokenizer.pad_token_id = token_id
                        break
                except:
                    continue

            if chosen_pad_token is None:
                # 如果找不到合适的token，使用eos_token
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                chosen_pad_token = tokenizer.eos_token

            print(f"   修复后pad_token_id: {tokenizer.pad_token_id}")
            print(f"   修复后pad_token: {repr(chosen_pad_token)}")
        else:
            print(f"✅ 正确: pad_token_id = 0 是真正的<unk>")
    else:
        print(f"✅ 正确: pad_token_id ({tokenizer.pad_token_id}) != 128009")

    # 最终验证
    print(f"\n🔧 最终tokenizer状态:")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  是否等于<|eot_id|>: {tokenizer.pad_token_id == 128009}")
    print(f"  是否为None: {tokenizer.pad_token_id is None}")

    print("✅ Tokenizer loaded")

    # 🚨 训练前最终tokenizer验证
    print(f"\n🚨 训练前最终tokenizer验证:")
    print(f"  pad_token: {repr(tokenizer.pad_token)}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  eos_token: {repr(tokenizer.eos_token)}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")

    # 测试tokenizer实际行为
    test_text = "Hello world"
    test_encoded = tokenizer(test_text, max_length=10, padding='max_length', truncation=True, return_tensors='pt')
    test_input_ids = test_encoded['input_ids'][0]
    print(f"  测试序列: {test_input_ids.tolist()}")

    # 检查padding token在实际序列中的表现
    padding_positions = (test_input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
    if len(padding_positions) > 0:
        pad_token_id = tokenizer.pad_token_id
        pad_token_text = tokenizer.decode([pad_token_id], skip_special_tokens=True)
        print(f"  padding token {pad_token_id} 解码为: '{pad_token_text}'")

        if pad_token_text.strip() not in ['', '<pad>', '<unk>']:
            print(f"  🚨 严重警告: padding token解码为有意义字符'{pad_token_text}'!")
            print(f"  这会导致训练标签包含大量无意义字符!")

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

    # 创建联合LoRA+MoE配置
    if is_main_process(rank):
        print(f"\nConfiguring Joint LoRA + MoE...")

    joint_config = JointLoRAMoEConfig(
        # LoRA配置
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        # MoE配置
        num_moe_experts=args.num_moe_experts,
        moe_hidden_dim=base_model.config.hidden_size,  # 将自动在模型初始化时设置

        # 文化损失配置
        use_culture_loss=use_culture_loss,
        culture_loss_weight=args.culture_loss_weight,

        # 其他配置
        dropout=0.1
    )

    # 创建联合模型
    model = JointLoRAMoEModel(base_model, joint_config)

    if is_main_process(rank):
        model.print_trainable_parameters()
        print("✅ Joint LoRA + MoE configured")

    # 确保所有参数在正确设备上（在DDP包装前）
    torch.cuda.empty_cache()

    # 使用DDP包装模型（仅在多GPU时）
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # 设置为True以处理动态创建的层
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )

        # 设置静态图以避免参数重复标记问题
        model._set_static_graph()

        if is_main_process(rank):
            print("✅ Model wrapped with DDP with static graph")

    # 设置分层优化器
    if hasattr(model, 'module'):
        param_groups = model.module.get_parameter_groups(
            base_lr=args.learning_rate_base,
            moe_lr=args.learning_rate_moe
        )
    else:
        param_groups = model.get_parameter_groups(
            base_lr=args.learning_rate_base,
            moe_lr=args.learning_rate_moe
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # 训练循环
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Starting joint training...")
        print("="*80 + "\n")

    best_eval_accuracy = 0.0
    best_model_dir = os.path.join(args.output_dir, 'best_joint_model')
    epoch_results = []

    for epoch in range(args.num_epochs):
        # 设置分布式采样器的epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process(rank):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_metrics = train_epoch_joint(
            model, train_loader, optimizer, device, tokenizer,
            num_accumulation_steps=args.gradient_accumulation_steps,
            rank=rank,
            use_culture_loss=use_culture_loss,
            culture_loss_weight=args.culture_loss_weight
        )

        if is_main_process(rank):
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    LM Loss: {train_metrics['lm_loss']:.4f}")
            print(f"    MoE Loss: {train_metrics['moe_loss']:.4f}")
            if use_culture_loss:
                print(f"    Culture Loss: {train_metrics['culture_loss']:.4f}")

        # 每eval_interval个epoch进行一次验证
        if (epoch + 1) % args.eval_interval == 0:
            # 验证
            val_metrics = evaluate_joint(
                model, val_loader, device, tokenizer, rank=rank,
                use_culture_loss=use_culture_loss,
                culture_loss_weight=args.culture_loss_weight
            )

            # 生成答案并评估准确率（只在主进程执行）
            if is_main_process(rank):
                gen_metrics = generate_and_evaluate_answers_joint(
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
                'train_culture_loss': train_metrics.get('culture_loss', 0),
                'eval_loss': val_metrics['loss'],
                'eval_lm_loss': val_metrics['lm_loss'],
                'eval_moe_loss': val_metrics['moe_loss'],
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
                'train_lm_loss': train_metrics['lm_loss'],
                'train_moe_loss': train_metrics['moe_loss'],
                'train_culture_loss': train_metrics.get('culture_loss', 0),
                'eval_loss': None,
                'eval_lm_loss': None,
                'eval_moe_loss': None,
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
            'training_mode': 'joint_lora_moe',
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'effective_batch_size': args.batch_size * world_size * args.gradient_accumulation_steps,
            'world_size': world_size,
            'learning_rate_base': args.learning_rate_base,
            'learning_rate_moe': args.learning_rate_moe,
            'max_length': args.max_length,
            'num_moe_experts': args.num_moe_experts,
            'use_culture_loss': use_culture_loss,
            'culture_loss_weight': args.culture_loss_weight,
            'lora_config': {
                'rank': args.lora_rank,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout
            },
            'eval_interval': args.eval_interval,
            'best_eval_accuracy': best_eval_accuracy,
            'architecture': 'joint_lora_moe_end_to_end'
        }

        with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print("\n" + "="*80)
        print("✅ Joint training completed!")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")
        print(f"\nFiles generated:")
        print(f"  - best_joint_model/ (Best model weights)")
        print(f"  - epoch_eval_results.json (Epoch-by-epoch results)")
        print(f"  - generated_answers.json (Generated answers on validation set)")
        print(f"  - config.json (Training configuration)")
        print(f"\nBest validation accuracy: {best_eval_accuracy:.4f}")
        print(f"Architecture: Joint LoRA + MoE End-to-End Training")
        print(f"MoE experts: {args.num_moe_experts}")
        print(f"Culture loss: {'enabled' if use_culture_loss else 'disabled'}")
        print("="*80)

    # 清理分布式训练
    cleanup_distributed()


if __name__ == "__main__":
    main()