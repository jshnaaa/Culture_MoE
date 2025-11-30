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

        # 正确处理labels masking - 只计算output部分的loss
        for i in range(labels.shape[0]):
            instruction = batch['instruction'][i] if isinstance(batch['instruction'], list) else batch['instruction']
            input_text = batch['input'][i] if isinstance(batch['input'], list) else batch['input']
            output_text = batch['output'][i] if isinstance(batch['output'], list) else batch['output']

            # 构建input部分（需要mask的部分）- 与数据集格式完全保持一致
            # 数据集中的full_input就是需要mask的部分
            if input_text:
                input_part = f"{instruction}\n{input_text}"  # 这是数据集中的full_input
            else:
                input_part = instruction  # 这是数据集中的full_input

            # 计算input部分的token长度 - 使用与数据集相同的tokenization方式
            input_tokens = tokenizer(input_part, add_special_tokens=False, truncation=False)['input_ids']
            input_length = len(input_tokens)

            # 注意：input_part就是数据集中的full_input，连接output的"\n"应该被mask
            # 我们需要加上连接符"\n"的token数量，因为这个\n也属于输入部分
            separator_tokens = tokenizer("\n", add_special_tokens=False, truncation=False)['input_ids']
            separator_length = len(separator_tokens)

            # 实际需要mask的长度：input_part + 连接符"\n"
            # 这样只有pure output部分会用于计算loss
            actual_input_length = input_length + separator_length

            # 计算完整文本的长度以验证 - 与数据集格式完全一致
            # 数据集构建逻辑：full_input = f"{instruction}\n{input_text}"，然后 full_text = f"{full_input}\n{output_text}"
            if input_text:
                full_input = f"{instruction}\n{input_text}"
                full_text = f"{full_input}\n{output_text}"  # instruction\ninput_text\noutput_text
            else:
                full_input = instruction
                full_text = f"{full_input}\n{output_text}"  # instruction\noutput_text

            full_tokens = tokenizer(full_text, add_special_tokens=False, truncation=False)['input_ids']
            full_length = len(full_tokens)

            # 更安全的masking策略 - 使用实际的input长度
            if actual_input_length < labels.shape[1]:
                # 确保至少保留一些output tokens用于训练
                max_mask_length = min(actual_input_length, labels.shape[1] - 5)  # 至少保留5个token用于output
                labels[i, :max_mask_length] = -100
            else:
                # 如果input太长，保留最后10个token用于训练
                labels[i, :-10] = -100

            # 调试信息：检查labels masking
            if batch_idx < 1 and i == 0:  # 只在第一个batch的第一个样本显示
                valid_labels = (labels[i] != -100).sum().item()
                total_labels = labels.shape[1]
                print(f"🔍 Batch {batch_idx}, Sample {i}:")
                print(f"  input_length={input_length}, actual_input_length={actual_input_length}")
                print(f"  valid_labels={valid_labels}/{total_labels}")
                # print(f"  instruction: {repr(instruction)}")
                # print(f"  input_text: {repr(input_text)}")
                # print(f"  output: {repr(output_text)}")

                if valid_labels == 0:
                    print(f"  ⚠️ WARNING: No valid labels for training!")

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

        # 调试：检查模型输出的logits
        if batch_idx < 1:  # 只在第一个batch显示
            logits = outputs.logits
            print(f"🔍 Batch {batch_idx} - Logits analysis:")
            print(f"  logits.shape: {logits.shape}")
            print(f"  logits contains NaN: {torch.isnan(logits).any()}")
            print(f"  logits contains Inf: {torch.isinf(logits).any()}")
            # print(f"  logits.min: {logits.min().item():.6f}")
            # print(f"  logits.max: {logits.max().item():.6f}")

            # 检查特定位置的logits（答案位置）
            # for i in range(min(2, logits.shape[0])):  # 前2个样本
            #     sample_labels = labels[i]
            #     valid_positions = (sample_labels != -100).nonzero().flatten()
            #     if len(valid_positions) > 0:
            #         answer_pos = valid_positions[0].item()  # 第一个有效答案位置
            #         answer_logits = logits[i, answer_pos]
            #         print(f"  Sample {i}, pos {answer_pos} logits: min={answer_logits.min().item():.6f}, max={answer_logits.max().item():.6f}")
            #         print(f"  Answer token {sample_labels[answer_pos].item()}: logit={answer_logits[sample_labels[answer_pos]].item():.6f}")

        # 获取各种损失
        lm_loss = outputs.loss  # 语言模型损失
        expert_weights = getattr(outputs, 'expert_weights', None)  # 专家权重

        # 确保moe_aux_loss有梯度连接
        moe_aux_loss = getattr(outputs, 'moe_aux_loss', None)
        if moe_aux_loss is None:
            # 使用requires_grad=True的零张量确保梯度连接
            moe_aux_loss = torch.tensor(0.0, device=device, dtype=lm_loss.dtype, requires_grad=True)

        # 调试：检查MoE相关输出
        if batch_idx < 3:  # 只在前3个batch显示详细调试信息
            print(f"🔍 Batch {batch_idx} - MoE analysis:")
            print(f"  lm_loss: {lm_loss.item() if lm_loss is not None else 'None'}")
            print(f"  moe_aux_loss: {moe_aux_loss.item() if moe_aux_loss is not None else 'None'}")
            print(f"  moe_aux_loss.requires_grad: {moe_aux_loss.requires_grad if moe_aux_loss is not None else 'None'}")
            if expert_weights is not None:
                print(f"  expert_weights.shape: {expert_weights.shape}")
                print(f"  expert_weights[0]: {expert_weights[0].tolist()}")
                print(f"  expert_weights contains NaN: {torch.isnan(expert_weights).any()}")
                print(f"  expert_weights.requires_grad: {expert_weights.requires_grad}")

            # 检查hidden_states（MoE输出）
            # if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            #     hidden_states = outputs.hidden_states
            #     print(f"  hidden_states.min: {hidden_states.min().item():.6f}")
            #     print(f"  hidden_states.max: {hidden_states.max().item():.6f}")
            #     print(f"  hidden_states contains NaN: {torch.isnan(hidden_states).any()}")
            #     print(f"  hidden_states contains Inf: {torch.isinf(hidden_states).any()}")  # MoE辅助损失

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

        # 调试：检查梯度连接
        if batch_idx < 3:
            print(f"🔍 Batch {batch_idx} - Gradient check:")
            print(f"  lm_loss.requires_grad: {lm_loss.requires_grad}")
            print(f"  moe_aux_loss.requires_grad: {moe_aux_loss.requires_grad}")
            print(f"  culture_loss.requires_grad: {culture_loss.requires_grad}")
            print(f"  total_batch_loss.requires_grad: {total_batch_loss.requires_grad}")
            print(f"  lm_loss.dtype: {lm_loss.dtype}")
            print(f"  total_batch_loss.dtype: {total_batch_loss.dtype}")

        # 检查 NaN/Inf loss
        if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
            print(f"❌ NaN or Inf total loss detected at batch {batch_idx}")
            print(f"  LM loss: {lm_loss.item()}, MoE loss: {moe_aux_loss.item()}, Culture loss: {culture_loss.item()}")
            continue

        # 调试信息：显示实际损失值
        if batch_idx < 5 or batch_idx % 50 == 0:  # 前5个batch和每50个batch
            print(f"🔍 Batch {batch_idx} - Actual loss values:")
            print(f"  Total: {total_batch_loss.item():.8f}, LM: {lm_loss.item():.8f}, MoE: {moe_aux_loss.item():.8f}")
            if expert_weights is not None:
                print(f"  Expert weights: {expert_weights.mean(dim=0).detach().cpu().numpy()}")
            else:
                print(f"  Expert weights: None")

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
            # 分别对不同组件进行梯度裁剪
            # 对MoE路由器使用更严格的梯度裁剪
            moe_params = []
            other_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'moe_layer.router' in name:
                        moe_params.append(param)
                    else:
                        other_params.append(param)

            # 极严格的路由器梯度裁剪和权重保护
            if moe_params:
                # 预先检查和清理NaN/Inf梯度
                for name, param in model.named_parameters():
                    if 'moe_layer.router' in name and param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"⚠️ Cleaning NaN/Inf gradient in {name}")
                            param.grad.zero_()  # 清零有问题的梯度

                # 极严格的梯度裁剪
                torch.nn.utils.clip_grad_norm_(moe_params, max_norm=0.05)  # 进一步降低到0.05

                # 权重更新后立即检查和修复
                for name, param in model.named_parameters():
                    if 'moe_layer.router' in name:
                        with torch.no_grad():
                            # 检查权重是否超出安全范围
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                print(f"⚠️ Post-update NaN/Inf in {name}, resetting")
                                if 'weight' in name:
                                    torch.nn.init.normal_(param, mean=0.0, std=0.0001)
                                elif 'bias' in name:
                                    torch.nn.init.constant_(param, 0.0)
                            else:
                                # 即使没有NaN/Inf，也要限制权重范围防止溢出
                                param.clamp_(-1.0, 1.0)  # 严格限制权重范围

            # 其他参数使用正常梯度裁剪
            if other_params:
                torch.nn.utils.clip_grad_norm_(other_params, max_norm=0.5)

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