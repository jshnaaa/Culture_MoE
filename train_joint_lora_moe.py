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

# 导入增强MoE损失函数
from enhanced_moe_losses import integrate_enhanced_moe_loss

# 复用现有的数据集类
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    load_and_process_data,
    load_and_process_data_8_1_1,
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


def compute_csl_culture_loss(expert_weights, shared_expert_outputs, router_expert_outputs, culture_labels, loss_weight=0.01):
    """
    计算CSL文化相似性损失（两组件版本）

    组件说明：
    1. L_culture_router: 路由专家文化相似性损失
    2. L_culture_sr: 共享-路由专家解耦损失
    注：已移除L_culture_share（共享专家文化无关损失）组件

    Args:
        expert_weights: 路由器专家权重 [B, num_experts] (wr_i)
        shared_expert_outputs: 共享专家输出 [B, hidden_dim] (es_i)
        router_expert_outputs: 路由专家融合输出 [B, hidden_dim] (er_i)
        culture_labels: 文化标签 [B]
        loss_weight: 损失权重

    Returns:
        dict: 包含总损失和各组件损失的字典
    """
    device = culture_labels.device if culture_labels is not None else torch.device('cuda')
    dtype = torch.float16

    # 初始化损失组件（两组件版本）
    L_culture_router = torch.tensor(0.0, device=device, dtype=dtype)
    L_culture_share = torch.tensor(0.0, device=device, dtype=dtype)  # 保持为0，不再计算
    L_culture_sr = torch.tensor(0.0, device=device, dtype=dtype)

    # 输入验证
    if culture_labels is None:
        return {
            'L_culture_router': L_culture_router,
            'L_culture_share': L_culture_share,
            'L_culture_sr': L_culture_sr,
            'L_culture_total': L_culture_router + L_culture_sr  # 两组件版本
        }

    batch_size = culture_labels.shape[0]

    # 处理单样本批次
    if batch_size < 2:
        # 对于单样本，只计算L_culture_sr（如果两种专家输出都可用）
        if shared_expert_outputs is not None and router_expert_outputs is not None:
            # 检查向量有效性
            if torch.norm(shared_expert_outputs) > 1e-8 and torch.norm(router_expert_outputs) > 1e-8:
                similarity = F.cosine_similarity(
                    shared_expert_outputs.unsqueeze(0),
                    router_expert_outputs.unsqueeze(0)
                )
                if not (torch.isnan(similarity) or torch.isinf(similarity)):
                    # 确保similarity是标量值，避免张量形状不匹配
                    similarity_scalar = similarity.item() if similarity.numel() == 1 else similarity.mean().item()
                    L_culture_sr = similarity_scalar * loss_weight

        return {
            'L_culture_router': L_culture_router,
            'L_culture_share': L_culture_share,
            'L_culture_sr': L_culture_sr,
            'L_culture_total': L_culture_router + L_culture_sr  # 两组件版本
        }

    # 计算L_culture_router（路由专家文化相似性损失）
    if expert_weights is not None:
        router_loss = torch.tensor(0.0, device=device, dtype=dtype)
        count_router = 0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                wr_i = expert_weights[i]
                wr_j = expert_weights[j]

                # 检查零向量
                if torch.norm(wr_i) < 1e-8 or torch.norm(wr_j) < 1e-8:
                    continue

                similarity = F.cosine_similarity(wr_i.unsqueeze(0), wr_j.unsqueeze(0))
                if torch.isnan(similarity) or torch.isinf(similarity):
                    continue

                # 确保similarity是标量值，避免张量形状不匹配
                similarity_scalar = similarity.item() if similarity.numel() == 1 else similarity.mean().item()

                if culture_labels[i] == culture_labels[j]:
                    # 相同文化，鼓励相似的专家权重
                    router_loss += (1.0 - similarity_scalar)
                else:
                    # 不同文化，惩罚相似的专家权重
                    router_loss += similarity_scalar

                count_router += 1

        if count_router > 0:
            L_culture_router = router_loss / count_router * loss_weight

    # L_culture_share组件已移除（共享专家文化无关损失）
    # 根据用户要求，CSL现在只包含两个组件：L_culture_router和L_culture_sr
    # L_culture_share保持为0

    # 计算L_culture_sr（共享-路由解耦损失）
    if shared_expert_outputs is not None and router_expert_outputs is not None:
        # 🔧 调试：检查专家输出状态
        print(f"🔍 CSL SR Debug: shared_shape={shared_expert_outputs.shape}, router_shape={router_expert_outputs.shape}")
        print(f"  Shared norm: {[torch.norm(shared_expert_outputs[i]).item() for i in range(min(3, batch_size))]}")
        print(f"  Router norm: {[torch.norm(router_expert_outputs[i]).item() for i in range(min(3, batch_size))]}")
        sr_loss = torch.tensor(0.0, device=device, dtype=dtype)
        count_sr = 0

        for i in range(batch_size):
            es_i = shared_expert_outputs[i]
            er_i = router_expert_outputs[i]

            # 检查零向量
            if torch.norm(es_i) < 1e-8 or torch.norm(er_i) < 1e-8:
                continue

            similarity = F.cosine_similarity(es_i.unsqueeze(0), er_i.unsqueeze(0))
            if torch.isnan(similarity) or torch.isinf(similarity):
                continue

            # 确保similarity是标量值，避免张量形状不匹配
            similarity_scalar = similarity.item() if similarity.numel() == 1 else similarity.mean().item()

            # 惩罚同一样本的共享和路由专家输出相似性
            sr_loss += similarity_scalar
            count_sr += 1

        if count_sr > 0:
            L_culture_sr = sr_loss / count_sr * loss_weight

    # 返回所有损失组件
    return {
        'L_culture_router': L_culture_router,
        'L_culture_share': L_culture_share,
        'L_culture_sr': L_culture_sr,
        'L_culture_total': L_culture_router + L_culture_sr  # 两组件版本
    }


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
        # 当batch_size=1时，使用专家权重的正则化损失来鼓励专家分化
        # 计算专家权重的熵，鼓励专家权重分布均匀（避免某个专家过于占主导）
        expert_probs = torch.softmax(expert_weights, dim=-1)  # [1, num_experts]
        entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=-1)  # [1]
        # 熵越大越好（分布越均匀），所以损失是负熵
        regularization_loss = -entropy.mean() * loss_weight
        return regularization_loss.to(dtype=torch.float16)

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

                # 确保similarity是标量值，避免广播问题
                similarity_scalar = similarity.item() if similarity.numel() == 1 else similarity.mean().item()
                culture_loss += (1.0 - similarity_scalar)
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


def train_epoch_joint(model, train_loader, optimizer, device, tokenizer,
                     num_accumulation_steps=1, rank=0, use_culture_loss=True, culture_loss_weight=0.01,
                     alpha=0.01, beta=0.01):
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
        labels = batch['labels'].to(device)  # 修复：使用collate_fn返回的正确键名

        # 🔧 检查batch是否有有效的训练标签，如果没有则跳过
        total_valid_labels = (labels != -100).sum().item()
        if total_valid_labels == 0:
            if rank == 0:  # 只在主进程打印
                print(f"⚠️ Batch {batch_idx}: 所有标签都被mask，跳过此batch")
            continue  # 跳过这个batch

        # ✅ 标签掩码已在数据集级别正确处理，不需要在训练时重复处理
        # 🔍 只添加调试信息来验证数据集的标签掩码是否正确
        if batch_idx < 1:  # 只在第一个batch显示
            for i in range(min(1, labels.shape[0])):  # 只检查第一个样本
                valid_labels = (labels[i] != -100).sum().item()
                total_labels = labels.shape[1]
                non_pad_labels = (labels[i] != tokenizer.pad_token_id).sum().item()

                # 数据集标签掩码验证（简化版，只在异常时打印）
                if valid_labels == 0:
                    print(f"❌ 警告: Batch {batch_idx} Sample {i} 没有有效训练标签!")
                # 注释掉有效标签过少的警告，因为单个token的标签是正常的
                # elif valid_labels < 3:
                #     print(f"⚠️ 警告: Batch {batch_idx} Sample {i} 有效训练标签太少 ({valid_labels})")

                # # 详细验证信息（注释掉）
                # print(f"🔍 数据集标签掩码验证 - Batch {batch_idx}, Sample {i}:")
                # print(f"  总标签数: {total_labels}")
                # print(f"  有效训练标签数: {valid_labels}")
                # print(f"  非pad标签数: {non_pad_labels}")
                # print(f"  有效标签比例: {valid_labels/non_pad_labels:.1%}")
                # if valid_labels >= 3:
                #     print(f"  ✅ 有效训练标签数量合理")

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
            culture_labels=culture_labels,
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

        # 🔧 增强MoE损失计算
        if use_culture_loss != 'false':
            # 使用增强MoE损失函数：L = L_h + L_balance = L_h + αL_aux + βL_o + γL_v
            # 获取模型输出的所有信息
            logits = outputs.logits
            expert_weights = getattr(outputs, 'expert_weights', None)
            expert_outputs = getattr(outputs, 'expert_outputs', {})
            soft_routing_scores = getattr(outputs, 'soft_routing_scores', expert_weights)
            activated_experts = getattr(outputs, 'activated_experts', list(expert_outputs.keys()))

            # 根据use_culture_loss参数决定损失函数模式
            if use_culture_loss == "ori":
                # 原始损失函数：L = L_h + L_balance (L_balance = αL_aux + βL_o + γL_v)
                enhanced_loss_dict = integrate_enhanced_moe_loss(
                    model_outputs=outputs,
                    labels=labels,
                    culture_labels=culture_labels,
                    use_culture_loss=False,
                    loss_weights={
                        "alpha": 1e-2,  # L_aux权重
                        "beta": 1e-3,   # L_o权重（原始）
                        "gamma": 1e-3,  # L_v权重
                        "use_cultural_aware": False,
                        "use_kl_loss": False
                    }
                )
                total_batch_loss = enhanced_loss_dict["L_total"]

            elif use_culture_loss == "new":
                # 新损失函数：L = L_h + L_balance (L_o使用文化感知版本)
                enhanced_loss_dict = integrate_enhanced_moe_loss(
                    model_outputs=outputs,
                    labels=labels,
                    culture_labels=culture_labels,
                    use_culture_loss=False,
                    loss_weights={
                        "alpha": 1e-2,  # L_aux权重
                        "beta": 1e-3,   # L_o权重（文化感知）
                        "gamma": 1e-3,  # L_v权重
                        "use_cultural_aware": True,
                        "use_kl_loss": False
                    }
                )
                total_batch_loss = enhanced_loss_dict["L_total"]

            elif use_culture_loss == "kl":
                # KL散度损失函数：L = L_h + L_balance (L_o使用KL散度文化损失)
                enhanced_loss_dict = integrate_enhanced_moe_loss(
                    model_outputs=outputs,
                    labels=labels,
                    culture_labels=culture_labels,
                    use_culture_loss=False,
                    loss_weights={
                        "alpha": 1e-2,  # L_aux权重
                        "beta": 1e-3,   # L_o权重（KL散度）
                        "gamma": 1e-3,  # L_v权重
                        "use_cultural_aware": False,
                        "use_kl_loss": True
                    }
                )
                total_batch_loss = enhanced_loss_dict["L_total"]

            elif use_culture_loss == "csl":
                # CSL文化相似性损失：L = L_h + L_balance + L_culture_csl
                # 首先获取基础损失（不包含文化损失）
                enhanced_loss_dict = integrate_enhanced_moe_loss(
                    model_outputs=outputs,
                    labels=labels,
                    culture_labels=culture_labels,
                    use_culture_loss=False,
                    loss_weights={
                        "alpha": 1e-2,  # L_aux权重
                        "beta": 0.0,    # 不使用原有的L_o
                        "gamma": 0.0,   # 不使用L_v
                        "use_cultural_aware": False,
                        "use_kl_loss": False
                    }
                )

                # 提取专家输出用于CSL计算
                expert_weights = getattr(outputs, 'expert_weights', None)
                shared_expert_outputs = getattr(outputs, 'shared_expert_outputs', None)
                router_expert_outputs = getattr(outputs, 'router_expert_outputs', None)

                # 计算CSL损失
                csl_loss_dict = compute_csl_culture_loss(
                    expert_weights=expert_weights,
                    shared_expert_outputs=shared_expert_outputs,
                    router_expert_outputs=router_expert_outputs,
                    culture_labels=culture_labels,
                    loss_weight=culture_loss_weight
                )

                # 组合总损失 (新公式: L_total = L_CE + ALPHA × L_aux + BETA × L_csl)
                lm_loss = enhanced_loss_dict["L_h"]
                aux_loss = enhanced_loss_dict["L_aux"]
                csl_total_loss = csl_loss_dict["L_culture_total"]

                # 🔧 增强CSL损失影响：如果CSL损失太小，进行放大
                if csl_total_loss.item() < 0.1:  # 如果CSL损失小于0.1，放大10倍
                    enhanced_csl_loss = csl_total_loss * 10.0
                    print(f"🔧 CSL损失太小({csl_total_loss.item():.6f})，放大10倍至{enhanced_csl_loss.item():.6f}")
                else:
                    enhanced_csl_loss = csl_total_loss

                total_batch_loss = lm_loss + alpha * aux_loss + beta * enhanced_csl_loss

                # 🔧 调试信息：输出损失组件以验证beta参数作用
                if batch_idx < 3 and rank == 0:  # 只在前几个batch和主进程输出
                    print(f"🔍 Batch {batch_idx} CSL Loss Debug:")
                    print(f"  LM Loss: {lm_loss.item():.6f}")
                    print(f"  Aux Loss: {aux_loss.item():.6f} (alpha={alpha})")
                    print(f"  CSL Total Loss: {csl_total_loss.item():.6f} (beta={beta})")
                    print(f"  CSL Router Loss: {csl_loss_dict['L_culture_router'].item():.6f}")
                    print(f"  CSL SR Loss: {csl_loss_dict['L_culture_sr'].item():.6f}")
                    print(f"  Final Total Loss: {total_batch_loss.item():.6f}")
                    print(f"  Beta contribution: {(beta * csl_total_loss).item():.6f}")

                    # 🔧 新增：检查专家权重和文化标签
                    if expert_weights is not None and culture_labels is not None:
                        print(f"  Expert weights shape: {expert_weights.shape}")
                        print(f"  Culture labels: {culture_labels.tolist()}")
                        print(f"  Expert weights mean: {expert_weights.mean(dim=1).tolist()}")

                    # 🔧 新增：检查CSL损失是否产生梯度
                    if csl_total_loss.requires_grad:
                        print(f"  CSL loss requires_grad: True")
                    else:
                        print(f"  ⚠️  CSL loss requires_grad: False (无梯度!)")

                    # 🔧 新增：检查专家输出状态
                    shared_available = shared_expert_outputs is not None
                    router_available = router_expert_outputs is not None
                    print(f"  Shared expert output: {'Available' if shared_available else 'Missing'}")
                    print(f"  Router expert output: {'Available' if router_available else 'Missing'}")

                # 更新损失字典以包含CSL组件
                enhanced_loss_dict["L_culture"] = csl_total_loss
                enhanced_loss_dict["L_culture_router"] = csl_loss_dict["L_culture_router"]
                enhanced_loss_dict["L_culture_share"] = csl_loss_dict["L_culture_share"]
                enhanced_loss_dict["L_culture_sr"] = csl_loss_dict["L_culture_sr"]
                enhanced_loss_dict["L_total"] = total_batch_loss

            elif use_culture_loss == "false" or use_culture_loss is False:
                # 简化损失函数：L = L_h + αL_aux (只有主任务损失和负载均衡)
                enhanced_loss_dict = integrate_enhanced_moe_loss(
                    model_outputs=outputs,
                    labels=labels,
                    culture_labels=culture_labels,
                    use_culture_loss=False,
                    loss_weights={
                        "alpha": 1e-2,  # L_aux权重
                        "beta": 0.0,    # 不使用L_o
                        "gamma": 0.0,   # 不使用L_v
                        "use_cultural_aware": False,
                        "use_kl_loss": False
                    }
                )
                total_batch_loss = enhanced_loss_dict["L_total"]

            else:
                # 默认使用原始损失函数
                enhanced_loss_dict = integrate_enhanced_moe_loss(
                    model_outputs=outputs,
                    labels=labels,
                    culture_labels=culture_labels,
                    use_culture_loss=False,
                    loss_weights={
                        "alpha": 1e-2,
                        "beta": 1e-3,
                        "gamma": 1e-3,
                        "use_cultural_aware": False,
                        "use_kl_loss": False
                    }
                )
                total_batch_loss = enhanced_loss_dict["L_total"]

            # 提取损失组件
            lm_loss = enhanced_loss_dict["L_h"]
            moe_aux_loss = enhanced_loss_dict["L_aux"]
            culture_loss = enhanced_loss_dict["L_culture"]

            # 记录各个损失组件用于监控
            l_balance = enhanced_loss_dict["L_balance"]
            l_o = enhanced_loss_dict["L_o"]
            l_v = enhanced_loss_dict["L_v"]

        else:
            # 简化损失计算：L = L_h + αL_aux
            lm_loss = outputs.loss  # 语言模型损失
            expert_weights = getattr(outputs, 'expert_weights', None)

            # 确保moe_aux_loss有梯度连接
            moe_aux_loss = getattr(outputs, 'moe_aux_loss', None)
            if moe_aux_loss is None:
                moe_aux_loss = torch.tensor(0.0, device=device, dtype=lm_loss.dtype, requires_grad=True)

            # 简化的总损失 (新公式: L_total = L_CE + ALPHA × L_aux)
            total_batch_loss = lm_loss + alpha * moe_aux_loss
            culture_loss = torch.tensor(0.0, device=device, dtype=lm_loss.dtype, requires_grad=True)

            # 设置占位符变量用于进度条显示
            l_balance = moe_aux_loss
            l_o = torch.tensor(0.0, device=device, dtype=lm_loss.dtype, requires_grad=True)
            l_v = torch.tensor(0.0, device=device, dtype=lm_loss.dtype, requires_grad=True)

        # 简化的梯度检查（仅在前3个batch）- 注释掉，专注tokenizer问题
        # if batch_idx < 3:
        #     if not total_batch_loss.requires_grad:
        #         print(f"⚠️ Batch {batch_idx} - Total loss missing gradients")

        # 检查 NaN/Inf loss - 跳过无效batch
        if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
            if rank == 0:
                print(f"⚠️ Batch {batch_idx}: NaN/Inf loss detected, 跳过此batch")
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

        # 更新进度条 - 显示增强损失组件
        postfix = {
            'loss': f"{total_batch_loss.item() * num_accumulation_steps:.6f}",
            'lm': f"{lm_loss.item():.6f}",
            'aux': f"{moe_aux_loss.item():.6f}"
        }

        if use_culture_loss:
            # 显示增强损失的各个组件
            postfix['bal'] = f"{l_balance.item():.4f}"  # L_balance
            postfix['ort'] = f"{l_o.item():.4f}"        # L_o (orthogonalization)
            postfix['var'] = f"{l_v.item():.4f}"        # L_v (routing variance)
            postfix['cul'] = f"{culture_loss.item():.4f}"  # L_culture
        else:
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


def evaluate_joint(model, val_loader, device, tokenizer, rank=0, use_culture_loss=True, culture_loss_weight=0.01, alpha=0.01, beta=0.01):
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
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # 修复：使用collate_fn返回的正确键名

            # 🔧 检查batch是否有有效的训练标签，如果没有则跳过
            total_valid_labels = (labels != -100).sum().item()
            if total_valid_labels == 0:
                if rank == 0:  # 只在主进程打印
                    print(f"⚠️ Eval Batch {batch_idx}: 所有标签都被mask，跳过此batch")
                continue  # 跳过这个batch

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
                culture_labels=culture_labels,
                return_dict=True
            )

            # 🔧 增强MoE损失计算（评估时）
            if use_culture_loss != 'false':
                # 根据use_culture_loss参数决定损失函数模式（评估时）
                if use_culture_loss == "ori":
                    enhanced_loss_dict = integrate_enhanced_moe_loss(
                        model_outputs=outputs,
                        labels=labels,
                        culture_labels=culture_labels,
                        use_culture_loss=False,
                        loss_weights={
                            "alpha": 1e-2, "beta": 1e-3, "gamma": 1e-3,
                            "use_cultural_aware": False, "use_kl_loss": False
                        }
                    )
                elif use_culture_loss == "new":
                    enhanced_loss_dict = integrate_enhanced_moe_loss(
                        model_outputs=outputs,
                        labels=labels,
                        culture_labels=culture_labels,
                        use_culture_loss=False,
                        loss_weights={
                            "alpha": 1e-2, "beta": 1e-3, "gamma": 1e-3,
                            "use_cultural_aware": True, "use_kl_loss": False
                        }
                    )
                elif use_culture_loss == "kl":
                    enhanced_loss_dict = integrate_enhanced_moe_loss(
                        model_outputs=outputs,
                        labels=labels,
                        culture_labels=culture_labels,
                        use_culture_loss=False,
                        loss_weights={
                            "alpha": 1e-2, "beta": 1e-3, "gamma": 1e-3,
                            "use_cultural_aware": False, "use_kl_loss": True
                        }
                    )
                elif use_culture_loss == "false" or use_culture_loss is False:
                    enhanced_loss_dict = integrate_enhanced_moe_loss(
                        model_outputs=outputs,
                        labels=labels,
                        culture_labels=culture_labels,
                        use_culture_loss=False,
                        loss_weights={
                            "alpha": 1e-2, "beta": 0.0, "gamma": 0.0,
                            "use_cultural_aware": False, "use_kl_loss": False
                        }
                    )
                else:
                    enhanced_loss_dict = integrate_enhanced_moe_loss(
                        model_outputs=outputs,
                        labels=labels,
                        culture_labels=culture_labels,
                        use_culture_loss=False,
                        loss_weights={
                            "alpha": 1e-2, "beta": 1e-3, "gamma": 1e-3,
                            "use_cultural_aware": False, "use_kl_loss": False
                        }
                    )

                total_batch_loss = enhanced_loss_dict["L_total"]
                lm_loss = enhanced_loss_dict["L_h"]
                moe_aux_loss = enhanced_loss_dict["L_aux"]
                culture_loss = enhanced_loss_dict["L_culture"]
            else:
                # 简化损失计算
                lm_loss = outputs.loss
                expert_weights = getattr(outputs, 'expert_weights', None)
                moe_aux_loss = getattr(outputs, 'moe_aux_loss', torch.tensor(0.0, device=device, dtype=torch.float16))
                culture_loss = torch.tensor(0.0, device=device, dtype=torch.float16)

                lm_loss = lm_loss.to(dtype=torch.float16)
                total_batch_loss = lm_loss + alpha * moe_aux_loss

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
    parser.add_argument("--data_id", type=str, required=True,
                        help="Data ID for dataset-specific optimizations")

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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval (every N epochs)")

    # 模型参数
    parser.add_argument("--backbone", type=str, default="qwen", choices=["llama", "qwen"],
                        help="Model backbone type")
    parser.add_argument("--num_moe_experts", type=int, default=4,
                        help="Number of MoE experts")
    parser.add_argument("--num_activated_experts", type=int, default=2,
                        help="Number of activated experts (top-k), if equal to num_moe_experts then dense mode")
    parser.add_argument("--use_culture_loss", type=str, default="new",
                        help="Whether to use culture loss")
    parser.add_argument("--culture_loss_weight", type=float, default=0.01,
                        help="Culture loss weight")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Load balancing loss coefficient")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="Culture specialty loss CSL coefficient")
    parser.add_argument("--use_lora", type=str, default="true",
                        help="Whether to enable pre-trained LoRA fine-tuning")
    parser.add_argument("--use_mask", type=str, default="true",
                        help="Whether to enable MASK mechanism with dual-path input processing")
    parser.add_argument("--use_shared", type=str, default="true",
                        help="Whether to use shared expert in MoE")
    parser.add_argument("--use_gate", type=str, default="true",
                        help="Whether to use gating network in MoE")

    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--pos_lora", type=str, default="att",
                        help="LoRA position: att=attention layer, ffn=FFN layer (default: att)")

    parser.add_argument("--memory_efficient", action='store_true',
                        help="Enable memory efficient training")

    args = parser.parse_args()

    # 转换字符串参数
    use_culture_loss = args.use_culture_loss.lower() if args.use_culture_loss.lower() in ['ori', 'new', 'kl', 'csl', 'false'] else 'ori'
    use_lora = args.use_lora.lower() == 'true'
    use_mask = args.use_mask.lower() == 'true'
    use_shared = args.use_shared.lower() == 'true'
    use_gate = args.use_gate.lower() == 'true'
    # 🔧 根据pos_lora参数决定LoRA挂载位置
    pos_lora = args.pos_lora.lower() if args.pos_lora else 'att'
    if pos_lora not in ['att', 'ffn']:
        print(f"⚠️ 无效的pos_lora参数: {pos_lora}，使用默认值'att'")
        pos_lora = 'att'

    # 🔧 根据backbone和data_id组合设置MoE影响权重
    if args.backbone == "llama":
        moe_influence_weight = 0.5  # LLaMA: 保持较强的MoE影响
        if is_main_process(rank):
            print(f"🔧 LLaMA backbone: MoE影响权重设置为 {moe_influence_weight}")
    elif args.backbone == "qwen":
        # Qwen根据不同数据集进行精细调整
        if args.data_id == "2":  # CulturalBench
            moe_influence_weight = 0.05  # 极低影响，避免干扰Qwen原有能力
        elif args.data_id in ["4", "1"]:  # CultureLLM或其他短文本数据集
            moe_influence_weight = 0.1   # 低影响
        elif args.data_id == "3":  # Normad
            moe_influence_weight = 0.2   # 中等影响，适合长文本
        else:
            moe_influence_weight = 0.2   # 默认值

        if is_main_process(rank):
            dataset_name = {"2": "CulturalBench", "3": "Normad", "4": "CultureLLM", "1": "其他"}.get(args.data_id, "未知")
            print(f"🔧 Qwen backbone + {dataset_name} (DATA_ID={args.data_id}): MoE影响权重设置为 {moe_influence_weight}")
    else:
        moe_influence_weight = 0.5  # 默认值
        if is_main_process(rank):
            print(f"🔧 未知backbone: 使用默认MoE影响权重 {moe_influence_weight}")

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
        print(f"Activated experts: {args.num_activated_experts} ({'dense mode' if args.num_activated_experts == args.num_moe_experts else f'top-{args.num_activated_experts}'})")
        print(f"Use culture loss: {use_culture_loss}")
        if use_culture_loss != 'false':
            print(f"Culture loss weight: {args.culture_loss_weight}")
        print(f"Use pre-trained LoRA: {use_lora}")
        print(f"Use MASK mechanism: {use_mask} ({'dual-path input processing' if use_mask else 'single-path input processing'})")
        print(f"Use shared expert: {use_shared}")
        print(f"Use gating network: {use_gate}")
        print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")
        print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    print("Loading tokenizer...")
    print("🚨 使用修改后的train_joint_lora_moe.py (版本标记: 2024-11-30)")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    # 🔧 修复Llama 3.1 tokenizer配置问题 - 使用官方padding token
    print(f"🔧 原始tokenizer状态: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")

    # 强制检查和修复pad_token配置，使用Llama 3.1官方的padding token
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128009:
        # Llama 3.1: 使用官方的finetune_right_pad_id
        print(f"🔧 检测到Llama 3.1模型，查找官方padding token...")

        # 查找Llama 3.1官方的padding token
        official_pad_token = "<|finetune_right_pad_id|>"
        try:
            pad_token_id = tokenizer.convert_tokens_to_ids(official_pad_token)

            # 检查这个token是否存在且有效
            if pad_token_id != tokenizer.unk_token_id and pad_token_id is not None:
                tokenizer.pad_token = official_pad_token
                tokenizer.pad_token_id = pad_token_id
                print(f"✅ Llama 3.1: 使用官方padding token: '{official_pad_token}' (id={pad_token_id})")
            else:
                raise ValueError("Official pad token not found or invalid")

        except Exception as e:
            print(f"⚠️ 无法找到官方padding token '{official_pad_token}': {e}")
            print(f"🔧 尝试查找其他Llama 3.1 padding相关token...")

            # 尝试查找其他可能的padding相关token
            alternative_pad_tokens = [
                "<|pad|>",
                "<|padding|>",
                "<|finetune_pad|>",
                "<|right_pad|>"
            ]

            found_alternative = False
            for alt_token in alternative_pad_tokens:
                try:
                    alt_token_id = tokenizer.convert_tokens_to_ids(alt_token)
                    if alt_token_id != tokenizer.unk_token_id and alt_token_id is not None:
                        tokenizer.pad_token = alt_token
                        tokenizer.pad_token_id = alt_token_id
                        print(f"✅ Llama 3.1: 使用替代padding token: '{alt_token}' (id={alt_token_id})")
                        found_alternative = True
                        break
                except:
                    continue

            if not found_alternative:
                print(f"⚠️ 找不到合适的padding token，使用安全的低频字符...")
                # 使用安全的低频字符作为最后的fallback
                safe_tokens = ['~', '`', '|', '^', '§', '¶', '†', '‡']
                for safe_token in safe_tokens:
                    try:
                        safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                        if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                            tokenizer.pad_token = safe_token
                            tokenizer.pad_token_id = safe_token_id
                            print(f"🔧 Llama 3.1: 使用安全字符 '{safe_token}' (id={safe_token_id}) 作为padding")
                            break
                    except:
                        continue

    elif tokenizer.pad_token is None:
        # 其他模型的标准配置
        tokenizer.pad_token = tokenizer.eos_token
        print(f"🔧 标准配置: pad_token = eos_token")
    else:
        # 对于已经有pad_token但可能配置错误的情况，也要检查
        if tokenizer.pad_token_id == 128009:
            print(f"🔧 检测到错误的pad_token配置(使用了<|eot_id|>)，强制修复...")

            # 对于Llama 3.1，优先尝试官方padding token
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128009:
                official_pad_token = "<|finetune_right_pad_id|>"
                try:
                    pad_token_id = tokenizer.convert_tokens_to_ids(official_pad_token)
                    if pad_token_id != tokenizer.unk_token_id and pad_token_id != 128009:
                        tokenizer.pad_token = official_pad_token
                        tokenizer.pad_token_id = pad_token_id
                        print(f"✅ 强制修复: 使用官方padding token '{official_pad_token}' (id={pad_token_id})")
                    else:
                        raise ValueError("Official pad token invalid")
                except:
                    # 如果官方token不可用，使用安全字符
                    safe_tokens = ['~', '`', '|', '^', '§', '¶']
                    for safe_token in safe_tokens:
                        try:
                            safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                            if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                                tokenizer.pad_token = safe_token
                                tokenizer.pad_token_id = safe_token_id
                                print(f"🔧 强制修复: 使用安全字符 '{safe_token}' (id={safe_token_id}) 作为padding")
                                break
                        except:
                            continue

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

    # 最终验证和测试
    print(f"\n🔧 最终tokenizer状态:")
    print(f"  pad_token: {repr(tokenizer.pad_token)}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  是否等于<|eot_id|>: {tokenizer.pad_token_id == 128009}")
    print(f"  是否为None: {tokenizer.pad_token_id is None}")

    # 🧪 测试padding行为
    print(f"\n🧪 测试padding行为:")
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

        # 验证padding token是否合理
        if pad_token_text.strip() in ['', '<pad>', '<|finetune_right_pad_id|>', '~', '`', '|', '^', '§', '¶']:
            print(f"  ✅ padding token合理: '{pad_token_text}'")
        else:
            print(f"  ⚠️ padding token可能有问题: '{pad_token_text}' (应该是空字符或特殊符号)")

        # 检查是否还在使用<|eot_id|>作为padding
        if pad_token_id == 128009:
            print(f"  🚨 严重错误: 仍在使用<|eot_id|>作为padding!")
        else:
            print(f"  ✅ 不再使用<|eot_id|>作为padding")
    else:
        print(f"  ℹ️ 测试序列中没有padding token")

    print("✅ Tokenizer loaded and validated")

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

    # 🔧 新增：检测是否为多数据集联合训练模式
    if ',' in args.train_file:
        # 多数据集模式：DATA_ID=24
        print("\n🔧 检测到多数据集联合训练模式")
        file_list = [f.strip() for f in args.train_file.split(',')]
        print(f"  - 数据集文件: {file_list}")
        print(f"  - 将分别对每个数据集进行8:1:1划分")

        all_train_datasets = []
        all_val_datasets = []
        all_test_datasets = []

        for idx, data_file in enumerate(file_list, 1):
            print(f"\n📁 处理数据集 {idx}: {os.path.basename(data_file)}")

            # 🔧 为每个数据集独立进行8:1:1划分，但不保存默认pkl文件
            # 临时禁用output_dir，避免覆盖原有的data_split_8_1_1.pkl
            datasets = load_and_process_data_8_1_1(
                data_file,
                tokenizer,
                max_length=args.max_length,
                output_dir=None,  # 不保存默认pkl文件
                seed=42  # 使用固定种子确保可重现性
            )

            # 🔧 手动保存独立的pkl文件，使用指定的命名格式
            import pickle
            pkl_filename = f"data_split_8_1_1_{idx}.pkl"
            pkl_path = os.path.join(args.output_dir, pkl_filename)

            # 获取Subset对象的indices
            train_indices = datasets['train'].indices
            val_indices = datasets['validation'].indices
            test_indices = datasets['test'].indices
            total_size = len(train_indices) + len(val_indices) + len(test_indices)

            split_info = {
                'data_path': data_file,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'train_size': len(datasets['train']),
                'val_size': len(datasets['validation']),
                'test_size': len(datasets['test']),
                'total_size': total_size,
                'seed': 42,
                'split_ratio': '8:1:1',
                'dataset_index': idx,
                'dataset_name': os.path.basename(data_file)
            }

            with open(pkl_path, 'wb') as f:
                pickle.dump(split_info, f)

            print(f"  ✅ 数据集 {idx} 划分完成:")
            print(f"    - 训练集: {len(datasets['train'])} 样本")
            print(f"    - 验证集: {len(datasets['validation'])} 样本")
            print(f"    - 测试集: {len(datasets['test'])} 样本")
            print(f"    - pkl文件: {pkl_filename}")

            # 收集所有数据集
            all_train_datasets.append(datasets['train'])
            all_val_datasets.append(datasets['validation'])
            all_test_datasets.append(datasets['test'])

        # 🔧 合并所有数据集
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(all_train_datasets)
        val_dataset = ConcatDataset(all_val_datasets)
        test_dataset = ConcatDataset(all_test_datasets)

        print(f"\n✅ 多数据集联合训练数据加载完成:")
        print(f"  - 总训练集: {len(train_dataset)} 样本")
        print(f"  - 总验证集: {len(val_dataset)} 样本")
        print(f"  - 总测试集: {len(test_dataset)} 样本")
        print(f"  - 已保存 {len(file_list)} 个独立的pkl文件")

    else:
        # 单数据集模式：原有逻辑
        print("\nLoading and processing data...")
        datasets = load_and_process_data_8_1_1(
            args.train_file,
            tokenizer,
            max_length=args.max_length,
            output_dir=args.output_dir
        )
        train_dataset = datasets['train']
        val_dataset = datasets['validation']
        test_dataset = datasets['test']  # 保存测试集引用，但暂时不在训练中使用
        print("✅ Data loaded with 8:1:1 split")

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
        num_workers=0,
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

    # 🔧 启用梯度检查点以节省内存
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
        if is_main_process(rank):
            print("✅ Gradient checkpointing enabled")
    elif hasattr(base_model, 'config'):
        base_model.config.use_cache = False  # 禁用KV cache以配合gradient checkpointing
        if is_main_process(rank):
            print("✅ KV cache disabled for gradient checkpointing")

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
        use_lora=use_lora,  # 是否启用预训练LoRA微调

        # MoE配置
        num_moe_experts=args.num_moe_experts,
        num_activated_experts=args.num_activated_experts,
        moe_hidden_dim=2048,  # 专家隐藏层大小
        moe_influence_weight=moe_influence_weight,  # 根据backbone设置的影响权重

        # 文化损失配置
        use_culture_loss=use_culture_loss,
        culture_loss_weight=args.culture_loss_weight,

        # MASK机制配置
        use_mask=use_mask,

        # 消融实验配置
        use_shared=use_shared,
        use_gate=use_gate,

        # 其他配置
        dropout=0.1
    )

    # 创建联合模型
    model = JointLoRAMoEModel(base_model, joint_config)

    # 🔧 确保梯度检查点在联合模型中仍然有效
    if hasattr(model.base_model, 'gradient_checkpointing_enable'):
        model.base_model.gradient_checkpointing_enable()
        if is_main_process(rank):
            print("✅ Gradient checkpointing re-enabled for joint model")

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
            culture_loss_weight=args.culture_loss_weight,
            alpha=args.alpha,
            beta=args.beta
        )

        if is_main_process(rank):
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    LM Loss: {train_metrics['lm_loss']:.4f}")
            print(f"    MoE Loss: {train_metrics['moe_loss']:.4f}")
            if use_culture_loss:
                print(f"    Culture Loss: {train_metrics['culture_loss']:.4f}")

            # 🔧 打印MoE层NaN统计信息
            try:
                # 获取实际的模型（处理DDP包装）
                actual_model = model.module if hasattr(model, 'module') else model
                if hasattr(actual_model, 'moe_layer'):
                    nan_rate, nan_count, total_calls = actual_model.moe_layer.get_nan_stats()
                    # print(f"    MoE NaN统计: {nan_count}/{total_calls} ({nan_rate:.2%}) - Epoch {epoch + 1}")

                    # 如果有NaN，显示更详细的信息
                    if nan_count > 0:
                        print(f"      ⚠️ 检测到 {nan_count} 次NaN，占总前向传播的 {nan_rate:.2%}")

                    # 重置统计计数器，为下一个epoch做准备
                    actual_model.moe_layer.reset_nan_stats()
            except Exception as e:
                print(f"    ⚠️ 无法获取MoE NaN统计: {e}")

        # 每eval_interval个epoch进行一次验证
        if (epoch + 1) % args.eval_interval == 0:
            # 验证
            val_metrics = evaluate_joint(
                model, val_loader, device, tokenizer, rank=rank,
                use_culture_loss=use_culture_loss,
                culture_loss_weight=args.culture_loss_weight,
                alpha=args.alpha,
                beta=args.beta
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
            'num_activated_experts': args.num_activated_experts,
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
        print(f"Activated experts: {args.num_activated_experts} ({'dense mode' if args.num_activated_experts == args.num_moe_experts else f'top-{args.num_activated_experts}'})")
        print(f"Use LoRA: {use_lora}")
        print(f"Culture loss: {use_culture_loss}")
        print("="*80)

    # 清理分布式训练
    cleanup_distributed()  # 🔧 修复：拼写错误，应为cleanup_distributed


if __name__ == "__main__":
    main()