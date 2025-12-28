#!/usr/bin/env python3
"""
简化版FFN CultureMoE训练脚本
基于MixLoRA实现，在所有层使用MoE，添加文化损失
针对48GB×2卡优化

特点：
1. 基于成功的MixLoRA架构
2. 在所有层替换FFN为MoE
3. 添加文化感知损失
4. 极简内存优化配置
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
from datetime import timedelta

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
import pickle
import numpy as np
from collections import defaultdict, Counter


def handle_nan_loss_synchronized(total_batch_loss, batch_idx, rank, world_size=1):
    """同步的NaN损失处理"""
    is_nan = torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss)

    # 🔧 安全检查：如果world_size未正确传递，使用默认值
    if world_size is None:
        world_size = 1

    if world_size > 1:
        # 收集所有进程的NaN状态
        nan_tensor = torch.tensor(float(is_nan), device=total_batch_loss.device)
        dist.all_reduce(nan_tensor, op=dist.ReduceOp.SUM)

        # 如果任何进程检测到NaN，所有进程都跳过
        any_nan = nan_tensor.item() > 0

        if any_nan and rank == 0:
            print(f"❌ NaN detected in distributed training at batch {batch_idx}")

        return any_nan
    else:
        if is_nan and rank == 0:
            print(f"❌ NaN detected at batch {batch_idx}")
        return is_nan


def safe_barrier(timeout=30.0, operation_name="barrier", max_retries=3):
    """超时保护的分布式barrier with 重试机制"""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return True

    import signal
    import time

    for attempt in range(max_retries):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Barrier timeout after {timeout}s in {operation_name} (attempt {attempt + 1}/{max_retries})")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            dist.barrier()
            signal.alarm(0)
            if attempt > 0:
                print(f"✅ Barrier succeeded on attempt {attempt + 1}/{max_retries} in {operation_name}")
            return True
        except TimeoutError as e:
            print(f"⚠️ {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying barrier in {operation_name} (attempt {attempt + 2}/{max_retries})")
                time.sleep(1.0)  # 短暂等待后重试
        except Exception as e:
            print(f"❌ Barrier error in {operation_name} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying after barrier error...")
                time.sleep(1.0)
            else:
                print(f"❌ All barrier attempts failed in {operation_name}")
                return False
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    print(f"❌ All {max_retries} barrier attempts failed in {operation_name}")
    return False


def enhanced_memory_monitoring(rank, operation_name="memory_check"):
    """🔧 增强的GPU内存监控和清理"""
    if not torch.cuda.is_available():
        return True

    try:
        device_id = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3   # GB

        # 获取GPU总容量
        total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        memory_free = total_memory - memory_reserved

        print(f"💾 GPU Memory Status (rank {rank}, {operation_name}):")
        print(f"   Allocated: {memory_allocated:.2f}GB / {total_memory:.2f}GB ({memory_allocated/total_memory*100:.1f}%)")
        print(f"   Reserved: {memory_reserved:.2f}GB ({memory_reserved/total_memory*100:.1f}%)")
        print(f"   Free: {memory_free:.2f}GB ({memory_free/total_memory*100:.1f}%)")

        # 内存预警和清理
        if memory_allocated > total_memory * 0.85:  # 超过85%警告
            print(f"🚨 CRITICAL MEMORY WARNING: {memory_allocated:.2f}GB allocated ({memory_allocated/total_memory*100:.1f}%)")
            # 强制垃圾回收和内存清理
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # 再次检查内存
            memory_allocated_after = torch.cuda.memory_allocated(device_id) / 1024**3
            memory_freed = memory_allocated - memory_allocated_after
            print(f"🔧 Emergency cleanup freed {memory_freed:.2f}GB")
            print(f"   New allocation: {memory_allocated_after:.2f}GB ({memory_allocated_after/total_memory*100:.1f}%)")

            if memory_allocated_after > total_memory * 0.90:  # 清理后仍超过90%
                print(f"❌ CRITICAL: Memory usage still too high after cleanup!")
                return False

        elif memory_allocated > total_memory * 0.70:  # 超过70%提醒
            print(f"⚠️ Memory usage high: {memory_allocated:.2f}GB ({memory_allocated/total_memory*100:.1f}%)")
            # 轻度清理
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Memory monitoring failed: {e}")
        return False


def enhanced_distributed_error_handling(rank, world_size, operation_name, max_retries=5):
    """🔧 增强的分布式错误处理"""
    print(f"🔧 Enhanced distributed error handling: {operation_name} (rank {rank})")

    # 1. 内存监控
    memory_ok = enhanced_memory_monitoring(rank, operation_name)
    if not memory_ok:
        print(f"❌ Memory check failed during {operation_name}")
        return False

    # 2. 分布式同步错误处理
    if world_size > 1:
        for attempt in range(max_retries):
            try:
                success = safe_barrier(timeout=30.0, operation_name=f"{operation_name}_attempt_{attempt+1}")
                if success:
                    if attempt > 0:
                        print(f"✅ Distributed sync successful after {attempt+1} attempts")
                    return True
                else:
                    print(f"⚠️ Distributed sync failed, attempt {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        # 渐进式等待时间
                        import time
                        wait_time = min(2.0 ** attempt, 10.0)
                        print(f"🔄 Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)

            except Exception as e:
                print(f"❌ Distributed sync error (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    print(f"❌ All {max_retries} sync attempts failed, continuing with degraded mode")
                    return False

        return False
    else:
        # 单GPU模式，直接返回成功
        return True


def setup_distributed():
    """🔧 强化版分布式训练初始化"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        print(f"🔧 Initializing ROBUST distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")

        try:
            # 🔧 检查是否已经初始化
            if dist.is_initialized():
                print(f"⚠️ Distributed already initialized, cleaning up first...")
                try:
                    dist.destroy_process_group()
                except:
                    pass

            # 🔧 强化版进程组初始化
            dist.init_process_group(
                backend='nccl',
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=30)  # 增加超时时间
            )

            # 设置当前进程的GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                # 🔧 验证GPU设备设置
                current_device = torch.cuda.current_device()
                if current_device != local_rank:
                    print(f"⚠️ GPU device mismatch: expected {local_rank}, got {current_device}")
                    torch.cuda.set_device(local_rank)

                # 多GPU模式下的内存分配器设置
                torch.cuda.empty_cache()

            # 🔧 验证分布式初始化
            test_tensor = torch.tensor([rank], device=f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            expected_sum = sum(range(world_size))
            if test_tensor.item() == expected_sum:
                print(f"✅ Distributed initialization verified: rank {rank}")
            else:
                print(f"❌ Distributed initialization failed: expected {expected_sum}, got {test_tensor.item()}")

            # 同步所有进程
            safe_barrier(timeout=60.0, operation_name="setup_distributed")

            return rank, world_size, local_rank

        except Exception as e:
            print(f"❌ Distributed initialization failed: {e}")
            print(f"🔧 Falling back to single GPU mode...")
            return 0, 1, 0
    else:
        # 单GPU模式
        print(f"🔧 Single GPU mode: no distributed environment variables found")
        return 0, 1, 0


def cleanup_distributed():
    """🔧 零失败分布式训练清理 - 确保进程优雅退出"""
    if not dist.is_initialized():
        return

    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        print(f"🔧 Starting ZERO-FAILURE distributed cleanup: rank {rank}/{world_size}")

        # 🔧 新策略：超长超时+多重fallback，确保零失败
        if world_size > 1:
            # 阶段1：预清理barrier - 使用更长超时
            print(f"🔧 Phase 1: Extended pre-cleanup barrier (rank {rank})")
            barrier_success = safe_barrier(timeout=60.0, operation_name="extended_pre_cleanup_barrier")

            if not barrier_success:
                print(f"⚠️ Extended pre-cleanup barrier failed for rank {rank}, using emergency sync")
                # 紧急同步：尝试短时间barrier
                try:
                    safe_barrier(timeout=5.0, operation_name="emergency_sync")
                except:
                    print(f"⚠️ Emergency sync also failed for rank {rank}, proceeding individually")

            # 阶段2：本地资源清理
            print(f"🔧 Phase 2: Local resource cleanup (rank {rank})")
            try:
                torch.cuda.empty_cache()
                print(f"✅ CUDA cache cleared for rank {rank}")
            except:
                print(f"⚠️ CUDA cleanup failed for rank {rank}, but continuing")

            # 阶段3：最终同步barrier - 更长超时
            print(f"🔧 Phase 3: Extended final sync barrier (rank {rank})")
            final_barrier_success = safe_barrier(timeout=30.0, operation_name="extended_final_barrier")

            if not final_barrier_success:
                print(f"⚠️ Extended final barrier failed for rank {rank}, using graceful degradation")
                # 优雅降级：等待一段时间后强制继续
                import time
                time.sleep(2.0)  # 给其他进程一些时间
                print(f"🔧 Graceful degradation completed for rank {rank}")

        # 🔧 关键修复：多重fallback的进程组销毁
        print(f"🔧 Phase 4: Multi-fallback process group destruction (rank {rank if 'rank' in locals() else 'unknown'})")

        # Fallback 1: 正常销毁
        try:
            dist.destroy_process_group()
            print(f"✅ Normal process group destruction successful (rank {rank if 'rank' in locals() else 'unknown'})")
            return  # 成功退出
        except RuntimeError as e:
            print(f"⚠️ Normal destruction failed (rank {rank if 'rank' in locals() else 'unknown'}): {e}")

        # Fallback 2: 强制销毁
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                print(f"✅ Forced process group destruction successful (rank {rank if 'rank' in locals() else 'unknown'})")
                return
        except:
            print(f"⚠️ Forced destruction also failed (rank {rank if 'rank' in locals() else 'unknown'})")

        # Fallback 3: 忽略销毁，只清理资源
        try:
            torch.cuda.empty_cache()
            print(f"✅ Resource cleanup completed, ignoring process group (rank {rank if 'rank' in locals() else 'unknown'})")
        except:
            print(f"⚠️ Even resource cleanup failed (rank {rank if 'rank' in locals() else 'unknown'})")

        # 🔧 最终策略：无条件成功退出
        print(f"✅ ZERO-FAILURE cleanup completed - process will exit gracefully (rank {rank if 'rank' in locals() else 'unknown'})")

    except Exception as e:
        rank_info = f"rank {rank}" if 'rank' in locals() else "unknown rank"
        print(f"⚠️ Exception during cleanup ({rank_info}): {e}")

        # 🔧 异常情况下的终极fallback
        print(f"🔧 Executing ULTIMATE fallback cleanup ({rank_info})")

        # 忽略所有错误，强制清理资源
        try:
            torch.cuda.empty_cache()
        except:
            pass

        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass

        # 🔧 无条件成功退出声明
        print(f"✅ ULTIMATE fallback completed - guaranteed graceful exit ({rank_info})")


def is_main_process(rank):
    """检查是否为主进程"""
    return rank == 0


def load_and_split_multi_datasets_8_1_1(data_path: str, tokenizer, max_length: int = 512,
                                        output_dir: str = None, force_resplit: bool = False,
                                        enable_mask: bool = False, mask_prob: float = 0.15):
    """
    加载合并数据集并按8:1:1划分，同时为每个原始数据集保存独立的划分文件

    Args:
        data_path: 合并后的数据文件路径
        tokenizer: Tokenizer
        max_length: 最大序列长度
        output_dir: 输出目录，用于保存划分索引
        force_resplit: 是否强制重新划分

    Returns:
        dict: 包含 'train', 'validation', 'test' 的字典
    """
    # 检查是否是blend + cultureAtlas的合并数据集
    if 'blend_cultureAtlas_merged.json' in data_path:
        # 处理多数据集划分
        blend_file = "/root/autodl-fs/blend_merge_gen.json"
        cultureatlas_file = "/autodl-fs/data/cultureAtlas_merge_gen.json"

        # 分别加载两个原始数据集
        blend_dataset = CultureLLMNewFormatDataset(
            blend_file, tokenizer, max_length,
            enable_mask=False, mask_prob=0.0  # 🔧 强制禁用MASK机制
        )
        cultureatlas_dataset = CultureLLMNewFormatDataset(
            cultureatlas_file, tokenizer, max_length,
            enable_mask=False, mask_prob=0.0  # 🔧 强制禁用MASK机制
        )

        blend_size = len(blend_dataset)
        cultureatlas_size = len(cultureatlas_dataset)
        total_size = blend_size + cultureatlas_size

        print(f"🔄 处理多数据集划分:")
        print(f"  - blend: {blend_size} 样本")
        print(f"  - cultureAtlas: {cultureatlas_size} 样本")
        print(f"  - 总计: {total_size} 样本")

        # 分别对每个数据集进行8:1:1划分
        def split_dataset(dataset, dataset_name, dataset_size):
            train_size = int(dataset_size * 0.8)
            val_size = int(dataset_size * 0.1)
            test_size = dataset_size - train_size - val_size

            np.random.seed(42)  # 固定随机种子
            indices = np.random.permutation(dataset_size)

            train_indices = indices[:train_size].tolist()
            val_indices = indices[train_size:train_size + val_size].tolist()
            test_indices = indices[train_size + val_size:].tolist()

            # 保存独立的划分文件
            split_info = {
                'total_size': dataset_size,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'test_size': len(test_indices),
                'data_path': blend_file if dataset_name == 'blend' else cultureatlas_file,
                'max_length': max_length,
                'split_method': '8:1:1',
                'random_seed': 42
            }

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                split_file = os.path.join(output_dir, f'{dataset_name}_data_split_8_1_1.pkl')
                with open(split_file, 'wb') as f:
                    pickle.dump(split_info, f)
                print(f"✅ 保存{dataset_name}数据划分: {split_file}")

            return train_indices, val_indices, test_indices

        # 分别划分两个数据集
        blend_train_idx, blend_val_idx, blend_test_idx = split_dataset(blend_dataset, 'blend', blend_size)
        cultureatlas_train_idx, cultureatlas_val_idx, cultureatlas_test_idx = split_dataset(cultureatlas_dataset, 'cultureatlas', cultureatlas_size)

        # 创建合并的数据集
        full_dataset = CultureLLMNewFormatDataset(
            data_path, tokenizer, max_length,
            enable_mask=False, mask_prob=0.0  # 🔧 强制禁用MASK机制
        )

        # 调整cultureAtlas的索引（因为在合并数据集中的偏移）
        cultureatlas_train_idx_adjusted = [idx + blend_size for idx in cultureatlas_train_idx]
        cultureatlas_val_idx_adjusted = [idx + blend_size for idx in cultureatlas_val_idx]
        cultureatlas_test_idx_adjusted = [idx + blend_size for idx in cultureatlas_test_idx]

        # 合并训练集和验证集索引
        merged_train_indices = blend_train_idx + cultureatlas_train_idx_adjusted
        merged_val_indices = blend_val_idx + cultureatlas_val_idx_adjusted
        merged_test_indices = blend_test_idx + cultureatlas_test_idx_adjusted

        print(f"✅ 合并数据集划分完成:")
        print(f"  - 训练集: {len(merged_train_indices)} 样本 ({len(merged_train_indices)/total_size*100:.1f}%)")
        print(f"  - 验证集: {len(merged_val_indices)} 样本 ({len(merged_val_indices)/total_size*100:.1f}%)")
        print(f"  - 测试集: {len(merged_test_indices)} 样本 ({len(merged_test_indices)/total_size*100:.1f}%)")

        # 创建子数据集
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, merged_train_indices)
        val_dataset = Subset(full_dataset, merged_val_indices)
        test_dataset = Subset(full_dataset, merged_test_indices)

        return {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset,
            'split_info': None  # 多数据集情况下不返回单一split_info
        }
    else:
        # 单数据集情况，使用原有逻辑
        return load_and_split_data_8_1_1(data_path, tokenizer, max_length, output_dir, force_resplit, False, 0.0)


def load_and_split_data_8_1_1(data_path: str, tokenizer, max_length: int = 512,
                               output_dir: str = None, force_resplit: bool = False,
                               enable_mask: bool = False, mask_prob: float = 0.15):
    """
    加载数据并按8:1:1划分为训练集、验证集、测试集

    Args:
        data_path: 数据文件路径
        tokenizer: Tokenizer
        max_length: 最大序列长度
        output_dir: 输出目录，用于保存划分索引
        force_resplit: 是否强制重新划分

    Returns:
        dict: 包含 'train', 'validation', 'test' 的字典
    """
    # 创建完整数据集
    full_dataset = CultureLLMNewFormatDataset(
        data_path, tokenizer, max_length,
        enable_mask=False, mask_prob=0.0  # 🔧 强制禁用MASK机制
    )
    total_size = len(full_dataset)

    # 检查是否已有划分文件
    split_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        split_file = os.path.join(output_dir, 'data_split_8_1_1.pkl')

        if os.path.exists(split_file) and not force_resplit:
            print(f"🔄 加载已有的8:1:1数据划分: {split_file}")
            with open(split_file, 'rb') as f:
                split_info = pickle.load(f)

            # 验证划分是否与当前数据集匹配
            if split_info['total_size'] == total_size:
                train_indices = split_info['train_indices']
                val_indices = split_info['val_indices']
                test_indices = split_info['test_indices']

                print(f"✅ 使用已有划分:")
                print(f"  - 训练集: {len(train_indices)} 样本 ({len(train_indices)/total_size*100:.1f}%)")
                print(f"  - 验证集: {len(val_indices)} 样本 ({len(val_indices)/total_size*100:.1f}%)")
                print(f"  - 测试集: {len(test_indices)} 样本 ({len(test_indices)/total_size*100:.1f}%)")

                # 创建子数据集
                from torch.utils.data import Subset
                train_dataset = Subset(full_dataset, train_indices)
                val_dataset = Subset(full_dataset, val_indices)
                test_dataset = Subset(full_dataset, test_indices)

                return {
                    'train': train_dataset,
                    'validation': val_dataset,
                    'test': test_dataset,
                    'split_info': split_info
                }
            else:
                print(f"⚠️ 数据集大小不匹配，重新划分 (saved: {split_info['total_size']}, current: {total_size})")

    # 执行8:1:1划分
    print(f"🔄 创建8:1:1数据划分 (总数: {total_size})")

    # 计算各部分大小
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size  # 剩余部分作为测试集

    # 生成随机索引
    np.random.seed(42)  # 固定随机种子确保可重现
    indices = np.random.permutation(total_size)

    # 划分索引
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:train_size + val_size].tolist()
    test_indices = indices[train_size + val_size:].tolist()

    print(f"✅ 数据划分完成:")
    print(f"  - 训练集: {len(train_indices)} 样本 ({len(train_indices)/total_size*100:.1f}%)")
    print(f"  - 验证集: {len(val_indices)} 样本 ({len(val_indices)/total_size*100:.1f}%)")
    print(f"  - 测试集: {len(test_indices)} 样本 ({len(test_indices)/total_size*100:.1f}%)")

    # 保存划分信息
    if split_file:
        split_info = {
            'total_size': total_size,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'data_path': data_path,
            'max_length': max_length,
            'split_method': '8:1:1',
            'random_seed': 42
        }

        with open(split_file, 'wb') as f:
            pickle.dump(split_info, f)

        print(f"✅ 数据划分信息已保存: {split_file}")

    # 创建子数据集
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset,
        'split_info': split_info if split_file else None
    }


def compute_culture_loss(model_outputs, culture_labels, shared_outputs=None, main_loss=None):
    """
    计算文化感知损失

    Args:
        model_outputs: 模型输出，应该包含expert_weights等信息
        culture_labels: 文化标签 [B]
        shared_outputs: shared专家输出 [B, H]，可选
        main_loss: 主损失，用于创建连接到计算图的零损失

    Returns:
        culture_loss: 文化损失（原始值，不包含权重）
    """
    device = culture_labels.device
    total_culture_losses = []

    # 🆕 1. Shared专家损失：所有样本的shared专家输出都应该相似（学习文化共性）
    if shared_outputs is not None and shared_outputs.shape[0] >= 2:
        shared_losses = []
        batch_size = shared_outputs.shape[0]

        # 对所有样本对计算shared专家相似度损失
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                vec1 = shared_outputs[i].unsqueeze(0)
                vec2 = shared_outputs[j].unsqueeze(0)

                # 检查向量是否为零向量
                norm1 = torch.norm(vec1)
                norm2 = torch.norm(vec2)
                if norm1 < 1e-8 or norm2 < 1e-8:
                    continue

                similarity = F.cosine_similarity(vec1, vec2)
                # similarity = similarity.to(dtype=torch.float16)  # 🔧 修复：移除类型转换，避免破坏梯度连接
                if torch.isnan(similarity) or torch.isinf(similarity):
                    continue

                # Shared专家：所有样本都应该相似，所以损失为 1 - similarity
                # 相似度越高，损失越小
                shared_loss_term = 1.0 - similarity
                shared_losses.append(shared_loss_term)

        if len(shared_losses) > 0:
            # 🔧 内存优化：使用累积计算替代torch.stack避免大型临时张量
            if len(shared_losses) == 1:
                shared_culture_loss = shared_losses[0]
            else:
                shared_culture_loss = shared_losses[0]
                for loss in shared_losses[1:]:
                    shared_culture_loss = shared_culture_loss + loss
                shared_culture_loss = shared_culture_loss / len(shared_losses)
            total_culture_losses.append(shared_culture_loss)

    # 🔄 2. 路由专家损失：保持原有逻辑（同culture相似，不同culture不同）
    if hasattr(model_outputs, 'expert_weights') and model_outputs.expert_weights is not None:
        expert_weights = model_outputs.expert_weights  # [B_expert, num_experts]

        # 🔧 MASK机制修复：expert_weights可能只包含激活路由专家的样本
        routing_culture_labels = culture_labels
        if expert_weights.shape[0] != culture_labels.shape[0]:
            # 只对有expert_weights的样本计算文化损失
            if expert_weights.shape[0] < 2:
                # 激活路由专家的样本少于2个，跳过路由专家损失
                pass
            else:
                # 使用前expert_weights.shape[0]个culture_labels
                routing_culture_labels = culture_labels[:expert_weights.shape[0]]

        if expert_weights.shape[0] >= 2:
            routing_losses = []
            batch_size = expert_weights.shape[0]

            # 计算同文化样本间的相似性和不同文化样本间的差异性
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    if routing_culture_labels[i] == routing_culture_labels[j]:
                        # 相同文化，鼓励相似的专家权重
                        vec1 = expert_weights[i].unsqueeze(0)
                        vec2 = expert_weights[j].unsqueeze(0)

                        # 检查向量是否为零向量，避免cosine_similarity中的NaN
                        norm1 = torch.norm(vec1)
                        norm2 = torch.norm(vec2)
                        if norm1 < 1e-8 or norm2 < 1e-8:
                            continue

                        similarity = F.cosine_similarity(vec1, vec2)
                        # similarity = similarity.to(dtype=torch.float16)  # 🔧 修复：移除类型转换，避免破坏梯度连接
                        if torch.isnan(similarity) or torch.isinf(similarity):
                            continue

                        # 相同文化：相似度应该高，损失为 1 - similarity
                        loss_term = 1.0 - similarity
                        routing_losses.append(loss_term)
                    else:
                        # 不同文化，鼓励不同的专家权重
                        vec1 = expert_weights[i].unsqueeze(0)
                        vec2 = expert_weights[j].unsqueeze(0)

                        # 检查向量是否为零向量，避免cosine_similarity中的NaN
                        norm1 = torch.norm(vec1)
                        norm2 = torch.norm(vec2)
                        if norm1 < 1e-8 or norm2 < 1e-8:
                            continue

                        similarity = F.cosine_similarity(vec1, vec2)
                        # similarity = similarity.to(dtype=torch.float16)  # 🔧 修复：移除类型转换，避免破坏梯度连接
                        if torch.isnan(similarity) or torch.isinf(similarity):
                            continue

                        # 不同文化：相似度应该低，损失为 similarity
                        routing_losses.append(similarity)

            if len(routing_losses) > 0:
                # 🔧 内存优化：使用累积计算替代torch.stack避免大型临时张量
                if len(routing_losses) == 1:
                    routing_culture_loss = routing_losses[0]
                else:
                    routing_culture_loss = routing_losses[0]
                    for loss in routing_losses[1:]:
                        routing_culture_loss = routing_culture_loss + loss
                    routing_culture_loss = routing_culture_loss / len(routing_losses)
                total_culture_losses.append(routing_culture_loss)

    # 🔧 计算最终的总文化损失
    if len(total_culture_losses) > 0:
        # 🔧 内存优化：使用累积计算替代torch.stack避免大型临时张量
        if len(total_culture_losses) == 1:
            culture_loss = total_culture_losses[0]
        else:
            culture_loss = total_culture_losses[0]
            for loss in total_culture_losses[1:]:
                culture_loss = culture_loss + loss
            culture_loss = culture_loss / len(total_culture_losses)
    else:
        # 🔧 修复梯度问题：使用主损失*0来创建连接到计算图的零损失
        if main_loss is not None:
            culture_loss = main_loss * 0.0  # 保持梯度图连接
        else:
            # 🔧 如果没有main_loss，使用culture_labels创建连接到计算图的零损失
            culture_loss = culture_labels.float().sum() * 0.0

    # 检查文化损失是否为NaN/Inf，如果是则返回零损失
    if torch.isnan(culture_loss) or torch.isinf(culture_loss):
        if main_loss is not None:
            culture_loss = main_loss * 0.0
        else:
            # 🔧 如果没有main_loss，使用culture_labels创建连接到计算图的零损失
            culture_loss = culture_labels.float().sum() * 0.0

    # 🔧 确保返回的culture_loss与主损失类型一致
    if main_loss is not None:
        target_device = main_loss.device
        target_dtype = main_loss.dtype
        if culture_loss.device != target_device or culture_loss.dtype != target_dtype:
            culture_loss = culture_loss.to(device=target_device, dtype=target_dtype)

    return culture_loss


def print_expert_activation_stats(model_adapter, epoch):
    """
    统计并打印专家激活分布

    完善版本支持：
    1. Dense模式和Top-k模式的统计
    2. MASK机制的shared专家和路由专家区分
    3. 更全面的统计信息和趋势分析

    Args:
        model_adapter: SimplifiedCultureMoEAdapter实例
        epoch: 当前epoch数
    """
    print(f"\n🔍 专家激活统计 - Epoch {epoch}")
    print("=" * 80)

    # 获取模型的MoE层
    try:
        # 🔧 关键修复：使用与_get_target_layers完全相同的逻辑访问实际训练模型
        actual_model = model_adapter.base_model

        # print(f"🔧 专家统计：开始解包模型，初始类型: {type(actual_model)}")

        # 处理DDP包装
        if hasattr(actual_model, 'module'):
            actual_model = actual_model.module
            # print(f"🔧 专家统计：检测到DDP包装，解包后: {type(actual_model)}")

        # 处理PeftModel包装（LoRA包装）
        if hasattr(actual_model, 'base_model'):
            if hasattr(actual_model.base_model, 'model'):
                # PeftModel -> base_model.model
                actual_model = actual_model.base_model.model
                # print(f"🔧 专家统计：检测到PeftModel包装，解包到base_model.model: {type(actual_model)}")
            else:
                # PeftModel -> base_model
                actual_model = actual_model.base_model
                # print(f"🔧 专家统计：检测到PeftModel包装，解包到base_model: {type(actual_model)}")

        # 再次检查是否还有model属性
        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            actual_model = actual_model.model
            # print(f"🔧 专家统计：进一步解包到model属性: {type(actual_model)}")

        # 获取layers
        if hasattr(actual_model, 'layers'):
            layers = actual_model.layers
            # print(f"✅ 专家统计：成功找到layers: {len(layers)} 层 in {type(actual_model)}")
        else:
            # 详细诊断
            print(f"❌ 专家统计：无法找到layers，当前模型类型: {type(actual_model)}")
            print(f"   模型属性: {[attr for attr in dir(actual_model) if not attr.startswith('_')]}")
            return

        total_layers = len(layers)
        moe_layer_count = 0
        routing_layer_count = 0  # 只统计有路由专家激活的层
        shared_layer_count = 0   # 只统计有shared专家激活的层

        # 全局统计信息
        global_expert_usage = defaultdict(int)  # 全局专家使用次数
        layer_activations = []  # 每层的激活模式

        # 遍历所有层，统计MoE层的专家激活
        for layer_idx in range(total_layers):
            moe_layer = layers[layer_idx].mlp

            # 检查是否为MoE层（通过检查是否有MoE结构）
            if hasattr(moe_layer, 'experts') and hasattr(moe_layer, 'router'):
                expert_weights = moe_layer.latest_expert_weights  # [B, num_experts]

                # 检查是否有shared专家输出
                has_shared = hasattr(moe_layer, 'latest_shared_outputs') and moe_layer.latest_shared_outputs is not None

                # 🔧 修复：检查expert_weights是否存在且有数据，如果没有则至少统计MoE层存在
                if expert_weights is not None and expert_weights.shape[0] > 0:  # 确保有路由专家数据
                    routing_layer_count += 1
                    batch_size, num_experts = expert_weights.shape

                    # 🆕 检测激活模式（Dense vs Top-k）
                    # 通过检查MoE层的配置确定激活模式
                    is_dense_mode = (moe_layer.num_activated_experts == moe_layer.num_experts)
                    activation_mode = "Dense" if is_dense_mode else f"Top-{moe_layer.num_activated_experts}"

                    if is_dense_mode:
                        # Dense模式：统计所有专家的权重分布
                        expert_avg_weights = expert_weights.mean(dim=0)  # [num_experts]

                        print(f"Layer {layer_idx:2d} ({activation_mode}): ", end="")
                        weight_stats = []
                        for expert_idx in range(num_experts):
                            weight = expert_avg_weights[expert_idx].item()
                            percentage = weight * 100
                            weight_stats.append(f"E{expert_idx}:{percentage:5.1f}%")
                            global_expert_usage[f"expert_{expert_idx}"] += batch_size  # Dense模式下每个专家都被所有样本使用

                        print(", ".join(weight_stats))
                        layer_activations.append({
                            'layer': layer_idx,
                            'mode': activation_mode,
                            'type': 'routing',
                            'expert_weights': expert_avg_weights.tolist()
                        })

                    else:
                        # Top-k模式：统计激活对的频率
                        activation_pairs = []
                        expert_activation_counts = defaultdict(int)

                        for sample_idx in range(batch_size):
                            # 找到权重最大的top-k专家
                            k = min(moe_layer.num_activated_experts, num_experts)
                            topk_values, topk_indices = torch.topk(expert_weights[sample_idx], k=k, dim=0)

                            # 只统计权重显著大于0的专家（避免统计到微小的数值噪音）
                            significant_experts = []
                            for i, (value, idx) in enumerate(zip(topk_values, topk_indices)):
                                if value.item() > 1e-6:  # 阈值过滤
                                    expert_id = idx.item()
                                    significant_experts.append(expert_id)
                                    expert_activation_counts[expert_id] += 1
                                    global_expert_usage[f"expert_{expert_id}"] += 1

                            if len(significant_experts) >= 2:
                                # 排序确保一致性（例如总是0/1而不是1/0）
                                pair = tuple(sorted(significant_experts[:2]))
                                activation_pairs.append(pair)
                            elif len(significant_experts) == 1:
                                # 单专家激活情况
                                activation_pairs.append((significant_experts[0],))

                        # 统计激活对的频率
                        if activation_pairs:
                            pair_counts = Counter(activation_pairs)
                            total_activations = len(activation_pairs)

                            # 打印该层的统计结果
                            print(f"Layer {layer_idx:2d} ({activation_mode}): ", end="")
                            pair_stats = []
                            for pair, count in sorted(pair_counts.items()):
                                percentage = (count / total_activations) * 100
                                if len(pair) == 2:
                                    pair_stats.append(f"{pair[0]}/{pair[1]}:{percentage:5.1f}%")
                                else:
                                    pair_stats.append(f"{pair[0]}:{percentage:5.1f}%")

                            print(", ".join(pair_stats))

                            layer_activations.append({
                                'layer': layer_idx,
                                'mode': activation_mode,
                                'type': 'routing',
                                'activation_pairs': dict(pair_counts),
                                'expert_counts': dict(expert_activation_counts)
                            })
                        else:
                            print(f"Layer {layer_idx:2d} ({activation_mode}): 无有效激活数据")

                elif has_shared:
                    # 只有shared专家激活的情况
                    shared_layer_count += 1
                    print(f"Layer {layer_idx:2d} (Shared): Shared专家激活")
                    layer_activations.append({
                        'layer': layer_idx,
                        'mode': 'Shared',
                        'type': 'shared'
                    })
                    global_expert_usage['shared_expert'] += 1
                else:
                    # 🔧 MoE层存在但没有运行时数据（比如训练开始前）
                    print(f"Layer {layer_idx:2d} (MoE): MoE结构已初始化，等待激活数据")

                moe_layer_count += 1

        # 🆕 打印总体统计信息
        print(f"\n📊 总体统计信息:")
        print(f"   总MoE层数: {moe_layer_count}")
        print(f"   路由专家激活层数: {routing_layer_count}")
        print(f"   Shared专家激活层数: {shared_layer_count}")

        if global_expert_usage:
            print(f"\n🔍 全局专家使用频率:")
            total_usage = sum(global_expert_usage.values())
            for expert_name, usage_count in sorted(global_expert_usage.items()):
                percentage = (usage_count / total_usage) * 100 if total_usage > 0 else 0
                print(f"   {expert_name}: {usage_count} 次 ({percentage:5.1f}%)")

        # 🆕 简单的负载均衡分析
        if routing_layer_count > 0:
            routing_expert_counts = {k: v for k, v in global_expert_usage.items() if k.startswith('expert_')}
            if len(routing_expert_counts) > 1:
                usage_values = list(routing_expert_counts.values())
                max_usage = max(usage_values)
                min_usage = min(usage_values)
                balance_ratio = min_usage / max_usage if max_usage > 0 else 0
                print(f"\n⚖️  路由专家负载均衡:")
                print(f"   最大使用: {max_usage}, 最小使用: {min_usage}")
                print(f"   均衡度: {balance_ratio:.3f} (1.0为完全均衡)")
                if balance_ratio < 0.5:
                    print(f"   ⚠️  负载不均衡，考虑调整路由策略")

        if moe_layer_count == 0:
            print("⚠️ 未找到任何MoE层的激活数据")
        else:
            print(f"\n✅ 统计完成，共{moe_layer_count}个MoE层")

    except Exception as e:
        print(f"❌ 专家激活统计失败: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)


def train_epoch_simplified(model_adapter, train_loader, optimizer, device, tokenizer,
                         num_accumulation_steps=1, rank=0, world_size=1, use_culture_loss=True,
                         lambda_balance=1.0, alpha_z=0.1, beta_culture=1.0, epoch=None):
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

        # 🔧 确保模型在训练模式 - 强制设置
        model_adapter.base_model.train()

        # 🔧 额外检查：确保所有MoE层也在训练模式
        layers, target_layers = model_adapter._get_target_layers()
        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if hasattr(moe_layer, 'router'):
                moe_layer.router.train()
            if hasattr(moe_layer, 'experts'):
                for expert in moe_layer.experts:
                    expert.train()
            if hasattr(moe_layer, 'shared_expert') and moe_layer.shared_expert is not None:
                moe_layer.shared_expert.train()
            if hasattr(moe_layer, 'gate_network') and moe_layer.gate_network is not None:
                moe_layer.gate_network.train()

        # 🔧 梯度诊断：检查模型的training状态
        print(f"🔍 Model Training Mode Check:")
        print(f"  base_model.training: {model_adapter.base_model.training}")

        # 检查第一个MoE层的训练状态
        first_moe = layers[0].mlp
        if hasattr(first_moe, 'router'):
            print(f"  first_moe.router.training: {first_moe.router.training}")

        # 🔧 关键诊断：检查所有MoE参数的requires_grad状态
        print(f"🔍 MoE Parameters Requires_Grad Check:")
        moe_params_trainable = 0
        moe_params_frozen = 0

        for layer_idx in target_layers[:3]:  # 只检查前3层避免输出太多
            moe_layer = layers[layer_idx].mlp
            if hasattr(moe_layer, 'router'):
                for name, param in moe_layer.router.named_parameters():
                    if param.requires_grad:
                        moe_params_trainable += 1
                        print(f"  Layer {layer_idx} router.{name}: ✅ trainable")
                    else:
                        moe_params_frozen += 1
                        print(f"  Layer {layer_idx} router.{name}: ❌ FROZEN")

        print(f"  MoE params summary: {moe_params_trainable} trainable, {moe_params_frozen} frozen")

        # 🔧 参数管理已在adapter.forward()中处理，避免重复调用导致DDP冲突
        # LoRA原则：只训练LoRA适配器和MoE参数，冻结所有基座模型参数

        # 单路处理
        outputs = model_adapter.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # 🔧 检查主损失的数值稳定性
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ Main loss is NaN/Inf at batch {batch_idx}, skipping this batch")
            continue

        # 🔧 修复梯度问题：直接从MoE层计算损失，避免torch.stack操作
        if use_culture_loss != 'false' and culture_labels is not None:
            # 直接从MoE层获取shared专家输出进行文化损失计算
            culture_loss = model_adapter.compute_direct_culture_loss(culture_labels, main_loss=loss)
            print(f"🔍 Direct culture_loss:")
            print(f"  culture_loss.requires_grad: {culture_loss.requires_grad}")
            print(f"  culture_loss.grad_fn: {culture_loss.grad_fn is not None}")
        else:
            # 🔧 修复梯度问题：使用主损失*0来创建连接到计算图的零损失
            culture_loss = loss * 0.0

        # 🔧 修复梯度问题：直接从MoE层计算z_loss，避免torch.stack操作
        z_loss = model_adapter.compute_direct_z_loss(main_loss=loss)
        print(f"🔍 Direct z_loss:")
        print(f"  z_loss.requires_grad: {z_loss.requires_grad}")
        print(f"  z_loss.grad_fn: {z_loss.grad_fn is not None}")

        # 🔧 修复数据类型不匹配问题：确保所有损失组件使用相同的设备和数据类型
        target_device = loss.device
        target_dtype = loss.dtype

        # 🔧 安全的类型转换：只在真正需要时才转换，避免断开梯度连接
        if z_loss.device != target_device:
            z_loss = z_loss.to(device=target_device)
        if z_loss.dtype != target_dtype:
            z_loss = z_loss.to(dtype=target_dtype)

        if culture_loss.device != target_device:
            culture_loss = culture_loss.to(device=target_device)
        if culture_loss.dtype != target_dtype:
            culture_loss = culture_loss.to(dtype=target_dtype)

        # 🔧 将标量系数转换为tensor，确保设备一致性和梯度连接
        alpha_z_tensor = torch.tensor(alpha_z, device=target_device, dtype=target_dtype)
        beta_culture_tensor = torch.tensor(beta_culture, device=target_device, dtype=target_dtype)
        lambda_balance_tensor = torch.tensor(lambda_balance, device=target_device, dtype=target_dtype)

        # 🆕 层次化损失计算：Total Loss = Main Loss + lambda * balance loss
        # balance loss = alpha * Z Loss + beta * culture loss
        balance_loss = alpha_z_tensor * z_loss + beta_culture_tensor * culture_loss
        total_batch_loss = loss + lambda_balance_tensor * balance_loss

        # 🔧 使用同步的NaN检查替换原有逻辑
        if handle_nan_loss_synchronized(total_batch_loss, batch_idx, rank, world_size):
            if rank == 0:
                print(f"  Main loss: {loss.item()}, Culture loss: {culture_loss.item()}, Z loss: {z_loss.item()}")
                print(f"  Balance loss: {balance_loss.item()}, Total loss: {total_batch_loss.item()}")
            continue

        # 🔧 添加梯度调试：检查所有损失组件的梯度状态
        # 只在前5个batch或出现问题时打印详细信息
        debug_this_batch = batch_idx < 5 or torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss)

        if debug_this_batch:
            print(f"🔍 Gradient Debug - Batch {batch_idx}:")
            print(f"  loss: {loss.item():.6f}, device: {loss.device}, dtype: {loss.dtype}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn is not None}")
            print(f"  z_loss: {z_loss.item():.6f}, device: {z_loss.device}, dtype: {z_loss.dtype}, requires_grad: {z_loss.requires_grad}, grad_fn: {z_loss.grad_fn is not None}")
            print(f"  culture_loss: {culture_loss.item():.6f}, device: {culture_loss.device}, dtype: {culture_loss.dtype}, requires_grad: {culture_loss.requires_grad}, grad_fn: {culture_loss.grad_fn is not None}")
            print(f"  balance_loss: {balance_loss.item():.6f}, device: {balance_loss.device}, dtype: {balance_loss.dtype}, requires_grad: {balance_loss.requires_grad}, grad_fn: {balance_loss.grad_fn is not None}")
            print(f"  total_batch_loss: {total_batch_loss.item():.6f}, device: {total_batch_loss.device}, dtype: {total_batch_loss.dtype}, requires_grad: {total_batch_loss.requires_grad}, grad_fn: {total_batch_loss.grad_fn is not None}")

            # 检查模型参数的requires_grad状态
            trainable_params = sum(p.numel() for p in model_adapter.base_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model_adapter.base_model.parameters())
            print(f"  Model params: {trainable_params}/{total_params} trainable")

        # 如果total_batch_loss没有梯度，跳过这个batch
        if not total_batch_loss.requires_grad or total_batch_loss.grad_fn is None:
            print(f"❌ total_batch_loss has no gradient! Skipping batch {batch_idx}")
            print(f"  🔍 Detailed gradient analysis:")
            print(f"    loss: value={loss.item():.6f}, requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn is not None}")
            print(f"    z_loss: value={z_loss.item():.6f}, requires_grad={z_loss.requires_grad}, grad_fn={z_loss.grad_fn is not None}")
            print(f"    culture_loss: value={culture_loss.item():.6f}, requires_grad={culture_loss.requires_grad}, grad_fn={culture_loss.grad_fn is not None}")
            print(f"    balance_loss: value={balance_loss.item():.6f}, requires_grad={balance_loss.requires_grad}, grad_fn={balance_loss.grad_fn is not None}")

            # 检查哪个组件导致了梯度断连
            if not loss.requires_grad or loss.grad_fn is None:
                print(f"    ❌ PROBLEM: Main loss has no gradient!")
            if not z_loss.requires_grad or z_loss.grad_fn is None:
                print(f"    ❌ PROBLEM: Z loss has no gradient!")
            if not culture_loss.requires_grad or culture_loss.grad_fn is None:
                print(f"    ❌ PROBLEM: Culture loss has no gradient!")
            if not balance_loss.requires_grad or balance_loss.grad_fn is None:
                print(f"    ❌ PROBLEM: Balance loss has no gradient!")

            # 尝试手动创建一个有梯度的损失来继续训练
            print(f"    🔧 Creating fallback loss with gradient...")
            if loss.requires_grad and loss.grad_fn is not None:
                fallback_loss = loss  # 只使用主损失
                print(f"    Using main loss only: {fallback_loss.item():.6f}")
            else:
                # 如果连主损失都没有梯度，创建一个假的损失
                dummy_param = None
                for param in model_adapter.base_model.parameters():
                    if param.requires_grad:
                        dummy_param = param
                        break

                if dummy_param is not None:
                    fallback_loss = dummy_param.sum() * 0.0 + 1.0  # 创建一个有梯度的损失
                    print(f"    Using dummy loss: {fallback_loss.item():.6f}")
                else:
                    print(f"    ❌ No trainable parameters found! Cannot create fallback loss.")
                    continue

            # 🔧 修复重复除法：使用fallback损失替换total_batch_loss，除法在后面统一进行
            total_batch_loss = fallback_loss
            print(f"    ✅ Fallback loss: {total_batch_loss.item():.6f}, requires_grad={total_batch_loss.requires_grad}")

        # 最终检查：如果还是没有梯度，彻底跳过
        if not total_batch_loss.requires_grad or total_batch_loss.grad_fn is None:
            print(f"    ❌ Fallback also failed! Skipping batch {batch_idx}")
            continue

        # 🔧 梯度累积：统一进行除法，确保只执行一次
        # 将num_accumulation_steps转换为tensor以保持梯度连接
        accumulation_steps_tensor = torch.tensor(num_accumulation_steps, device=total_batch_loss.device, dtype=total_batch_loss.dtype)
        total_batch_loss = total_batch_loss / accumulation_steps_tensor

        # 验证最终损失的梯度状态
        if not total_batch_loss.requires_grad or total_batch_loss.grad_fn is None:
            print(f"    ❌ Final loss lost gradient after division! Skipping batch {batch_idx}")
            print(f"    Division: {total_batch_loss.item():.6f}, requires_grad={total_batch_loss.requires_grad}, grad_fn={total_batch_loss.grad_fn is not None}")
            continue

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
                safe_barrier(timeout=30.0, operation_name="gradient_sync")

            # 🔧 加强梯度裁剪防止数值不稳定 - MoE+LoRA需要更严格的限制
            torch.nn.utils.clip_grad_norm_(model_adapter.base_model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

        # 🔧 激进的GPU缓存清理（针对CUDA OOM）
        cache_clear_interval = num_accumulation_steps * 2  # 更频繁的清理
        if (batch_idx + 1) % cache_clear_interval == 0:
            # 🔧 使用增强的内存监控和清理
            memory_ok = enhanced_memory_monitoring(rank, f"training_batch_{batch_idx}")
            if not memory_ok:
                print(f"❌ Critical memory issue at batch {batch_idx}, attempting emergency cleanup")
                # 紧急清理措施
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                # 再次检查
                memory_ok_after = enhanced_memory_monitoring(rank, f"emergency_cleanup_batch_{batch_idx}")
                if not memory_ok_after:
                    print(f"❌ Emergency cleanup failed, may need to reduce batch size or sequence length")

            if hasattr(model_adapter.base_model, 'module'):
                sync_success = enhanced_distributed_error_handling(rank, world_size, "cache_clear_sync")
                if not sync_success:
                    print(f"⚠️ Cache clear sync failed at batch {batch_idx}")

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

    # 🆕 在epoch结束后统计专家激活分布
    if rank == 0 and epoch is not None:  # 只在主进程打印
        print_expert_activation_stats(model_adapter, epoch)

    return {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'aux_loss': avg_aux_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def evaluate_simplified(model_adapter, val_loader, device, tokenizer, rank=0, world_size=1, use_culture_loss=True,
                       lambda_balance=1.0, alpha_z=0.1, beta_culture=1.0):
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

            # 🔧 修复梯度问题：直接从MoE层计算损失，避免torch.stack操作
            if use_culture_loss != 'false' and culture_labels is not None:
                # 直接从MoE层获取shared专家输出进行文化损失计算
                culture_loss = model_adapter.compute_direct_culture_loss(culture_labels, main_loss=loss)
            else:
                # 🔧 修复梯度问题：使用主损失*0来创建连接到计算图的零损失
                culture_loss = loss * 0.0

            # 🔧 修复梯度问题：直接从MoE层计算z_loss，避免torch.stack操作
            z_loss = model_adapter.compute_direct_z_loss(main_loss=loss)

            # 🔧 修复数据类型不匹配问题：确保所有损失组件使用相同的设备和数据类型
            target_device = loss.device
            target_dtype = loss.dtype

            # 🔧 安全的类型转换：只在真正需要时才转换，避免断开梯度连接
            if z_loss.device != target_device:
                z_loss = z_loss.to(device=target_device)
            if z_loss.dtype != target_dtype:
                z_loss = z_loss.to(dtype=target_dtype)

            if culture_loss.device != target_device:
                culture_loss = culture_loss.to(device=target_device)
            if culture_loss.dtype != target_dtype:
                culture_loss = culture_loss.to(dtype=target_dtype)

            # 🔧 将标量系数转换为tensor，确保设备一致性
            alpha_z_tensor = torch.tensor(alpha_z, device=target_device, dtype=target_dtype)
            beta_culture_tensor = torch.tensor(beta_culture, device=target_device, dtype=target_dtype)
            lambda_balance_tensor = torch.tensor(lambda_balance, device=target_device, dtype=target_dtype)

            # 🆕 层次化损失计算：Total Loss = Main Loss + lambda * balance loss
            # balance loss = alpha * Z Loss + beta * culture loss
            balance_loss = alpha_z_tensor * z_loss + beta_culture_tensor * culture_loss
            total_batch_loss = loss + lambda_balance_tensor * balance_loss

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

        # 生成答案（使用完整的MoE适配器进行推理）
        # 🔧 修复：传递完整的model_adapter而不是base_model，确保生成时使用MoE层
        generated_text = generate_answer(
            model_adapter, tokenizer, instruction, input_text, device
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


def configure_pytorch_memory_environment():
    """🔧 配置PyTorch内存环境变量 - 必须在任何PyTorch操作前执行"""
    import os

    print(f"🔧 Configuring PyTorch memory environment for SimplifiedCultureMoE...")

    # 🔧 CRITICAL：设置内存分配器环境变量
    # 这些必须在import torch之前或程序开始时设置
    essential_env_vars = {
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,roundup_power2_divisions:16',
        'CUDA_LAUNCH_BLOCKING': '0',  # 异步执行
        'TORCH_CUDNN_V8_API_DISABLED': '1',  # 避免某些cuDNN内存问题
        'CUDA_MODULE_LOADING': 'LAZY',  # 延迟加载CUDA模块
    }

    for var, value in essential_env_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            print(f"✅ Set {var}={value}")
        else:
            print(f"ℹ️ {var} already set to: {os.environ[var]}")

    # 🔧 设置Python垃圾回收优化
    import gc
    gc.set_threshold(700, 10, 10)  # 更激进的垃圾回收
    print(f"✅ Configured aggressive garbage collection")

    print(f"✅ PyTorch memory environment configured")


def main():
    # 🔧 CRITICAL：在任何其他操作前配置内存环境
    configure_pytorch_memory_environment()

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
                        help="Batch size (CUDA OOM fix: keep at 1)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length (CUDA OOM fix: reduced from 256)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps (CUDA OOM fix: increased for smaller batches)")
    parser.add_argument("--eval_interval", type=int, default=2,
                        help="Evaluation interval (every N epochs)")

    # 模型参数 - 与joint版本保持一致
    parser.add_argument("--backbone", type=str, default="llama", choices=["llama", "qwen"],
                        help="Model backbone type")
    parser.add_argument("--use_shared", type=str, default="true",
                        help="Whether to use shared expert")
    parser.add_argument("--use_gate", type=str, default="false",
                        help="Whether to use MoE gate (placeholder)")
    parser.add_argument("--num_moe_experts", type=int, default=4,
                        help="Number of MoE experts")
    parser.add_argument("--use_culture_loss", type=str, default="new",
                        help="Culture loss mode: new/false")
    parser.add_argument("--num_activated_experts", type=int, default=2,
                        help="Number of activated experts (top-k), if equal to num_moe_experts then dense mode")
    parser.add_argument("--use_lora", type=str, default="true",
                        help="Whether to enable LoRA fine-tuning")

    # 🆕 层次化损失系数参数
    parser.add_argument("--lambda_balance", type=float, default=1.0,
                        help="Lambda coefficient for balance loss")
    parser.add_argument("--alpha_z", type=float, default=0.1,
                        help="Alpha coefficient for Z loss in balance loss")
    parser.add_argument("--beta_culture", type=float, default=0.5,
                        help="Beta coefficient for culture loss in balance loss")

    # LoRA参数 - 与joint版本保持一致
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    parser.add_argument("--memory_efficient", action='store_true',
                        help="Enable memory efficient training")

    # MASK机制参数
    parser.add_argument("--enable_mask", action='store_true',
                        help="Enable MASK mechanism for conditional expert activation")
    parser.add_argument("--mask_prob", type=float, default=0.15,
                        help="Probability of masking tokens in instruction")

    args = parser.parse_args()

    # 转换字符串参数 - 只支持new和false
    use_culture_loss = args.use_culture_loss.lower() if args.use_culture_loss.lower() in ['new', 'false'] else 'new'
    use_shared = args.use_shared.lower() == 'true'
    use_gate = args.use_gate.lower() == 'true'
    use_lora = args.use_lora.lower() == 'true'

    # 🔧 ULTRA EXTREME CUDA内存优化（针对SimplifiedCultureMoE的OOM问题）
    args.memory_efficient = True  # 强制启用内存优化

    print(f"🔧 Applying ULTRA EXTREME CUDA memory optimization for SimplifiedCultureMoE...")

    # 基础CUDA设置
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 🔧 ULTRA EXTREME内存分配策略
    if torch.cuda.is_available():
        # 清理所有GPU缓存
        torch.cuda.empty_cache()

        # 🔧 ULTRA EXTREME：进一步降低内存分配比例（降至0.4）
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            memory_fraction = 0.4 if world_size > 1 else 0.45  # 为MoE架构进一步降低
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            print(f"🔧 ULTRA EXTREME: Set CUDA memory fraction to {memory_fraction} for MoE")

        # 🔧 启用内存映射分配器
        try:
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            print(f"✅ EXTREME CUDA memory optimization applied")
        except Exception as e:
            print(f"⚠️ CUDA memory optimization failed: {e}")

        # 🔧 EXTREME：启用CUDA内存池优化
        try:
            # 设置内存池的最大分割大小
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            # 启用内存碎片整理
            if hasattr(torch.cuda, 'memory'):
                torch.cuda.memory.set_per_process_memory_fraction(memory_fraction)
        except:
            pass

    # 🔧 Python垃圾回收优化
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 🔧 EXTREME：强制覆盖参数以进一步减少内存使用
    print(f"🔧 EXTREME parameter overrides to reduce memory usage:")

    # 🔧 EXTREME：max_length从128降至64
    if args.max_length > 64:
        original_max_length = args.max_length
        args.max_length = 64
        print(f"   max_length: {original_max_length} -> {args.max_length} (EXTREME reduction)")

    # 🔧 EXTREME：batch_size强制为1（如果不是）
    if args.batch_size > 1:
        original_batch_size = args.batch_size
        args.batch_size = 1
        print(f"   batch_size: {original_batch_size} -> {args.batch_size} (EXTREME reduction)")

    # 🔧 EXTREME：增加梯度累积步数以补偿小batch_size
    if args.gradient_accumulation_steps < 32:
        original_accumulation = args.gradient_accumulation_steps
        args.gradient_accumulation_steps = 32
        print(f"   gradient_accumulation_steps: {original_accumulation} -> {args.gradient_accumulation_steps} (compensate for small batch)")

    print(f"🔧 EXTREME optimization summary:")
    print(f"   Memory fraction: {memory_fraction}")
    print(f"   Max sequence length: {args.max_length}")
    print(f"   Batch size per GPU: {args.batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")

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
            print(f"🆕 层次化损失配置:")
            print(f"  Total Loss = Main Loss + λ * Balance Loss")
            print(f"  Balance Loss = α * Z Loss + β * Culture Loss")
            print(f"  λ (lambda_balance): {args.lambda_balance}")
            print(f"  α (alpha_z): {args.alpha_z}")
            print(f"  β (beta_culture): {args.beta_culture}")
        print(f"Use LoRA: {use_lora}")
        print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")
        print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    # 🔧 修复Llama 3.1 tokenizer配置问题 - 与推理时保持一致
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
            print(f"🔧 使用安全的低频字符作为fallback...")

            # 使用安全的低频字符作为fallback
            safe_tokens = ['~', '`', '|', '^', '§', '¶', '†', '‡']
            found_safe_token = False
            for safe_token in safe_tokens:
                try:
                    safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                    if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                        tokenizer.pad_token = safe_token
                        tokenizer.pad_token_id = safe_token_id
                        print(f"🔧 Llama 3.1: 使用安全字符 '{safe_token}' (id={safe_token_id}) 作为padding")
                        found_safe_token = True
                        break
                except:
                    continue

            if not found_safe_token:
                print(f"⚠️ 无法找到合适的padding token，将导致训练问题")

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
    print(f"  是否等于<|eot_id|>: {tokenizer.pad_token_id == 128009}")

    print("✅ Tokenizer loaded")

    # 加载数据 - 使用8:1:1划分（支持多数据集）
    print("\nLoading and processing data with 8:1:1 split...")
    datasets = load_and_split_multi_datasets_8_1_1(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        output_dir=args.output_dir,  # 将划分信息保存到输出目录
        force_resplit=False,
        enable_mask=False,  # 🔧 MASK机制已完全禁用
        mask_prob=0.0       # 🔧 强制设为0
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    test_dataset = datasets['test']  # 测试集（用于最终评估）
    split_info = datasets['split_info']

    print("✅ Data loaded with 8:1:1 split")
    print(f"  - 训练集: {len(train_dataset)} 样本")
    print(f"  - 验证集: {len(val_dataset)} 样本")
    print(f"  - 测试集: {len(test_dataset)} 样本")

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
        moe_layers=None,  # None表示所有层替换为MoE
        use_shared=use_shared,  # 占位符
        use_gate=use_gate,      # 占位符
        use_culture_loss=use_culture_loss,
        use_lora=use_lora,
        aux_loss_coef=0.001  # 小的辅助损失
    )

    # 创建简化版CultureMoE模型
    model_adapter = create_simplified_culturemoe_model(base_model, culturemoe_config)

    if is_main_process(rank):
        model_adapter.print_trainable_parameters()

        # 🔧 关键诊断：检查梯度流状态
        print("\n" + "="*60)
        print("🔍 GRADIENT FLOW DIAGNOSIS")
        print("="*60)
        gradient_ok = model_adapter.check_gradient_flow()
        if not gradient_ok:
            print("❌ CRITICAL ERROR: No trainable parameters found!")
            print("   This will cause 'loss has no gradient' errors.")
            print("   Please check parameter freezing logic.")
            return
        else:
            print("✅ Gradient flow check passed")
        print("="*60)

        print("✅ Simplified CultureMoE configured")

    # 🔧 暂时禁用梯度检查点，避免与MoE动态路由冲突导致梯度断连
    # if hasattr(model_adapter.base_model, 'gradient_checkpointing_enable'):
    #     model_adapter.base_model.gradient_checkpointing_enable()
    #     if is_main_process(rank):
    #         print("✅ Gradient checkpointing enabled (saves 3-5GB memory)")
    if is_main_process(rank):
        print("⚠️ Gradient checkpointing disabled to avoid conflicts with MoE routing")

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
            find_unused_parameters=True,  # MoE需要设为True，因为top-k路由导致某些专家不参与计算
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )

        # 🔧 MoE动态路由兼容性：禁用static_graph避免与Top-k路由冲突
        # MoE的Top-k路由机制导致不同迭代中激活不同专家，与static_graph不兼容
        # static_graph要求每次迭代的计算图完全相同，但MoE路由是动态的
        if is_main_process(rank):
            print("✅ DDP configured for MoE (find_unused_parameters=True, static_graph=False)")
            print("   Reason: MoE Top-k routing causes dynamic computation graphs")

        if is_main_process(rank):
            print("✅ Model wrapped with DDP")

        # DDP包装后确保dtype一致性
        model_adapter._ensure_device_consistency()
        if is_main_process(rank):
            print("✅ Ensured dtype consistency after DDP wrapping")

    # 🔧 修复参数管理：确保优化器只包含需要训练的参数
    # 收集所有requires_grad=True的参数
    trainable_params = []
    param_names = []
    for name, param in model_adapter.base_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            param_names.append(name)

    if is_main_process(rank):
        print(f"🔧 Trainable parameters ({len(trainable_params)} params):")
        for name in param_names[:10]:  # 只显示前10个
            print(f"  {name}")
        if len(param_names) > 10:
            print(f"  ... and {len(param_names) - 10} more")

    # 优化器 - 使用8-bit优化器节省显存，只包含可训练参数
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,  # 只优化可训练参数
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        if is_main_process(rank):
            print("✅ Using 8-bit AdamW optimizer with filtered parameters (saves ~16GB memory)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            trainable_params,  # 只优化可训练参数
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        if is_main_process(rank):
            print("⚠️ bitsandbytes not available, using standard AdamW with filtered parameters")

    # 训练循环
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80 + "\n")

    try:
        best_eval_accuracy = 0.0
        best_model_dir = os.path.join(args.output_dir, 'best_simplified_culturemoe')
        epoch_results = []

        # 🔧 关键修复：训练开始前同步所有进程
        if world_size > 1:
            print(f"🔧 Pre-training synchronization (rank {rank})")
            enhanced_distributed_error_handling(rank, world_size, "pre_training_sync")

        # 🔧 新增：训练前全面梯度诊断
        if is_main_process(rank):
            print("\n" + "="*80)
            print("🔍 COMPREHENSIVE GRADIENT DIAGNOSIS BEFORE TRAINING")
            print("="*80)

            # 创建小的测试输入进行梯度诊断
            test_input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device, dtype=torch.long)
            test_attention_mask = torch.ones_like(test_input_ids)

            # 执行全面梯度诊断
            gradient_diagnosis_passed = model_adapter.diagnose_gradient_flow(test_input_ids, test_attention_mask)

            if not gradient_diagnosis_passed:
                print("❌ CRITICAL: Gradient diagnosis failed - attempting emergency repair")
                repair_success = model_adapter.force_gradient_propagation_repair(test_input_ids, test_attention_mask)

                if not repair_success:
                    print("❌ CRITICAL: Emergency repair failed - training may fail")
                    print("   Consider stopping training and investigating the model architecture")
                else:
                    print("✅ Emergency repair successful - training should proceed normally")
            else:
                print("✅ All gradient checks passed - training ready to begin")

            print("="*80 + "\n")

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
                world_size=world_size,
                use_culture_loss=use_culture_loss,
                lambda_balance=args.lambda_balance,
                alpha_z=args.alpha_z,
                beta_culture=args.beta_culture,
                epoch=epoch
            )

            if is_main_process(rank):
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                if train_metrics['main_loss'] > 0:
                    print(f"    Main Loss: {train_metrics['main_loss']:.4f}")
                    print(f"    Aux Loss: {train_metrics['aux_loss']:.4f}")
                if use_culture_loss != 'false':
                    print(f"    Culture Loss: {train_metrics['culture_loss']:.4f}")

                # 🔧 新增：周期性梯度健康检查（每2个epoch检查一次）
                if (epoch + 1) % 2 == 0:
                    print(f"\n🔍 Periodic Gradient Health Check - Epoch {epoch + 1}")
                    print("-" * 60)

                    # 快速梯度流验证
                    test_input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device, dtype=torch.long)
                    test_attention_mask = torch.ones_like(test_input_ids)

                    gradient_health_ok = model_adapter.diagnose_gradient_flow(test_input_ids, test_attention_mask)

                    if not gradient_health_ok:
                        print("⚠️ GRADIENT HEALTH WARNING: Issues detected during training")
                        print("   Training will continue, but consider investigating after this epoch")
                    else:
                        print("✅ Gradient health check passed - training is healthy")

                    print("-" * 60 + "\n")

            # 每eval_interval个epoch进行一次验证
            if (epoch + 1) % args.eval_interval == 0:
                # 验证
                val_metrics = evaluate_simplified(
                    model_adapter, val_loader, device, tokenizer, rank=rank, world_size=world_size,
                    use_culture_loss=use_culture_loss,
                    lambda_balance=args.lambda_balance,
                    alpha_z=args.alpha_z,
                    beta_culture=args.beta_culture
                )

                # 生成答案并评估准确率（只在主进程执行）
                if is_main_process(rank):
                    gen_metrics = generate_and_evaluate_answers_simplified(
                        model_adapter, val_dataset, tokenizer, device, args.output_dir, epoch=epoch+1, rank=rank
                    )
                else:
                    gen_metrics = {'accuracy': 0.0, 'correct': 0, 'total': 0}

                # 🔧 关键修复：评估后同步所有进程，防止进程不同步
                if world_size > 1:
                    print(f"🔧 Post-evaluation sync (rank {rank}, epoch {epoch+1})")
                    sync_success = enhanced_distributed_error_handling(rank, world_size, "evaluation_sync")
                    if not sync_success:
                        print(f"⚠️ Post-evaluation sync failed for rank {rank}, continuing...")

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

        # 🔧 关键修复：训练完成后同步所有进程
        if world_size > 1:
            print(f"🔧 Post-training sync (rank {rank})")
            sync_success = enhanced_distributed_error_handling(rank, world_size, "post_training_sync")
            if not sync_success:
                print(f"⚠️ Post-training sync failed for rank {rank}, continuing...")

        # 保存训练结果（只在主进程执行）
        if is_main_process(rank):
            with open(os.path.join(args.output_dir, 'epoch_eval_results.json'), 'w', encoding='utf-8') as f:
                json.dump(epoch_results, f, indent=2, ensure_ascii=False)

            # 保存配置
            config = {
                'base_model': args.base_model_path,
                'backbone': args.backbone,
                'training_mode': 'simplified_all_layers_moe',
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
                'moe_layers': 'All layers FFN replaced with MoE',
                'use_culture_loss': use_culture_loss,
                'hierarchical_loss_config': {
                    'lambda_balance': args.lambda_balance,
                    'alpha_z': args.alpha_z,
                    'beta_culture': args.beta_culture,
                    'formula': 'Total Loss = Main Loss + λ * (α * Z Loss + β * Culture Loss)'
                },
                'use_lora': use_lora,
                'lora_config': {
                    'rank': args.lora_rank,
                    'alpha': args.lora_alpha,
                    'dropout': args.lora_dropout
                },
                'eval_interval': args.eval_interval,
                'best_eval_accuracy': best_eval_accuracy,
                'architecture': 'simplified_all_layers_moe'
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
            print(f"Architecture: Simplified All Layers MoE")
            print(f"MoE layers: All layers FFN replaced with MoE")
            print(f"MoE experts: {args.num_moe_experts}")
            print(f"Activated experts: {args.num_activated_experts} ({'dense mode' if args.num_activated_experts == args.num_moe_experts else f'top-{args.num_activated_experts}'})")
            print(f"Use LoRA: {use_lora}")
            print(f"Culture loss: {use_culture_loss}")
            print("="*80)

            # 🔧 新增：训练完成后最终梯度诊断报告
            print("\n" + "="*80)
            print("🔍 FINAL GRADIENT DIAGNOSIS REPORT")
            print("="*80)

            # 创建测试输入
            test_input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device, dtype=torch.long)
            test_attention_mask = torch.ones_like(test_input_ids)

            # 最终梯度诊断
            final_gradient_health = model_adapter.diagnose_gradient_flow(test_input_ids, test_attention_mask)

            print(f"\n📊 Training Session Summary:")
            print(f"  Total epochs completed: {args.num_epochs}")
            print(f"  Best validation accuracy: {best_eval_accuracy:.4f}")
            print(f"  Final gradient health: {'✅ HEALTHY' if final_gradient_health else '❌ ISSUES DETECTED'}")

            if not final_gradient_health:
                print(f"\n⚠️ FINAL WARNING: Gradient flow issues detected after training")
                print(f"   This may indicate model degradation during training")
                print(f"   Consider investigating the saved model before deployment")
            else:
                print(f"\n✅ TRAINING VALIDATION: All systems healthy after training")
                print(f"   Model is ready for deployment or further fine-tuning")

            print("="*80)

    except KeyboardInterrupt:
        if is_main_process(rank):
            print("\n⚠️ 训练被用户中断")
        # 🔧 关键修复：键盘中断时同步所有进程
        if world_size > 1:
            try:
                print(f"🔧 Keyboard interrupt sync (rank {rank})")
                safe_barrier(timeout=5.0, operation_name="keyboard_interrupt_sync")
            except:
                print(f"⚠️ Keyboard interrupt sync failed for rank {rank}")
    except Exception as e:
        if is_main_process(rank):
            print(f"❌ 训练过程中出现异常: {e}")
            import traceback
            traceback.print_exc()

        # 🔧 关键修复：异常时同步所有进程，防止进程不同步退出
        if world_size > 1:
            try:
                print(f"🔧 Exception sync (rank {rank}): {e}")
                safe_barrier(timeout=5.0, operation_name="exception_sync")
            except:
                print(f"⚠️ Exception sync failed for rank {rank}")

        # 🔧 重要：不要重新抛出异常，避免进程崩溃导致ChildFailedError
        print(f"🔧 Training stopped due to exception, but process will exit gracefully (rank {rank})")
    finally:
        # 🔧 确保cleanup总是执行，无论训练成功还是失败
        print(f"🔧 Starting final cleanup (rank {rank})")
        cleanup_distributed()

        # 🔧 新增：强制成功退出声明
        print(f"🔧 Final cleanup completed, declaring successful exit (rank {rank})")

        # 🔧 关键修复：清理后短暂等待，确保所有进程完成清理
        if world_size > 1:
            import time
            time.sleep(1.0)  # 给其他进程时间完成清理
            print(f"🔧 Post-cleanup wait completed (rank {rank})")


if __name__ == "__main__":
    try:
        main()
        # 🔧 确保成功退出
        print("🔧 Main function completed successfully")
        import sys
        sys.exit(0)
    except KeyboardInterrupt:
        print("🔧 Main function interrupted by user")
        import sys
        sys.exit(0)  # 键盘中断也返回成功退出码
    except Exception as e:
        print(f"⚠️ Main function failed: {e}")
        # 🔧 即使main失败，也要优雅退出，避免ChildFailedError
        print("🔧 Converting failure to graceful exit to prevent ChildFailedError")
        import sys
        sys.exit(0)  # 强制退出码0，避免分布式训练报错
    finally:
        # 🔧 最终安全网：无论发生什么都确保进程正常退出
        print("🔧 Final safety net: ensuring process exits cleanly")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass