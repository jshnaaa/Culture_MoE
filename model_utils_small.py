#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model utilities for compressed weight handling
支持权重压缩和智能加载的工具函数
"""

import os
import gzip
import torch
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def save_moe_weights_compressed(weights: Dict[str, torch.Tensor],
                              save_path: str,
                              compression_type: str = "fp16") -> str:
    """
    保存压缩的MoE权重

    Args:
        weights: MoE权重字典
        save_path: 保存路径
        compression_type: 压缩类型 ("fp16", "gzip", "fp16_gzip")

    Returns:
        实际保存的文件路径
    """
    save_path = Path(save_path)

    if compression_type == "fp16":
        # FP16压缩
        compressed_weights = {k: v.half() for k, v in weights.items()}
        save_file = save_path.parent / f"{save_path.stem}_fp16{save_path.suffix}"
        torch.save(compressed_weights, save_file)
        logger.info(f"Saved FP16 compressed weights to {save_file}")

    elif compression_type == "gzip":
        # Gzip压缩
        save_file = save_path.parent / f"{save_path.name}.gz"
        with gzip.open(save_file, 'wb') as f:
            torch.save(weights, f)
        logger.info(f"Saved gzip compressed weights to {save_file}")

    elif compression_type == "fp16_gzip":
        # FP16 + Gzip双重压缩
        compressed_weights = {k: v.half() for k, v in weights.items()}
        save_file = save_path.parent / f"{save_path.stem}_fp16{save_path.suffix}.gz"
        with gzip.open(save_file, 'wb') as f:
            torch.save(compressed_weights, f)
        logger.info(f"Saved FP16+gzip compressed weights to {save_file}")

    else:
        # 普通保存
        torch.save(weights, save_path)
        save_file = save_path
        logger.info(f"Saved uncompressed weights to {save_file}")

    return str(save_file)

def load_moe_weights_smart(weight_path: str,
                          target_device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
    """
    智能加载MoE权重，自动检测和处理压缩格式

    Args:
        weight_path: 权重文件路径
        target_device: 目标设备

    Returns:
        加载的权重字典
    """
    weight_path = Path(weight_path)

    # 如果文件不存在，尝试寻找压缩版本
    if not weight_path.exists():
        logger.warning(f"Weight file {weight_path} not found, searching for compressed versions...")

        # 搜索可能的压缩文件
        compressed_variants = [
            weight_path.parent / f"{weight_path.stem}_fp16{weight_path.suffix}",
            weight_path.parent / f"{weight_path.name}.gz",
            weight_path.parent / f"{weight_path.stem}_fp16{weight_path.suffix}.gz",
        ]

        for variant in compressed_variants:
            if variant.exists():
                weight_path = variant
                logger.info(f"Found compressed version: {weight_path}")
                break
        else:
            raise FileNotFoundError(f"No weight file found for {weight_path}")

    # 检测文件类型并加载
    file_name = weight_path.name.lower()

    try:
        if file_name.endswith('.gz'):
            # Gzip压缩文件
            logger.info(f"Loading gzip compressed weights from {weight_path}")
            with gzip.open(weight_path, 'rb') as f:
                weights = torch.load(f, map_location='cpu')
        else:
            # 普通文件
            logger.info(f"Loading weights from {weight_path}")
            weights = torch.load(weight_path, map_location='cpu')

        # 检查是否为FP16并转换
        if weights and isinstance(weights, dict):
            first_tensor = next(iter(weights.values()))
            if hasattr(first_tensor, 'dtype') and first_tensor.dtype == torch.float16:
                logger.info("Converting FP16 weights to FP32")
                weights = {k: v.float() for k, v in weights.items()}

        # 移动到目标设备
        if target_device is not None:
            logger.info(f"Moving weights to device: {target_device}")
            weights = {k: v.to(target_device) for k, v in weights.items()}

        logger.info(f"Successfully loaded weights with {len(weights)} components")
        return weights

    except Exception as e:
        logger.error(f"Failed to load weights from {weight_path}: {e}")
        raise

def get_compression_info(weight_path: str) -> Dict[str, Any]:
    """
    获取权重文件的压缩信息

    Args:
        weight_path: 权重文件路径

    Returns:
        包含压缩信息的字典
    """
    weight_path = Path(weight_path)

    if not weight_path.exists():
        return {"exists": False, "error": "File not found"}

    info = {
        "exists": True,
        "file_path": str(weight_path),
        "file_size_mb": weight_path.stat().st_size / (1024 * 1024),
        "is_gzip": weight_path.name.endswith('.gz'),
        "is_fp16": '_fp16' in weight_path.name,
        "compression_type": "none"
    }

    # 确定压缩类型
    if info["is_gzip"] and info["is_fp16"]:
        info["compression_type"] = "fp16_gzip"
    elif info["is_gzip"]:
        info["compression_type"] = "gzip"
    elif info["is_fp16"]:
        info["compression_type"] = "fp16"

    return info

def find_best_weight_file(base_path: str) -> Optional[str]:
    """
    在给定目录中寻找最佳的权重文件（优先压缩版本）

    Args:
        base_path: 基础路径（可能包含或不包含具体文件名）

    Returns:
        找到的最佳权重文件路径，如果未找到则返回None
    """
    base_path = Path(base_path)

    # 如果是具体文件路径
    if base_path.suffix == '.pth':
        candidates = [
            base_path.parent / f"{base_path.stem}_fp16{base_path.suffix}.gz",  # 最优：FP16+Gzip
            base_path.parent / f"{base_path.stem}_fp16{base_path.suffix}",     # 次优：FP16
            base_path.parent / f"{base_path.name}.gz",                         # 第三：Gzip
            base_path,                                                         # 最后：原文件
        ]
    else:
        # 如果是目录，寻找moe_weights相关文件
        candidates = [
            base_path / "moe_weights_fp16.pth.gz",
            base_path / "moe_weights_fp16.pth",
            base_path / "moe_weights.pth.gz",
            base_path / "moe_weights.pth",
        ]

    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Found weight file: {candidate}")
            return str(candidate)

    logger.warning(f"No weight file found in {base_path}")
    return None

def estimate_memory_usage(weight_path: str) -> Dict[str, float]:
    """
    估算权重文件的内存使用量

    Args:
        weight_path: 权重文件路径

    Returns:
        内存使用估算（MB）
    """
    info = get_compression_info(weight_path)

    if not info["exists"]:
        return {"error": "File not found"}

    file_size_mb = info["file_size_mb"]

    # 估算加载后的内存使用
    if info["compression_type"] == "fp16_gzip":
        # FP16+Gzip: 文件大小 * 2 (解压) * 2 (FP16->FP32转换)
        estimated_memory = file_size_mb * 4
    elif info["compression_type"] == "fp16":
        # FP16: 文件大小 * 2 (FP16->FP32转换)
        estimated_memory = file_size_mb * 2
    elif info["compression_type"] == "gzip":
        # Gzip: 文件大小 * 2-3 (解压比例)
        estimated_memory = file_size_mb * 2.5
    else:
        # 无压缩: 文件大小
        estimated_memory = file_size_mb

    return {
        "file_size_mb": file_size_mb,
        "estimated_memory_mb": estimated_memory,
        "compression_type": info["compression_type"]
    }

# 兼容性函数：保持与原有代码的接口一致
def load_moe_weights(weight_path: str, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    兼容性函数：与原有的load_moe_weights接口保持一致
    """
    return load_moe_weights_smart(weight_path, device)

if __name__ == "__main__":
    # 测试代码
    import argparse

    parser = argparse.ArgumentParser(description="Test model utils")
    parser.add_argument("--weight_path", type=str, help="Weight file path to test")
    parser.add_argument("--action", type=str, choices=["info", "memory", "find"],
                       default="info", help="Action to perform")

    args = parser.parse_args()

    if args.weight_path:
        if args.action == "info":
            info = get_compression_info(args.weight_path)
            print(f"Weight file info: {info}")
        elif args.action == "memory":
            memory = estimate_memory_usage(args.weight_path)
            print(f"Memory estimation: {memory}")
        elif args.action == "find":
            best_file = find_best_weight_file(args.weight_path)
            print(f"Best weight file: {best_file}")
    else:
        print("Please provide --weight_path argument")