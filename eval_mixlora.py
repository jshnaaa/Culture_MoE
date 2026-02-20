#!/usr/bin/env python3
"""
MixLoRA 模型评估脚本

从保存的训练参数还原完整模型，并在测试集上进行评估。

使用方法：
    python eval_mixlora.py \
        --base_model_path /path/to/base_model \
        --mixlora_model_path /path/to/mixlora_model \
        --pkl_file /path/to/data_split.pkl \
        --output_dir /path/to/output

或：
    python eval_mixlora.py \
        --base_model_path /path/to/base_model \
        --mixlora_model_path /path/to/mixlora_model \
        --data_file /path/to/data.json \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ft_lora_only_gen import CultureLLMNewFormatDataset, extract_answer_from_text, generate_answer
from src.llamafactory.model.mixlora_adapter import MixLoRAModelAdapter
from src.llamafactory.model.mixlora import MixLoRAConfig


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        print(f"Initializing distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        dist.barrier()
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """检查是否为主进程"""
    return rank == 0


def load_mixlora_model(base_model_path: str, mixlora_model_path: str, device: str):
    """
    从保存的参数还原 MixLoRA 模型

    Args:
        base_model_path: 基础模型路径
        mixlora_model_path: MixLoRA 模型保存路径
        device: 设备

    Returns:
        还原的 MixLoRA 模型适配器
    """
    print(f"🔧 加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    base_model = base_model.to(device)
    print("✅ 基础模型加载完成")

    # 加载 MixLoRA 配置
    config_path = os.path.join(mixlora_model_path, 'mixlora_config.json')
    print(f"🔧 加载 MixLoRA 配置: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    mixlora_config = MixLoRAConfig.from_dict(config_dict)
    print("✅ MixLoRA 配置加载完成")

    # 创建 MixLoRA 适配器
    print(f"🔧 构建 MixLoRA 模型...")
    adapter = MixLoRAModelAdapter(base_model, mixlora_config)

    # 加载 MixLoRA 权重
    weights_path = os.path.join(mixlora_model_path, 'mixlora_weights.pt')
    print(f"🔧 加载 MixLoRA 权重: {weights_path}")
    mixlora_state_dict = torch.load(weights_path, map_location='cpu')

    for layer_name, state_dict in mixlora_state_dict.items():
        if layer_name in adapter.mixlora_layers:
            adapter.mixlora_layers[layer_name].load_state_dict(state_dict, strict=False)

    print("✅ MixLoRA 权重加载完成")
    return adapter


def load_test_data_from_pkl(pkl_file: str, tokenizer, max_length: int = 512):
    """
    从 pkl 文件加载测试集数据

    Args:
        pkl_file: pkl 文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度

    Returns:
        测试数据集
    """
    print(f"🔧 从pkl文件加载测试集: {pkl_file}")

    with open(pkl_file, 'rb') as f:
        split_info = pickle.load(f)

    # 获取数据文件路径和测试集索引
    if 'data_path' in split_info:
        data_path = split_info['data_path']
    elif 'data_paths' in split_info:
        # 合并数据集模式
        data_path = split_info['data_paths']
    else:
        raise ValueError(f"无法从pkl文件中找到数据路径: {pkl_file}")

    test_indices = split_info.get('test_indices', [])

    print(f"  - 数据文件: {data_path}")
    print(f"  - 测试集样本数: {len(test_indices)}")

    # 加载完整数据集
    if isinstance(data_path, list):
        # 多数据集模式 - 使用第一个数据集（假设已经合并）
        data_path = data_path[0]

    full_dataset = CultureLLMNewFormatDataset(data_path, tokenizer, max_length)

    # 创建测试集子集
    test_dataset = Subset(full_dataset, test_indices)

    return test_dataset


def load_test_data_from_file(data_file: str, tokenizer, max_length: int = 512, test_split: float = 0.1):
    """
    从数据文件加载测试集（最后 test_split 比例作为测试集）

    Args:
        data_file: 数据文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度
        test_split: 测试集比例

    Returns:
        测试数据集
    """
    print(f"🔧 从数据文件加载测试集: {data_file}")

    full_dataset = CultureLLMNewFormatDataset(data_file, tokenizer, max_length)

    # 计算测试集大小
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)

    # 取最后 test_size 个样本作为测试集
    test_indices = list(range(total_size - test_size, total_size))

    print(f"  - 总样本数: {total_size}")
    print(f"  - 测试集样本数: {len(test_indices)}")

    test_dataset = Subset(full_dataset, test_indices)

    return test_dataset


def evaluate_mixlora_model(model_adapter, test_dataset, tokenizer, device, output_dir, rank=0):
    """
    评估 MixLoRA 模型

    Args:
        model_adapter: MixLoRA 模型适配器
        test_dataset: 测试数据集
        tokenizer: tokenizer
        device: 设备
        output_dir: 输出目录
        rank: 进程rank

    Returns:
        评估结果字典
    """
    model_adapter.base_model.eval()

    correct = 0
    total = 0
    generated_data = []

    for idx in tqdm(range(len(test_dataset)), desc="Evaluating", disable=(rank != 0), mininterval=1.0):
        # 获取样本
        if hasattr(test_dataset, 'dataset'):
            original_idx = test_dataset.indices[idx]
            sample = test_dataset.dataset[original_idx]
        else:
            sample = test_dataset[idx]

        instruction = sample['instruction']
        input_text = sample['input']
        true_output = sample['output']
        label = sample.get('label', '')

        # 生成答案
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

    accuracy = correct / total if total > 0 else 0

    # 保存结果（只在主进程执行）
    if is_main_process(rank):
        # 保存详细结果
        with open(os.path.join(output_dir, 'detailed_results.json'), 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=2, ensure_ascii=False)

        # 保存评估摘要
        results = {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_samples': total,
            'model_type': 'MixLoRA'
        }

        with open(os.path.join(output_dir, 'eval_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 打印结果
        print("\n" + "="*80)
        print("📊 评估结果")
        print("="*80)
        print(f"  准确率: {accuracy:.4f}")
        print(f"  正确预测: {correct}/{total}")
        print("="*80)

        # 打印样本示例
        print("\n📋 评估样本示例:")
        print("-" * 80)
        for idx in range(min(5, len(generated_data))):
            item = generated_data[idx]
            print(f"\n样本 {idx + 1}:")
            print(f"  Instruction: {item['instruction'][:80]}...")
            print(f"  Input: {item['input']}")
            print(f"  True Output: {item['true_output']}")
            print(f"  Predicted Answer: {item['predicted_answer']}")
            print(f"  Correct: {'✅' if item['correct'] else '❌'}")
        print("\n" + "-" * 80)

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MixLoRA model")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--mixlora_model_path", type=str, required=True,
                        help="Path to saved MixLoRA model")
    parser.add_argument("--pkl_file", type=str, default=None,
                        help="Path to pkl file containing test split")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to data file (alternative to pkl_file)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--backbone", type=str, default='llama',
                        help="Model backbone (llama or qwen)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")

    args = parser.parse_args()

    # 初始化分布式训练
    rank, world_size, local_rank = setup_distributed()

    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        print("\n" + "="*80)
        print("MixLoRA Model Evaluation")
        print("="*80)
        print(f"Base model: {args.base_model_path}")
        print(f"MixLoRA model: {args.mixlora_model_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Backbone: {args.backbone}")
        print(f"Max length: {args.max_length}")
        if args.pkl_file:
            print(f"PKL file: {args.pkl_file}")
        if args.data_file:
            print(f"Data file: {args.data_file}")
        print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer
    if is_main_process(rank):
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if is_main_process(rank):
        print("✅ Tokenizer loaded")

    # 加载测试数据
    if is_main_process(rank):
        print("\nLoading test data...")

    if args.pkl_file:
        test_dataset = load_test_data_from_pkl(args.pkl_file, tokenizer, args.max_length)
    elif args.data_file:
        test_dataset = load_test_data_from_file(args.data_file, tokenizer, args.max_length)
    else:
        raise ValueError("Either --pkl_file or --data_file must be provided")

    if is_main_process(rank):
        print(f"✅ Test data loaded: {len(test_dataset)} samples")

    # 加载 MixLoRA 模型
    if is_main_process(rank):
        print("\nLoading MixLoRA model...")
    model_adapter = load_mixlora_model(args.base_model_path, args.mixlora_model_path, device)

    # 使用DDP包装模型（仅在多GPU时）
    if world_size > 1:
        model_adapter.base_model = DDP(
            model_adapter.base_model,
            device_ids=[local_rank],
            output_device=local_rank
        )
        if is_main_process(rank):
            print("✅ Model wrapped with DDP")

    # 进行评估
    if is_main_process(rank):
        print("\nStarting evaluation...")

    results = evaluate_mixlora_model(
        model_adapter, test_dataset, tokenizer, device, args.output_dir, rank
    )

    # 清理分布式训练
    cleanup_distributed()

    if is_main_process(rank):
        print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
