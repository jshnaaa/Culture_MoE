#!/usr/bin/env python3
"""
联合训练CultureMoE模型评估脚本

专门用于评估通过train_joint_lora_moe.py训练的联合LoRA+MoE模型
支持8:1:1数据集划分的测试集评估和完整数据集评估

使用方法：
    python eval_joint_culturemoe.py \
        --base_model_path /path/to/base_model \
        --joint_model_path /path/to/best_joint_model \
        --data_file /path/to/data.json \
        --data_id 2 \
        --output_dir /path/to/output

特点：
    1. 支持pkl文件测试集 (data_id=0) 和完整数据集 (data_id=1-5)
    2. 加载联合训练的LoRA+MoE模型
    3. 生成式评估，提取数字答案并计算准确率
    4. 分布式评估支持
"""

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.joint_lora_moe_model import JointLoRAMoEModel, JointLoRAMoEConfig

# 复用现有的数据集和工具函数
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    extract_answer_from_text,
    generate_answer,
    dynamic_padding_collate_fn
)


def setup_distributed():
    """初始化分布式评估"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        print(f"Initializing distributed evaluation: rank={rank}, world_size={world_size}, local_rank={local_rank}")

        # 初始化进程组
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        # 设置当前进程的GPU
        torch.cuda.set_device(local_rank)

        # 同步所有进程
        dist.barrier()

        return rank, world_size, local_rank
    else:
        # 单GPU模式
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式评估"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """检查是否为主进程"""
    return rank == 0


def load_test_dataset_from_pkl(pkl_file_path: str, tokenizer, max_length: int = 512):
    """
    从pkl文件加载测试集

    Args:
        pkl_file_path: pkl文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度

    Returns:
        测试数据集
    """
    print(f"Loading test dataset from pkl file: {pkl_file_path}")

    with open(pkl_file_path, 'rb') as f:
        split_info = pickle.load(f)

    # 获取原始数据路径和测试集索引
    original_data_path = split_info['data_path']
    test_indices = split_info['test_indices']

    print(f"Original data path: {original_data_path}")
    print(f"Test set size: {len(test_indices)}")

    # 创建完整数据集
    full_dataset = CultureLLMNewFormatDataset(original_data_path, tokenizer, max_length)

    # 创建测试集子集
    test_dataset = Subset(full_dataset, test_indices)

    return test_dataset


def load_test_dataset_from_file(data_file_path: str, tokenizer, max_length: int = 512):
    """
    从完整数据文件加载测试集（使用全部数据）

    Args:
        data_file_path: 数据文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度

    Returns:
        测试数据集
    """
    print(f"Loading test dataset from full data file: {data_file_path}")

    test_dataset = CultureLLMNewFormatDataset(data_file_path, tokenizer, max_length)

    print(f"Test set size: {len(test_dataset)}")

    return test_dataset


def load_joint_model(base_model_path: str, joint_model_path: str, device: str,
                    num_moe_experts: int = 4, num_activated_experts: int = 2,
                    use_shared: bool = True, use_gate: bool = True, use_mask: bool = True,
                    use_culture_loss: str = "csl", lora_rank: int = 16, lora_alpha: int = 32):
    """
    加载联合训练的LoRA+MoE模型

    Args:
        base_model_path: 基础模型路径
        joint_model_path: 联合模型路径 (best_joint_model/)
        device: 设备
        其他参数: 模型配置参数

    Returns:
        加载的联合模型
    """
    print(f"Loading base model from: {base_model_path}")

    # 加载基础模型
    load_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': None,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True
    }

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    base_model = base_model.to(device)

    print(f"Loading joint model from: {joint_model_path}")

    # 读取保存的配置
    config_file = os.path.join(joint_model_path, 'joint_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        print("✅ Loaded saved joint model configuration")
        print(f"  - Saved config keys: {list(saved_config.keys())}")
    else:
        print("⚠️ No saved config found, using default configuration")
        saved_config = {}

    # 🔧 修复：创建联合模型配置，支持消融评估
    # 消融评估模式：使用传入的参数覆盖保存的配置，允许动态禁用/启用组件
    # 根据训练脚本的save_model方法，保存的配置包含：
    # lora_rank, lora_alpha, lora_dropout, lora_target_modules,
    # num_moe_experts, moe_hidden_dim, moe_intermediate_dim,
    # use_culture_loss, culture_loss_weight, dropout

    # 🔧 消融评估：优先使用传入的参数，实现真正的组件控制
    joint_config = JointLoRAMoEConfig(
        # LoRA配置 - 从保存的配置中读取，确保与训练时一致
        lora_rank=saved_config.get('lora_rank', lora_rank),
        lora_alpha=saved_config.get('lora_alpha', lora_alpha),
        lora_dropout=saved_config.get('lora_dropout', 0.1),
        lora_target_modules=saved_config.get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
        use_lora=True,  # 评估时总是启用LoRA

        # MoE配置 - 支持消融评估的动态配置
        num_moe_experts=saved_config.get('num_moe_experts', num_moe_experts),
        num_activated_experts=num_activated_experts,  # 这个参数不在保存的配置中，使用传入值
        moe_hidden_dim=saved_config.get('moe_hidden_dim', 4096),  # 从保存的配置读取
        moe_intermediate_dim=saved_config.get('moe_intermediate_dim', None),  # 从保存的配置读取
        moe_influence_weight=saved_config.get('moe_influence_weight', 0.1),  # 从配置读取，确保一致

        # 🔧 消融评估关键配置：使用传入参数而非保存配置，实现真正的组件控制
        use_shared=use_shared,  # 消融评估：是否启用共享专家
        use_gate=use_gate,      # 消融评估：是否启用门控网络
        use_mask=use_mask,      # 消融评估：是否启用MASK机制

        # 文化损失配置 - 使用传入参数支持消融评估
        use_culture_loss=use_culture_loss,  # 消融评估：文化损失类型
        culture_loss_weight=saved_config.get('culture_loss_weight', 0.01),

        # 其他配置 - 从保存的配置中读取
        dropout=saved_config.get('dropout', 0.1)
    )

    print(f"✅ Joint model config created (支持消融评估):")
    print(f"  - LoRA: rank={joint_config.lora_rank}, alpha={joint_config.lora_alpha}, dropout={joint_config.lora_dropout}")
    print(f"  - LoRA target modules: {joint_config.lora_target_modules}")
    print(f"  - MoE: experts={joint_config.num_moe_experts}, activated={joint_config.num_activated_experts}")
    print(f"  - MoE hidden_dim: {joint_config.moe_hidden_dim}, intermediate_dim: {joint_config.moe_intermediate_dim}")
    print(f"  - Culture loss: {joint_config.use_culture_loss}, weight: {joint_config.culture_loss_weight}")
    print(f"  - Dropout: {joint_config.dropout}")
    print(f"  🔧 消融评估配置:")
    print(f"    - 共享专家: {joint_config.use_shared}")
    print(f"    - 门控网络: {joint_config.use_gate}")
    print(f"    - MASK机制: {joint_config.use_mask}")

    # 🔧 修复：先加载LoRA权重到base_model，再创建联合模型
    # 这个顺序很重要，必须先应用LoRA，再创建JointLoRAMoEModel
    lora_path = os.path.join(joint_model_path, 'lora_weights')
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from: {lora_path}")

        # 检查LoRA权重目录的内容
        import os
        lora_files = os.listdir(lora_path)
        print(f"  - LoRA files found: {lora_files}")

        try:
            from peft import PeftModel
            # 加载LoRA权重到基础模型
            base_model = PeftModel.from_pretrained(base_model, lora_path)
            print("✅ LoRA weights loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load LoRA weights: {e}")
            print("  - Using base model without LoRA")
    else:
        print(f"⚠️ LoRA weights directory not found: {lora_path}")
        print("  - Using base model without LoRA")

    # 创建联合模型（基于已加载LoRA的base_model）
    print("Creating JointLoRAMoEModel...")
    try:
        joint_model = JointLoRAMoEModel(base_model, joint_config)
        print("✅ JointLoRAMoEModel created successfully")
    except Exception as e:
        print(f"❌ Failed to create JointLoRAMoEModel: {e}")
        raise e

    # 加载MoE权重
    moe_weights_path = os.path.join(joint_model_path, 'moe_weights.pt')
    if os.path.exists(moe_weights_path):
        print(f"Loading MoE weights from: {moe_weights_path}")
        try:
            moe_state_dict = torch.load(moe_weights_path, map_location=device)
            print(f"  - MoE state dict keys: {list(moe_state_dict.keys())}")

            # 🔧 修复：检查MoE层是否存在，并正确加载权重
            if hasattr(joint_model, 'moe_layer') and joint_model.moe_layer is not None:
                # 确保state_dict的键匹配
                missing_keys, unexpected_keys = joint_model.moe_layer.load_state_dict(moe_state_dict, strict=False)
                if missing_keys:
                    print(f"  ⚠️ Missing keys in MoE state dict: {missing_keys}")
                if unexpected_keys:
                    print(f"  ⚠️ Unexpected keys in MoE state dict: {unexpected_keys}")

                print("✅ MoE weights loaded successfully")

                # 🔧 验证MoE层参数是否正确加载
                moe_param_count = sum(p.numel() for p in joint_model.moe_layer.parameters())
                print(f"  - MoE layer parameters count: {moe_param_count:,}")
            else:
                print("❌ joint_model.moe_layer not found or is None")
                print("  - Check if JointLoRAMoEModel was created correctly")

        except Exception as e:
            print(f"❌ Failed to load MoE weights: {e}")
            print(f"  - Error details: {str(e)}")
            print("  - Using randomly initialized MoE layer")
    else:
        print(f"⚠️ MoE weights file not found: {moe_weights_path}")
        print("  - Using randomly initialized MoE layer")

    # 🔧 最终验证：检查模型结构和参数
    print("✅ Joint model loading completed")
    print(f"\n🔍 Final model verification:")
    print(f"  - Model type: {type(joint_model).__name__}")
    print(f"  - Has base_model: {hasattr(joint_model, 'base_model')}")
    print(f"  - Has moe_layer: {hasattr(joint_model, 'moe_layer')}")

    if hasattr(joint_model, 'base_model'):
        base_model_type = type(joint_model.base_model).__name__
        print(f"  - Base model type: {base_model_type}")

        # 检查是否是PEFT模型（LoRA已加载）
        if 'PeftModel' in base_model_type:
            print(f"  ✅ LoRA adapter is loaded (PeftModel detected)")
        else:
            print(f"  ⚠️ No LoRA adapter detected (base model type: {base_model_type})")

    if hasattr(joint_model, 'moe_layer') and joint_model.moe_layer is not None:
        moe_type = type(joint_model.moe_layer).__name__
        print(f"  - MoE layer type: {moe_type}")

        # 检查MoE专家数量
        if hasattr(joint_model.moe_layer, 'experts'):
            expert_count = len(joint_model.moe_layer.experts)
            print(f"  - Number of experts: {expert_count}")

        # 检查路由器
        if hasattr(joint_model.moe_layer, 'router'):
            router_type = type(joint_model.moe_layer.router).__name__
            print(f"  - Router type: {router_type}")

    # 计算总参数量
    total_params = sum(p.numel() for p in joint_model.parameters())
    trainable_params = sum(p.numel() for p in joint_model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Trainable ratio: {trainable_params/total_params:.2%}")

    return joint_model


def evaluate_joint_model(model, test_loader, tokenizer, device, rank=0):
    """
    评估联合模型

    Args:
        model: 联合模型
        test_loader: 测试数据加载器
        tokenizer: tokenizer
        device: 设备
        rank: 进程rank

    Returns:
        评估结果字典
    """
    model.eval()

    correct = 0
    total = 0
    detailed_results = []

    # 获取实际的模型（处理DDP包装）
    actual_model = model.module if isinstance(model, DDP) else model

    pbar = tqdm(test_loader, desc="Evaluating", disable=(rank != 0), mininterval=1.0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            batch_size = len(batch['instruction'])

            for i in range(batch_size):
                # 获取单个样本数据
                if isinstance(batch['instruction'], list):
                    instruction = batch['instruction'][i]
                    input_text = batch['input'][i]
                    true_output = batch['output'][i]
                    label = batch['label'][i]
                else:
                    instruction = batch['instruction']
                    input_text = batch['input']
                    true_output = batch['output']
                    label = batch['label']

                # 生成答案
                generated_text = generate_answer(
                    actual_model, tokenizer, instruction, input_text, device
                )

                # 提取答案
                predicted_answer = extract_answer_from_text(generated_text)

                # 判断正确性
                is_correct = (predicted_answer == true_output)
                if is_correct:
                    correct += 1
                total += 1

                # 保存详细结果
                detailed_results.append({
                    'sample_id': total - 1,
                    'instruction': instruction,
                    'input': input_text,
                    'true_output': true_output,
                    'label': label,
                    'generated_text': generated_text,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct
                })

                # 更新进度条
                if total % 10 == 0:
                    current_accuracy = correct / total if total > 0 else 0
                    pbar.set_postfix({
                        'accuracy': f'{current_accuracy:.4f}',
                        'correct': f'{correct}/{total}'
                    })

    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct_predictions': correct,
        'total_samples': total,
        'detailed_results': detailed_results
    }


def main():
    # 初始化分布式评估
    rank, world_size, local_rank = setup_distributed()

    parser = argparse.ArgumentParser(description="Joint CultureMoE Model Evaluation")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--joint_model_path", type=str, required=True,
                        help="Path to joint model (best_joint_model/)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to data file or 'pkl_split' for pkl file")
    parser.add_argument("--pkl_file", type=str, default="",
                        help="Path to pkl split file (for data_id=0)")
    parser.add_argument("--data_id", type=str, required=True,
                        help="Data ID (0=pkl, 1-5=full dataset)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    # 模型参数
    parser.add_argument("--backbone", type=str, default="qwen", choices=["llama", "qwen"],
                        help="Model backbone type")
    parser.add_argument("--num_moe_experts", type=int, default=4,
                        help="Number of MoE experts")
    parser.add_argument("--num_activated_experts", type=int, default=2,
                        help="Number of activated experts")
    parser.add_argument("--use_shared", type=str, default="true",
                        help="Whether to use shared expert")
    parser.add_argument("--use_gate", type=str, default="true",
                        help="Whether to use MoE gate")
    parser.add_argument("--use_culture_loss", type=str, default="csl",
                        help="Culture loss type")
    parser.add_argument("--use_mask", type=str, default="false",
                        help="Placeholder parameter")

    # 评估参数
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")

    args = parser.parse_args()

    # 转换字符串参数
    use_shared = args.use_shared.lower() == 'true'
    use_gate = args.use_gate.lower() == 'true'

    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        print("\n" + "="*80)
        print("Joint CultureMoE Model Evaluation")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Base model: {args.base_model_path}")
        print(f"Joint model: {args.joint_model_path}")
        print(f"Data ID: {args.data_id}")
        print(f"Backbone: {args.backbone}")
        print(f"MoE experts: {args.num_moe_experts}")
        print(f"Activated experts: {args.num_activated_experts}")
        print(f"Use shared: {use_shared}")
        print(f"Use gate: {use_gate}")
        print(f"Culture loss: {args.use_culture_loss}")
        print(f"Output directory: {args.output_dir}")
        print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    if is_main_process(rank):
        print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    # 🔧 修复tokenizer配置（复用训练脚本的逻辑）
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128009:
        # Llama 3.1: 尝试使用官方padding token
        official_pad_token = "<|finetune_right_pad_id|>"
        try:
            pad_token_id = tokenizer.convert_tokens_to_ids(official_pad_token)
            if pad_token_id != tokenizer.unk_token_id and pad_token_id is not None:
                tokenizer.pad_token = official_pad_token
                tokenizer.pad_token_id = pad_token_id
            else:
                raise ValueError("Official pad token not found")
        except:
            # 使用安全的低频字符
            safe_tokens = ['~', '`', '|', '^']
            for safe_token in safe_tokens:
                try:
                    safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                    if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                        tokenizer.pad_token = safe_token
                        tokenizer.pad_token_id = safe_token_id
                        break
                except:
                    continue
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    if is_main_process(rank):
        print("✅ Tokenizer loaded")

    # 加载测试数据集
    if is_main_process(rank):
        print("Loading test dataset...")

    if args.data_id == "0":
        # 使用pkl文件测试集
        if not args.pkl_file or not os.path.exists(args.pkl_file):
            raise ValueError(f"pkl file not found: {args.pkl_file}")
        test_dataset = load_test_dataset_from_pkl(args.pkl_file, tokenizer, args.max_length)
    else:
        # 使用完整数据集
        if not os.path.exists(args.data_file):
            raise ValueError(f"Data file not found: {args.data_file}")
        test_dataset = load_test_dataset_from_file(args.data_file, tokenizer, args.max_length)

    if is_main_process(rank):
        print("✅ Test dataset loaded")

    # 创建分布式采样器
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    # 创建数据加载器
    def collate_fn(batch):
        return dynamic_padding_collate_fn(batch, tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 加载联合模型
    if is_main_process(rank):
        print("Loading joint model...")

    model = load_joint_model(
        args.base_model_path,
        args.joint_model_path,
        device,
        num_moe_experts=args.num_moe_experts,
        num_activated_experts=args.num_activated_experts,
        use_shared=use_shared,
        use_gate=use_gate,
        use_mask=use_mask,  # 🔧 添加消融评估支持：MASK机制控制
        use_culture_loss=args.use_culture_loss
    )

    # 使用DDP包装模型（仅在多GPU时）
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False
        )

        if is_main_process(rank):
            print("✅ Model wrapped with DDP")

    # 开始评估
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Starting evaluation...")
        print("="*80 + "\n")

    eval_results = evaluate_joint_model(model, test_loader, tokenizer, device, rank)

    # 保存结果（只在主进程执行）
    if is_main_process(rank):
        # 保存评估结果
        results_file = os.path.join(args.output_dir, 'eval_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': eval_results['accuracy'],
                'correct_predictions': eval_results['correct_predictions'],
                'total_samples': eval_results['total_samples'],
                'model_config': {
                    'backbone': args.backbone,
                    'num_moe_experts': args.num_moe_experts,
                    'num_activated_experts': args.num_activated_experts,
                    'use_shared': use_shared,
                    'use_gate': use_gate,
                    'use_mask': use_mask,  # 🔧 添加MASK机制配置记录
                    'use_culture_loss': args.use_culture_loss
                },
                'ablation_study': {
                    'shared_expert_enabled': use_shared,
                    'gate_network_enabled': use_gate,
                    'mask_mechanism_enabled': use_mask,
                    'culture_loss_type': args.use_culture_loss,
                    'ablation_note': '消融评估：可通过参数控制组件启用/禁用'
                },
                'data_config': {
                    'data_id': args.data_id,
                    'use_pkl_split': args.data_id == "0"
                }
            }, f, indent=2, ensure_ascii=False)

        # 保存详细结果
        detailed_file = os.path.join(args.output_dir, 'detailed_results.json')
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results['detailed_results'], f, indent=2, ensure_ascii=False)

        print("\n" + "="*80)
        print("✅ Evaluation completed!")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")
        print(f"\n📊 Evaluation Summary:")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Correct predictions: {eval_results['correct_predictions']}")
        print(f"  Total samples: {eval_results['total_samples']}")
        print(f"\n🔧 消融评估配置:")
        print(f"  共享专家: {use_shared}")
        print(f"  门控网络: {use_gate}")
        print(f"  MASK机制: {use_mask}")
        print(f"  文化损失: {args.use_culture_loss}")
        print(f"\nFiles generated:")
        print(f"  - eval_results.json (Summary results)")
        print(f"  - detailed_results.json (Detailed predictions)")
        print("="*80)

        # 显示前几个预测示例
        print("\n📋 Sample predictions:")
        for i in range(min(3, len(eval_results['detailed_results']))):
            result = eval_results['detailed_results'][i]
            correct_mark = '✅' if result['is_correct'] else '❌'
            print(f"  Sample {i+1}: True={result['true_output']}, Pred={result['predicted_answer']}, Generated='{result['generated_text'][:20]}...' {correct_mark}")

    # 清理分布式评估
    cleanup_distributed()


if __name__ == "__main__":
    main()