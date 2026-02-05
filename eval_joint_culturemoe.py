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


def parse_soft_answers(output_str: str) -> list:
    """
    解析可能包含多个答案的output字符串（用于soft accuracy）

    例如:
        "1, 2" -> ["1", "2"]
        "3, 4, 5" -> ["3", "4", "5"]
        "1" -> ["1"]

    Args:
        output_str: 输出字符串，可能包含逗号分隔的多个答案

    Returns:
        答案列表
    """
    if not output_str:
        return []

    # 分割并清理空格
    answers = [ans.strip() for ans in str(output_str).split(',') if ans.strip()]
    return answers


def generate_answer_with_shared_control(model, tokenizer, instruction: str, input_text: str,
                                      instruction_mask: str = None, device: str = 'cuda',
                                      use_mask: bool = True, use_shared: bool = None,
                                      use_moe: bool = None, use_gate: bool = None,
                                      max_new_tokens: int = 5) -> str:
    """
    支持推理时共享专家控制的生成答案函数

    Args:
        model: 联合训练模型
        tokenizer: tokenizer
        instruction: 指令
        input_text: 输入文本
        instruction_mask: MASK版本指令（可选）
        device: 设备
        use_mask: 是否启用MASK机制
        use_shared: 推理时是否使用共享专家（None使用训练配置，True/False覆盖配置进行消融研究）
        use_moe: 推理时是否使用MoE结构（None使用训练配置，False=仅使用shared专家，跳过router+路由专家）
        use_gate: 推理时是否使用门控网络（None使用训练配置，True/False覆盖配置进行消融研究）
        max_new_tokens: 最大生成token数

    Returns:
        生成的文本
    """
    # 构建输入 - 与训练时格式保持一致
    if input_text:
        full_input = f"{instruction}\n{input_text}"
        if instruction_mask and use_mask:
            full_input_mask = f"{instruction_mask}\n{input_text}"
        else:
            full_input_mask = full_input
    else:
        full_input = instruction
        full_input_mask = instruction_mask if instruction_mask and use_mask else instruction

    # 🔧 关键：保持与训练时完全一致的格式
    full_input = full_input.rstrip()
    full_input_mask = full_input_mask.rstrip()

    # Tokenize输入
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512, padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 🔧 MASK机制：如果启用且有instruction_mask，同时tokenize mask版本
    inputs_mask = None
    if use_mask and instruction_mask and instruction_mask != instruction:
        inputs_mask = tokenizer(full_input_mask, return_tensors="pt", truncation=True, max_length=512, padding=False)
        inputs_mask = {k: v.to(device) for k, v in inputs_mask.items()}

    # 确保attention_mask存在
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    if inputs_mask and 'attention_mask' not in inputs_mask:
        inputs_mask['attention_mask'] = torch.ones_like(inputs_mask['input_ids'])

    with torch.no_grad():
        # 🔧 关键修改：调用模型的generate方法，传递use_shared参数
        # 检查模型是否支持use_shared参数
        try:
            # 尝试使用联合训练模型的generate方法，传递MASK机制和共享专家控制参数
            if hasattr(model, 'generate') and hasattr(model, 'moe_layer'):
                # 这是联合训练模型，支持完整的参数控制
                generate_kwargs = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask'),
                    'max_new_tokens': max_new_tokens,
                    'min_new_tokens': 1,
                    'pad_token_id': tokenizer.pad_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'do_sample': False,
                    'num_beams': 1,
                    'early_stopping': True,
                    'repetition_penalty': 1.0
                }

                # 🔧 如果启用MASK机制且有mask版本输入，传递双路输入
                if use_mask and inputs_mask is not None:
                    generate_kwargs.update({
                        'input_ids_mask': inputs_mask['input_ids'],
                        'attention_mask_mask': inputs_mask['attention_mask']
                    })

                # 🔧 关键：传递推理时消融控制参数
                if use_shared is not None:
                    generate_kwargs['use_shared'] = use_shared
                if use_moe is not None:
                    generate_kwargs['use_moe'] = use_moe
                if use_gate is not None:
                    generate_kwargs['use_gate'] = use_gate

                outputs = model.generate(**generate_kwargs)

            else:
                # 回退到标准generate方法
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    num_beams=1,
                    early_stopping=True
                )

        except Exception as e:
            print(f"⚠️ Generate with shared control failed: {e}")
            # 回退到标准generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
                early_stopping=True
            )

    # 解码生成的部分
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]

    if len(generated_ids) > 0:
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    else:
        generated_text = ""

    return generated_text


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


def load_test_dataset_from_pkl(pkl_file_path: str, tokenizer, max_length: int = 512, use_mask: bool = True):
    """
    从pkl文件加载测试集

    Args:
        pkl_file_path: pkl文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度
        use_mask: 是否启用MASK机制（推理时消融研究）

    Returns:
        测试数据集和数据集信息
    """
    print(f"Loading test dataset from pkl file: {pkl_file_path}")

    with open(pkl_file_path, 'rb') as f:
        split_info = pickle.load(f)

    # 获取原始数据路径和测试集索引
    original_data_path = split_info['data_path']
    test_indices = split_info['test_indices']
    train_indices = split_info.get('train_indices', [])
    val_indices = split_info.get('val_indices', [])

    # 🔧 新增：获取数据集信息，用于多数据集模式的标识
    dataset_index = split_info.get('dataset_index', None)
    dataset_name = split_info.get('dataset_name', os.path.basename(original_data_path))

    print(f"Original data path: {original_data_path}")
    if dataset_index is not None:
        print(f"Dataset index: {dataset_index} ({dataset_name})")
    print(f"Train set size: {len(train_indices)} ({len(train_indices)/len(train_indices + val_indices + test_indices)*100:.1f}%)")
    print(f"Validation set size: {len(val_indices)} ({len(val_indices)/len(train_indices + val_indices + test_indices)*100:.1f}%)")
    print(f"Test set size: {len(test_indices)} ({len(test_indices)/len(train_indices + val_indices + test_indices)*100:.1f}%)")

    # 🔧 新增：检查数据集划分的一致性
    total_samples = len(train_indices) + len(val_indices) + len(test_indices)
    print(f"Total samples: {total_samples}")
    print(f"Split ratio: {len(train_indices)/total_samples:.1f}:{len(val_indices)/total_samples:.1f}:{len(test_indices)/total_samples:.1f}")

    # 创建完整数据集，传递use_mask参数控制MASK机制
    full_dataset = CultureLLMNewFormatDataset(original_data_path, tokenizer, max_length, use_mask_inference=use_mask)

    # 🔧 调试：检查数据集中是否有country字段
    if len(full_dataset) > 0:
        sample_item = full_dataset[0]
        has_country = 'country' in sample_item
        print(f"🔍 数据集country字段检查: {has_country}")
        if has_country:
            print(f"    第一个样本的country: {sample_item.get('country', 'None')}")

    # 🔧 验证数据集大小一致性
    if len(full_dataset) != total_samples:
        print(f"⚠️ Warning: Full dataset size ({len(full_dataset)}) != split total ({total_samples})")

    # 创建测试集子集
    test_dataset = Subset(full_dataset, test_indices)

    # 🔧 返回数据集和元信息，用于多数据集模式
    return test_dataset, {
        'dataset_index': dataset_index,
        'dataset_name': dataset_name,
        'original_data_path': original_data_path,
        'test_size': len(test_indices)
    }


def load_test_dataset_from_file(data_file_path: str, tokenizer, max_length: int = 512, use_mask: bool = True):
    """
    从完整数据文件加载测试集（使用全部数据）

    Args:
        data_file_path: 数据文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度
        use_mask: 是否启用MASK机制（推理时消融研究）

    Returns:
        测试数据集
    """
    print(f"Loading test dataset from full data file: {data_file_path}")

    test_dataset = CultureLLMNewFormatDataset(data_file_path, tokenizer, max_length, use_mask_inference=use_mask)

    print(f"Test set size: {len(test_dataset)}")

    # 🔧 调试：检查测试数据集中是否有country字段
    if len(test_dataset) > 0:
        sample_item = test_dataset[0]
        has_country = 'country' in sample_item
        print(f"🔍 测试数据集country字段检查: {has_country}")
        if has_country:
            print(f"    第一个样本的country: {sample_item.get('country', 'None')}")

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

    # 🔧 修复：创建联合模型配置，只使用JointLoRAMoEConfig支持的参数
    # JointLoRAMoEConfig不包含use_shared和use_gate参数，这些将在模型创建后处理
    joint_config = JointLoRAMoEConfig(
        # LoRA配置 - 从保存的配置中读取，确保与训练时一致
        lora_rank=saved_config.get('lora_rank', lora_rank),
        lora_alpha=saved_config.get('lora_alpha', lora_alpha),
        lora_dropout=saved_config.get('lora_dropout', 0.1),
        lora_target_modules=saved_config.get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
        use_lora=True,  # 评估时总是启用LoRA

        # MoE配置 - 从保存的配置中读取，确保与训练时一致
        num_moe_experts=saved_config.get('num_moe_experts', num_moe_experts),
        num_activated_experts=num_activated_experts,  # 这个参数不在保存的配置中，使用传入值
        moe_hidden_dim=saved_config.get('moe_hidden_dim', 4096),  # 从保存的配置读取
        moe_intermediate_dim=saved_config.get('moe_intermediate_dim', None),  # 从保存的配置读取
        moe_influence_weight=saved_config.get('moe_influence_weight', 0.1),  # 从配置读取，确保一致

        # MASK机制配置 - 支持消融评估
        use_mask=use_mask,  # 消融评估：是否启用MASK机制

        # 文化损失配置 - 使用传入参数支持消融评估
        use_culture_loss=use_culture_loss,  # 消融评估：文化损失类型
        culture_loss_weight=saved_config.get('culture_loss_weight', 0.01),

        # 其他配置 - 从保存的配置中读取
        dropout=saved_config.get('dropout', 0.1)
    )

    # 🔧 消融评估参数记录（仅记录当前支持的参数）
    ablation_config = {
        'use_shared': use_shared,  # 记录参数但当前模型架构可能不支持动态切换
        'use_gate': use_gate,      # 记录参数但当前模型架构可能不支持动态切换
        'use_mask': use_mask       # 实际支持的消融评估：MASK机制
    }

    print(f"✅ Joint model config created (支持消融评估):")
    print(f"  - LoRA: rank={joint_config.lora_rank}, alpha={joint_config.lora_alpha}, dropout={joint_config.lora_dropout}")
    print(f"  - LoRA target modules: {joint_config.lora_target_modules}")
    print(f"  - MoE: experts={joint_config.num_moe_experts}, activated={joint_config.num_activated_experts}")
    print(f"  - MoE hidden_dim: {joint_config.moe_hidden_dim}, intermediate_dim: {joint_config.moe_intermediate_dim}")
    print(f"  - Culture loss: {joint_config.use_culture_loss}, weight: {joint_config.culture_loss_weight}")
    print(f"  - Dropout: {joint_config.dropout}")
    print(f"  🔧 消融评估配置:")
    print(f"    - 共享专家: {ablation_config['use_shared']} (参数已记录)")
    print(f"    - 门控网络: {ablation_config['use_gate']} (参数已记录)")
    print(f"    - MASK机制: {ablation_config['use_mask']} (✅ 实际支持)")
    if not use_shared or not use_gate:
        print(f"    ⚠️ 注意: use_shared和use_gate的消融评估可能需要模型架构的进一步支持")

    # 🔧 修复：根据配置决定LoRA加载策略，避免重复应用
    lora_path = os.path.join(joint_model_path, 'lora_weights')
    lora_loaded_externally = False

    if os.path.exists(lora_path):
        print(f"Found LoRA weights at: {lora_path}")
        lora_files = os.listdir(lora_path)
        print(f"  - LoRA files found: {lora_files}")

        # 检查是否应该外部加载LoRA（推荐方式）
        # 外部加载可以确保与训练时的加载顺序完全一致
        try:
            from peft import PeftModel
            print("🔧 Loading LoRA externally (before JointLoRAMoEModel creation)")
            base_model = PeftModel.from_pretrained(base_model, lora_path)
            lora_loaded_externally = True
            print("✅ LoRA weights loaded successfully")

            # 🔧 关键：既然已经外部加载LoRA，需要告诉JointLoRAMoEModel不要再次应用
            joint_config.use_lora = False  # 避免重复应用LoRA
            print("🔧 Set joint_config.use_lora=False to avoid duplicate LoRA application")

        except Exception as e:
            print(f"❌ Failed to load LoRA externally: {e}")
            print("  - Will let JointLoRAMoEModel handle LoRA loading internally")
            lora_loaded_externally = False
            # 保持joint_config.use_lora=True，让JointLoRAMoEModel内部处理
    else:
        print(f"⚠️ LoRA weights directory not found: {lora_path}")
        print("  - Will proceed without LoRA or let JointLoRAMoEModel handle it")

    # 创建联合模型
    print("Creating JointLoRAMoEModel...")
    try:
        joint_model = JointLoRAMoEModel(base_model, joint_config)
        print("✅ JointLoRAMoEModel created successfully")

        # 🔧 验证LoRA加载状态
        if lora_loaded_externally:
            print("🔧 LoRA was loaded externally before model creation")
        elif joint_config.use_lora:
            print("🔧 LoRA will be applied internally by JointLoRAMoEModel")
        else:
            print("🔧 No LoRA will be applied (base model only)")

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
            # 🔧 新增：验证LoRA适配器的具体信息
            if hasattr(joint_model.base_model, 'peft_config'):
                peft_config = joint_model.base_model.peft_config
                print(f"    - LoRA config keys: {list(peft_config.keys())}")
                if 'default' in peft_config:
                    config = peft_config['default']
                    print(f"    - LoRA rank: {getattr(config, 'r', 'unknown')}")
                    print(f"    - LoRA alpha: {getattr(config, 'lora_alpha', 'unknown')}")
                    print(f"    - Target modules: {getattr(config, 'target_modules', 'unknown')}")
        else:
            print(f"  ❌ CRITICAL: No LoRA adapter detected! (base model type: {base_model_type})")
            print(f"    - This will cause significant performance degradation!")

    if hasattr(joint_model, 'moe_layer') and joint_model.moe_layer is not None:
        moe_type = type(joint_model.moe_layer).__name__
        print(f"  - MoE layer type: {moe_type}")

        # 检查MoE专家数量
        if hasattr(joint_model.moe_layer, 'experts'):
            expert_count = len(joint_model.moe_layer.experts)
            print(f"  - Number of experts: {expert_count}")

            # 🔧 新增：验证MoE专家的参数状态
            expert_param_count = sum(p.numel() for p in joint_model.moe_layer.experts.parameters())
            print(f"  - MoE experts parameters: {expert_param_count:,}")

            # 🔧 修复：检查第一个专家的完整线性层参数（而非LoRA参数）
            # JointLoRAMoEModel使用完整的线性层专家，不是LoRA结构
            if expert_count > 0:
                first_expert = joint_model.moe_layer.experts[0]
                if hasattr(first_expert, 'gate_proj') and hasattr(first_expert, 'up_proj') and hasattr(first_expert, 'down_proj'):
                    print(f"  ✅ MoE experts have complete linear layer structure")
                    # 检查参数是否为零（可能表示加载失败）
                    gate_proj_norm = first_expert.gate_proj.weight.norm().item()
                    up_proj_norm = first_expert.up_proj.weight.norm().item()
                    down_proj_norm = first_expert.down_proj.weight.norm().item()
                    print(f"    - First expert gate_proj norm: {gate_proj_norm:.6f}")
                    print(f"    - First expert up_proj norm: {up_proj_norm:.6f}")
                    print(f"    - First expert down_proj norm: {down_proj_norm:.6f}")
                    if gate_proj_norm < 1e-6 or up_proj_norm < 1e-6 or down_proj_norm < 1e-6:
                        print(f"    ⚠️ Warning: Some MoE expert parameters seem to be zero-initialized")
                elif hasattr(first_expert, 'gate_lora_A'):
                    # 这是旧的LoRA专家结构，不应该出现在JointLoRAMoEModel中
                    print(f"  ❌ CRITICAL: Found LoRA expert structure, expected complete linear layers!")
                    print(f"    - This indicates architecture mismatch between training and evaluation")
                else:
                    print(f"  ❌ CRITICAL: MoE experts missing expected linear layer structure!")
                    print(f"    - Expected: gate_proj, up_proj, down_proj")
                    print(f"    - Found attributes: {[attr for attr in dir(first_expert) if not attr.startswith('_')]}")

        # 检查路由器
        if hasattr(joint_model.moe_layer, 'router'):
            router_type = type(joint_model.moe_layer.router).__name__
            print(f"  - Router type: {router_type}")

            # 🔧 新增：检查路由器参数
            router_param_count = sum(p.numel() for p in joint_model.moe_layer.router.parameters())
            print(f"  - Router parameters: {router_param_count:,}")

            # 检查路由器权重
            if hasattr(joint_model.moe_layer.router, 'linear'):
                router_weight_norm = joint_model.moe_layer.router.linear.weight.norm().item()
                print(f"  - Router weight norm: {router_weight_norm:.6f}")
                if router_weight_norm < 1e-6:
                    print(f"    ⚠️ Warning: Router parameters seem to be zero-initialized")
    else:
        print(f"  ❌ CRITICAL: No MoE layer found! Model will behave like base LoRA only!")

    # 计算总参数量
    total_params = sum(p.numel() for p in joint_model.parameters())
    trainable_params = sum(p.numel() for p in joint_model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Trainable ratio: {trainable_params/total_params:.2%}")

    # 🔧 修复：模型完整性检查，验证JointLoRAMoEModel架构
    print(f"\n🔧 Model integrity check:")
    base_lora_ok = hasattr(joint_model, 'base_model') and 'PeftModel' in type(joint_model.base_model).__name__
    moe_layer_ok = hasattr(joint_model, 'moe_layer') and joint_model.moe_layer is not None

    if base_lora_ok and moe_layer_ok:
        # 进一步检查MoE专家是否为完整线性层结构
        moe_experts_ok = False
        if hasattr(joint_model.moe_layer, 'experts') and len(joint_model.moe_layer.experts) > 0:
            first_expert = joint_model.moe_layer.experts[0]
            moe_experts_ok = (hasattr(first_expert, 'gate_proj') and
                            hasattr(first_expert, 'up_proj') and
                            hasattr(first_expert, 'down_proj'))

        if moe_experts_ok:
            print(f"  ✅ JointLoRAMoEModel appears to be correctly loaded (LoRA + Complete Linear MoE)")
        else:
            print(f"  ❌ CRITICAL: MoE experts have wrong structure - expected complete linear layers!")
            print(f"    - This will cause severe performance degradation!")
    elif base_lora_ok:
        print(f"  ❌ Model is missing MoE layer - will perform like LoRA-only!")
    elif moe_layer_ok:
        print(f"  ❌ Model is missing LoRA adapter - will perform like base model!")
    else:
        print(f"  ❌ Model is missing both LoRA adapter and MoE layer!")

    return joint_model


def evaluate_joint_model(model, test_loader, tokenizer, device, rank=0, use_mask=True, use_shared=None, use_moe=None, use_gate=None, group_by_country=False, data_id=""):
    """
    🔧 修复版本：评估联合模型，支持MASK机制双路输入和推理时消融控制

    Args:
        model: 联合模型
        test_loader: 测试数据加载器
        tokenizer: tokenizer
        device: 设备
        rank: 进程rank
        use_mask: 是否使用MASK机制
        use_shared: 是否使用共享专家
        use_moe: 是否使用MoE结构
        use_gate: 是否使用门控网络
        group_by_country: 是否按country分组统计
        data_id: 数据集ID，用于决定是否使用soft accuracy（40使用soft accuracy）
        tokenizer: tokenizer
        device: 设备
        rank: 进程rank
        use_mask: 是否启用MASK机制
        use_shared: 推理时是否使用共享专家（None使用训练配置，True/False覆盖配置进行消融研究）
        use_gate: 推理时是否使用门控网络（None使用训练配置，True/False覆盖配置进行消融研究）
        group_by_country: 是否按country分组统计结果（用于blend数据集）

    Returns:
        评估结果字典
    """
    model.eval()

    # 🔧 关键修复：设置消融配置，确保参数真正影响模型计算
    actual_model = model.module if isinstance(model, DDP) else model
    if hasattr(actual_model, 'set_ablation_config'):
        actual_model.set_ablation_config(use_shared=use_shared, use_gate=use_gate, use_mask=use_mask)
        if rank == 0:
            print(f"🔧 消融配置已设置: use_shared={use_shared}, use_gate={use_gate}, use_mask={use_mask}")
    else:
        if rank == 0:
            print("⚠️  模型不支持消融配置，使用默认行为")

    # 🔧 调试：检查country分组状态
    if rank == 0 and group_by_country:
        print(f"🌍 Country分组统计已启用，将收集country字段统计信息")

    # 🔧 新增：判断是否使用soft accuracy（DATA_ID=40）
    use_soft_accuracy = (data_id == "40")
    if rank == 0 and use_soft_accuracy:
        print(f"🎯 Accuracy mode: Soft accuracy (multiple correct answers allowed)")

    correct = 0
    total = 0
    detailed_results = []

    # 🔧 新增：country分组统计
    country_stats = {}  # {country: {'correct': 0, 'total': 0, 'accuracy': 0.0}}

    pbar = tqdm(test_loader, desc="Evaluating", disable=(rank != 0), mininterval=1.0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            batch_size = len(batch['instruction'])

            # 🔧 调试：检查前几个batch的country字段
            if batch_idx < 2 and rank == 0 and group_by_country:
                print(f"🔍 Batch {batch_idx}: batch_size={batch_size}")
                print(f"    batch keys: {list(batch.keys())}")
                if 'country' in batch:
                    print(f"    batch['country'] type: {type(batch['country'])}")
                    print(f"    batch['country'] length: {len(batch['country']) if hasattr(batch['country'], '__len__') else 'N/A'}")
                    if hasattr(batch['country'], '__len__') and len(batch['country']) > 0:
                        print(f"    first few countries: {batch['country'][:min(3, len(batch['country']))]}")

            for i in range(batch_size):
                # 获取单个样本数据
                if isinstance(batch['instruction'], list):
                    instruction = batch['instruction'][i]
                    input_text = batch['input'][i]
                    true_output = batch['output'][i]
                    label = batch['label'][i]
                    # 🔧 新增：获取country字段（用于blend数据集分组统计）
                    country = batch.get('country', [None] * batch_size)[i] if 'country' in batch else None
                else:
                    instruction = batch['instruction']
                    input_text = batch['input']
                    true_output = batch['output']
                    label = batch['label']
                    # 🔧 新增：获取country字段（用于blend数据集分组统计）
                    country = batch.get('country', None) if 'country' in batch else None

                # 🔧 修复：支持MASK机制的生成答案
                # 从batch中获取instruction_mask（如果使用修复版数据集）
                instruction_mask = None
                if hasattr(batch, 'get') and 'instruction_mask' in batch:
                    instruction_mask = batch['instruction_mask'][i] if isinstance(batch['instruction_mask'], list) else batch['instruction_mask']
                elif hasattr(batch, 'instruction_mask'):
                    instruction_mask = batch.instruction_mask[i] if hasattr(batch.instruction_mask, '__getitem__') else batch.instruction_mask

                # 🔧 修复：支持推理时共享专家控制的生成答案
                # 使用改进的generate_answer函数，支持use_shared和use_moe参数
                generated_text = generate_answer_with_shared_control(
                    actual_model, tokenizer, instruction, input_text,
                    instruction_mask=instruction_mask, device=device,
                    use_mask=use_mask, use_shared=use_shared,
                    use_moe=use_moe, use_gate=use_gate
                )

                # 提取答案
                predicted_answer = extract_answer_from_text(generated_text)

                # 判断正确性
                if use_soft_accuracy:
                    # Soft accuracy: 预测答案在可能的正确答案集合中即可
                    possible_answers = parse_soft_answers(true_output)
                    is_correct = (predicted_answer in possible_answers)
                else:
                    # Exact accuracy: 精确匹配
                    is_correct = (predicted_answer == true_output)

                # 🔧 新增：前10个样本的详细调试输出
                if total < 10 and rank == 0:
                    print(f"\n📋 Sample {total + 1} debug:")
                    print(f"  Instruction: {instruction[:100]}...")
                    print(f"  Input: {input_text[:50]}...")
                    print(f"  True output: '{true_output}'")
                    if use_soft_accuracy:
                        print(f"  Possible answers: {parse_soft_answers(true_output)}")
                    print(f"  Generated text: '{generated_text}'")
                    print(f"  Predicted answer: '{predicted_answer}'")
                    print(f"  Correct: {is_correct}")
                if is_correct:
                    correct += 1
                total += 1

                # 🔧 新增：更新country分组统计
                if group_by_country:
                    # 🔧 调试：检查前几个样本的country字段
                    if total <= 3 and rank == 0:
                        print(f"🔍 Sample {total}: country='{country}', batch has country field: {'country' in batch}")
                        if 'country' in batch:
                            print(f"    batch['country'] type: {type(batch['country'])}, content: {batch['country'][:3] if hasattr(batch['country'], '__len__') and len(batch['country']) > 3 else batch['country']}")

                    if country is not None:
                        if country not in country_stats:
                            country_stats[country] = {'correct': 0, 'total': 0, 'accuracy': 0.0}

                        country_stats[country]['total'] += 1
                        if is_correct:
                            country_stats[country]['correct'] += 1
                        country_stats[country]['accuracy'] = country_stats[country]['correct'] / country_stats[country]['total']

                # 保存详细结果
                result_item = {
                    'sample_id': total - 1,
                    'instruction': instruction,
                    'input': input_text,
                    'true_output': true_output,
                    'label': label,
                    'generated_text': generated_text,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct
                }
                # 🔧 新增：如果有country字段，也保存到结果中
                if country is not None:
                    result_item['country'] = country
                detailed_results.append(result_item)

                # 更新进度条 - 降低更新频率
                if total % 50 == 0:  # 每50个样本更新一次
                    current_accuracy = correct / total if total > 0 else 0
                    pbar.set_postfix({
                        'accuracy': f'{current_accuracy:.4f}',
                        'correct': f'{correct}/{total}'
                    })

    accuracy = correct / total if total > 0 else 0

    # 🔧 新增：准备返回结果
    result = {
        'accuracy': accuracy,
        'correct_predictions': correct,
        'total_samples': total,
        'detailed_results': detailed_results
    }

    # 🔧 新增：如果启用了country分组统计，添加分组结果
    if group_by_country and country_stats:
        result['country_stats'] = country_stats

    return result


def quick_model_test(model, tokenizer, device):
    """
    快速测试模型的基本功能
    """
    print(f"\n🧪 Quick model functionality test:")

    # 简单的测试样本
    test_instruction = "Give me the answer from 1 to 4: What is 1+1?"
    test_input = "This is a simple math question."

    try:
        generated_text = generate_answer(model, tokenizer, test_instruction, test_input, device, max_new_tokens=10)
        predicted_answer = extract_answer_from_text(generated_text)

        print(f"  Test input: '{test_instruction}'")
        print(f"  Generated: '{generated_text}'")
        print(f"  Extracted: '{predicted_answer}'")

        if generated_text.strip():
            print(f"  ✅ Model can generate text")
        else:
            print(f"  ❌ Model generates empty text!")

        if predicted_answer in ['1', '2', '3', '4']:
            print(f"  ✅ Model generates valid answers")
        else:
            print(f"  ⚠️ Model doesn't generate expected answer format")

    except Exception as e:
        print(f"  ❌ Model test failed: {e}")


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
    parser.add_argument("--use_moe", type=str, default="true",
                        help="Whether to use MoE structure (router+routing experts). "
                             "false=only use shared expert (ablation study)")
    parser.add_argument("--use_gate", type=str, default="true",
                        help="Whether to use MoE gate network for inference-time ablation study")
    parser.add_argument("--use_culture_loss", type=str, default="csl",
                        help="Culture loss type")
    parser.add_argument("--use_mask", type=str, default="true",
                        help="Whether to use MASK mechanism for ablation study")

    # 评估参数
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")

    args = parser.parse_args()

    # 转换字符串参数
    use_shared = args.use_shared.lower() == 'true'
    use_moe = args.use_moe.lower() == 'true'  # 🔧 添加use_moe参数转换
    use_gate = args.use_gate.lower() == 'true'
    use_mask = args.use_mask.lower() == 'true'  # 🔧 添加use_mask参数转换

    # 🔧 消融控制参数说明：
    # - use_shared: 推理时是否使用共享专家
    # - use_moe: 推理时是否使用MoE结构（router+路由专家）
    # - use_gate: 推理时是否使用门控网络
    # None时使用训练配置，True/False时覆盖配置进行消融研究
    use_shared_for_inference = use_shared  # 直接使用USE_SHARED参数进行推理时控制
    use_moe_for_inference = use_moe  # 直接使用USE_MOE参数进行推理时控制
    use_gate_for_inference = use_gate  # 直接使用USE_GATE参数进行推理时控制

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
        print(f"Use MoE: {use_moe}")  # 🔧 添加use_moe参数显示
        print(f"Use gate: {use_gate}")
        print(f"Use mask: {use_mask}")  # 🔧 添加use_mask参数显示
        print(f"Use shared inference: {use_shared} (消融研究: 推理时是否使用共享专家)")  # 🔧 使用USE_SHARED参数控制推理
        print(f"Use MoE inference: {use_moe} (消融研究: 推理时是否使用MoE结构router+路由专家)")  # 🔧 使用USE_MOE参数控制推理
        print(f"Use gate inference: {use_gate} (消融研究: 推理时是否使用门控网络)")  # 🔧 使用USE_GATE参数控制推理
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
        print(f"🔧 MASK机制状态: {'启用' if use_mask else '禁用'} (推理时消融研究)")

    if args.data_id == "0":
        # 使用pkl文件测试集
        if not args.pkl_file or not os.path.exists(args.pkl_file):
            raise ValueError(f"pkl file not found: {args.pkl_file}")
        test_dataset, dataset_metadata = load_test_dataset_from_pkl(args.pkl_file, tokenizer, args.max_length, use_mask=use_mask)

        if is_main_process(rank):
            if dataset_metadata.get('dataset_index') is not None:
                print(f"✅ 加载数据集 {dataset_metadata['dataset_index']}: {dataset_metadata['dataset_name']}")
                print(f"  - 测试集大小: {dataset_metadata['test_size']} 样本")
            else:
                print(f"✅ 加载单数据集测试集: {dataset_metadata['test_size']} 样本")
    else:
        # 使用完整数据集
        if not os.path.exists(args.data_file):
            raise ValueError(f"Data file not found: {args.data_file}")
        test_dataset = load_test_dataset_from_file(args.data_file, tokenizer, args.max_length, use_mask=use_mask)

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

    # 🔧 新增：快速模型功能测试
    if is_main_process(rank):
        actual_model = model.module if isinstance(model, DDP) else model
        quick_model_test(actual_model, tokenizer, device)

    # 开始评估
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Starting evaluation...")
        print("="*80 + "\n")

    # 🔧 新增：检查是否需要按country分组统计（DATA_ID=16的blend数据集）
    group_by_country = (args.data_id == "16")

    if group_by_country:
        print(f"🌍 启用country分组统计 (DATA_ID={args.data_id})")
    else:
        print(f"📊 使用标准评估模式 (DATA_ID={args.data_id})")

    eval_results = evaluate_joint_model(model, test_loader, tokenizer, device, rank, use_mask=use_mask, use_shared=use_shared_for_inference, use_moe=use_moe_for_inference, use_gate=use_gate_for_inference, group_by_country=group_by_country, data_id=args.data_id)

    # 保存结果（只在主进程执行）
    if is_main_process(rank):
        # 保存评估结果
        results_file = os.path.join(args.output_dir, 'eval_results.json')

        # 🔧 新增：准备保存的结果数据
        save_data = {
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
                'moe_structure_enabled': use_moe,
                'gate_network_enabled': use_gate,
                'mask_mechanism_enabled': use_mask,
                'culture_loss_type': args.use_culture_loss,
                'shared_expert_inference_actual': use_shared_for_inference,
                'moe_structure_inference_actual': use_moe_for_inference,
                'gate_network_inference_actual': use_gate_for_inference,
                'ablation_note': '消融评估：可通过USE_SHARED、USE_MOE和USE_GATE参数控制推理时是否使用共享专家、MoE结构和门控网络'
            },
            'data_config': {
                'data_id': args.data_id,
                'use_pkl_split': args.data_id == "0",
                'group_by_country': group_by_country
            }
        }

        # 🔧 新增：如果有country分组统计，添加到结果中
        if 'country_stats' in eval_results:
            save_data['country_stats'] = eval_results['country_stats']

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

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

        # 🔧 新增：显示country分组统计结果
        if 'country_stats' in eval_results:
            print(f"\n🌍 Country-wise Statistics (DATA_ID=16 blend dataset):")
            country_stats = eval_results['country_stats']
            for country, stats in sorted(country_stats.items()):
                print(f"  {country}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.4f})")

        print(f"\n🔧 消融评估配置:")
        print(f"  共享专家: {use_shared}")
        print(f"  门控网络: {use_gate}")
        print(f"  MASK机制: {use_mask}")
        print(f"  文化损失: {args.use_culture_loss}")
        if group_by_country:
            print(f"  Country分组: 启用 (DATA_ID={args.data_id})")
        print(f"\nFiles generated:")
        print(f"  - eval_results.json (Summary results)")
        print(f"  - detailed_results.json (Detailed predictions)")
        print("="*80)

        # 统计生成问题
        detailed_results = eval_results['detailed_results']
        empty_generations = sum(1 for r in detailed_results if not r['generated_text'].strip())
        non_digit_generations = sum(1 for r in detailed_results if not r['predicted_answer'])

        print(f"\n📊 生成质量统计:")
        print(f"  空生成: {empty_generations}")
        print(f"  未提取到数字: {non_digit_generations}")

        # 只在存在生成问题时显示问题样本
        if empty_generations > 0 or non_digit_generations > 0:
            print(f"\n⚠️ 发现生成问题，显示前3个有问题的样本:")
            problem_count = 0
            for i, result in enumerate(detailed_results):
                if problem_count >= 3:
                    break
                if (not result['generated_text'].strip() or
                    not result['predicted_answer'] or
                    result['predicted_answer'] not in ['1', '2', '3', '4']):
                    print(f"  问题样本 {i+1}: True={result['true_output']}, Pred='{result['predicted_answer']}', Generated='{result['generated_text'][:30]}...'")
                    problem_count += 1
        else:
            print("✅ 所有样本都成功生成了有效的数字答案")
            # 只显示少数几个正确的示例
            print(f"\n📋 随机正确样本 (前3个):")
            for i in range(min(3, len(detailed_results))):
                result = detailed_results[i]
                if result['is_correct']:
                    print(f"  样本 {i+1}: True={result['true_output']}, Pred={result['predicted_answer']} ✅")

    # 清理分布式评估
    cleanup_distributed()


if __name__ == "__main__":
    main()