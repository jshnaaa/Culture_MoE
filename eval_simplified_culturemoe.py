#!/usr/bin/env python3
"""
简化版CultureMoE推理评估脚本 - 支持消融实验

使用方法：
    # 完整模型评估
    python eval_simplified_culturemoe.py \
        --model_path /path/to/best_simplified_culturemoe \
        --data_file /path/to/data.json \
        --output_dir /path/to/results

    # 消融实验：关闭shared专家
    python eval_simplified_culturemoe.py \
        --model_path /path/to/best_simplified_culturemoe \
        --data_file /path/to/data.json \
        --output_dir /path/to/results \
        --disable_shared \
        --experiment_name "no_shared"

    # 消融实验：关闭gate网络
    python eval_simplified_culturemoe.py \
        --model_path /path/to/best_simplified_culturemoe \
        --data_file /path/to/data.json \
        --output_dir /path/to/results \
        --disable_gate \
        --experiment_name "no_gate"
"""

import argparse
import json
import os
import sys
import pickle
from typing import Dict, List, Optional
import numpy as np
from torch.utils.data import Subset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.simplified_culturemoe import SimplifiedCultureMoEConfig
from src.llamafactory.model.simplified_culturemoe_adapter import SimplifiedCultureMoEAdapter, MoEFFNLoRA
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    load_and_process_data,
    extract_answer_from_text,
    generate_answer,
    dynamic_padding_collate_fn
)


def load_fixed_test_split(data_file: str, tokenizer, max_length: int,
                          split_file: str, split_type: str = 'test'):
    """
    加载固定的数据集划分（测试集）

    Args:
        data_file: 数据文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度
        split_file: 划分文件路径
        split_type: 划分类型 ('test', 'validation')

    Returns:
        测试集数据集对象
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"数据划分文件不存在: {split_file}")

    # 加载划分信息
    with open(split_file, 'rb') as f:
        split_info = pickle.load(f)

    print(f"✅ 加载数据划分信息: {split_file}")
    print(f"  - 划分方法: {split_info.get('split_method', 'unknown')}")
    print(f"  - 总样本数: {split_info['total_size']}")
    print(f"  - 训练集: {split_info['train_size']} 样本")
    print(f"  - 验证集: {split_info['val_size']} 样本")
    print(f"  - 测试集: {split_info['test_size']} 样本")

    # 验证数据文件是否匹配
    if split_info.get('data_path') != data_file:
        print(f"⚠️ 数据文件路径不匹配:")
        print(f"  保存的: {split_info.get('data_path')}")
        print(f"  当前的: {data_file}")

    # 创建完整数据集
    full_dataset = CultureLLMNewFormatDataset(
        data_file, tokenizer, max_length
    )

    if len(full_dataset) != split_info['total_size']:
        raise ValueError(f"数据集大小不匹配: 保存的={split_info['total_size']}, 当前={len(full_dataset)}")

    # 获取对应的索引
    if split_type == 'test':
        indices = split_info['test_indices']
        print(f"✅ 使用测试集: {len(indices)} 样本")
    elif split_type == 'validation':
        indices = split_info['val_indices']
        print(f"✅ 使用验证集: {len(indices)} 样本")
    else:
        raise ValueError(f"不支持的split_type: {split_type}")

    # 创建子数据集
    subset = Subset(full_dataset, indices)

    return subset, split_info


class SimplifiedCultureMoEEvaluator:
    """简化版CultureMoE评估器 - 支持消融实验"""

    def __init__(self, model_path: str, config_path: str = None):
        """
        初始化评估器

        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径，如果为None则从model_path/simplified_culturemoe_config.json加载
        """
        self.model_path = model_path
        self.config_path = config_path or os.path.join(model_path, 'simplified_culturemoe_config.json')

        # 加载配置
        self.config = self._load_config()

        # 初始化模型和tokenizer
        self.tokenizer = None
        self.model_adapter = None

        # 消融实验控制
        self.disable_shared = False
        self.disable_mask = False  # 新增MASK机制控制（占位符）
        self.disable_gate = False
        self.disable_culture_loss = False  # 新增文化损失控制

    def _load_config(self) -> SimplifiedCultureMoEConfig:
        """加载模型配置"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # 创建配置对象
        config = SimplifiedCultureMoEConfig(
            lora_rank=config_dict.get('lora_rank', 16),
            lora_alpha=config_dict.get('lora_alpha', 32),
            lora_dropout=config_dict.get('lora_dropout', 0.1),
            num_moe_experts=config_dict.get('num_moe_experts', 4),
            num_activated_experts=config_dict.get('num_activated_experts', 2),
            use_shared=config_dict.get('use_shared', False),
            use_gate=config_dict.get('use_gate', False),
            use_culture_loss=config_dict.get('use_culture_loss', 'new'),
            culture_loss_weight=config_dict.get('culture_loss_weight', 0.01),
            use_lora=config_dict.get('use_lora', True)
        )

        print(f"✅ 加载配置: {self.config_path}")
        print(f"  - MoE专家数: {config.num_moe_experts}")
        print(f"  - 激活专家数: {config.num_activated_experts}")
        print(f"  - 使用共享专家: {config.use_shared}")
        print(f"  - 使用Gate网络: {config.use_gate}")
        print(f"  - LoRA配置: rank={config.lora_rank}, alpha={config.lora_alpha}")

        return config

    def load_model(self, base_model_path: str, device: str = 'cuda'):
        """加载模型和tokenizer"""
        print(f"\n🔄 加载模型...")
        print(f"  基础模型: {base_model_path}")
        print(f"  训练权重: {self.model_path}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        # 🔧 关键修复：保持与训练时完全一致的tokenizer配置
        # 训练时使用的是默认配置：pad_token = eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        # 验证tokenizer配置（确保与训练时一致）
        print(f"✅ Tokenizer配置（与训练时一致）: pad_token_id={self.tokenizer.pad_token_id}, eos_token_id={self.tokenizer.eos_token_id}")
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            print(f"✅ pad_token_id == eos_token_id，与训练时配置一致")

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        base_model = base_model.to(device)

        # 先加载LoRA权重，然后创建MoE适配器
        self.base_model = base_model
        self._load_trained_weights_and_create_adapter()

        # 设置为评估模式
        self.model_adapter.base_model.eval()

        print("✅ 模型加载完成")

    def _load_trained_weights_and_create_adapter(self):
        """正确顺序加载权重并创建适配器"""
        # 🔧 关键修复：直接加载完整的模型，而不是分别加载LoRA和MoE权重
        # 问题：如果分别加载，创建适配器时会重新初始化MoE层，导致训练好的MoE权重丢失

        # 1. 先检查是否有保存的完整模型
        full_model_path = os.path.join(self.model_path, 'full_model.pt')
        if os.path.exists(full_model_path):
            try:
                print(f"🔍 检测到完整模型文件: {full_model_path}")
                print("🔍 尝试加载完整模型...")
                full_model = torch.load(full_model_path, map_location='cpu')

                # 加载基础模型
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path or self.model_path,
                    torch_dtype=torch.float16,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                base_model = base_model.to(self.device)

                # 加载完整模型的权重
                missing_keys, unexpected_keys = base_model.load_state_dict(full_model, strict=False)
                print(f"✅ 加载完整模型: 缺失{len(missing_keys)}个, 多余{len(unexpected_keys)}个键")

                # 创建MoE适配器
                print("🔧 创建MoE适配器...")
                self.model_adapter = SimplifiedCultureMoEAdapter(base_model, self.config)
                self.model_adapter.base_model.eval()

                return
            except Exception as e:
                print(f"⚠️ 完整模型加载失败: {e}")
                import traceback
                traceback.print_exc()

        # 2. 回退到原来的LoRA+MoE加载逻辑
        # 🔧 但要避免重新初始化MoE层导致权重丢失

        # 加载原始base模型（无LoRA）
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path or self.model_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        base_model = base_model.to(self.device)
        print("✅ 原始base模型加载完成")

        # 加载LoRA权重
        lora_path = os.path.join(self.model_path, 'lora_weights')
        if os.path.exists(lora_path):
            try:
                from peft import PeftModel
                base_model = PeftModel.from_pretrained(base_model, lora_path)
                print("✅ 加载LoRA权重")
            except Exception as e:
                print(f"⚠️ LoRA权重加载失败: {e}")

        # 创建MoE适配器（基于已加载LoRA的base_model）
        print("🔧 创建MoE适配器...")
        self.model_adapter = SimplifiedCultureMoEAdapter(base_model, self.config)
        self.model_adapter.base_model.eval()

        # 3. 🔧 尝试加载MoE权重，但不替换已有的MoE层
        # 关键修改：直接在base_model.module上加载MoE权重，而不是在backbone_model上
        moe_path = os.path.join(self.model_path, 'moe_weights.pt')
        if os.path.exists(moe_path):
            try:
                moe_state_dict = torch.load(moe_path, map_location='cpu')

                # 🔧 关键修复：直接在MoE适配器上加载权重
                # 因为model_adapter已经包含了替换后的MoE层，直接在其上加载权重
                model_to_load = self.model_adapter
                print(f"🔍 使用model_adapter进行权重加载: {type(model_to_load)}")

                missing_keys = []
                loaded_keys = []

                print(f"🔍 尝试加载{len(moe_state_dict)}个MoE权重参数...")
                print(f"🔍 目标模型类型: {type(model_to_load)}")

                # 🔧 调试：打印一些参数名称以了解结构
                sample_names = list(moe_state_dict.keys())[:3]
                print(f"🔍 示例参数名称: {sample_names}")

                # 🔧 先检查模型结构，打印一些层的名称以便调试
                print("🔍 检查模型结构...")
                if hasattr(model_to_load, 'layers') and len(model_to_load.layers) > 0:
                    layer0 = model_to_load.layers[0]
                    if hasattr(layer0, 'mlp'):
                        print(f"  - 第0层MLP类型: {type(layer0.mlp)}")
                        if hasattr(layer0.mlp, 'experts'):
                            print(f"  - 检测到experts: {type(layer0.mlp.experts)}")
                        if hasattr(layer0.mlp, 'router'):
                            print(f"  - 检测到router: {type(layer0.mlp.router)}")

                for name, param in moe_state_dict.items():
                    try:
                        # 🔧 修复：更智能的参数路径匹配
                        target_param = None

                        # 清理参数名称，去掉常见的包装前缀
                        clean_name = name
                        prefixes_to_remove = [
                            'base_model.model.model.',
                            'base_model.model.',
                            'model.model.',
                            'model.',
                            'base_model.',
                            'backbone_model.'
                        ]

                        for prefix in prefixes_to_remove:
                            if clean_name.startswith(prefix):
                                clean_name = clean_name[len(prefix):]
                                break

                        # 尝试在backbone_model上查找清理后的参数名
                        try:
                            target_param = model_to_load
                            for attr in clean_name.split('.'):
                                target_param = getattr(target_param, attr)
                        except AttributeError:
                            target_param = None

                        # 如果还是找不到，尝试原始名称
                        if target_param is None:
                            try:
                                target_param = model_to_load
                                for attr in name.split('.'):
                                    target_param = getattr(target_param, attr)
                            except AttributeError:
                                target_param = None

                        if target_param is not None:
                            # 确保target_param是参数张量而不是模型对象
                            if hasattr(target_param, 'shape') and hasattr(target_param, 'data'):
                                # 确保形状匹配
                                if target_param.shape == param.shape:
                                    # 🔧 添加权重加载前后的检查
                                    old_norm = target_param.data.norm().item()
                                    target_param.data.copy_(param.data)
                                    new_norm = target_param.data.norm().item()
                                    loaded_keys.append(name)
                                    print(f"  ✅ 加载: {name} (shape: {param.shape}, norm: {old_norm:.4f} → {new_norm:.4f})")
                                else:
                                    print(f"  ❌ 形状不匹配: {name} 期望{target_param.shape}, 得到{param.shape}")
                                    missing_keys.append(name)
                            else:
                                missing_keys.append(name)
                                print(f"  ❌ 找到对象但不是参数张量: {name} (type: {type(target_param)})")
                        else:
                            missing_keys.append(name)
                            print(f"  ❌ 未找到: {name} (尝试的路径: {clean_name})")

                    except Exception as e:
                        missing_keys.append(name)
                        print(f"  ❌ 加载失败: {name} (错误: {e})")

                print(f"\n📊 MoE权重加载结果:")
                print(f"  - 成功加载: {len(loaded_keys)}个")
                print(f"  - 未找到: {len(missing_keys)}个")

                if missing_keys:
                    print(f"⚠️ 警告：{len(missing_keys)}个MoE权重未加载！")
                    print("  这可能导致MoE层无法正常工作！")
                    print(f"  未加载的权重: {missing_keys[:5]}...")  # 只显示前5个
                else:
                    print("✅ 所有MoE权重加载成功")

            except Exception as e:
                print(f"⚠️ MoE权重加载失败: {e}")
                import traceback
                traceback.print_exc()

    def set_ablation_config(self, disable_shared: bool = False, disable_mask: bool = False,
                           disable_gate: bool = False, disable_culture_loss: bool = False):
        """设置消融实验配置"""
        self.disable_shared = disable_shared
        self.disable_mask = disable_mask
        self.disable_gate = disable_gate
        self.disable_culture_loss = disable_culture_loss

        if disable_shared or disable_mask or disable_gate or disable_culture_loss:
            print(f"\n🔧 消融实验配置:")
            print(f"  - 禁用共享专家: {disable_shared}")
            print(f"  - 禁用MASK机制: {disable_mask} (占位符)")
            print(f"  - 禁用Gate网络: {disable_gate}")
            print(f"  - 禁用文化损失: {disable_culture_loss}")

            # 应用消融配置到模型
            self._apply_ablation_to_model()

    def _apply_ablation_to_model(self):
        """将消融配置应用到模型"""
        layers, target_layers = self.model_adapter._get_target_layers()

        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFNLoRA):
                # 设置消融标志
                moe_layer.ablation_disable_shared = self.disable_shared
                moe_layer.ablation_disable_mask = self.disable_mask  # MASK机制占位符
                moe_layer.ablation_disable_gate = self.disable_gate

        print(f"✅ 消融配置已应用到{len(target_layers)}个MoE层")

    def save_validation_split(self, dataset, val_indices: List[int], output_dir: str):
        """保存验证集划分索引"""
        split_file = os.path.join(output_dir, 'validation_split.pkl')

        split_info = {
            'val_indices': val_indices,
            'total_size': len(dataset),
            'val_size': len(val_indices)
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(split_file, 'wb') as f:
            pickle.dump(split_info, f)

        print(f"✅ 验证集划分已保存: {split_file}")
        print(f"  - 总样本数: {split_info['total_size']}")
        print(f"  - 验证集大小: {split_info['val_size']}")

    def load_validation_split(self, output_dir: str) -> Optional[List[int]]:
        """加载验证集划分索引"""
        split_file = os.path.join(output_dir, 'validation_split.pkl')

        if not os.path.exists(split_file):
            return None

        with open(split_file, 'rb') as f:
            split_info = pickle.load(f)

        print(f"✅ 加载验证集划分: {split_file}")
        print(f"  - 总样本数: {split_info['total_size']}")
        print(f"  - 验证集大小: {split_info['val_size']}")

        return split_info['val_indices']

    def evaluate(self, data_file: str, output_dir: str, max_length: int = 384,
                 val_split: float = 0.1, device: str = 'cuda', use_fixed_split: bool = True,
                 split_file: str = None):
        """评估模型性能"""
        print(f"\n📊 开始评估...")
        print(f"  数据文件: {data_file}")
        print(f"  输出目录: {output_dir}")
        print(f"  使用固定数据集划分: {use_fixed_split}")
        print(f"  数据划分文件: {split_file}")

        # 加载测试集
        if use_fixed_split and split_file:
            # 🔍 验证假设：同时测试验证集和测试集
            print(f"🔍 验证假设：训练时验证集 vs 推理时测试集的差异")

            # 加载验证集
            val_dataset_for_comparison, _ = load_fixed_test_split(
                data_file, self.tokenizer, max_length, split_file, split_type='validation'
            )
            print(f"🔍 验证集大小: {len(val_dataset_for_comparison)} 样本")

            # 使用固定的测试集划分
            test_dataset, split_info = load_fixed_test_split(
                data_file, self.tokenizer, max_length, split_file, split_type='test'
            )
            val_dataset = test_dataset  # 在消融实验中，我们在测试集上评估
            print(f"🔍 关键信息：推理脚本使用的是测试集（{len(test_dataset)}样本）")
            print(f"✅ 使用固定测试集进行评估")
        else:
            # 回退到原来的逻辑（兼容性）
            print("⚠️ 未指定数据划分文件，使用原有的验证集划分逻辑")
            full_dataset = CultureLLMNewFormatDataset(
                data_file, self.tokenizer, max_length
            )

            # 处理验证集划分
            if use_fixed_split:
                val_indices = self.load_validation_split(output_dir)
                if val_indices is None:
                    # 第一次运行，创建并保存验证集划分
                    print("🔄 创建新的验证集划分...")
                    val_size = int(len(full_dataset) * val_split)
                    indices = list(range(len(full_dataset)))
                    np.random.seed(42)  # 固定随机种子确保可重现
                    np.random.shuffle(indices)
                    val_indices = indices[:val_size]
                    self.save_validation_split(full_dataset, val_indices, output_dir)
            else:
                # 使用随机划分
                val_size = int(len(full_dataset) * val_split)
                indices = list(range(len(full_dataset)))
                np.random.shuffle(indices)
                val_indices = indices[:val_size]

            # 创建验证集
            val_dataset = Subset(full_dataset, val_indices)

        # 创建数据加载器
        def collate_fn(batch):
            return dynamic_padding_collate_fn(batch, self.tokenizer)

        val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # 评估损失
        eval_metrics = self._evaluate_loss(val_loader, device)

        # 生成答案并评估准确率
        gen_metrics = self._evaluate_generation(val_dataset, device, output_dir)

        # 合并结果
        results = {
            **eval_metrics,
            **gen_metrics,
            'config': {
                'disable_shared': self.disable_shared,
                'disable_mask': self.disable_mask,
                'disable_gate': self.disable_gate,
                'disable_culture_loss': self.disable_culture_loss,
                'num_moe_experts': self.config.num_moe_experts,
                'num_activated_experts': self.config.num_activated_experts,
                'use_shared': self.config.use_shared and not self.disable_shared,
                'use_mask': True and not self.disable_mask,  # 占位符，目前默认为True
                'use_gate': self.config.use_gate and not self.disable_gate,
                'use_culture_loss': True and not self.disable_culture_loss  # 占位符，目前默认为True
            }
        }

        return results

    def _evaluate_loss(self, val_loader, device):
        """评估模型损失"""
        self.model_adapter.base_model.eval()
        total_loss = 0
        total_culture_loss = 0
        num_batches = 0

        print("🔄 计算验证损失...")

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating Loss"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # 正确处理labels masking
                for i in range(labels.shape[0]):
                    instruction = batch['instruction'][i] if isinstance(batch['instruction'], list) else batch['instruction']
                    input_text = batch['input'][i] if isinstance(batch['input'], list) else batch['input']

                    if input_text:
                        input_part = f"{instruction}\n{input_text}\n"
                    else:
                        input_part = f"{instruction}\n"

                    input_tokens = self.tokenizer(input_part, add_special_tokens=False)['input_ids']
                    input_length = len(input_tokens)

                    if input_length < labels.shape[1]:
                        labels[i, :input_length] = -100

                # 获取input_type（评估时为兼容模式）
                input_type = batch.get('input_type', None)
                if input_type is not None:
                    input_type = input_type.to(device)

                # 前向传播
                outputs = self.model_adapter.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    input_type=input_type  # 🆕 MASK机制
                )

                loss = outputs.loss
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()

                # 计算文化损失（如果有专家权重且未被禁用）
                if (hasattr(outputs, 'expert_weights') and outputs.expert_weights is not None
                    and not self.disable_culture_loss):
                    culture_labels = None
                    if 'label' in batch:
                        if isinstance(batch['label'], list):
                            label_ints = [int(label) if label.isdigit() else 0 for label in batch['label']]
                            culture_labels = torch.tensor(label_ints, dtype=torch.long, device=device)
                        else:
                            culture_labels = batch['label'].to(device)

                    if culture_labels is not None:
                        from train_simplified_culturemoe import compute_culture_loss
                        culture_loss = compute_culture_loss(outputs, culture_labels, 0.01)
                        total_culture_loss += culture_loss.item()

                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

        return {
            'eval_loss': avg_loss,
            'eval_culture_loss': avg_culture_loss,
            'num_batches': num_batches
        }

    def _evaluate_generation(self, val_dataset, device, output_dir):
        """评估生成质量和准确率"""
        self.model_adapter.base_model.eval()

        correct = 0
        total = 0
        generated_data = []

        print("🔄 生成答案并评估准确率...")

        for idx in tqdm(range(len(val_dataset)), desc="Generating"):
            # 获取样本
            if hasattr(val_dataset, 'dataset'):
                original_idx = val_dataset.indices[idx]
                sample = val_dataset.dataset[original_idx]
            else:
                sample = val_dataset[idx]

            instruction = sample['instruction']
            input_text = sample['input']
            true_output = sample['output']
            label = sample['label']

            # 生成答案（评估时使用原始instruction，不使用mask）
            generated_text = generate_answer(
                self.model_adapter, self.tokenizer, instruction, input_text, device
            )

            # 提取答案
            predicted_answer = extract_answer_from_text(generated_text)

            # 比对答案
            is_correct = predicted_answer == true_output
            if is_correct:
                correct += 1
            total += 1

            # 保存生成数据
            generated_data.append({
                'instruction': instruction,
                'input': input_text,
                'true_output': true_output,
                'label': label,
                'generated_text': generated_text,
                'predicted_answer': predicted_answer,
                'correct': is_correct
            })

        accuracy = correct / total if total > 0 else 0

        # 保存生成结果
        os.makedirs(output_dir, exist_ok=True)

        # 根据消融实验配置命名文件
        suffix = ""
        if self.disable_shared:
            suffix += "_no_shared"
        if self.disable_mask:
            suffix += "_no_mask"
        if self.disable_gate:
            suffix += "_no_gate"
        if self.disable_culture_loss:
            suffix += "_no_culture_loss"
        if not suffix:
            suffix = "_full"

        results_file = os.path.join(output_dir, f'generated_answers{suffix}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 生成结果已保存: {results_file}")

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }


def main():
    parser = argparse.ArgumentParser(description="简化版CultureMoE推理评估 - 支持消融实验")

    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型路径（包含权重和配置）")
    parser.add_argument("--data_file", type=str, required=True,
                        help="评估数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")

    # 模型参数
    parser.add_argument("--base_model_path", type=str,
                        help="基础模型路径（如果与训练时不同）")
    parser.add_argument("--max_length", type=int, default=384,
                        help="最大序列长度")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--device", type=str, default='cuda',
                        help="设备")

    # 消融实验参数
    parser.add_argument("--disable_shared", action='store_true',
                        help="禁用共享专家")
    parser.add_argument("--disable_mask", action='store_true',
                        help="禁用MASK机制（占位符）")
    parser.add_argument("--disable_gate", action='store_true',
                        help="禁用Gate网络")
    parser.add_argument("--disable_culture_loss", action='store_true',
                        help="禁用文化损失")
    parser.add_argument("--experiment_name", type=str, default="",
                        help="实验名称（用于文件命名）")
    parser.add_argument("--use_fixed_split", action='store_true', default=True,
                        help="使用固定的验证集划分（默认启用）")
    parser.add_argument("--split_file", type=str, default="",
                        help="数据划分文件路径（8:1:1划分信息）")

    args = parser.parse_args()

    print("="*80)
    print("简化版CultureMoE推理评估")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"数据文件: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"消融实验: disable_shared={args.disable_shared}, disable_mask={args.disable_mask}, disable_gate={args.disable_gate}, disable_culture_loss={args.disable_culture_loss}")
    if args.experiment_name:
        print(f"实验名称: {args.experiment_name}")
    print("="*80)

    # 推断基础模型路径
    if not args.base_model_path:
        # 尝试从模型路径中找到基础模型配置
        config_file = os.path.join(args.model_path, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                args.base_model_path = config.get('base_model', config.get('_name_or_path', ''))

        if not args.base_model_path:
            # 使用常见的默认路径
            possible_paths = [
                "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct",
                "/Users/yzl/models/Meta-Llama-3.1-8B-Instruct",
                "meta-llama/Meta-Llama-3.1-8B-Instruct"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    args.base_model_path = path
                    break

        if not args.base_model_path:
            print("❌ 无法推断基础模型路径，请使用 --base_model_path 指定")
            return

    print(f"基础模型路径: {args.base_model_path}")

    try:
        # 创建评估器
        evaluator = SimplifiedCultureMoEEvaluator(args.model_path)

        # 加载模型
        evaluator.load_model(args.base_model_path, args.device)

        # 设置消融实验配置
        evaluator.set_ablation_config(args.disable_shared, args.disable_mask,
                                     args.disable_gate, args.disable_culture_loss)

        # 评估模型
        results = evaluator.evaluate(
            args.data_file,
            args.output_dir,
            args.max_length,
            args.val_split,
            args.device,
            args.use_fixed_split,
            args.split_file if args.split_file else None
        )

        # 打印结果
        print("\n" + "="*80)
        print("📊 评估结果")
        print("="*80)
        print(f"验证损失: {results['eval_loss']:.4f}")
        print(f"文化损失: {results['eval_culture_loss']:.4f}")
        print(f"准确率: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
        print(f"配置:")
        print(f"  - 共享专家: {'启用' if results['config']['use_shared'] else '禁用'}")
        print(f"  - MASK机制: {'启用' if results['config']['use_mask'] else '禁用'} (占位符)")
        print(f"  - Gate网络: {'启用' if results['config']['use_gate'] else '禁用'}")
        print(f"  - 文化损失: {'启用' if results['config']['use_culture_loss'] else '禁用'}")

        # 保存结果
        experiment_suffix = ""
        if args.disable_shared:
            experiment_suffix += "_no_shared"
        if args.disable_mask:
            experiment_suffix += "_no_mask"
        if args.disable_gate:
            experiment_suffix += "_no_gate"
        if args.disable_culture_loss:
            experiment_suffix += "_no_culture_loss"
        if args.experiment_name:
            experiment_suffix += f"_{args.experiment_name}"
        if not experiment_suffix:
            experiment_suffix = "_full"

        results_file = os.path.join(args.output_dir, f'evaluation_results{experiment_suffix}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 评估完成！结果已保存到: {results_file}")
        print("="*80)

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()