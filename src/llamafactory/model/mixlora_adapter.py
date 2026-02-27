#!/usr/bin/env python3
"""
MixLoRA Model Adapter

将MixLoRA集成到现有的Transformer模型中，支持LLaMA和Qwen模型。

主要功能：
1. 自动识别和替换FFN层为MixLoRA层
2. 在注意力层添加普通LoRA适配器
3. 管理训练过程中的辅助损失
4. 提供模型保存和加载功能
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from .mixlora import MixLoRAConfig, MixLoRALayer, compute_mixlora_total_loss

logger = logging.getLogger(__name__)


class MixLoRAModelAdapter:
    """
    MixLoRA模型适配器

    负责将MixLoRA集成到预训练模型中。
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        mixlora_config: MixLoRAConfig
    ):
        """
        初始化MixLoRA模型适配器

        Args:
            base_model: 基础预训练模型
            mixlora_config: MixLoRA配置
        """
        self.base_model = base_model
        self.mixlora_config = mixlora_config
        self.mixlora_layers = nn.ModuleDict()
        self.aux_info_cache = []

        # 识别模型架构
        self.model_type = self._identify_model_type()
        logger.info(f"Detected model type: {self.model_type}")

        # 应用MixLoRA
        self._apply_mixlora()

        # 应用注意力LoRA（如果配置了）
        if mixlora_config.attention_target_modules:
            self._apply_attention_lora()

    def _identify_model_type(self) -> str:
        """识别模型类型"""
        model_class_name = self.base_model.__class__.__name__.lower()

        if 'llama' in model_class_name:
            return 'llama'
        elif 'qwen' in model_class_name:
            return 'qwen'
        else:
            # 尝试从config中获取
            config = getattr(self.base_model, 'config', None)
            if config:
                model_type = getattr(config, 'model_type', '').lower()
                if model_type in ['llama', 'qwen2']:
                    return model_type

            logger.warning(f"Unknown model type: {model_class_name}, assuming llama-like")
            return 'llama'

    def _get_layer_path(self) -> Tuple[str, str]:
        """
        获取模型层的路径

        Returns:
            layers_attr: 层属性名称 (如 'layers')
            layer_class_name: 层类名称 (如 'LlamaDecoderLayer')
        """
        if self.model_type == 'llama':
            return 'model.layers', 'LlamaDecoderLayer'
        elif self.model_type in ['qwen', 'qwen2']:
            return 'model.layers', 'Qwen2DecoderLayer'
        else:
            # 默认假设为llama-like
            return 'model.layers', 'DecoderLayer'

    def _get_ffn_modules(self, layer) -> Dict[str, nn.Module]:
        """
        获取FFN模块

        Args:
            layer: Transformer层

        Returns:
            FFN模块字典
        """
        ffn_modules = {}

        # 尝试不同的FFN结构
        ffn_attr_names = ['mlp', 'feed_forward', 'ffn']

        ffn = None
        for attr_name in ffn_attr_names:
            if hasattr(layer, attr_name):
                ffn = getattr(layer, attr_name)
                break

        if ffn is None:
            raise ValueError(f"Cannot find FFN module in layer: {type(layer)}")

        # 获取FFN内部的线性层
        for name, module in ffn.named_modules():
            if isinstance(module, nn.Linear):
                # 过滤掉嵌套的模块（只要直接子模块）
                if '.' not in name:
                    ffn_modules[name] = module

        if not ffn_modules:
            # 如果没有找到，尝试直接获取常见的FFN层名称
            common_ffn_names = [
                'gate_proj', 'up_proj', 'down_proj',  # LLaMA
                'w1', 'w2', 'w3',  # 一些实现
                'fc1', 'fc2'  # 通用FFN
            ]

            for name in common_ffn_names:
                if hasattr(ffn, name):
                    module = getattr(ffn, name)
                    if isinstance(module, nn.Linear):
                        ffn_modules[name] = module

        if not ffn_modules:
            raise ValueError(f"Cannot find linear layers in FFN: {type(ffn)}")

        logger.debug(f"Found FFN modules: {list(ffn_modules.keys())}")
        return ffn_modules

    def _apply_mixlora(self):
        """将MixLoRA应用到模型的FFN层"""
        layers_attr, _ = self._get_layer_path()

        # 获取模型层
        model_layers = self.base_model
        for attr in layers_attr.split('.'):
            model_layers = getattr(model_layers, attr)

        logger.info(f"Applying MixLoRA to {len(model_layers)} layers")

        # 为每一层应用MixLoRA
        for layer_idx, layer in enumerate(model_layers):
            try:
                # 获取FFN模块
                ffn_modules = self._get_ffn_modules(layer)

                # 创建LoRA配置
                lora_config = LoraConfig(
                    r=self.mixlora_config.lora_rank,
                    lora_alpha=self.mixlora_config.lora_alpha,
                    lora_dropout=self.mixlora_config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM"
                )

                # 创建MixLoRA层
                mixlora_layer = MixLoRALayer(
                    layer_idx=layer_idx,
                    base_ffn_layers=ffn_modules,
                    lora_config=lora_config,
                    num_experts=self.mixlora_config.num_experts,
                    top_k=self.mixlora_config.top_k,
                    target_modules=self.mixlora_config.ffn_target_modules
                )

                # 保存MixLoRA层
                self.mixlora_layers[f'layer_{layer_idx}'] = mixlora_layer

                # 替换原始FFN的前向传播
                self._replace_ffn_forward(layer, layer_idx)

                logger.debug(f"Applied MixLoRA to layer {layer_idx}")

            except Exception as e:
                logger.error(f"Failed to apply MixLoRA to layer {layer_idx}: {e}")
                raise

        logger.info(f"Successfully applied MixLoRA to {len(self.mixlora_layers)} layers")

    def _replace_ffn_forward(self, layer, layer_idx: int):
        """
        替换FFN层的前向传播 - 永久替换，不使用临时替换

        Args:
            layer: Transformer层
            layer_idx: 层索引
        """
        mixlora_layer = self.mixlora_layers[f'layer_{layer_idx}']

        # 获取FFN模块
        ffn_module = getattr(layer, 'mlp', None)
        if ffn_module is None:
            for ffn_attr in ['feed_forward', 'ffn']:
                if hasattr(layer, ffn_attr):
                    ffn_module = getattr(layer, ffn_attr)
                    break

        if ffn_module is None:
            logger.warning(f"Layer {layer_idx}: Cannot find FFN module, skipping MixLoRA")
            return

        # 保存原始的FFN forward方法
        original_ffn_forward = ffn_module.forward

        # 创建永久性的MixLoRA FFN forward方法
        def mixlora_ffn_forward(ffn_input):
            """使用MixLoRA替换FFN计算"""
            try:
                # 确保输入张量的设备和数据类型正确
                if not torch.is_tensor(ffn_input):
                    return original_ffn_forward(ffn_input)

                # 检查输入维度
                if len(ffn_input.shape) != 3:  # 期望 [batch, seq, hidden]
                    return original_ffn_forward(ffn_input)

                # 调用MixLoRA层
                mixlora_output, aux_info = mixlora_layer(ffn_input)

                # 缓存辅助信息（仅在训练时）
                if self.base_model.training:
                    self.aux_info_cache.append(aux_info)

                # 确保输出维度与输入匹配
                if mixlora_output.shape != ffn_input.shape:
                    return original_ffn_forward(ffn_input)

                return mixlora_output
            except Exception as e:
                logger.warning(f"MixLoRA FFN failed: {e}, using original")
                return original_ffn_forward(ffn_input)

        # 永久性替换FFN的forward方法
        ffn_module.forward = mixlora_ffn_forward
        logger.debug(f"Layer {layer_idx}: FFN forward permanently replaced with MixLoRA")

    def _apply_attention_lora(self):
        """在注意力层应用普通LoRA"""
        # 检查是否需要应用注意力LoRA
        if self.mixlora_config.attention_target_modules is None or not self.mixlora_config.attention_target_modules:
            logger.info("Skipping attention LoRA (no target modules specified)")
            return

        # 创建注意力LoRA配置
        attention_lora_config = LoraConfig(
            r=self.mixlora_config.lora_rank,
            lora_alpha=self.mixlora_config.lora_alpha,
            lora_dropout=self.mixlora_config.lora_dropout,
            target_modules=self.mixlora_config.attention_target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 应用LoRA到注意力层
        self.base_model = get_peft_model(self.base_model, attention_lora_config)
        logger.info(f"Applied LoRA to attention modules: {self.mixlora_config.attention_target_modules}")

    def forward(self, *args, **kwargs):
        """前向传播包装器"""
        # 清空辅助信息缓存
        self.aux_info_cache = []

        # 执行模型前向传播
        outputs = self.base_model(*args, **kwargs)

        # 如果是训练模式且有标签，计算总损失
        if self.base_model.training and 'labels' in kwargs and kwargs['labels'] is not None:
            main_loss = outputs.loss
            total_loss, loss_info = compute_mixlora_total_loss(
                main_loss=main_loss,
                aux_info_list=self.aux_info_cache,
                aux_loss_coef=self.mixlora_config.aux_loss_coef
            )

            # 替换损失
            outputs.loss = total_loss

            # 添加损失信息到输出
            if hasattr(outputs, 'loss_info'):
                outputs.loss_info = loss_info
            else:
                # 如果输出对象不支持添加属性，记录到日志
                logger.debug(f"Loss info: {loss_info}")

        return outputs

    def save_mixlora(self, save_directory: str):
        """
        保存MixLoRA权重和配置 - 只保存可训练参数

        Args:
            save_directory: 保存目录
        """
        os.makedirs(save_directory, exist_ok=True)

        # 保存MixLoRA配置
        config_path = os.path.join(save_directory, 'mixlora_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.mixlora_config.to_dict(), f, indent=2, ensure_ascii=False)

        # 只保存可训练的MixLoRA权重（不包括冻结的基础层）
        mixlora_state_dict = {}
        total_trainable_params = 0
        total_saved_params = 0
        dtype_counter = {}  # 统计不同数据类型的参数

        for layer_name, mixlora_layer in self.mixlora_layers.items():
            layer_state_dict = {}

            # 只保存可训练参数
            for param_name, param in mixlora_layer.named_parameters():
                if param.requires_grad:
                    # 确保参数名称不包含冻结的基础层
                    if not any(frozen_key in param_name for frozen_key in ['base_ffn_layers']):
                        # 🔧 修复：确保使用float16保存，减小文件大小
                        param_to_save = param.cpu().clone().detach()
                        if param_to_save.dtype == torch.float32:
                            param_to_save = param_to_save.half()  # 转换为float16

                        layer_state_dict[param_name] = param_to_save
                        total_trainable_params += param.numel()
                        total_saved_params += param.numel()

                        # 统计数据类型
                        dtype_str = str(param_to_save.dtype)
                        dtype_counter[dtype_str] = dtype_counter.get(dtype_str, 0) + param.numel()

            # 只有当层有可训练参数时才保存
            if layer_state_dict:
                mixlora_state_dict[layer_name] = layer_state_dict

        weights_path = os.path.join(save_directory, 'mixlora_weights.pt')
        torch.save(mixlora_state_dict, weights_path)

        # 计算文件大小
        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        # 🔧 修复：根据实际数据类型计算期望大小
        param_size_mb = total_saved_params * 2 / (1024 * 1024)  # float16占2字节

        logger.info(f"MixLoRA saved to: {save_directory}")
        logger.info(f"  Trainable parameters saved: {total_saved_params:,}")
        logger.info(f"  Parameter dtypes: {dtype_counter}")
        logger.info(f"  File size: {file_size_mb:.2f} MB")
        logger.info(f"  Expected size (float16): {param_size_mb:.2f} MB")

        # 警告如果文件过大
        if file_size_mb > 200:  # 如果超过200MB
            logger.warning(f"⚠️  MixLoRA weights file is larger than expected: {file_size_mb:.2f} MB")
            logger.warning("This may indicate that frozen base model weights are being saved")
            logger.warning(f"Expected size for {total_saved_params:,} parameters: ~{param_size_mb:.2f} MB")
        else:
            logger.info("✅ MixLoRA weights saved successfully (trainable parameters only)")

    @classmethod
    def load_mixlora(
        cls,
        base_model: PreTrainedModel,
        load_directory: str
    ) -> 'MixLoRAModelAdapter':
        """
        加载MixLoRA权重和配置

        Args:
            base_model: 基础模型
            load_directory: 加载目录

        Returns:
            MixLoRA模型适配器
        """
        # 加载配置
        config_path = os.path.join(load_directory, 'mixlora_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        mixlora_config = MixLoRAConfig.from_dict(config_dict)

        # 创建适配器
        adapter = cls(base_model, mixlora_config)

        # 加载权重
        weights_path = os.path.join(load_directory, 'mixlora_weights.pt')
        mixlora_state_dict = torch.load(weights_path, map_location='cpu')

        total_loaded_params = 0
        for layer_name, state_dict in mixlora_state_dict.items():
            if layer_name in adapter.mixlora_layers:
                # 使用strict=False，因为我们只加载可训练参数
                # 冻结的基础层权重不在保存的状态字典中
                missing_keys, unexpected_keys = adapter.mixlora_layers[layer_name].load_state_dict(
                    state_dict, strict=False
                )

                # 统计加载的参数数量
                for param in state_dict.values():
                    total_loaded_params += param.numel()

                # 记录缺失的键（这些应该是冻结的基础层，这是正常的）
                if missing_keys:
                    logger.debug(f"Layer {layer_name} - Missing keys (frozen base layers): {len(missing_keys)} keys")
                if unexpected_keys:
                    logger.warning(f"Layer {layer_name} - Unexpected keys: {unexpected_keys}")

        logger.info(f"MixLoRA loaded from: {load_directory}")
        logger.info(f"  Loaded parameters: {total_loaded_params:,}")
        logger.info("✅ MixLoRA weights loaded successfully (trainable parameters only)")
        return adapter

    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        total_params = 0
        trainable_params = 0

        # 统计基础模型参数
        for param in self.base_model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        # 统计MixLoRA参数
        mixlora_params = 0
        for mixlora_layer in self.mixlora_layers.values():
            for param in mixlora_layer.parameters():
                mixlora_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()

        trainable_percentage = 100 * trainable_params / total_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"MixLoRA parameters: {mixlora_params:,}")
        print(f"Trainable percentage: {trainable_percentage:.4f}%")


def create_mixlora_model(
    base_model: PreTrainedModel,
    mixlora_config: Optional[MixLoRAConfig] = None,
    **config_kwargs
) -> MixLoRAModelAdapter:
    """
    创建MixLoRA模型

    Args:
        base_model: 基础预训练模型
        mixlora_config: MixLoRA配置
        **config_kwargs: 配置参数

    Returns:
        MixLoRA模型适配器
    """
    if mixlora_config is None:
        mixlora_config = MixLoRAConfig(**config_kwargs)

    adapter = MixLoRAModelAdapter(base_model, mixlora_config)
    return adapter