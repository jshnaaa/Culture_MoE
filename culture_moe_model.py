#!/usr/bin/env python3
"""
CultureMoE模型实现
基于LoRA的MoE架构，在每个transformer层的FFN中添加4个路由专家和1个共享专家
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import re
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


class Router(nn.Module):
    """路由网络，计算专家选择权重"""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        # 简单的线性层作为路由器
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            gate_logits: [batch_size, seq_len, num_experts] 专家权重logits
            gate_probs: [batch_size, seq_len, num_experts] 专家权重概率
        """
        # 计算每个token对每个专家的权重
        gate_logits = self.gate(hidden_states)  # [batch_size, seq_len, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        return gate_logits, gate_probs


class Gate(nn.Module):
    """轻量级融合网络，融合共享专家和路由专家的输出"""

    def __init__(self, intermediate_size: int):
        super().__init__()
        # 🔧 修复：使用轻量级融合方式，大幅减少参数
        # 使用可学习的权重进行加权平均，而不是全连接层
        self.shared_weight = nn.Parameter(torch.ones(1))
        self.routed_weight = nn.Parameter(torch.ones(1))

    def forward(self, shared_output: torch.Tensor, routed_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_output: [batch_size, seq_len, intermediate_size] 共享专家输出
            routed_output: [batch_size, seq_len, intermediate_size] 路由专家加权输出
        Returns:
            fused_output: [batch_size, seq_len, intermediate_size] 融合后的输出
        """
        # 🔧 使用可学习的加权平均，参数量从4亿降到2个
        # 归一化权重
        total_weight = torch.abs(self.shared_weight) + torch.abs(self.routed_weight)
        shared_norm_weight = torch.abs(self.shared_weight) / (total_weight + 1e-8)
        routed_norm_weight = torch.abs(self.routed_weight) / (total_weight + 1e-8)

        # 加权融合
        fused_output = shared_norm_weight * shared_output + routed_norm_weight * routed_output
        return fused_output


class LoRAExpert(nn.Module):
    """单个LoRA专家"""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # LoRA参数
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, in_features]
        Returns:
            output: [batch_size, seq_len, out_features]
        """
        # LoRA变换: x @ A @ B
        lora_output = x @ self.lora_A @ self.lora_B
        return lora_output * (self.alpha / self.rank)


class CultureMoEFFN(nn.Module):
    """CultureMoE的FFN层，替换原始FFN"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_routing_experts: int = 4,
        num_activated_experts: int = 2,
        use_shared: bool = True,
        use_gate: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_routing_experts = num_routing_experts
        self.num_activated_experts = num_activated_experts
        self.use_shared = use_shared
        self.use_gate = use_gate

        # 路由网络
        self.router = Router(hidden_size, num_routing_experts)

        # 路由专家（LoRA）
        self.routing_experts = nn.ModuleList([
            LoRAExpert(hidden_size, intermediate_size, lora_rank, lora_alpha)
            for _ in range(num_routing_experts)
        ])

        # 共享专家（LoRA）
        if use_shared:
            self.shared_expert = LoRAExpert(hidden_size, intermediate_size, lora_rank, lora_alpha)

        # 融合门控
        if use_gate and use_shared:
            self.gate = Gate(intermediate_size)  # 轻量级门控网络

        # 输出投影层：将intermediate_size映射回hidden_size
        self.output_projection = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]  # 修复：应该返回hidden_size
            aux_info: 辅助信息，包含路由权重等
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 计算路由权重
        gate_logits, gate_probs = self.router(hidden_states)

        # Top-k选择
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.num_activated_experts, dim=-1)
        # 重新归一化
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 计算路由专家输出
        expert_outputs = []
        for i, expert in enumerate(self.routing_experts):
            expert_output = expert(hidden_states)  # [batch_size, seq_len, intermediate_size]
            expert_outputs.append(expert_output)

        # 堆叠所有专家输出 [num_experts, batch_size, seq_len, intermediate_size]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=0)

        # 使用向量化操作进行加权求和，保持梯度连接
        # top_k_indices: [batch_size, seq_len, num_activated_experts]
        # top_k_probs: [batch_size, seq_len, num_activated_experts]

        # 创建one-hot编码矩阵用于选择专家
        # [batch_size, seq_len, num_activated_experts, num_experts]
        # 确保与专家输出的设备和类型一致
        expert_mask = torch.zeros(batch_size, seq_len, self.num_activated_experts, self.num_routing_experts,
                                 device=stacked_expert_outputs.device, dtype=stacked_expert_outputs.dtype)

        # 使用scatter创建one-hot mask
        expert_mask.scatter_(3, top_k_indices.unsqueeze(-1), 1.0)

        # 计算加权专家输出
        # expert_mask: [batch_size, seq_len, num_activated_experts, num_experts]
        # stacked_expert_outputs: [num_experts, batch_size, seq_len, intermediate_size]
        # 重排维度以便广播: [batch_size, seq_len, num_experts, intermediate_size]
        expert_outputs_reshaped = stacked_expert_outputs.permute(1, 2, 0, 3)

        # 应用mask和权重
        # expert_mask: [batch_size, seq_len, num_activated_experts, num_experts, 1]
        # expert_outputs_reshaped: [batch_size, seq_len, 1, num_experts, intermediate_size]
        weighted_outputs = expert_mask.unsqueeze(-1) * expert_outputs_reshaped.unsqueeze(2)

        # 应用top-k权重
        # top_k_probs: [batch_size, seq_len, num_activated_experts, 1, 1]
        weighted_outputs = weighted_outputs * top_k_probs.unsqueeze(-1).unsqueeze(-1)

        # 求和得到最终路由输出
        routed_output = weighted_outputs.sum(dim=(2, 3))  # [batch_size, seq_len, intermediate_size]

        # 计算共享专家输出
        if self.use_shared:
            shared_output = self.shared_expert(hidden_states)
        else:
            shared_output = None

        # 融合输出
        if self.use_gate and self.use_shared:
            intermediate_output = self.gate(shared_output, routed_output)
        elif self.use_shared:
            # 简单相加
            intermediate_output = shared_output + routed_output
        else:
            intermediate_output = routed_output

        # 投影到hidden_size维度
        final_output = self.output_projection(intermediate_output)

        # 收集辅助信息
        aux_info = {
            'gate_probs': gate_probs,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs,
            'expert_outputs': expert_outputs,
            'shared_output': shared_output,
            'routed_output': routed_output
        }

        return final_output, aux_info


class CultureMoEModel(nn.Module):
    """CultureMoE模型包装器"""

    def __init__(
        self,
        base_model,
        config: Dict
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # 获取模型配置
        model_config = base_model.config
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = getattr(model_config, 'intermediate_size', 4 * self.hidden_size)

        # 确定要替换的层
        if config['backbone'] == 'llama':
            self.num_layers = 32
            self.layer_attr = 'layers'
            self.ffn_attr = 'mlp'
        elif config['backbone'] == 'qwen':
            self.num_layers = 28
            self.layer_attr = 'layers'
            self.ffn_attr = 'mlp'
        else:
            raise ValueError(f"Unsupported backbone: {config['backbone']}")

        # 替换每一层的FFN为CultureMoE - 确保设备一致性
        self.culture_moe_layers = nn.ModuleList()

        # 获取基座模型各层的设备分布
        transformer = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
        layers = getattr(transformer, self.layer_attr)

        print(f"检查基座模型层的设备分布:")
        for layer_idx in range(min(5, len(layers))):  # 只检查前5层
            layer_device = next(layers[layer_idx].parameters()).device
            print(f"  基座模型第{layer_idx}层在: {layer_device}")

        for layer_idx in range(self.num_layers):
            moe_ffn = CultureMoEFFN(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_routing_experts=config['num_moe_experts'],
                num_activated_experts=config['num_activated_experts'],
                use_shared=config['use_shared'],
                use_gate=config['use_gate'],
                lora_rank=config['lora_rank'],
                lora_alpha=config['lora_alpha']
            )

            # 获取对应基座模型层的设备和数据类型，确保完全一致
            if layer_idx < len(layers):
                target_device = next(layers[layer_idx].parameters()).device
                target_dtype = next(layers[layer_idx].parameters()).dtype

                # 同时设置设备和数据类型
                moe_ffn = moe_ffn.to(device=target_device, dtype=target_dtype)

                if layer_idx < 3:  # 只打印前3层的设备和类型分配
                    print(f"MoE层{layer_idx}移动到{target_device}, 数据类型: {target_dtype}")

            self.culture_moe_layers.append(moe_ffn)

        # 冻结基座模型参数
        self._freeze_base_model()

        # 添加注意力层LoRA（如果启用）
        if config.get('apply_to_attention', False):
            self._add_attention_lora()

    def _freeze_base_model(self):
        """选择性冻结基座模型参数，保持梯度流"""
        # 获取模型结构
        if hasattr(self.base_model, 'model'):
            model = self.base_model.model
        else:
            model = self.base_model

        # 冻结embedding层（不需要训练）
        if hasattr(model, 'embed_tokens'):
            for param in model.embed_tokens.parameters():
                param.requires_grad = False
            print("Embedding layer frozen")

        # 冻结位置编码等
        if hasattr(model, 'embed_positions'):
            for param in model.embed_positions.parameters():
                param.requires_grad = False

        # 冻结LayerNorm
        if hasattr(model, 'norm'):
            for param in model.norm.parameters():
                param.requires_grad = False

        # 冻结LM Head（如果存在）
        if hasattr(self.base_model, 'lm_head'):
            for param in self.base_model.lm_head.parameters():
                param.requires_grad = False

        # 🔧 关键修复：保留所有transformer层的梯度，确保MoE层能接收到梯度
        # 不冻结transformer layers，让梯度能够流通到MoE层
        layers = getattr(model, self.layer_attr)
        trainable_transformer_params = 0
        for layer in layers:
            for param in layer.parameters():
                if param.requires_grad:
                    trainable_transformer_params += param.numel()

        print(f"Transformer layers kept trainable for gradient flow")
        print(f"Trainable transformer parameters: {trainable_transformer_params:,}")
        print("Only embedding and output layers frozen")

    def _add_attention_lora(self):
        """为注意力层添加LoRA"""
        lora_config = LoraConfig(
            r=self.config['lora_rank'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        print("Attention LoRA added")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """前向传播"""
        # 获取基座模型的transformer层
        if hasattr(self.base_model, 'model'):
            transformer = self.base_model.model
        else:
            transformer = self.base_model

        # 存储MoE辅助信息
        moe_aux_info = []

        # Hook函数来替换FFN输出
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                # input[0]是FFN的输入hidden_states
                hidden_states = input[0]

                # 确保MoE层和输入在同一设备且数据类型一致
                moe_layer = self.culture_moe_layers[layer_idx]
                moe_device = next(moe_layer.parameters()).device
                moe_dtype = next(moe_layer.parameters()).dtype

                # 检查设备和数据类型一致性
                if hidden_states.device != moe_device or hidden_states.dtype != moe_dtype:
                    print(f"不匹配警告: hidden_states({hidden_states.device}, {hidden_states.dtype}) vs MoE层{layer_idx}({moe_device}, {moe_dtype})")
                    # 将MoE层移动到hidden_states的设备和数据类型
                    moe_layer = moe_layer.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    self.culture_moe_layers[layer_idx] = moe_layer

                # 通过MoE层计算输出，保持梯度连接
                moe_output, aux_info = moe_layer(hidden_states)
                moe_aux_info.append(aux_info)

                # 直接返回MoE输出，保持梯度图完整
                return moe_output
            return hook_fn

        # 注册hooks
        hooks = []
        layers = getattr(transformer, self.layer_attr)
        for layer_idx, layer in enumerate(layers):
            ffn = getattr(layer, self.ffn_attr)
            hook = ffn.register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)

        try:
            # 执行前向传播
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

            # 添加MoE辅助信息
            outputs.moe_aux_info = moe_aux_info

            return outputs
        finally:
            # 清理hooks
            for hook in hooks:
                hook.remove()


def create_culture_moe_model(base_model_path: str, config: Dict) -> CultureMoEModel:
    """创建CultureMoE模型"""

    # 加载基座模型 - 适配48GB*2卡配置
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "40GB", 1: "40GB"}  # 为每张卡预留充足空间
    )

    # 启用gradient checkpointing以节省显存
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # 创建CultureMoE模型
    model = CultureMoEModel(base_model, config)

    return model


def extract_answer_from_text(text: str) -> str:
    """从生成的文本中提取数字答案"""
    # 查找文本中的阿拉伯数字
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return numbers[-1]  # 返回最后一个数字
    return ""