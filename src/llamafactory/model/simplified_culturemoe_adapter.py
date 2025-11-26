# src/llamafactory/model/simplified_culturemoe_adapter.py
"""
简化版CultureMoE适配器
基于MixLoRA实现，只在指定层使用MoE，添加文化损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import math

from .simplified_culturemoe import SimplifiedCultureMoEConfig


class LoRALinear(nn.Module):
    """LoRA线性层"""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA参数
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class SimplifiedMoEExpert(nn.Module):
    """简化的MoE专家"""

    def __init__(self, base_layer, config: SimplifiedCultureMoEConfig):
        super().__init__()
        self.base_layer = base_layer

        # 冻结基础层
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # 添加LoRA适配器（在正确设备上）
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device

        self.lora_adapter = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        ).to(device)

    def forward(self, x):
        # 基础输出 + LoRA适配
        base_output = self.base_layer(x)
        lora_output = self.lora_adapter(x)
        return base_output + lora_output


class SimplifiedMoELayer(nn.Module):
    """简化的MoE层，基于MixLoRA实现"""

    def __init__(self, original_mlp, config: SimplifiedCultureMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_routing_experts
        self.top_k = config.top_k
        self.aux_loss_coef = config.aux_loss_coef

        # 获取原始MLP的维度和设备
        self.hidden_size = original_mlp.gate_proj.in_features
        self.intermediate_size = original_mlp.gate_proj.out_features
        self.device = original_mlp.gate_proj.weight.device

        # 创建路由器（在正确设备上）
        self.router = nn.Linear(self.hidden_size, self.num_experts).to(self.device)

        # 创建专家 - 每个专家都基于原始MLP + LoRA
        self.experts = nn.ModuleList([
            self._create_expert_from_mlp(original_mlp, config)
            for _ in range(self.num_experts)
        ])

        # 激活函数
        self.act_fn = original_mlp.act_fn

        # 确保所有专家在正确设备上
        self.experts = self.experts.to(self.device)

    def _create_expert_from_mlp(self, original_mlp, config):
        """从原始MLP创建专家"""
        expert = nn.Module()

        # 复制原始层并添加LoRA
        expert.gate_proj = SimplifiedMoEExpert(original_mlp.gate_proj, config)
        expert.up_proj = SimplifiedMoEExpert(original_mlp.up_proj, config)
        expert.down_proj = SimplifiedMoEExpert(original_mlp.down_proj, config)
        expert.act_fn = original_mlp.act_fn

        # 确保专家在正确设备上
        expert = expert.to(self.device)

        return expert

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 路由决策
        router_logits = self.router(hidden_states.view(-1, hidden_dim))  # [B*L, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 重新归一化

        # 专家计算 - 简化版本，避免复杂的调度
        expert_outputs = []
        for i in range(self.num_experts):
            # 找到使用当前专家的token
            expert_mask = (top_k_indices == i).any(dim=-1)  # [B*L]

            if expert_mask.any():
                # 计算专家输出
                expert = self.experts[i]
                expert_input = hidden_states.view(-1, hidden_dim)[expert_mask]  # [num_tokens, hidden_dim]

                # SwiGLU FFN
                gate_output = expert.act_fn(expert.gate_proj(expert_input))
                up_output = expert.up_proj(expert_input)
                expert_output = expert.down_proj(gate_output * up_output)

                expert_outputs.append((expert_mask, expert_output, i))

        # 合并专家输出
        final_output = torch.zeros_like(hidden_states.view(-1, hidden_dim))

        # 确保有专家输出，否则返回原始输入
        if not expert_outputs:
            # 如果没有专家被选中，返回原始输入（fallback）
            return hidden_states

        for expert_mask, expert_output, expert_idx in expert_outputs:
            # 获取该专家的权重
            expert_weights = torch.zeros(batch_size * seq_len, device=hidden_states.device)
            for k in range(self.top_k):
                mask_k = (top_k_indices[:, k] == expert_idx)
                expert_weights[mask_k] = top_k_probs[mask_k, k]

            # 应用权重
            weighted_output = expert_output * expert_weights[expert_mask].unsqueeze(-1)
            final_output[expert_mask] += weighted_output

        final_output = final_output.view(batch_size, seq_len, hidden_dim)

        # 计算辅助损失（但不返回，为了兼容性）
        aux_loss = self._compute_aux_loss(router_probs)

        # 为了兼容性，只返回输出张量，就像普通MLP一样
        return final_output

    def _compute_aux_loss(self, router_probs):
        """计算负载均衡辅助损失"""
        # 计算每个专家的使用频率
        expert_freq = router_probs.mean(dim=0)  # [num_experts]

        # 负载均衡损失：鼓励均匀分布
        aux_loss = torch.var(expert_freq) * self.aux_loss_coef

        return aux_loss


class SimplifiedCultureMoEAdapter:
    """简化版CultureMoE适配器"""

    def __init__(self, base_model, config: SimplifiedCultureMoEConfig):
        self.base_model = base_model
        self.config = config

        # 替换指定层的MLP为MoE
        self._replace_mlp_with_moe()

        # 冻结非LoRA参数
        self._freeze_non_lora_parameters()

        # 确保所有参数在同一设备上
        self._ensure_device_consistency()

    def _replace_mlp_with_moe(self):
        """替换指定层的MLP为MoE"""
        # 处理DDP包装的模型
        model_to_modify = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_modify, 'model'):
            # LlamaForCausalLM
            layers = model_to_modify.model.layers
        else:
            # LlamaModel
            layers = model_to_modify.layers

        for layer_idx in self.config.moe_layers:
            if layer_idx < len(layers):
                original_mlp = layers[layer_idx].mlp
                moe_layer = SimplifiedMoELayer(original_mlp, self.config)
                layers[layer_idx].mlp = moe_layer
                print(f"✅ Replaced layer {layer_idx} MLP with SimplifiedMoE")

    def _freeze_non_lora_parameters(self):
        """冻结非LoRA参数"""
        # 处理DDP包装的模型
        model_to_freeze = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        for name, param in model_to_freeze.named_parameters():
            if 'lora_' not in name and 'router' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _ensure_device_consistency(self):
        """确保所有参数在同一设备上"""
        # 处理DDP包装的模型
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 获取基础模型的设备
        base_device = next(model_to_check.parameters()).device

        # 验证所有MoE层都在正确设备上
        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        for layer_idx in self.config.moe_layers:
            if layer_idx < len(layers):
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, SimplifiedMoELayer):
                    # 确保MoE层在正确设备上
                    layers[layer_idx].mlp = moe_layer.to(base_device)
                    print(f"✅ Ensured MoE layer {layer_idx} is on device {base_device}")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """前向传播"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 简化版本：不收集复杂的MoE辅助信息
        # 只添加一个简单的专家权重用于文化损失计算
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            # 创建一个虚拟的专家权重用于文化损失
            batch_size = input_ids.shape[0]
            # 简单的均匀分布权重
            dummy_weights = torch.ones(batch_size, self.config.num_routing_experts, device=input_ids.device)
            dummy_weights = F.softmax(dummy_weights, dim=-1)
            outputs.expert_weights = dummy_weights

        return outputs

    def save_model(self, save_path: str):
        """保存LoRA权重"""
        import os

        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        # 只保存LoRA参数
        lora_state_dict = {}
        # 处理DDP包装的模型
        model_to_save = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        for name, param in model_to_save.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_state_dict[name] = param.data

        # 保存LoRA权重
        lora_path = os.path.join(save_path, 'lora_weights.pt')
        torch.save(lora_state_dict, lora_path)

        # 保存配置
        config_path = os.path.join(save_path, 'simplified_culturemoe_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"✅ Simplified CultureMoE weights saved to {save_path}")
        print(f"   - LoRA weights: {lora_path}")
        print(f"   - Config: {config_path}")

    def print_trainable_parameters(self):
        """打印可训练参数统计"""
        total_params = 0
        trainable_params = 0

        # 处理DDP包装的模型
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        for param in model_to_check.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || "
              f"Total params: {total_params:,} || "
              f"Trainable%: {100 * trainable_params / total_params:.4f}%")

        # 详细统计
        lora_params = 0
        router_params = 0

        for name, param in model_to_check.named_parameters():
            if param.requires_grad:
                if 'lora_' in name:
                    lora_params += param.numel()
                elif 'router' in name:
                    router_params += param.numel()

        print(f"  - LoRA params: {lora_params:,}")
        print(f"  - Router params: {router_params:,}")
        print(f"  - MoE layers: {self.config.moe_layers}")
        print(f"  - Routing experts per layer: {self.config.num_routing_experts}")


def create_simplified_culturemoe_model(base_model, config: SimplifiedCultureMoEConfig):
    """创建简化版CultureMoE模型"""

    # 创建适配器
    adapter = SimplifiedCultureMoEAdapter(base_model, config)

    return adapter