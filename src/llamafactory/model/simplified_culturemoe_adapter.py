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

        # 添加LoRA适配器
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_adapter = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )

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

        # 获取原始MLP的维度
        self.hidden_size = original_mlp.gate_proj.in_features
        self.intermediate_size = original_mlp.gate_proj.out_features

        # 创建路由器
        self.router = nn.Linear(self.hidden_size, self.num_experts)

        # 创建专家 - 每个专家都基于原始MLP + LoRA
        self.experts = nn.ModuleList([
            self._create_expert_from_mlp(original_mlp, config)
            for _ in range(self.num_experts)
        ])

        # 激活函数
        self.act_fn = original_mlp.act_fn

    def _create_expert_from_mlp(self, original_mlp, config):
        """从原始MLP创建专家"""
        expert = nn.Module()

        # 复制原始层并添加LoRA
        expert.gate_proj = SimplifiedMoEExpert(original_mlp.gate_proj, config)
        expert.up_proj = SimplifiedMoEExpert(original_mlp.up_proj, config)
        expert.down_proj = SimplifiedMoEExpert(original_mlp.down_proj, config)
        expert.act_fn = original_mlp.act_fn

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

        # 计算辅助损失
        aux_loss = self._compute_aux_loss(router_probs)

        return final_output, {
            'router_logits': router_logits,
            'router_probs': router_probs,
            'aux_loss': aux_loss,
            'expert_weights': router_probs  # 用于文化损失计算
        }

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

    def _replace_mlp_with_moe(self):
        """替换指定层的MLP为MoE"""
        if hasattr(self.base_model, 'model'):
            # LlamaForCausalLM
            layers = self.base_model.model.layers
        else:
            # LlamaModel
            layers = self.base_model.layers

        for layer_idx in self.config.moe_layers:
            if layer_idx < len(layers):
                original_mlp = layers[layer_idx].mlp
                moe_layer = SimplifiedMoELayer(original_mlp, self.config)
                layers[layer_idx].mlp = moe_layer
                print(f"✅ Replaced layer {layer_idx} MLP with SimplifiedMoE")

    def _freeze_non_lora_parameters(self):
        """冻结非LoRA参数"""
        for name, param in self.base_model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """前向传播"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 收集MoE辅助信息
        aux_losses = []
        expert_weights_list = []

        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        else:
            layers = self.base_model.layers

        for layer_idx in self.config.moe_layers:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                if hasattr(layer, '_moe_aux_info'):
                    aux_info = layer._moe_aux_info
                    if 'aux_loss' in aux_info:
                        aux_losses.append(aux_info['aux_loss'])
                    if 'expert_weights' in aux_info:
                        expert_weights_list.append(aux_info['expert_weights'])

        # 添加辅助损失到主损失
        if aux_losses and hasattr(outputs, 'loss') and outputs.loss is not None:
            total_aux_loss = sum(aux_losses)
            outputs.loss = outputs.loss + total_aux_loss

        # 添加专家权重信息（用于文化损失）
        if expert_weights_list:
            # 取最后一个MoE层的专家权重
            outputs.expert_weights = expert_weights_list[-1]

        return outputs

    def save_model(self, save_path: str):
        """保存LoRA权重"""
        import os

        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        # 只保存LoRA参数
        lora_state_dict = {}
        for name, param in self.base_model.named_parameters():
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

        for param in self.base_model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || "
              f"Total params: {total_params:,} || "
              f"Trainable%: {100 * trainable_params / total_params:.4f}%")

        # 详细统计
        lora_params = 0
        router_params = 0

        for name, param in self.base_model.named_parameters():
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

    # 修改前向传播以支持MoE辅助信息收集
    def _modified_forward_hook(module, input, output):
        """Hook函数，用于收集MoE层的辅助信息"""
        if hasattr(module, 'mlp') and isinstance(module.mlp, SimplifiedMoELayer):
            # 重新计算MLP输出并收集辅助信息
            hidden_states = input[0]  # 输入到layer的hidden_states

            # 先通过attention
            residual = hidden_states
            hidden_states = module.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, _, _ = module.self_attn(
                hidden_states=hidden_states,
                attention_mask=input[1] if len(input) > 1 else None,
                position_ids=input[2] if len(input) > 2 else None,
            )
            hidden_states = residual + hidden_states

            # MLP (MoE)
            residual = hidden_states
            hidden_states = module.post_attention_layernorm(hidden_states)
            mlp_output, aux_info = module.mlp(hidden_states)
            hidden_states = residual + mlp_output

            # 保存辅助信息
            module._moe_aux_info = aux_info

            # 返回修改后的输出
            return (hidden_states,)

    # 为指定的MoE层注册hook
    if hasattr(base_model, 'model'):
        layers = base_model.model.layers
    else:
        layers = base_model.layers

    for layer_idx in config.moe_layers:
        if layer_idx < len(layers):
            layers[layer_idx].register_forward_hook(_modified_forward_hook)

    # 创建适配器
    adapter = SimplifiedCultureMoEAdapter(base_model, config)

    return adapter