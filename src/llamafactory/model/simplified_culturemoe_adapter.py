# src/llamafactory/model/simplified_culturemoe_adapter.py
"""
简化版CultureMoE适配器
严格基于MixLoRA实现 + 文化损失
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

        # 初始化 - 与MixLoRA一致
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class SimplifiedMixLoRALayer(nn.Module):
    """简化的MixLoRA层，严格基于MixLoRA + 文化损失"""

    def __init__(self, original_mlp, config: SimplifiedCultureMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_routing_experts
        self.top_k = config.top_k

        # 保存原始MLP（冻结）
        self.shared_ffn = original_mlp
        # 冻结共享FFN
        for param in self.shared_ffn.parameters():
            param.requires_grad = False

        # 获取维度
        self.hidden_size = original_mlp.gate_proj.in_features
        self.intermediate_size = original_mlp.gate_proj.out_features
        self.device = original_mlp.gate_proj.weight.device
        self.dtype = original_mlp.gate_proj.weight.dtype

        # 简单的线性路由器（与MixLoRA一致）
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # 路由器初始化（与MixLoRA一致）
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        with torch.no_grad():
            self.router.weight.data.clamp_(-0.1, 0.1)

        # 创建LoRA专家（每个专家只是LoRA适配器）
        self.lora_experts = nn.ModuleList([
            self._create_lora_expert(config)
            for _ in range(self.num_experts)
        ])

        # 移动到正确设备
        self.router = self.router.to(device=self.device, dtype=self.dtype)
        self.lora_experts = self.lora_experts.to(device=self.device, dtype=self.dtype)

        # 保存最新的router logits用于文化损失
        self.latest_router_logits = None
        self.latest_batch_size = None
        self.latest_seq_len = None

    def _create_lora_expert(self, config):
        """创建LoRA专家（只是LoRA适配器）"""
        expert = nn.Module()

        # 为每个FFN层添加LoRA适配器
        expert.gate_proj_lora = LoRALinear(
            self.hidden_size, self.intermediate_size,
            config.lora_rank, config.lora_alpha, config.lora_dropout
        )
        expert.up_proj_lora = LoRALinear(
            self.hidden_size, self.intermediate_size,
            config.lora_rank, config.lora_alpha, config.lora_dropout
        )
        expert.down_proj_lora = LoRALinear(
            self.intermediate_size, self.hidden_size,
            config.lora_rank, config.lora_alpha, config.lora_dropout
        )

        return expert

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 输入预处理（与MixLoRA一致）
        hidden_states = torch.clamp(hidden_states, min=-5.0, max=5.0)
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # 确保router在正确设备和dtype上
        if self.router.weight.device != hidden_flat.device or self.router.weight.dtype != hidden_flat.dtype:
            self.router = self.router.to(device=hidden_flat.device, dtype=hidden_flat.dtype)

        # 限制router权重（与MixLoRA一致）
        with torch.no_grad():
            self.router.weight.data.clamp_(-2.0, 2.0)

        # 计算路由logits
        router_logits = self.router(hidden_flat)
        router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)

        # 保存router logits用于文化损失
        if self.training:
            self.latest_router_logits = router_logits.detach().clone()
            self.latest_batch_size = batch_size
            self.latest_seq_len = seq_len

        # 检查异常值
        if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
            router_logits = torch.zeros_like(router_logits)

        # Top-K选择（与MixLoRA一致）
        try:
            top_k_logits, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        except:
            selected_experts = torch.arange(self.top_k, device=router_logits.device).unsqueeze(0).expand(router_logits.size(0), -1)
            top_k_logits = router_logits[:, :self.top_k]

        # 数值稳定的softmax
        try:
            top_k_logits_max = top_k_logits.max(dim=-1, keepdim=True)[0]
            top_k_logits_stable = top_k_logits - top_k_logits_max
            expert_weights = F.softmax(top_k_logits_stable, dim=-1)
        except:
            expert_weights = torch.ones_like(top_k_logits) / self.top_k

        # 检查权重异常值
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            expert_weights = torch.ones_like(expert_weights) / self.top_k

        # 1. 共享FFN计算（与MixLoRA一致）
        shared_output = self._compute_shared_ffn(hidden_states)

        # 2. LoRA专家计算
        expert_output = self._compute_lora_experts(
            hidden_states, selected_experts, expert_weights
        )

        # 3. 组合输出
        final_output = shared_output + expert_output

        return final_output

    def _compute_shared_ffn(self, hidden_states):
        """计算共享FFN输出（与MixLoRA一致）"""
        # 使用原始的共享FFN
        gate_output = self.shared_ffn.act_fn(self.shared_ffn.gate_proj(hidden_states))
        up_output = self.shared_ffn.up_proj(hidden_states)
        shared_output = self.shared_ffn.down_proj(gate_output * up_output)
        return shared_output

    def _compute_lora_experts(self, hidden_states, selected_experts, expert_weights):
        """计算LoRA专家输出"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 初始化专家输出
        expert_output = torch.zeros_like(hidden_states)

        # 重塑为[batch_size * seq_len, hidden_dim]以匹配路由器输出
        hidden_flat = hidden_states.view(-1, hidden_dim)
        expert_output_flat = expert_output.view(-1, hidden_dim)

        # 为每个选中的专家计算LoRA输出
        for expert_idx in range(self.num_experts):
            # 找到使用当前专家的token位置
            expert_mask = (selected_experts == expert_idx).any(dim=-1)  # [batch_size * seq_len]

            if expert_mask.any():
                # 获取该专家的权重
                expert_weight_mask = (selected_experts == expert_idx).float()  # [batch_size * seq_len, top_k]
                weights = (expert_weights * expert_weight_mask).sum(dim=-1)  # [batch_size * seq_len]

                # 只对使用该专家的token进行计算
                if expert_mask.any():
                    # 计算LoRA专家输出
                    expert = self.lora_experts[expert_idx]

                    # LoRA FFN计算 - 对所有token计算，然后只应用到相关位置
                    gate_lora = expert.gate_proj_lora(hidden_states)
                    up_lora = expert.up_proj_lora(hidden_states)
                    down_lora = expert.down_proj_lora(self.shared_ffn.act_fn(gate_lora) * up_lora)

                    # 重塑并应用权重
                    down_lora_flat = down_lora.view(-1, hidden_dim)
                    weighted_output = down_lora_flat * weights.unsqueeze(-1)  # [batch_size * seq_len, hidden_dim]

                    # 累加到专家输出
                    expert_output_flat += weighted_output

        # 重塑回原始维度
        expert_output = expert_output_flat.view(batch_size, seq_len, hidden_dim)
        return expert_output

    def get_router_probs_for_culture_loss(self):
        """获取路由概率用于文化损失计算"""
        if self.latest_router_logits is None:
            return None

        # 计算概率分布
        router_probs = F.softmax(self.latest_router_logits, dim=-1)

        # 平均到batch维度
        avg_probs = router_probs.mean(dim=0, keepdim=True)  # [1, num_experts]

        # 扩展到batch size
        if self.latest_batch_size is not None:
            expert_weights = avg_probs.expand(self.latest_batch_size, -1)
            return expert_weights.to(dtype=torch.float16)
        else:
            return avg_probs.to(dtype=torch.float16)


class SimplifiedCultureMoEAdapter:
    """简化版CultureMoE适配器，严格基于MixLoRA + 文化损失"""

    def __init__(self, base_model, config: SimplifiedCultureMoEConfig):
        self.base_model = base_model
        self.config = config

        # 替换所有层的MLP为MixLoRA层
        self._replace_mlp_with_mixlora()

        # 冻结非LoRA参数
        self._freeze_non_lora_parameters()

        # 确保设备一致性
        self._ensure_device_consistency()

    def _replace_mlp_with_mixlora(self):
        """替换所有层的MLP为MixLoRA层"""
        # 处理DDP包装的模型
        model_to_modify = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_modify, 'model'):
            # LlamaForCausalLM
            layers = model_to_modify.model.layers
        else:
            # LlamaModel
            layers = model_to_modify.layers

        print(f"🔄 Replacing ALL {len(layers)} layers with MixLoRA (like MixLoRA)")

        # 替换所有层的MLP为MixLoRA层
        for layer_idx in range(len(layers)):
            original_mlp = layers[layer_idx].mlp
            mixlora_layer = SimplifiedMixLoRALayer(original_mlp, self.config)
            layers[layer_idx].mlp = mixlora_layer
            if layer_idx % 5 == 0 or layer_idx == len(layers) - 1:
                print(f"✅ Replaced layer {layer_idx}/{len(layers)-1} MLP with MixLoRA")

    def _freeze_non_lora_parameters(self):
        """冻结非LoRA参数"""
        model_to_freeze = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        for name, param in model_to_freeze.named_parameters():
            if 'lora_' not in name and 'router' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _ensure_device_consistency(self):
        """确保设备一致性"""
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 获取基础模型的设备和数据类型
        base_param = next(model_to_check.parameters())
        base_device = base_param.device
        base_dtype = base_param.dtype

        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        # 确保所有MixLoRA层在正确设备上
        for layer_idx in range(len(layers)):
            mixlora_layer = layers[layer_idx].mlp
            if isinstance(mixlora_layer, SimplifiedMixLoRALayer):
                mixlora_layer.router = mixlora_layer.router.to(device=base_device, dtype=base_dtype)
                mixlora_layer.lora_experts = mixlora_layer.lora_experts.to(device=base_device, dtype=base_dtype)

                if layer_idx % 10 == 0 or layer_idx == len(layers) - 1:
                    print(f"✅ Ensured device consistency for MixLoRA layer {layer_idx}/{len(layers)-1}")

    def get_expert_weights_for_culture_loss(self):
        """收集所有MixLoRA层的专家权重用于文化损失计算"""
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        # 收集所有层的路由概率
        all_router_probs = []
        target_batch_size = None

        for layer_idx in range(len(layers)):
            mixlora_layer = layers[layer_idx].mlp
            if isinstance(mixlora_layer, SimplifiedMixLoRALayer):
                expert_weights = mixlora_layer.get_router_probs_for_culture_loss()
                if expert_weights is not None:
                    if target_batch_size is None:
                        target_batch_size = expert_weights.shape[0]
                    all_router_probs.append(expert_weights)

        if all_router_probs and target_batch_size is not None:
            # 对所有层的专家权重求平均
            layer_avg_probs = torch.stack(all_router_probs, dim=0).mean(dim=0)
            return layer_avg_probs.to(dtype=torch.float16)
        else:
            return None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """前向传播"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 为文化损失计算收集专家权重
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            expert_weights = self.get_expert_weights_for_culture_loss()
            if expert_weights is not None:
                outputs.expert_weights = expert_weights

        return outputs

    def save_model(self, save_path: str):
        """保存LoRA权重"""
        import os

        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        # 只保存LoRA参数
        lora_state_dict = {}
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

    def get_accumulated_z_loss(self):
        """计算z-loss用于稳定路由器"""
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        total_z_loss = None
        mixlora_layer_count = 0

        # 收集所有MixLoRA层的z-loss
        for layer_idx in range(len(layers)):
            mixlora_layer = layers[layer_idx].mlp
            if isinstance(mixlora_layer, SimplifiedMixLoRALayer) and hasattr(mixlora_layer, 'latest_router_logits'):
                if mixlora_layer.latest_router_logits is not None:
                    # 计算z-loss：惩罚过大的logits值
                    z_loss = 0.001 * (mixlora_layer.latest_router_logits ** 2).mean()
                    z_loss = z_loss.to(dtype=torch.float16)

                    # 检查数值稳定性
                    if torch.isnan(z_loss) or torch.isinf(z_loss):
                        z_loss = torch.tensor(0.0, device=mixlora_layer.latest_router_logits.device, dtype=torch.float16)

                    if total_z_loss is None:
                        total_z_loss = z_loss
                    else:
                        total_z_loss += z_loss

                    mixlora_layer_count += 1

        # 如果没有找到任何MixLoRA层，返回零损失
        if total_z_loss is None:
            device = next(model_to_check.parameters()).device
            total_z_loss = torch.tensor(0.0, device=device, dtype=torch.float16)
        else:
            # 平均化z-loss
            if mixlora_layer_count > 1:
                total_z_loss = total_z_loss / mixlora_layer_count

        return total_z_loss

    def print_trainable_parameters(self):
        """打印可训练参数统计"""
        total_params = 0
        trainable_params = 0

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
        print(f"  - Architecture: MixLoRA + Culture Loss")
        print(f"  - Routing experts: {self.config.num_routing_experts}")


def create_simplified_culturemoe_model(base_model, config: SimplifiedCultureMoEConfig):
    """创建简化版CultureMoE模型"""
    adapter = SimplifiedCultureMoEAdapter(base_model, config)
    return adapter