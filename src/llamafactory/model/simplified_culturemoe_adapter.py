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

        # 初始化 - 使用更小的scale防止梯度爆炸
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.002)  # 进一步降低初始化scale
        nn.init.zeros_(self.lora_B.weight)

    def to(self, *args, **kwargs):
        """重写to方法确保dtype正确传播"""
        super().to(*args, **kwargs)
        self.lora_A = self.lora_A.to(*args, **kwargs)
        self.lora_B = self.lora_B.to(*args, **kwargs)
        return self

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

        # 添加LoRA适配器（在正确设备和数据类型上）
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # 强制使用torch.float16以确保与模型一致
        if dtype == torch.float32:
            dtype = torch.float16

        self.lora_adapter = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        ).to(device=device, dtype=dtype)

        # 添加LayerNorm用于稳定专家输出（MoE标准做法）
        self.expert_ln = nn.LayerNorm(out_features, dtype=dtype, device=device)

    def forward(self, x):
        # 基础输出 + LoRA适配
        base_output = self.base_layer(x)
        lora_output = self.lora_adapter(x)
        combined_output = base_output + lora_output

        # 应用LayerNorm稳定输出（MoE标准做法）
        return self.expert_ln(combined_output)


class SimplifiedMoELayer(nn.Module):
    """简化的MoE层，基于MixLoRA实现"""

    def __init__(self, original_mlp, config: SimplifiedCultureMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_routing_experts
        self.top_k = config.top_k
        self.aux_loss_coef = config.aux_loss_coef

        # 保存最新的router_logits用于z-loss计算
        self.latest_router_logits = None
        # 保存batch和序列信息用于文化损失计算
        self.latest_batch_size = None
        self.latest_seq_len = None

        # 获取原始MLP的维度、设备和数据类型
        self.hidden_size = original_mlp.gate_proj.in_features
        self.intermediate_size = original_mlp.gate_proj.out_features
        self.device = original_mlp.gate_proj.weight.device
        self.dtype = original_mlp.gate_proj.weight.dtype

        # 强制使用torch.float16以确保与模型一致
        if self.dtype == torch.float32:
            self.dtype = torch.float16

        # 创建路由器（在正确设备和数据类型上）
        self.router = nn.Linear(self.hidden_size, self.num_experts, dtype=self.dtype, device=self.device)

        # 使用更小的初始化scale防止router logits爆炸
        nn.init.normal_(self.router.weight, mean=0.0, std=0.002)  # 进一步降低router初始化scale
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

        # 创建专家 - 每个专家都基于原始MLP + LoRA
        self.experts = nn.ModuleList([
            self._create_expert_from_mlp(original_mlp, config)
            for _ in range(self.num_experts)
        ])

        # 激活函数
        self.act_fn = original_mlp.act_fn

        # 添加pre-MoE LayerNorm用于稳定输入（Switch Transformer做法）
        self.pre_moe_ln = nn.LayerNorm(self.hidden_size, dtype=self.dtype, device=self.device)

        # 确保所有专家在正确设备和数据类型上
        self.experts = self.experts.to(device=self.device, dtype=self.dtype)

    def _create_expert_from_mlp(self, original_mlp, config):
        """从原始MLP创建专家"""
        expert = nn.Module()

        # 复制原始层并添加LoRA
        expert.gate_proj = SimplifiedMoEExpert(original_mlp.gate_proj, config)
        expert.up_proj = SimplifiedMoEExpert(original_mlp.up_proj, config)
        expert.down_proj = SimplifiedMoEExpert(original_mlp.down_proj, config)
        expert.act_fn = original_mlp.act_fn

        # 确保专家在正确设备和数据类型上
        expert = expert.to(device=self.device, dtype=self.dtype)

        return expert

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 应用pre-MoE LayerNorm稳定输入（Switch Transformer标准做法）
        hidden_states = self.pre_moe_ln(hidden_states)

        # 强制确保router的dtype与输入一致（激进的防御性检查）
        if hasattr(self, 'router'):
            if self.router.weight.dtype != hidden_states.dtype:
                # 重新创建router以确保正确的dtype
                self.router = nn.Linear(self.hidden_size, self.num_experts,
                                      dtype=hidden_states.dtype, device=hidden_states.device)
                # 使用小的初始化scale
                nn.init.normal_(self.router.weight, mean=0.0, std=0.002)
                if self.router.bias is not None:
                    nn.init.zeros_(self.router.bias)
                # 确保可训练
                for param in self.router.parameters():
                    param.requires_grad = True

        # 路由决策
        router_logits = self.router(hidden_states.view(-1, hidden_dim))  # [B*L, num_experts]

        # 关键修复：Clamp router logits防止爆炸（Switch Transformer标准做法）
        router_logits = torch.clamp(router_logits, -10.0, 10.0)

        # 保存router_logits的detached副本用于z-loss计算（避免梯度图问题）
        if self.training:
            self.latest_router_logits = router_logits.detach().clone()
            # 保存batch和序列信息
            self.latest_batch_size = batch_size
            self.latest_seq_len = seq_len

        router_probs = F.softmax(router_logits, dim=-1)

        # 检查softmax结果的数值稳定性
        if torch.isnan(router_probs).any() or torch.isinf(router_probs).any():
            # 如果出现NaN/Inf，使用均匀分布作为fallback
            router_probs = torch.ones_like(router_probs) / router_probs.shape[-1]

        # Top-K选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # 安全的重新归一化
        sum_probs = top_k_probs.sum(dim=-1, keepdim=True)
        # 避免除零错误
        sum_probs = torch.clamp(sum_probs, min=1e-8)
        top_k_probs = top_k_probs / sum_probs

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

        # 合并专家输出 - 使用与hidden_states相同的设备和dtype
        final_output = torch.zeros_like(hidden_states.view(-1, hidden_dim))

        # 确保有专家输出，否则返回原始输入
        if not expert_outputs:
            # 如果没有专家被选中，返回原始输入（fallback）
            return hidden_states

        for expert_mask, expert_output, expert_idx in expert_outputs:
            # 获取该专家的权重 - 强制使用torch.float16确保一致性
            expert_weights = torch.zeros(batch_size * seq_len, device=hidden_states.device, dtype=torch.float16)
            for k in range(self.top_k):
                mask_k = (top_k_indices[:, k] == expert_idx)
                expert_weights[mask_k] = top_k_probs[mask_k, k]

            # 应用权重 - 确保dtype匹配
            expert_weights_selected = expert_weights[expert_mask].unsqueeze(-1)
            # 将expert_weights转换为与expert_output相同的dtype
            expert_weights_selected = expert_weights_selected.to(dtype=expert_output.dtype)
            weighted_output = expert_output * expert_weights_selected
            final_output[expert_mask] += weighted_output

        final_output = final_output.view(batch_size, seq_len, hidden_dim)

        # 计算辅助损失（但不返回，为了兼容性）
        aux_loss = self._compute_aux_loss(router_probs)

        # z-loss将在get_accumulated_z_loss中从保存的router_logits计算

        # 为了兼容性，只返回输出张量，就像普通MLP一样
        return final_output

    def _compute_aux_loss(self, router_probs):
        """计算负载均衡辅助损失"""
        # 计算每个专家的使用频率
        expert_freq = router_probs.mean(dim=0)  # [num_experts]

        # 负载均衡损失：鼓励均匀分布
        aux_loss = torch.var(expert_freq) * self.aux_loss_coef

        return aux_loss

    def _compute_z_loss(self, router_logits):
        """计算z-loss用于稳定router logits（Google PaLM方法）"""
        # z-loss惩罚过大的logits值，防止router爆炸
        z_loss = 0.001 * (router_logits ** 2).mean()

        # 确保z_loss为float16并检查数值稳定性
        z_loss = z_loss.to(dtype=torch.float16)
        if torch.isnan(z_loss) or torch.isinf(z_loss):
            z_loss = torch.tensor(0.0, device=router_logits.device, dtype=torch.float16)

        return z_loss


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
        """替换所有层的MLP为MoE，与MixLoRA保持一致"""
        # 处理DDP包装的模型
        model_to_modify = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_modify, 'model'):
            # LlamaForCausalLM
            layers = model_to_modify.model.layers
        else:
            # LlamaModel
            layers = model_to_modify.layers

        print(f"🔄 Replacing ALL {len(layers)} layers with SimplifiedMoE (like MixLoRA)")

        # 替换所有层的MLP为MoE
        for layer_idx in range(len(layers)):
            original_mlp = layers[layer_idx].mlp
            moe_layer = SimplifiedMoELayer(original_mlp, self.config)
            layers[layer_idx].mlp = moe_layer
            if layer_idx % 5 == 0 or layer_idx == len(layers) - 1:  # 每5层打印一次进度
                print(f"✅ Replaced layer {layer_idx}/{len(layers)-1} MLP with SimplifiedMoE")

        # 更新配置中的moe_layers为所有层
        self.config.moe_layers = list(range(len(layers)))

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
        """确保所有参数在同一设备和数据类型上"""
        # 处理DDP包装的模型
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 获取基础模型的设备和数据类型
        base_param = next(model_to_check.parameters())
        base_device = base_param.device
        base_dtype = base_param.dtype

        # 强制使用torch.float16以确保与模型一致
        if base_dtype == torch.float32:
            base_dtype = torch.float16

        # 验证所有MoE层都在正确设备和数据类型上
        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        # 处理所有MoE层
        for layer_idx in range(len(layers)):
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, SimplifiedMoELayer):
                # 强制重新创建router以确保正确的dtype（强制使用float16）
                moe_layer.router = nn.Linear(moe_layer.hidden_size, moe_layer.num_experts, dtype=torch.float16, device=base_device)

                # 使用小的初始化scale防止router logits爆炸
                nn.init.normal_(moe_layer.router.weight, mean=0.0, std=0.002)
                if moe_layer.router.bias is not None:
                    nn.init.zeros_(moe_layer.router.bias)

                # 确保router参数可训练
                for param in moe_layer.router.parameters():
                    param.requires_grad = True

                # 确保所有专家也在正确设备和数据类型上
                moe_layer.experts = moe_layer.experts.to(device=base_device, dtype=base_dtype)

                # 强制重新创建LoRA适配器以确保正确的dtype
                for expert in moe_layer.experts:
                    if hasattr(expert, 'gate_proj') and hasattr(expert.gate_proj, 'lora_adapter'):
                        expert.gate_proj.lora_adapter = expert.gate_proj.lora_adapter.to(device=base_device, dtype=base_dtype)
                    if hasattr(expert, 'up_proj') and hasattr(expert.up_proj, 'lora_adapter'):
                        expert.up_proj.lora_adapter = expert.up_proj.lora_adapter.to(device=base_device, dtype=base_dtype)
                    if hasattr(expert, 'down_proj') and hasattr(expert.down_proj, 'lora_adapter'):
                        expert.down_proj.lora_adapter = expert.down_proj.lora_adapter.to(device=base_device, dtype=base_dtype)

                # 更新MoE层的设备和dtype记录
                moe_layer.device = base_device
                moe_layer.dtype = base_dtype

                if layer_idx % 10 == 0 or layer_idx == len(layers) - 1:  # 每10层打印一次进度
                    print(f"✅ Ensured device consistency for MoE layer {layer_idx}/{len(layers)-1} on device {base_device}")

    def get_expert_weights_for_culture_loss(self):
        """收集所有MoE层的专家权重用于文化损失计算"""
        # 处理DDP包装的模型
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        # 使用第一个有效的MoE层来确定目标维度
        target_batch_size = None
        target_seq_len = None
        num_experts = self.config.num_routing_experts

        # 收集所有层的路由概率，平均后作为专家权重
        all_router_probs = []

        for layer_idx in range(len(layers)):
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, SimplifiedMoELayer) and hasattr(moe_layer, 'latest_router_logits'):
                if moe_layer.latest_router_logits is not None:
                    router_logits = moe_layer.latest_router_logits  # [batch_size * seq_len, num_experts]

                    # 从第一个有效层确定目标维度
                    if target_batch_size is None and hasattr(moe_layer, 'latest_batch_size'):
                        target_batch_size = moe_layer.latest_batch_size
                        target_seq_len = moe_layer.latest_seq_len

                    # 从保存的router_logits计算概率分布
                    router_probs = F.softmax(router_logits, dim=-1)  # [batch_size * seq_len, num_experts]

                    # 简化处理：对所有token求平均，得到每个专家的全局使用概率
                    avg_probs = router_probs.mean(dim=0, keepdim=True)  # [1, num_experts]
                    all_router_probs.append(avg_probs)

        if all_router_probs and target_batch_size is not None:
            # 对所有层的专家权重求平均
            layer_avg_probs = torch.stack(all_router_probs, dim=0).mean(dim=0)  # [1, num_experts]

            # 扩展到目标batch_size
            expert_weights = layer_avg_probs.expand(target_batch_size, -1)  # [batch_size, num_experts]
            return expert_weights.to(dtype=torch.float16)
        elif all_router_probs:
            # 如果没有batch_size信息，假设batch_size=1
            layer_avg_probs = torch.stack(all_router_probs, dim=0).mean(dim=0)  # [1, num_experts]
            return layer_avg_probs.to(dtype=torch.float16)
        else:
            # 如果没有找到任何路由信息，返回None
            return None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """前向传播"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 为文化损失计算收集真实的专家权重
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

    def get_accumulated_z_loss(self):
        """从最新的router_logits计算z-loss"""
        # 处理DDP包装的模型
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        # 收集所有MoE层的z-loss
        total_z_loss = None
        moe_layer_count = 0

        # 处理所有层
        for layer_idx in range(len(layers)):
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, SimplifiedMoELayer) and hasattr(moe_layer, 'latest_router_logits'):
                if moe_layer.latest_router_logits is not None:
                    # 从保存的router_logits计算z-loss
                    z_loss = moe_layer._compute_z_loss(moe_layer.latest_router_logits)

                    if total_z_loss is None:
                        total_z_loss = z_loss
                    else:
                        # 确保设备和dtype一致
                        z_loss = z_loss.to(device=total_z_loss.device, dtype=total_z_loss.dtype)
                        total_z_loss += z_loss

                    moe_layer_count += 1

        # 如果没有找到任何MoE层，返回零损失
        if total_z_loss is None:
            # 使用模型参数的设备和dtype
            device = next(model_to_check.parameters()).device
            total_z_loss = torch.tensor(0.0, device=device, dtype=torch.float16)
        else:
            # 平均化z-loss
            if moe_layer_count > 1:
                total_z_loss = total_z_loss / moe_layer_count

            # 检查数值稳定性
            if torch.isnan(total_z_loss) or torch.isinf(total_z_loss):
                device = next(model_to_check.parameters()).device
                total_z_loss = torch.tensor(0.0, device=device, dtype=torch.float16)

        return total_z_loss

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
        print(f"  - MoE layers: ALL layers ({len(self.config.moe_layers)} layers total)")
        print(f"  - Routing experts per layer: {self.config.num_routing_experts}")
        print(f"  - Total experts: {len(self.config.moe_layers) * self.config.num_routing_experts}")


def create_simplified_culturemoe_model(base_model, config: SimplifiedCultureMoEConfig):
    """创建简化版CultureMoE模型"""

    # 创建适配器
    adapter = SimplifiedCultureMoEAdapter(base_model, config)

    return adapter