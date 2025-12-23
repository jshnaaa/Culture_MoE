# src/llamafactory/model/simplified_culturemoe_adapter.py
"""
简化版CultureMoE适配器
纯MoE架构：只在最后两层替换FFN为MoE结构
不使用MixLoRA，仅在指定层使用MoE专家
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import math

from .simplified_culturemoe import SimplifiedCultureMoEConfig


class MoEExpert(nn.Module):
    """MoE专家层 - 完整的FFN结构"""

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # 完整的FFN结构（与原始FFN相同）
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = act_fn  # 使用原始模型的激活函数
        self.dropout = nn.Dropout(dropout)

        # 保守的权重初始化
        self._init_weights()

    def _init_weights(self):
        """保守的权重初始化"""
        # 使用较小的标准差初始化
        std = 0.02 / math.sqrt(2 * self.hidden_dim)

        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std * 0.5)  # 输出层更保守

    def forward(self, x):
        """前向传播"""
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros_like(x)

        # 限制输入范围
        x = torch.clamp(x, min=-10.0, max=10.0)

        try:
            # FFN计算
            gate_output = self.act_fn(self.gate_proj(x))
            up_output = self.up_proj(x)

            # 限制中间结果
            gate_output = torch.clamp(gate_output, min=-15.0, max=15.0)
            up_output = torch.clamp(up_output, min=-15.0, max=15.0)

            # 组合
            intermediate = gate_output * up_output
            intermediate = torch.clamp(intermediate, min=-20.0, max=20.0)

            # Dropout
            intermediate = self.dropout(intermediate)

            # 最终投影
            output = self.down_proj(intermediate)
            output = torch.clamp(output, min=-10.0, max=10.0)

            # 检查输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                return torch.zeros_like(x)

            return output

        except Exception as e:
            print(f"⚠️ MoEExpert forward failed: {e}")
            return torch.zeros_like(x)


class MoERouter(nn.Module):
    """MoE路由器"""

    def __init__(self, hidden_dim: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # 路由器网络
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # 保守的初始化
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)

    def forward(self, x):
        """
        路由计算

        Args:
            x: [B, L, H] 输入隐藏状态

        Returns:
            expert_weights: [B, L, num_experts] 专家权重
            router_logits: [B, L, num_experts] 路由logits
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 限制输入
        x = torch.clamp(x, min=-10.0, max=10.0)

        try:
            # 计算路由logits
            router_logits = self.router(x)  # [B, L, num_experts]
            router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)

            # 数值稳定的softmax
            max_logits = torch.max(router_logits, dim=-1, keepdim=True)[0]
            shifted_logits = router_logits - max_logits
            shifted_logits = torch.clamp(shifted_logits, min=-20.0, max=0.0)

            # 计算专家权重
            expert_weights = F.softmax(shifted_logits, dim=-1)

            # 检查结果
            if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
                # 使用均匀分布作为fallback
                expert_weights = torch.ones_like(expert_weights) / self.num_experts
                router_logits = torch.zeros_like(router_logits)

            return expert_weights, router_logits

        except Exception as e:
            print(f"⚠️ Router forward failed: {e}")
            # 安全的fallback
            expert_weights = torch.ones(batch_size, seq_len, self.num_experts,
                                      device=x.device, dtype=x.dtype) / self.num_experts
            router_logits = torch.zeros(batch_size, seq_len, self.num_experts,
                                      device=x.device, dtype=x.dtype)
            return expert_weights, router_logits


class MoEFFN(nn.Module):
    """MoE FFN层 - 替换原始FFN"""

    def __init__(self, original_ffn, config: SimplifiedCultureMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.num_activated_experts = config.num_activated_experts

        # 获取原始FFN的参数
        self.hidden_dim = original_ffn.gate_proj.in_features
        self.intermediate_dim = original_ffn.gate_proj.out_features
        self.act_fn = original_ffn.act_fn

        # 创建路由器
        self.router = MoERouter(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            dropout=config.lora_dropout
        )

        # 创建专家
        self.experts = nn.ModuleList([
            MoEExpert(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                act_fn=self.act_fn,
                dropout=config.lora_dropout
            ) for _ in range(self.num_experts)
        ])

        # 保存最新的专家权重用于文化损失
        self.latest_expert_weights = None

    def forward(self, hidden_states):
        """
        前向传播

        Args:
            hidden_states: [B, L, H] 输入隐藏状态

        Returns:
            output: [B, L, H] 输出隐藏状态
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 检查输入
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            print("⚠️ NaN/Inf in MoE input, using zeros")
            return torch.zeros_like(hidden_states)

        try:
            # 1. 路由计算
            expert_weights, router_logits = self.router(hidden_states)

            # 保存专家权重用于文化损失（平均到序列维度）
            self.latest_expert_weights = expert_weights.mean(dim=1)  # [B, num_experts]

            # 2. Top-k选择
            if self.num_activated_experts == self.num_experts:
                # Dense模式：使用所有专家
                selected_expert_weights = expert_weights
                selected_experts = torch.arange(self.num_experts, device=hidden_states.device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                k = self.num_experts
            else:
                # Top-k模式
                k = min(self.num_activated_experts, self.num_experts)
                top_k_logits, top_k_indices = torch.topk(router_logits, k=k, dim=-1)  # [B, L, k]

                # 重新归一化top-k权重
                top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, L, k]

                # 创建稀疏权重矩阵
                selected_expert_weights = torch.zeros_like(expert_weights)  # [B, L, num_experts]
                selected_expert_weights.scatter_(-1, top_k_indices, top_k_weights)
                selected_experts = top_k_indices

            # 3. 专家计算
            if self.num_activated_experts == self.num_experts:
                # Dense模式：计算所有专家
                expert_outputs = []
                for expert_idx in range(self.num_experts):
                    expert_output = self.experts[expert_idx](hidden_states)  # [B, L, H]
                    expert_outputs.append(expert_output)
                expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, H, num_experts]

                # 加权组合
                weights = selected_expert_weights.unsqueeze(-2)  # [B, L, 1, num_experts]
                final_output = torch.sum(expert_outputs * weights, dim=-1)  # [B, L, H]
            else:
                # Top-k模式：只计算被选中的专家
                final_output = torch.zeros_like(hidden_states)

                # 重塑为[B*L, H]便于处理
                hidden_flat = hidden_states.view(-1, hidden_dim)  # [B*L, H]
                output_flat = final_output.view(-1, hidden_dim)  # [B*L, H]
                weights_flat = selected_expert_weights.view(-1, self.num_experts)  # [B*L, num_experts]

                # 对每个专家计算
                for expert_idx in range(self.num_experts):
                    # 找到使用此专家的token
                    expert_mask = weights_flat[:, expert_idx] > 1e-8

                    if expert_mask.any():
                        # 获取对应的输入
                        expert_input = hidden_flat[expert_mask]  # [num_tokens, H]

                        # 计算专家输出
                        expert_output = self.experts[expert_idx](expert_input)  # [num_tokens, H]

                        # 获取权重
                        expert_weight = weights_flat[:, expert_idx][expert_mask].unsqueeze(-1)  # [num_tokens, 1]

                        # 加权累加
                        output_flat[expert_mask] += expert_output * expert_weight

                # 重塑回原始形状
                final_output = output_flat.view(batch_size, seq_len, hidden_dim)

            # 4. 最终检查
            if torch.isnan(final_output).any() or torch.isinf(final_output).any():
                print("⚠️ NaN/Inf in MoE output, using zeros")
                final_output = torch.zeros_like(hidden_states)

            return final_output

        except Exception as e:
            print(f"⚠️ MoE forward failed: {e}, using zeros")
            return torch.zeros_like(hidden_states)

    def get_aux_loss(self):
        """计算辅助损失"""
        if self.latest_expert_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        try:
            # 负载均衡损失
            expert_usage = self.latest_expert_weights.mean(dim=0)  # [num_experts]
            target_usage = torch.ones_like(expert_usage) / self.num_experts
            balance_loss = F.mse_loss(expert_usage, target_usage)

            # 检查数值稳定性
            if torch.isnan(balance_loss) or torch.isinf(balance_loss):
                balance_loss = torch.tensor(0.0, device=balance_loss.device)

            return balance_loss * 0.01  # 小的权重

        except Exception as e:
            print(f"⚠️ Aux loss computation failed: {e}")
            return torch.tensor(0.0, device=next(self.parameters()).device)


class SimplifiedCultureMoEAdapter:
    """简化版CultureMoE适配器 - 纯MoE架构"""

    def __init__(self, base_model, config: SimplifiedCultureMoEConfig):
        self.base_model = base_model
        self.config = config

        # 只替换最后两层的FFN为MoE
        self._replace_last_layers_with_moe()

        # 应用LoRA到注意力层
        if config.use_lora:
            self._apply_attention_lora()

        # 冻结非训练参数
        self._freeze_non_trainable_parameters()

        # 确保设备一致性
        self._ensure_device_consistency()

    def _get_target_layers(self):
        """获取目标层索引（最后两层）"""
        # 处理DDP包装的模型
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_check, 'model'):
            # LlamaForCausalLM
            layers = model_to_check.model.layers
        else:
            # LlamaModel
            layers = model_to_check.layers

        total_layers = len(layers)

        # 最后两层
        target_layers = [total_layers - 2, total_layers - 1]

        return layers, target_layers

    def _replace_last_layers_with_moe(self):
        """只替换最后两层的FFN为MoE"""
        layers, target_layers = self._get_target_layers()

        print(f"🔄 Replacing FFN in last 2 layers ({target_layers}) with MoE (Pure MoE Architecture)")

        for layer_idx in target_layers:
            original_ffn = layers[layer_idx].mlp
            moe_ffn = MoEFFN(original_ffn, self.config)
            layers[layer_idx].mlp = moe_ffn
            print(f"✅ Replaced layer {layer_idx} FFN with MoE ({self.config.num_moe_experts} experts)")

    def _apply_attention_lora(self):
        """应用LoRA到注意力层"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # LoRA配置
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 只针对注意力层
                bias="none",
            )

            # 应用LoRA
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"✅ Applied LoRA to attention layers (rank={self.config.lora_rank})")

        except ImportError:
            print("⚠️ PEFT not available, skipping LoRA")
        except Exception as e:
            print(f"⚠️ LoRA application failed: {e}")

    def _freeze_non_trainable_parameters(self):
        """冻结非训练参数"""
        model_to_freeze = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        for name, param in model_to_freeze.named_parameters():
            # 只训练LoRA参数和MoE参数
            if any(keyword in name.lower() for keyword in ['lora', 'experts', 'router']):
                param.requires_grad = True
            else:
                param.requires_grad = False

        print("✅ Frozen non-trainable parameters (only LoRA + MoE trainable)")

    def _ensure_device_consistency(self):
        """确保设备一致性"""
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 获取基础模型的设备
        base_device = next(model_to_check.parameters()).device

        layers, target_layers = self._get_target_layers()

        # 确保MoE层在正确设备上
        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFN):
                moe_layer = moe_layer.to(device=base_device)

        print(f"✅ Ensured device consistency for MoE layers on {base_device}")

    def get_expert_weights_for_culture_loss(self):
        """获取专家权重用于文化损失计算"""
        layers, target_layers = self._get_target_layers()

        # 收集MoE层的专家权重
        expert_weights_list = []

        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFN) and moe_layer.latest_expert_weights is not None:
                expert_weights_list.append(moe_layer.latest_expert_weights)

        if expert_weights_list:
            # 对所有MoE层的权重求平均
            avg_expert_weights = torch.stack(expert_weights_list, dim=0).mean(dim=0)
            return avg_expert_weights.to(dtype=torch.float16)
        else:
            return None

    def get_accumulated_z_loss(self):
        """计算累积的z-loss"""
        layers, target_layers = self._get_target_layers()

        total_aux_loss = None
        moe_layer_count = 0

        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFN):
                aux_loss = moe_layer.get_aux_loss()
                if total_aux_loss is None:
                    total_aux_loss = aux_loss
                else:
                    total_aux_loss += aux_loss
                moe_layer_count += 1

        if total_aux_loss is None:
            device = next(self.base_model.parameters()).device
            total_aux_loss = torch.tensor(0.0, device=device, dtype=torch.float16)
        elif moe_layer_count > 1:
            total_aux_loss = total_aux_loss / moe_layer_count

        return total_aux_loss.to(dtype=torch.float16)

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
        """保存模型权重"""
        import os

        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        # 保存LoRA权重（如果有）
        if hasattr(self.base_model, 'save_pretrained'):
            lora_path = os.path.join(save_path, 'lora_weights')
            self.base_model.save_pretrained(lora_path)

        # 保存MoE权重
        moe_state_dict = {}
        model_to_save = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        for name, param in model_to_save.named_parameters():
            if any(keyword in name.lower() for keyword in ['experts', 'router']) and param.requires_grad:
                moe_state_dict[name] = param.data

        if moe_state_dict:
            moe_path = os.path.join(save_path, 'moe_weights.pt')
            torch.save(moe_state_dict, moe_path)

        # 保存配置
        config_path = os.path.join(save_path, 'simplified_culturemoe_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"✅ Simplified CultureMoE (Pure MoE) weights saved to {save_path}")

    def print_trainable_parameters(self):
        """打印可训练参数统计"""
        total_params = 0
        trainable_params = 0
        lora_params = 0
        moe_params = 0

        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        for name, param in model_to_check.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'lora' in name.lower():
                    lora_params += param.numel()
                elif any(keyword in name.lower() for keyword in ['experts', 'router']):
                    moe_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || "
              f"Total params: {total_params:,} || "
              f"Trainable%: {100 * trainable_params / total_params:.4f}%")

        print(f"  - LoRA params (attention): {lora_params:,}")
        print(f"  - MoE params (experts+router): {moe_params:,}")
        print(f"  - Architecture: Pure MoE (Last 2 Layers FFN Only)")
        print(f"  - MoE experts: {self.config.num_moe_experts}")
        print(f"  - Activated experts: {self.config.num_activated_experts}")


def create_simplified_culturemoe_model(base_model, config: SimplifiedCultureMoEConfig):
    """创建简化版CultureMoE模型"""
    adapter = SimplifiedCultureMoEAdapter(base_model, config)
    return adapter