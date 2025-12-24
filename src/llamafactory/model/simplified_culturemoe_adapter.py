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


class LoRAExpert(nn.Module):
    """LoRA专家层 - 为FFN的每个线性层添加LoRA分支"""

    def __init__(self, original_ffn, lora_rank: int = 16, lora_alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.original_ffn = original_ffn  # 保持原始FFN不变
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank

        # 获取原始FFN的维度和数据类型
        self.hidden_dim = original_ffn.gate_proj.in_features
        self.intermediate_dim = original_ffn.gate_proj.out_features

        # 为每个FFN线性层创建LoRA分支
        # gate_proj LoRA: hidden_dim -> intermediate_dim
        self.gate_lora_A = nn.Linear(self.hidden_dim, lora_rank, bias=False)
        self.gate_lora_B = nn.Linear(lora_rank, self.intermediate_dim, bias=False)

        # up_proj LoRA: hidden_dim -> intermediate_dim
        self.up_lora_A = nn.Linear(self.hidden_dim, lora_rank, bias=False)
        self.up_lora_B = nn.Linear(lora_rank, self.intermediate_dim, bias=False)

        # down_proj LoRA: intermediate_dim -> hidden_dim
        self.down_lora_A = nn.Linear(self.intermediate_dim, lora_rank, bias=False)
        self.down_lora_B = nn.Linear(lora_rank, self.hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # 🔧 获取原始FFN的数据类型和设备，确保LoRA层使用相同类型
        target_dtype = original_ffn.gate_proj.weight.dtype
        target_device = original_ffn.gate_proj.weight.device

        # 将所有LoRA层移动到正确设备和数据类型
        self.to(device=target_device, dtype=target_dtype)

        # LoRA权重初始化
        self._init_lora_weights()

    def _init_lora_weights(self):
        """LoRA权重初始化"""
        # A矩阵使用高斯初始化，B矩阵初始化为0
        nn.init.kaiming_uniform_(self.gate_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_lora_B.weight)

        nn.init.kaiming_uniform_(self.up_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_lora_B.weight)

        nn.init.kaiming_uniform_(self.down_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_lora_B.weight)

    def forward(self, x):
        """前向传播：原始FFN输出 + LoRA输出"""
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros_like(x)

        # 限制输入范围
        x = torch.clamp(x, min=-10.0, max=10.0)

        try:
            # 原始FFN前向传播
            gate_output = self.original_ffn.act_fn(self.original_ffn.gate_proj(x))
            up_output = self.original_ffn.up_proj(x)

            # LoRA分支计算
            gate_lora = self.gate_lora_B(self.gate_lora_A(x)) * self.scaling
            up_lora = self.up_lora_B(self.up_lora_A(x)) * self.scaling

            # 组合原始输出和LoRA输出
            gate_combined = gate_output + gate_lora
            up_combined = up_output + up_lora

            # 限制中间结果
            gate_combined = torch.clamp(gate_combined, min=-15.0, max=15.0)
            up_combined = torch.clamp(up_combined, min=-15.0, max=15.0)

            # FFN的激活和组合
            intermediate = gate_combined * up_combined
            intermediate = torch.clamp(intermediate, min=-20.0, max=20.0)

            # Dropout
            intermediate = self.dropout(intermediate)

            # down_proj: 原始 + LoRA
            down_original = self.original_ffn.down_proj(intermediate)
            down_lora = self.down_lora_B(self.down_lora_A(intermediate)) * self.scaling
            output = down_original + down_lora

            output = torch.clamp(output, min=-10.0, max=10.0)

            # 检查输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                return torch.zeros_like(x)

            return output

        except Exception as e:
            print(f"⚠️ LoRAExpert forward failed: {e}")
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


class MoEFFNLoRA(nn.Module):
    """MoE FFN层 - 使用LoRA专家，支持shared专家和gate机制"""

    def __init__(self, original_ffn, config: SimplifiedCultureMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.num_activated_experts = config.num_activated_experts
        self.original_ffn = original_ffn  # 保持原始FFN
        self.use_shared = config.use_shared
        self.use_gate = config.use_gate

        # 获取原始FFN的参数
        self.hidden_dim = original_ffn.gate_proj.in_features
        self.intermediate_dim = original_ffn.gate_proj.out_features

        # 创建路由器
        self.router = MoERouter(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            dropout=config.lora_dropout
        )

        # 创建路由LoRA专家
        self.experts = nn.ModuleList([
            LoRAExpert(
                original_ffn=original_ffn,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                dropout=config.lora_dropout
            ) for _ in range(self.num_experts)
        ])

        # 创建shared专家（如果启用）
        if self.use_shared:
            self.shared_expert = LoRAExpert(
                original_ffn=original_ffn,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )

        # 创建gate网络（如果启用）
        if self.use_gate:
            # gate网络：输入两个专家输出，输出融合权重
            self.gate_network = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 2, bias=False),  # 输出2个权重：[shared_weight, routed_weight]
                nn.Softmax(dim=-1)
            )
            # Gate网络权重初始化
            for layer in self.gate_network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

        # 🔧 获取原始FFN的数据类型和设备，统一设置所有MoE组件
        target_dtype = original_ffn.gate_proj.weight.dtype
        target_device = original_ffn.gate_proj.weight.device

        # 将所有MoE组件移动到正确设备和数据类型
        self.router.to(device=target_device, dtype=target_dtype)
        for expert in self.experts:
            expert.to(device=target_device, dtype=target_dtype)
        if self.use_shared:
            self.shared_expert.to(device=target_device, dtype=target_dtype)
        if self.use_gate:
            self.gate_network.to(device=target_device, dtype=target_dtype)

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
            print("⚠️ NaN/Inf in MoE input, using original FFN")
            return self.original_ffn(hidden_states)

        try:
            # 1. 计算原始FFN输出作为基础
            original_output = self.original_ffn(hidden_states)

            # 2. 路由计算（为路由专家）
            expert_weights, router_logits = self.router(hidden_states)

            # 保存专家权重用于文化损失（平均到序列维度）
            self.latest_expert_weights = expert_weights.mean(dim=1)  # [B, num_experts]

            # 3. Top-k选择（路由专家）
            if self.num_activated_experts == self.num_experts:
                # Dense模式：使用所有专家
                selected_expert_weights = expert_weights
            else:
                # Top-k模式
                k = min(self.num_activated_experts, self.num_experts)
                top_k_logits, top_k_indices = torch.topk(router_logits, k=k, dim=-1)  # [B, L, k]

                # 重新归一化top-k权重
                top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, L, k]

                # 创建稀疏权重矩阵
                selected_expert_weights = torch.zeros_like(expert_weights)  # [B, L, num_experts]
                selected_expert_weights.scatter_(-1, top_k_indices, top_k_weights)

            # 4. 计算路由专家的LoRA增量
            if self.num_activated_experts == self.num_experts:
                # Dense模式：计算所有专家的增量
                expert_deltas = []
                for expert_idx in range(self.num_experts):
                    expert_output = self.experts[expert_idx](hidden_states)
                    expert_delta = expert_output - original_output
                    expert_deltas.append(expert_delta)
                expert_deltas = torch.stack(expert_deltas, dim=-1)  # [B, L, H, num_experts]

                # 加权组合增量
                weights = selected_expert_weights.unsqueeze(-2)  # [B, L, 1, num_experts]
                routed_delta = torch.sum(expert_deltas * weights, dim=-1)  # [B, L, H]
            else:
                # Top-k模式：只计算被选中专家的增量
                routed_delta = torch.zeros_like(hidden_states)

                # 重塑为[B*L, H]便于处理
                hidden_flat = hidden_states.view(-1, self.hidden_dim)
                delta_flat = routed_delta.view(-1, self.hidden_dim)
                weights_flat = selected_expert_weights.view(-1, self.num_experts)

                # 对每个专家计算
                for expert_idx in range(self.num_experts):
                    expert_mask = weights_flat[:, expert_idx] > 1e-8

                    if expert_mask.any():
                        expert_input = hidden_flat[expert_mask]
                        original_input = original_output.view(-1, self.hidden_dim)[expert_mask]

                        expert_output = self.experts[expert_idx](expert_input)
                        expert_delta = expert_output - original_input

                        expert_weight = weights_flat[:, expert_idx][expert_mask].unsqueeze(-1)
                        delta_flat[expert_mask] += expert_delta * expert_weight

                routed_delta = delta_flat.view(batch_size, seq_len, self.hidden_dim)

            # 5. 计算shared专家输出（如果启用）
            if self.use_shared:
                shared_output = self.shared_expert(hidden_states)
                shared_delta = shared_output - original_output
            else:
                shared_delta = torch.zeros_like(hidden_states)

            # 6. 融合shared专家和路由专家的输出
            if self.use_shared and self.use_gate:
                # 使用gate网络进行融合
                shared_final = original_output + shared_delta
                routed_final = original_output + routed_delta

                # 将两个专家输出拼接作为gate输入
                gate_input = torch.cat([shared_final, routed_final], dim=-1)  # [B, L, 2*H]
                gate_weights = self.gate_network(gate_input)  # [B, L, 2]

                # 加权融合
                final_output = (gate_weights[..., 0:1] * shared_final +
                              gate_weights[..., 1:2] * routed_final)

            elif self.use_shared:
                # 固定权重融合：0.1*shared + 0.9*routed
                final_output = original_output + 0.1 * shared_delta + 0.9 * routed_delta

            else:
                # 仅使用路由专家
                final_output = original_output + routed_delta

            # 7. 最终检查
            if torch.isnan(final_output).any() or torch.isinf(final_output).any():
                print("⚠️ NaN/Inf in MoE output, using original FFN")
                final_output = self.original_ffn(hidden_states)

            return final_output

        except Exception as e:
            print(f"⚠️ MoE forward failed: {e}, using original FFN")
            return self.original_ffn(hidden_states)

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

        # 🔧 实现"单一真源"原则 - 彻底解包PeftModel
        self.backbone_model = self._extract_backbone_model(base_model)

        # 只替换最后两层的FFN为MoE
        self._replace_last_layers_with_moe()

        # 应用LoRA到注意力层
        if config.use_lora:
            self._apply_attention_lora()

        # 冻结非训练参数
        self._freeze_non_trainable_parameters()

        # 确保设备一致性
        self._ensure_device_consistency()

    def _extract_backbone_model(self, model):
        """从各种包装中提取真正的backbone模型"""
        # 处理DDP包装
        if hasattr(model, 'module'):
            model = model.module

        # 处理PeftModel包装
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                print("🔧 检测到PeftModel，提取backbone...")
                backbone = model.base_model.model
                print(f"✅ 成功提取backbone: {type(backbone)}")
                return backbone
        except ImportError:
            pass

        # 检查是否已经是backbone模型
        if hasattr(model, 'layers'):
            print(f"✅ 直接使用模型: {type(model)}")
            return model
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            print(f"✅ 提取model.layers: {type(model.model)}")
            return model.model
        else:
            print(f"⚠️ 未知模型结构，直接使用: {type(model)}")
            return model

    def _get_target_layers(self):
        """获取目标层索引（最后8层）"""
        # 🔧 使用单一真源 - 直接从backbone_model获取layers
        print(f"🔍 Using backbone model: {type(self.backbone_model)}")

        # 直接访问backbone模型的layers
        if hasattr(self.backbone_model, 'layers'):
            layers = self.backbone_model.layers
            print(f"✅ Found layers directly: {len(layers)} layers")
        elif hasattr(self.backbone_model, 'model') and hasattr(self.backbone_model.model, 'layers'):
            layers = self.backbone_model.model.layers
            print(f"✅ Found layers via model: {len(layers)} layers")
        else:
            raise AttributeError(f"Cannot find layers in backbone model type: {type(self.backbone_model)}")

        total_layers = len(layers)

        # 最后8层
        target_layers = list(range(total_layers - 8, total_layers))

        return layers, target_layers


    def _replace_last_layers_with_moe(self):
        """替换最后8层的FFN为LoRA MoE"""
        layers, target_layers = self._get_target_layers()

        print(f"🔄 Replacing FFN in last 8 layers ({target_layers}) with LoRA MoE (Pure LoRA MoE Architecture)")

        for layer_idx in target_layers:
            original_ffn = layers[layer_idx].mlp
            moe_ffn = MoEFFNLoRA(original_ffn, self.config)
            layers[layer_idx].mlp = moe_ffn

            # 构建配置信息字符串
            config_info = f"{self.config.num_moe_experts} experts, rank={self.config.lora_rank}"
            if self.config.use_shared:
                config_info += ", +shared"
            if self.config.use_gate:
                config_info += ", +gate"

            print(f"✅ Replaced layer {layer_idx} FFN with LoRA MoE ({config_info})")

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

        try:
            layers, target_layers = self._get_target_layers()

            # 确保MoE层在正确设备上
            for layer_idx in target_layers:
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA):
                    moe_layer = moe_layer.to(device=base_device)

            print(f"✅ Ensured device consistency for MoE layers on {base_device}")

        except Exception as e:
            print(f"⚠️ Device consistency check failed: {e}")
            print(f"✅ Skipping device consistency check - MoE layers already properly configured on {base_device}")

            # 作为fallback，直接遍历所有参数确保设备一致性
            for name, param in model_to_check.named_parameters():
                if param.device != base_device:
                    param.data = param.data.to(base_device)
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(base_device)

            print(f"✅ Fallback device consistency completed on {base_device}")

    def get_expert_weights_for_culture_loss(self):
        """获取专家权重用于文化损失计算"""
        layers, target_layers = self._get_target_layers()

        # 收集MoE层的专家权重
        expert_weights_list = []

        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFNLoRA) and moe_layer.latest_expert_weights is not None:
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
            if isinstance(moe_layer, MoEFFNLoRA):
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
        print(f"  - Architecture: Pure LoRA MoE (Last 8 Layers FFN)")
        print(f"  - MoE experts: {self.config.num_moe_experts}")
        print(f"  - Activated experts: {self.config.num_activated_experts}")
        print(f"  - LoRA rank: {self.config.lora_rank}")
        print(f"  - LoRA alpha: {self.config.lora_alpha}")
        print(f"  - Shared expert: {'enabled' if self.config.use_shared else 'disabled'}")
        print(f"  - Gate network: {'enabled' if self.config.use_gate else 'disabled'}")
        if self.config.use_shared and not self.config.use_gate:
            print(f"  - Fusion weights: 0.1*shared + 0.9*routed")


def create_simplified_culturemoe_model(base_model, config: SimplifiedCultureMoEConfig):
    """创建简化版CultureMoE模型"""
    adapter = SimplifiedCultureMoEAdapter(base_model, config)
    return adapter