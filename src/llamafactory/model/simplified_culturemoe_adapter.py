# src/llamafactory/model/simplified_culturemoe_adapter.py
"""
简化版CultureMoE适配器
纯MoE架构：在所有层替换FFN为MoE结构
不使用MixLoRA，在所有层使用MoE专家
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import math

from .simplified_culturemoe import SimplifiedCultureMoEConfig


class LoRAExpert(nn.Module):
    """LoRA专家层 - 标准LoRA实现：output = original + lora_delta"""

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn,
                 lora_rank: int = 16, lora_alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn

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
        """前向传播：计算LoRA增量"""
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ LoRAExpert input NaN/Inf detected, returning zero delta")
            return x * 0.0

        # 限制输入范围
        x = torch.clamp(x, min=-10.0, max=10.0)

        try:
            # LoRA分支计算 - 只计算增量
            gate_lora = self.gate_lora_B(self.gate_lora_A(x)) * self.scaling
            up_lora = self.up_lora_B(self.up_lora_A(x)) * self.scaling

            # 限制中间结果
            gate_lora = torch.clamp(gate_lora, min=-15.0, max=15.0)
            up_lora = torch.clamp(up_lora, min=-15.0, max=15.0)

            # FFN的激活和组合（LoRA增量）
            intermediate_lora = self.act_fn(gate_lora) * up_lora
            intermediate_lora = torch.clamp(intermediate_lora, min=-20.0, max=20.0)

            # Dropout
            intermediate_lora = self.dropout(intermediate_lora)

            # down_proj LoRA增量
            lora_delta = self.down_lora_B(self.down_lora_A(intermediate_lora)) * self.scaling
            lora_delta = torch.clamp(lora_delta, min=-10.0, max=10.0)

            # 检查输出
            if torch.isnan(lora_delta).any() or torch.isinf(lora_delta).any():
                print(f"⚠️ LoRAExpert NaN/Inf detected, returning zero delta")
                return x * 0.0

            return lora_delta

        except Exception as e:
            print(f"⚠️ LoRAExpert forward failed: {e}, returning zero delta")
            return x * 0.0


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
                expert_weights = expert_weights * 0.0 + (1.0 / self.num_experts)
                router_logits = router_logits * 0.0

            return expert_weights, router_logits

        except Exception as e:
            print(f"⚠️ Router forward failed: {e}")
            # 安全的fallback - 使用x.new_*方法保持梯度连接
            expert_weights = x.new_ones(batch_size, seq_len, self.num_experts) / self.num_experts
            router_logits = x.new_zeros(batch_size, seq_len, self.num_experts)
            return expert_weights, router_logits


class MoEFFNLoRA(nn.Module):
    """MoE FFN层 - 使用LoRA专家，支持shared专家和gate机制"""

    def __init__(self, original_ffn, config: SimplifiedCultureMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.num_activated_experts = config.num_activated_experts
        self.original_ffn = original_ffn  # 保持原始FFN（只计算一次）
        self.use_shared = config.use_shared
        self.use_gate = config.use_gate

        # 🔧 消融实验控制标志
        self.ablation_disable_shared = False
        self.ablation_disable_gate = False

        # 获取原始FFN的参数
        self.hidden_dim = original_ffn.gate_proj.in_features
        self.intermediate_dim = original_ffn.gate_proj.out_features

        # 创建路由器
        self.router = MoERouter(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            dropout=config.lora_dropout
        )

        # 🔧 修复：创建独立的路由LoRA专家（不共享original_ffn）
        self.experts = nn.ModuleList([
            LoRAExpert(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                act_fn=original_ffn.act_fn,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                dropout=config.lora_dropout
            ) for _ in range(self.num_experts)
        ])

        # 🔧 修复：创建独立的shared专家（如果启用）
        if self.use_shared:
            self.shared_expert = LoRAExpert(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                act_fn=original_ffn.act_fn,
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

        # 🆕 保存shared专家的输出用于文化损失计算
        self.latest_shared_outputs = None

        # 🔧 关键修复：保存当前批次的梯度连接数据
        self.current_expert_weights = None
        self.current_shared_outputs = None

    def forward(self, hidden_states):
        """
        前向传播 - 简化版，移除MASK机制

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
            # 🔧 关键修复：清空当前批次数据，确保不使用过期的梯度数据
            self.current_expert_weights = None
            self.current_shared_outputs = None

            # 1. 计算原始FFN输出作为基础
            original_output = self.original_ffn(hidden_states)

            # 2. 路由计算（为路由专家）
            expert_weights, router_logits = self.router(hidden_states)

            # 🔧 关键修复：保存当前批次的有梯度expert_weights
            if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
                # 如果包含NaN，使用均匀分布（保持梯度连接）
                uniform_weights = expert_weights * 0.0 + (1.0 / expert_weights.shape[2])
                self.current_expert_weights = uniform_weights.mean(dim=1)  # [B, num_experts] - 有梯度
                self.latest_expert_weights = uniform_weights.mean(dim=1).detach()  # 无梯度版本用于统计
                print(f"🔍 MoE forward debug (NaN case): current_expert_weights set, requires_grad: {self.current_expert_weights.requires_grad}")
            else:
                self.current_expert_weights = expert_weights.mean(dim=1)  # [B, num_experts] - 有梯度
                self.latest_expert_weights = expert_weights.mean(dim=1).detach()  # 无梯度版本用于统计
                print(f"🔍 MoE forward debug: current_expert_weights set, shape: {self.current_expert_weights.shape}, requires_grad: {self.current_expert_weights.requires_grad}")

            # 验证expert_weights的梯度状态
            print(f"  original expert_weights: requires_grad: {expert_weights.requires_grad}, grad_fn: {expert_weights.grad_fn is not None}")
            print(f"  current_expert_weights: requires_grad: {self.current_expert_weights.requires_grad}, grad_fn: {self.current_expert_weights.grad_fn is not None}")

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
                selected_expert_weights = expert_weights * 0.0  # [B, L, num_experts]
                selected_expert_weights.scatter_(-1, top_k_indices, top_k_weights)

            # 4. 计算路由专家的LoRA增量（新架构：专家只计算LoRA增量）
            if self.num_activated_experts == self.num_experts:
                # Dense模式：计算所有专家的LoRA增量
                expert_deltas = []
                for expert_idx in range(self.num_experts):
                    expert_lora_delta = self.experts[expert_idx](hidden_states)  # 专家直接返回LoRA增量
                    expert_deltas.append(expert_lora_delta)
                expert_deltas = torch.stack(expert_deltas, dim=-1)  # [B, L, H, num_experts]

                # 加权组合LoRA增量
                weights = selected_expert_weights.unsqueeze(-2)  # [B, L, 1, num_experts]
                routed_delta = torch.sum(expert_deltas * weights, dim=-1)  # [B, L, H]
            else:
                # Top-k模式：只计算被选中专家的LoRA增量
                routed_delta = hidden_states * 0.0

                # 重塑为[B*L, H]便于处理
                hidden_flat = hidden_states.view(-1, self.hidden_dim)
                delta_flat = routed_delta.view(-1, self.hidden_dim)
                weights_flat = selected_expert_weights.view(-1, self.num_experts)

                # 对每个专家计算LoRA增量
                for expert_idx in range(self.num_experts):
                    expert_mask = weights_flat[:, expert_idx] > 1e-8

                    if expert_mask.any():
                        expert_input = hidden_flat[expert_mask]
                        # 🔧 新架构：专家直接返回LoRA增量，无需减去原始输出
                        expert_lora_delta = self.experts[expert_idx](expert_input)

                        expert_weight = weights_flat[:, expert_idx][expert_mask].unsqueeze(-1)
                        delta_flat[expert_mask] += expert_lora_delta * expert_weight

                routed_delta = delta_flat.view(batch_size, seq_len, self.hidden_dim)

            # 5. 计算shared专家LoRA增量（如果启用且未被消融）
            if self.use_shared and not self.ablation_disable_shared:
                shared_delta = self.shared_expert(hidden_states)  # 🔧 新架构：shared专家直接返回LoRA增量

                # 🔧 关键修复：保存当前批次的有梯度shared_outputs
                # shared专家的"输出"实际是原始FFN + LoRA增量
                shared_output_for_culture_loss = original_output + shared_delta
                self.current_shared_outputs = shared_output_for_culture_loss.mean(dim=1)  # [B, H] - 有梯度
                self.latest_shared_outputs = shared_output_for_culture_loss.mean(dim=1).detach()  # 无梯度版本用于统计
            else:
                shared_delta = hidden_states * 0.0
                self.current_shared_outputs = None
                self.latest_shared_outputs = None

            # 6. 融合shared专家和路由专家的输出
            use_shared_effective = self.use_shared and not self.ablation_disable_shared
            use_gate_effective = self.use_gate and not self.ablation_disable_gate

            if use_shared_effective and use_gate_effective:
                # 使用gate网络进行融合
                shared_final = original_output + shared_delta
                routed_final = original_output + routed_delta

                # 将两个专家输出拼接作为gate输入
                gate_input = torch.cat([shared_final, routed_final], dim=-1)  # [B, L, 2*H]
                gate_weights = self.gate_network(gate_input)  # [B, L, 2]

                # 加权融合
                final_output = (gate_weights[..., 0:1] * shared_final +
                              gate_weights[..., 1:2] * routed_final)

            elif use_shared_effective:
                # 当有shared专家但没有gate时，使用简单相加（按照用户要求）
                final_output = original_output + shared_delta + routed_delta

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
            # 🔧 修复梯度问题：寻找一个requires_grad=True的参数创建连接到计算图的零损失
            for param in self.parameters():
                if param.requires_grad:
                    return param.sum() * 0.0
            # 如果没有可训练参数，创建一个简单的零tensor，使用float32确保类型一致
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32, requires_grad=True)

        try:
            # 🔧 添加数值稳定性检查：在计算前检查latest_expert_weights
            if torch.isnan(self.latest_expert_weights).any() or torch.isinf(self.latest_expert_weights).any():
                # 如果expert_weights包含NaN或Inf，使用零损失
                for param in self.parameters():
                    if param.requires_grad:
                        return param.sum() * 0.0
                return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)

            # 负载均衡损失
            expert_usage = self.latest_expert_weights.mean(dim=0)  # [num_experts]
            target_usage = expert_usage * 0.0 + (1.0 / self.num_experts)
            balance_loss = F.mse_loss(expert_usage, target_usage)

            # 检查数值稳定性
            if torch.isnan(balance_loss) or torch.isinf(balance_loss):
                # 🔧 修复梯度问题：寻找一个requires_grad=True的参数创建连接到计算图的零损失
                for param in self.parameters():
                    if param.requires_grad:
                        balance_loss = param.sum() * 0.0
                        break
                else:
                    # 如果没有可训练参数，创建一个简单的零tensor，使用float32确保类型一致
                    balance_loss = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32, requires_grad=True)

            return balance_loss * 0.01  # 小的权重

        except Exception as e:
            # 🔧 减少重复打印：使用类属性来限制错误信息打印次数
            if not hasattr(self.__class__, '_aux_loss_error_count'):
                self.__class__._aux_loss_error_count = 0

            if self.__class__._aux_loss_error_count < 5:  # 只打印前5次错误
                print(f"⚠️ Aux loss computation failed: {e}")
                self.__class__._aux_loss_error_count += 1
            elif self.__class__._aux_loss_error_count == 5:
                print(f"⚠️ Aux loss computation failed (suppressing further similar errors): {e}")
                self.__class__._aux_loss_error_count += 1

            # 🔧 修复梯度问题：寻找一个requires_grad=True的参数创建连接到计算图的零损失
            for param in self.parameters():
                if param.requires_grad:
                    return param.sum() * 0.0
            # 如果没有可训练参数，创建一个简单的零tensor，使用float32确保类型一致
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32, requires_grad=True)



class SimplifiedCultureMoEAdapter:
    """简化版CultureMoE适配器 - 纯MoE架构"""

    def __init__(self, base_model, config: SimplifiedCultureMoEConfig):
        self.base_model = base_model
        self.config = config

        # 🔧 缓存解包结果，避免重复调用
        self._cached_layers = None
        self._cached_target_layers = None

        # 🔧 修复架构问题：正确的初始化顺序
        # 1. 先应用LoRA到注意力层（如果启用）
        if config.use_lora:
            self._apply_attention_lora()

        # 2. 然后在包装后的模型上替换FFN为MoE
        self._replace_all_layers_with_moe()

        # 3. 冻结非训练参数
        self._freeze_non_trainable_parameters()

        # 4. 确保设备一致性
        self._ensure_device_consistency()


    def _get_target_layers(self, force_refresh=False):
        """获取目标层索引（所有层）"""
        # 🔧 使用缓存避免重复解包和打印
        if not force_refresh and self._cached_layers is not None and self._cached_target_layers is not None:
            return self._cached_layers, self._cached_target_layers

        # 🔧 关键修复：正确处理所有包装层，确保访问到实际的训练模型
        actual_model = self.base_model

        # 只在首次调用或强制刷新时打印调试信息
        if self._cached_layers is None or force_refresh:
            print(f"🔧 开始解包模型，初始类型: {type(actual_model)}")

        # 处理DDP包装
        if hasattr(actual_model, 'module'):
            actual_model = actual_model.module
            if self._cached_layers is None or force_refresh:
                print(f"🔧 检测到DDP包装，解包后: {type(actual_model)}")

        # 处理PeftModel包装（LoRA包装）
        if hasattr(actual_model, 'base_model'):
            if hasattr(actual_model.base_model, 'model'):
                # PeftModel -> base_model.model
                actual_model = actual_model.base_model.model
                if self._cached_layers is None or force_refresh:
                    print(f"🔧 检测到PeftModel包装，解包到base_model.model: {type(actual_model)}")
            else:
                # PeftModel -> base_model
                actual_model = actual_model.base_model
                if self._cached_layers is None or force_refresh:
                    print(f"🔧 检测到PeftModel包装，解包到base_model: {type(actual_model)}")

        # 再次检查是否还有model属性
        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            actual_model = actual_model.model
            if self._cached_layers is None or force_refresh:
                print(f"🔧 进一步解包到model属性: {type(actual_model)}")

        # 获取layers
        if hasattr(actual_model, 'layers'):
            layers = actual_model.layers
            if self._cached_layers is None or force_refresh:
                print(f"✅ 成功找到layers: {len(layers)} 层 in {type(actual_model)}")
        else:
            # 详细诊断
            print(f"❌ 无法找到layers，当前模型类型: {type(actual_model)}")
            print(f"   模型属性: {[attr for attr in dir(actual_model) if not attr.startswith('_')]}")
            raise AttributeError(f"Cannot find layers in actual model type: {type(actual_model)}")

        total_layers = len(layers)

        # 🆕 所有层都嵌入MoE
        target_layers = list(range(total_layers))

        if self._cached_layers is None or force_refresh:
            print(f"🔧 目标层范围: {target_layers[:5]}...{target_layers[-5:]} (共{total_layers}层)")

        # 🔧 缓存结果
        self._cached_layers = layers
        self._cached_target_layers = target_layers

        return layers, target_layers


    def _replace_all_layers_with_moe(self):
        """替换所有层的FFN为LoRA MoE"""
        layers, target_layers = self._get_target_layers()

        print(f"🔄 Replacing FFN in ALL {len(target_layers)} layers with LoRA MoE (Pure LoRA MoE Architecture)")

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
        """冻结非训练参数，但保留计算loss所必需的关键层"""
        model_to_freeze = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 🔧 关键修复：保留计算loss必需的层和我们要训练的参数
        trainable_keywords = [
            'lora',          # LoRA参数
            'experts',       # MoE专家参数
            'router',        # MoE路由器参数
            'lm_head',       # 输出投影层（计算loss必需！）
            'embed_tokens',  # 词嵌入层
            'norm',          # 标准化层
            'embed',         # 其他嵌入层的可能命名
            'head'           # 其他head层的可能命名
        ]

        trainable_count = 0
        frozen_count = 0

        for name, param in model_to_freeze.named_parameters():
            # 检查参数名是否包含可训练的关键词
            should_train = any(keyword in name.lower() for keyword in trainable_keywords)

            if should_train:
                param.requires_grad = True
                trainable_count += 1
                # 打印关键层的状态
                if any(key in name.lower() for key in ['lm_head', 'embed_tokens', 'norm']):
                    print(f"🔧 Keeping trainable: {name} (critical for loss computation)")
            else:
                param.requires_grad = False
                frozen_count += 1

        print(f"✅ Parameter freeze completed:")
        print(f"  - Trainable parameters: {trainable_count}")
        print(f"  - Frozen parameters: {frozen_count}")
        print(f"  - Critical layers (lm_head, embed_tokens, norm) kept trainable")

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
        try:
            layers, target_layers = self._get_target_layers()

            # 🔧 关键修复：优先收集有梯度的current数据
            current_weights_list = []
            fallback_weights_list = []

            for layer_idx in target_layers:
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA):
                    # 优先使用有梯度的current数据
                    if moe_layer.current_expert_weights is not None:
                        current_weights_list.append(moe_layer.current_expert_weights)
                    # 备用：使用无梯度的latest数据
                    elif moe_layer.latest_expert_weights is not None:
                        fallback_weights_list.append(moe_layer.latest_expert_weights)

            # 优先返回有梯度的数据
            print(f"🔍 get_expert_weights debug:")
            print(f"  current_weights_list: {len(current_weights_list)} items")
            print(f"  fallback_weights_list: {len(fallback_weights_list)} items")

            if current_weights_list:
                expert_weights_list = current_weights_list
                print(f"  using current_weights_list (有梯度)")
            else:
                expert_weights_list = fallback_weights_list
                print(f"  using fallback_weights_list (无梯度)")

            if expert_weights_list:
                # 对所有MoE层的权重求平均
                avg_expert_weights = torch.stack(expert_weights_list, dim=0).mean(dim=0)
                # return avg_expert_weights.to(dtype=torch.float16)  # 🔧 修复：移除类型转换，避免破坏梯度连接
                return avg_expert_weights
            else:
                return None
        except Exception as e:
            print(f"⚠️ 获取专家权重失败: {e}")
            return None

    def get_shared_outputs_for_culture_loss(self):
        """获取shared专家输出用于文化损失计算"""
        try:
            layers, target_layers = self._get_target_layers()

            # 🔧 关键修复：优先收集有梯度的current数据
            current_outputs_list = []
            fallback_outputs_list = []

            for layer_idx in target_layers:
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA):
                    # 优先使用有梯度的current数据
                    if moe_layer.current_shared_outputs is not None:
                        current_outputs_list.append(moe_layer.current_shared_outputs)
                    # 备用：使用无梯度的latest数据
                    elif moe_layer.latest_shared_outputs is not None:
                        fallback_outputs_list.append(moe_layer.latest_shared_outputs)

            # 优先返回有梯度的数据
            if current_outputs_list:
                shared_outputs_list = current_outputs_list
            else:
                shared_outputs_list = fallback_outputs_list

            if shared_outputs_list:
                # 对所有MoE层的shared输出求平均
                avg_shared_outputs = torch.stack(shared_outputs_list, dim=0).mean(dim=0)
                # return avg_shared_outputs.to(dtype=torch.float16)  # 🔧 修复：移除类型转换，避免破坏梯度连接
                return avg_shared_outputs
            else:
                return None
        except Exception as e:
            print(f"⚠️ 获取shared专家输出失败: {e}")
            return None

    def get_accumulated_z_loss(self, main_loss=None):
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
                    total_aux_loss = total_aux_loss + aux_loss
                moe_layer_count += 1

        if total_aux_loss is None:
            # 🔧 修复梯度问题：使用主损失*0来创建连接到计算图的零损失
            if main_loss is not None:
                total_aux_loss = main_loss * 0.0
            else:
                # 🔧 如果没有main_loss，寻找一个requires_grad=True的参数创建连接到计算图的零损失
                dummy_param = None
                for param in self.base_model.parameters():
                    if param.requires_grad:
                        dummy_param = param
                        break

                if dummy_param is not None:
                    total_aux_loss = dummy_param.sum() * 0.0
                else:
                    # 如果没有可训练参数，创建一个简单的零tensor（这种情况不应该发生）
                    total_aux_loss = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32, requires_grad=True)
        elif moe_layer_count > 1:
            total_aux_loss = total_aux_loss / moe_layer_count

        # 🔧 添加最终的NaN检查
        if torch.isnan(total_aux_loss).any() or torch.isinf(total_aux_loss).any():
            # 如果最终结果包含NaN，使用主损失创建零损失
            if main_loss is not None:
                total_aux_loss = main_loss * 0.0
            else:
                # 寻找可训练参数
                for param in self.base_model.parameters():
                    if param.requires_grad:
                        total_aux_loss = param.sum() * 0.0
                        break
                else:
                    total_aux_loss = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32, requires_grad=True)

        # 🔧 确保返回的损失与主损失类型一致
        if main_loss is not None:
            target_device = main_loss.device
            target_dtype = main_loss.dtype
            if total_aux_loss.device != target_device or total_aux_loss.dtype != target_dtype:
                total_aux_loss = total_aux_loss.to(device=target_device, dtype=target_dtype)

        return total_aux_loss

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        前向传播

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 🔧 关键修复：为文化损失计算收集有梯度的数据
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            # 收集有梯度的expert_weights
            expert_weights = self.get_expert_weights_for_culture_loss()
            print(f"🔍 Adapter forward debug:")
            print(f"  expert_weights: {expert_weights is not None}")
            if expert_weights is not None:
                print(f"    shape: {expert_weights.shape}, requires_grad: {expert_weights.requires_grad}, grad_fn: {expert_weights.grad_fn is not None}")
                outputs.expert_weights = expert_weights

            # 收集有梯度的shared_outputs
            shared_outputs = self.get_shared_outputs_for_culture_loss()
            print(f"  shared_outputs: {shared_outputs is not None}")
            if shared_outputs is not None:
                print(f"    shape: {shared_outputs.shape}, requires_grad: {shared_outputs.requires_grad}, grad_fn: {shared_outputs.grad_fn is not None}")
                outputs.shared_outputs = shared_outputs

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
        critical_params = 0

        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        print("🔍 Critical layer status check:")
        critical_layers_found = []

        for name, param in model_to_check.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'lora' in name.lower():
                    lora_params += param.numel()
                elif any(keyword in name.lower() for keyword in ['experts', 'router']):
                    moe_params += param.numel()
                elif any(keyword in name.lower() for keyword in ['lm_head', 'embed_tokens', 'norm']):
                    critical_params += param.numel()

            # 检查关键层
            if any(keyword in name.lower() for keyword in ['lm_head', 'embed_tokens', 'norm']):
                status = "✅ Trainable" if param.requires_grad else "❌ FROZEN"
                print(f"  {name}: {status} ({param.numel():,} params)")
                critical_layers_found.append((name, param.requires_grad))

        if not critical_layers_found:
            print("  ⚠️  No critical layers found! This might cause gradient issues.")

        print(f"\nTrainable params: {trainable_params:,} || "
              f"Total params: {total_params:,} || "
              f"Trainable%: {100 * trainable_params / total_params:.4f}%")

        print(f"  - LoRA params (attention): {lora_params:,}")
        print(f"  - MoE params (experts+router): {moe_params:,}")
        print(f"  - Critical params (lm_head, embed, norm): {critical_params:,}")
        print(f"  - Architecture: Pure LoRA MoE (All Layers FFN)")
        print(f"  - MoE experts: {self.config.num_moe_experts}")
        print(f"  - Activated experts: {self.config.num_activated_experts}")
        print(f"  - LoRA rank: {self.config.lora_rank}")
        print(f"  - LoRA alpha: {self.config.lora_alpha}")
        print(f"  - Shared expert: {'enabled' if self.config.use_shared else 'disabled'}")
        print(f"  - Gate network: {'enabled' if self.config.use_gate else 'disabled'}")
        if self.config.use_shared and not self.config.use_gate:
            print(f"  - Fusion method: simple addition (shared + routed)")

        # 🔧 关键检查：确保有足够的可训练参数用于loss计算
        if critical_params == 0:
            print("❌ WARNING: No critical layers are trainable! This will cause gradient issues.")
            print("   lm_head layer must be trainable for loss computation.")
        else:
            print(f"✅ Critical layers are trainable ({critical_params:,} params)")

    def check_gradient_flow(self):
        """检查模型的梯度流状态"""
        print("\n🔍 Gradient flow diagnosis:")
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 检查关键层的梯度状态
        critical_layers = ['lm_head', 'embed_tokens', 'norm']
        for layer_name in critical_layers:
            found = False
            for name, param in model_to_check.named_parameters():
                if layer_name in name.lower():
                    found = True
                    print(f"  {name}:")
                    print(f"    requires_grad: {param.requires_grad}")
                    print(f"    shape: {param.shape}")
                    print(f"    device: {param.device}")
                    print(f"    dtype: {param.dtype}")
                    break
            if not found:
                print(f"  ❌ {layer_name}: NOT FOUND")

        # 统计可训练参数
        trainable_count = sum(1 for param in model_to_check.parameters() if param.requires_grad)
        total_count = sum(1 for param in model_to_check.parameters())
        print(f"\n  📊 Trainable parameters: {trainable_count}/{total_count}")

        return trainable_count > 0


def create_simplified_culturemoe_model(base_model, config: SimplifiedCultureMoEConfig):
    """创建简化版CultureMoE模型"""
    adapter = SimplifiedCultureMoEAdapter(base_model, config)
    return adapter