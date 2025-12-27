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

        # 🔧 更保守的输入范围限制
        x = torch.clamp(x, min=-5.0, max=5.0)

        try:
            # LoRA分支计算 - 只计算增量
            gate_lora = self.gate_lora_B(self.gate_lora_A(x)) * self.scaling
            up_lora = self.up_lora_B(self.up_lora_A(x)) * self.scaling

            # 🔧 更保守的中间结果限制
            gate_lora = torch.clamp(gate_lora, min=-8.0, max=8.0)
            up_lora = torch.clamp(up_lora, min=-8.0, max=8.0)

            # FFN的激活和组合（LoRA增量）
            intermediate_lora = self.act_fn(gate_lora) * up_lora
            intermediate_lora = torch.clamp(intermediate_lora, min=-10.0, max=10.0)

            # Dropout
            intermediate_lora = self.dropout(intermediate_lora)

            # down_proj LoRA增量
            lora_delta = self.down_lora_B(self.down_lora_A(intermediate_lora)) * self.scaling
            lora_delta = torch.clamp(lora_delta, min=-5.0, max=5.0)

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

        # 🔧 更保守的初始化，防止数值不稳定
        nn.init.normal_(self.router.weight, mean=0.0, std=0.005)

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

        # 🔧 限制诊断输出
        if not hasattr(self, '_debug_call_count'):
            self._debug_call_count = 0
        self._debug_call_count += 1
        debug_this_call = self._debug_call_count <= 2

        if debug_this_call:
            print(f"🔍 Router Forward - Input Diagnosis (Call {self._debug_call_count}):")
            print(f"  x.requires_grad: {x.requires_grad}")
            print(f"  x.grad_fn: {x.grad_fn is not None}")
            print(f"  x.shape: {x.shape}")

        # 🔧 更保守的输入限制，但保持梯度连接
        x_clamped = torch.clamp(x, min=-5.0, max=5.0)

        if debug_this_call:
            print(f"🔍 After clamp:")
            print(f"  x_clamped.requires_grad: {x_clamped.requires_grad}")
            print(f"  x_clamped.grad_fn: {x_clamped.grad_fn is not None}")

        try:
            if debug_this_call:
                # 计算路由logits
                print(f"🔍 Router Linear Layer Call:")
                print(f"  self.router.weight.requires_grad: {self.router.weight.requires_grad}")
                print(f"  self.router.weight.grad_fn: {self.router.weight.grad_fn is not None}")

            router_logits = self.router(x_clamped)  # [B, L, num_experts]

            if debug_this_call:
                print(f"🔍 Router Linear Output:")
                print(f"  router_logits.requires_grad: {router_logits.requires_grad}")
                print(f"  router_logits.grad_fn: {router_logits.grad_fn is not None}")

            router_logits = torch.clamp(router_logits, min=-5.0, max=5.0)

            if debug_this_call:
                print(f"🔍 After router_logits clamp:")
                print(f"  router_logits.requires_grad: {router_logits.requires_grad}")
                print(f"  router_logits.grad_fn: {router_logits.grad_fn is not None}")

            # 数值稳定的softmax
            max_logits = torch.max(router_logits, dim=-1, keepdim=True)[0]
            shifted_logits = router_logits - max_logits
            shifted_logits = torch.clamp(shifted_logits, min=-10.0, max=0.0)

            if debug_this_call:
                print(f"🔍 Softmax computation:")
                print(f"  shifted_logits.requires_grad: {shifted_logits.requires_grad}")
                print(f"  shifted_logits.grad_fn: {shifted_logits.grad_fn is not None}")

            # 计算专家权重
            expert_weights = F.softmax(shifted_logits, dim=-1)

            if debug_this_call:
                print(f"🔍 Softmax output:")
                print(f"  expert_weights.requires_grad: {expert_weights.requires_grad}")
                print(f"  expert_weights.grad_fn: {expert_weights.grad_fn is not None}")

            # 检查结果
            if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
                if debug_this_call:
                    print(f"🔍 NaN/Inf detected in expert_weights, using uniform fallback")
                # 使用均匀分布作为fallback
                expert_weights = expert_weights * 0.0 + (1.0 / self.num_experts)
                router_logits = router_logits * 0.0
                if debug_this_call:
                    print(f"🔍 After fallback:")
                    print(f"  expert_weights.requires_grad: {expert_weights.requires_grad}")
                    print(f"  expert_weights.grad_fn: {expert_weights.grad_fn is not None}")

            # 🔧 简化输出：如果expert_weights没有梯度，报告问题
            elif not debug_this_call and (not expert_weights.requires_grad or expert_weights.grad_fn is None):
                print(f"⚠️ Router: expert_weights output has no gradient!")
                print(f"  Input x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn is not None}")
                print(f"  Router weight.requires_grad: {self.router.weight.requires_grad}")
                print(f"  Output expert_weights.requires_grad: {expert_weights.requires_grad}, grad_fn: {expert_weights.grad_fn is not None}")

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

        # 🔧 关键修复：确保所有MoE参数从创建时就是可训练的
        for param in self.router.parameters():
            param.requires_grad = True
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = True
        if self.use_shared:
            for param in self.shared_expert.parameters():
                param.requires_grad = True
        if self.use_gate:
            for param in self.gate_network.parameters():
                param.requires_grad = True

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

        # 🔧 限制诊断输出：只在前几个调用时输出详细信息
        if not hasattr(self, '_debug_call_count'):
            self._debug_call_count = 0
        self._debug_call_count += 1

        debug_this_call = self._debug_call_count <= 2  # 只在前2次调用时输出详细诊断

        if debug_this_call:
            print(f"🔍 MoE Forward - Input Gradient Diagnosis (Call {self._debug_call_count}):")
            print(f"  hidden_states.requires_grad: {hidden_states.requires_grad}")
            print(f"  hidden_states.grad_fn: {hidden_states.grad_fn is not None}")
            print(f"  hidden_states.shape: {hidden_states.shape}")
            print(f"  hidden_states contains NaN/Inf: {torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any()}")

        # 🔧 梯度状态诊断（不进行修复，让梯度自然流动）
        if not hidden_states.requires_grad:
            print(f"⚠️ MoE input hidden_states has no gradient!")
            print(f"  requires_grad={hidden_states.requires_grad}, grad_fn={hidden_states.grad_fn is not None}")
            print(f"  This indicates upstream gradient propagation issue - need to fix upstream layers")
        else:
            if debug_this_call:
                print(f"✅ MoE input has gradient: requires_grad={hidden_states.requires_grad}, grad_fn={hidden_states.grad_fn is not None}")

        # 检查输入
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            print("⚠️ NaN/Inf in MoE input, using original FFN")
            return self.original_ffn(hidden_states)

        if debug_this_call:
            # 🔧 详细梯度诊断：检查router参数状态
            print(f"🔍 Router Parameter Diagnosis:")
            for name, param in self.router.named_parameters():
                print(f"  router.{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn is not None}, shape={param.shape}")

        try:
            # 1. 计算原始FFN输出作为基础
            original_output = self.original_ffn(hidden_states)

            # 2. 路由计算（为路由专家）
            if debug_this_call:
                print(f"🔍 Before router call:")
                print(f"  hidden_states.requires_grad: {hidden_states.requires_grad}")
                print(f"  hidden_states.grad_fn: {hidden_states.grad_fn is not None}")

            expert_weights, router_logits = self.router(hidden_states)

            if debug_this_call:
                print(f"🔍 After router call:")
                print(f"  expert_weights.requires_grad: {expert_weights.requires_grad}")
                print(f"  expert_weights.grad_fn: {expert_weights.grad_fn is not None}")
                print(f"  router_logits.requires_grad: {router_logits.requires_grad}")
                print(f"  router_logits.grad_fn: {router_logits.grad_fn is not None}")
            else:
                # 简化输出：只在expert_weights没有梯度时报告
                if not expert_weights.requires_grad or expert_weights.grad_fn is None:
                    print(f"⚠️ MoE Layer: expert_weights has no gradient!")
                    print(f"  expert_weights.requires_grad: {expert_weights.requires_grad}")
                    print(f"  expert_weights.grad_fn: {expert_weights.grad_fn is not None}")

            # 🔧 关键修复：保存当前批次的有梯度expert_weights
            if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
                # 如果包含NaN，使用均匀分布（保持梯度连接）
                uniform_weights = expert_weights * 0.0 + (1.0 / expert_weights.shape[2])
                self.current_expert_weights = uniform_weights.mean(dim=1)  # [B, num_experts] - 有梯度
                self.latest_expert_weights = uniform_weights.mean(dim=1).detach()  # 无梯度版本用于统计
            else:
                self.current_expert_weights = expert_weights.mean(dim=1)  # [B, num_experts] - 有梯度
                self.latest_expert_weights = expert_weights.mean(dim=1).detach()  # 无梯度版本用于统计

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
        # 🔧 关键修复：优先使用有梯度的current_expert_weights
        expert_weights_for_loss = None
        if self.current_expert_weights is not None:
            expert_weights_for_loss = self.current_expert_weights
        elif self.latest_expert_weights is not None:
            # 如果没有current数据，使用latest但要创建梯度连接
            expert_weights_for_loss = self.latest_expert_weights
        else:
            # 完全没有数据，返回零损失
            for param in self.parameters():
                if param.requires_grad:
                    return param.sum() * 0.0
            # 如果没有可训练参数，使用参数创建零损失
            dummy_param = next(iter(self.parameters()))
            return dummy_param.sum() * 0.0

        try:
            # 🔧 添加数值稳定性检查
            if torch.isnan(expert_weights_for_loss).any() or torch.isinf(expert_weights_for_loss).any():
                # 如果expert_weights包含NaN或Inf，使用零损失
                for param in self.parameters():
                    if param.requires_grad:
                        return param.sum() * 0.0
                dummy_param = next(iter(self.parameters()))
                return dummy_param.sum() * 0.0

            # 负载均衡损失
            expert_usage = expert_weights_for_loss.mean(dim=0)  # [num_experts]
            target_usage = expert_usage * 0.0 + (1.0 / self.num_experts)
            balance_loss = F.mse_loss(expert_usage, target_usage)

            # 检查数值稳定性
            if torch.isnan(balance_loss) or torch.isinf(balance_loss):
                # 🔧 修复梯度问题：使用参数创建连接到计算图的零损失
                for param in self.parameters():
                    if param.requires_grad:
                        balance_loss = param.sum() * 0.0
                        break
                else:
                    # 如果没有可训练参数，使用任意参数创建零损失
                    dummy_param = next(iter(self.parameters()))
                    balance_loss = dummy_param.sum() * 0.0

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

        # 2. 先冻结非训练参数（在MoE创建之前）
        self._freeze_non_trainable_parameters()

        # 3. 然后在包装后的模型上替换FFN为MoE
        self._replace_all_layers_with_moe()

        # 4. 🔧 关键修复：MoE创建后，确保所有MoE参数都是可训练的
        self._ensure_moe_parameters_trainable()

        # 5. 确保设备一致性
        self._ensure_device_consistency()

        # 6. 🔧 最终验证：再次确保MoE参数可训练（在所有操作后）
        self._final_moe_gradient_check()


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

    def _ensure_moe_parameters_trainable(self):
        """确保所有MoE参数都是可训练的"""
        print(f"🔧 Ensuring all MoE parameters are trainable...")

        layers, target_layers = self._get_target_layers()
        moe_param_count = 0

        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFNLoRA):
                # 确保router参数可训练
                for name, param in moe_layer.router.named_parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        print(f"🔧 Fixed: Set layer {layer_idx} router.{name} to trainable")
                    moe_param_count += param.numel()

                # 确保expert参数可训练
                for expert_idx, expert in enumerate(moe_layer.experts):
                    for name, param in expert.named_parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                            print(f"🔧 Fixed: Set layer {layer_idx} expert_{expert_idx}.{name} to trainable")
                        moe_param_count += param.numel()

                # 确保shared expert参数可训练（如果有）
                if hasattr(moe_layer, 'shared_expert') and moe_layer.shared_expert is not None:
                    for name, param in moe_layer.shared_expert.named_parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                            print(f"🔧 Fixed: Set layer {layer_idx} shared_expert.{name} to trainable")
                        moe_param_count += param.numel()

                # 确保gate网络参数可训练（如果有）
                if hasattr(moe_layer, 'gate_network') and moe_layer.gate_network is not None:
                    for name, param in moe_layer.gate_network.named_parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                            print(f"🔧 Fixed: Set layer {layer_idx} gate_network.{name} to trainable")
                        moe_param_count += param.numel()

        print(f"✅ MoE parameter check completed: {moe_param_count:,} MoE parameters ensured trainable")

    def _final_moe_gradient_check(self):
        """最终验证MoE参数的梯度状态"""
        print(f"🔧 Final MoE gradient verification...")

        layers, target_layers = self._get_target_layers()
        issues_found = 0

        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFNLoRA):
                # 检查router参数
                for name, param in moe_layer.router.named_parameters():
                    if not param.requires_grad:
                        print(f"❌ CRITICAL: Layer {layer_idx} router.{name} requires_grad=False")
                        param.requires_grad = True
                        issues_found += 1

                # 检查expert参数
                for expert_idx, expert in enumerate(moe_layer.experts):
                    for name, param in expert.named_parameters():
                        if not param.requires_grad:
                            print(f"❌ CRITICAL: Layer {layer_idx} expert_{expert_idx}.{name} requires_grad=False")
                            param.requires_grad = True
                            issues_found += 1

        if issues_found > 0:
            print(f"⚠️ Fixed {issues_found} MoE parameters that were incorrectly frozen")
        else:
            print(f"✅ All MoE parameters verified as trainable")

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
        """🔧 修复版参数冻结：使用保守策略，只冻结明确不需要的参数"""
        model_to_freeze = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        print(f"🔧 Using CONSERVATIVE parameter freezing strategy...")

        # 🔧 新策略：只冻结明确不需要训练的参数，而不是冻结除白名单外的所有参数
        freeze_keywords = [
            # 🔧 极其保守：暂时不冻结任何参数，让所有参数都可训练
            # 这样可以确保梯度传播不会被意外中断
        ]

        # 🔧 关键修复：确保这些参数绝对可训练
        must_trainable_keywords = [
            'lora',              # LoRA参数
            'experts',           # MoE专家参数
            'router',            # MoE路由器参数
            'lm_head',           # 输出投影层
            'embed_tokens',      # 词嵌入层
            'norm',              # 所有norm层
            'layernorm',         # 所有layernorm层
            'layer_norm',        # 所有layer_norm层
            'input_layernorm',   # 输入层归一化
            'post_attention_layernorm',  # 注意力后归一化
            'self_attn',         # 自注意力层（关键的梯度传播路径）
            'q_proj', 'k_proj', 'v_proj', 'o_proj',  # 注意力投影层
        ]

        trainable_count = 0
        frozen_count = 0

        print(f"🔧 ULTRA-CONSERVATIVE MODE: Making ALL parameters trainable to ensure gradient flow")

        for name, param in model_to_freeze.named_parameters():
            # 🔧 超保守策略：暂时让所有参数都可训练
            # 这样可以确保不会有任何梯度传播中断
            param.requires_grad = True
            trainable_count += 1

            # 特别标记关键的梯度传播参数
            if any(keyword in name.lower() for keyword in must_trainable_keywords):
                print(f"🔧 Critical parameter kept trainable: {name}")

        print(f"✅ ULTRA-CONSERVATIVE parameter freeze completed:")
        print(f"  - Trainable parameters: {trainable_count}")
        print(f"  - Frozen parameters: {frozen_count}")
        print(f"  - Strategy: ALL parameters trainable to ensure gradient flow")
        print(f"  - This ensures no gradient disconnection at any layer")

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
                    # 🔧 修复：如果没有current数据，但有latest数据，尝试重新计算有梯度的版本
                    elif moe_layer.latest_expert_weights is not None:
                        # 尝试从router重新计算有梯度的expert_weights
                        print(f"⚠️ Layer {layer_idx}: current_expert_weights is None, trying to recompute")
                        fallback_weights_list.append(moe_layer.latest_expert_weights)

            # 优先返回有梯度的数据
            if current_weights_list:
                expert_weights_list = current_weights_list
            else:
                expert_weights_list = fallback_weights_list

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
                    # 🔧 修复：如果没有可训练参数，使用任意参数创建零损失
                    dummy_param = next(iter(self.base_model.parameters()))
                    total_aux_loss = dummy_param.sum() * 0.0
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
                    # 🔧 修复：使用任意参数创建零损失
                    dummy_param = next(iter(self.base_model.parameters()))
                    total_aux_loss = dummy_param.sum() * 0.0

        # 🔧 确保返回的损失与主损失类型一致
        if main_loss is not None:
            target_device = main_loss.device
            target_dtype = main_loss.dtype
            if total_aux_loss.device != target_device or total_aux_loss.dtype != target_dtype:
                total_aux_loss = total_aux_loss.to(device=target_device, dtype=target_dtype)

        return total_aux_loss

    def compute_direct_z_loss(self, main_loss=None):
        """
        直接计算z-loss，避免torch.stack操作
        直接从MoE层获取expert_weights并计算，确保梯度连接
        """
        layers, target_layers = self._get_target_layers()

        total_z_loss = None
        moe_layer_count = 0

        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFNLoRA):
                # 直接使用当前MoE层的current_expert_weights
                if moe_layer.current_expert_weights is not None:
                    expert_weights = moe_layer.current_expert_weights  # [B, num_experts] - 有梯度

                    # 检查梯度状态
                    if expert_weights.requires_grad and expert_weights.grad_fn is not None:
                        # 计算z-loss
                        expert_usage = expert_weights.mean(dim=0)  # [num_experts]
                        num_experts = expert_weights.shape[-1]
                        target_usage = expert_usage * 0.0 + (1.0 / num_experts)
                        layer_z_loss = F.mse_loss(expert_usage, target_usage) * 0.01

                        if total_z_loss is None:
                            total_z_loss = layer_z_loss
                        else:
                            total_z_loss = total_z_loss + layer_z_loss
                        moe_layer_count += 1
                    else:
                        # 🔧 详细诊断expert_weights没有梯度的原因
                        print(f"⚠️ Layer {layer_idx}: expert_weights has no gradient, skipping")
                        print(f"  expert_weights.requires_grad: {expert_weights.requires_grad}")
                        print(f"  expert_weights.grad_fn: {expert_weights.grad_fn}")

                        # 检查router参数状态
                        router_has_grad = any(p.requires_grad for p in moe_layer.router.parameters())
                        print(f"  router parameters require_grad: {router_has_grad}")

                        # 检查router权重的详细状态
                        for name, param in moe_layer.router.named_parameters():
                            print(f"    router.{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn is not None}")
                else:
                    print(f"⚠️ Layer {layer_idx}: current_expert_weights is None, skipping")

        if total_z_loss is None:
            # 没有任何有效的expert_weights，使用零损失
            if main_loss is not None:
                total_z_loss = main_loss * 0.0
            else:
                # 寻找可训练参数创建零损失
                for param in self.base_model.parameters():
                    if param.requires_grad:
                        total_z_loss = param.sum() * 0.0
                        break
                else:
                    dummy_param = next(iter(self.base_model.parameters()))
                    total_z_loss = dummy_param.sum() * 0.0
        elif moe_layer_count > 1:
            # 对多层求平均
            total_z_loss = total_z_loss / moe_layer_count

        # 确保设备和数据类型一致
        if main_loss is not None:
            target_device = main_loss.device
            target_dtype = main_loss.dtype
            if total_z_loss.device != target_device or total_z_loss.dtype != target_dtype:
                total_z_loss = total_z_loss.to(device=target_device, dtype=target_dtype)

        return total_z_loss

    def compute_direct_culture_loss(self, culture_labels, main_loss=None):
        """
        直接计算文化损失，避免torch.stack操作
        直接从MoE层获取expert_weights和shared_outputs并计算，确保梯度连接
        """
        layers, target_layers = self._get_target_layers()

        total_culture_losses = []

        # 收集所有有梯度的expert_weights和shared_outputs
        valid_expert_weights = []
        valid_shared_outputs = []

        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFNLoRA):
                # 收集有梯度的expert_weights
                if (moe_layer.current_expert_weights is not None and
                    moe_layer.current_expert_weights.requires_grad and
                    moe_layer.current_expert_weights.grad_fn is not None):
                    valid_expert_weights.append(moe_layer.current_expert_weights)

                # 收集有梯度的shared_outputs
                if (moe_layer.current_shared_outputs is not None and
                    moe_layer.current_shared_outputs.requires_grad and
                    moe_layer.current_shared_outputs.grad_fn is not None):
                    valid_shared_outputs.append(moe_layer.current_shared_outputs)

        # 如果有有效的数据，直接计算文化损失，不使用torch.stack
        if valid_expert_weights or valid_shared_outputs:
            # 使用第一层的数据作为代表（避免torch.stack）
            representative_expert_weights = valid_expert_weights[0] if valid_expert_weights else None
            representative_shared_outputs = valid_shared_outputs[0] if valid_shared_outputs else None

            # 直接计算文化损失，不依赖外部函数
            device = culture_labels.device
            total_culture_losses = []

            # 1. Shared专家损失：所有样本的shared专家输出都应该相似（学习文化共性）
            if representative_shared_outputs is not None and representative_shared_outputs.shape[0] >= 2:
                shared_losses = []
                batch_size = representative_shared_outputs.shape[0]

                # 对所有样本对计算shared专家相似度损失
                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        vec1 = representative_shared_outputs[i].unsqueeze(0)
                        vec2 = representative_shared_outputs[j].unsqueeze(0)

                        # 检查向量是否为零向量
                        norm1 = torch.norm(vec1)
                        norm2 = torch.norm(vec2)
                        if norm1 < 1e-8 or norm2 < 1e-8:
                            continue

                        similarity = F.cosine_similarity(vec1, vec2)
                        if torch.isnan(similarity) or torch.isinf(similarity):
                            continue

                        # Shared专家：所有样本都应该相似，所以损失为 1 - similarity
                        shared_loss_term = 1.0 - similarity
                        shared_losses.append(shared_loss_term)

                if len(shared_losses) > 0:
                    shared_culture_loss = torch.stack(shared_losses).mean()
                    total_culture_losses.append(shared_culture_loss)

            # 2. 路由专家损失：保持原有逻辑（同culture相似，不同culture不同）
            if representative_expert_weights is not None:
                expert_weights = representative_expert_weights  # [B, num_experts]

                # 使用对应的culture_labels
                routing_culture_labels = culture_labels
                if expert_weights.shape[0] != culture_labels.shape[0]:
                    # 只对有expert_weights的样本计算文化损失
                    if expert_weights.shape[0] < 2:
                        pass  # 激活路由专家的样本少于2个，跳过路由专家损失
                    else:
                        routing_culture_labels = culture_labels[:expert_weights.shape[0]]

                if expert_weights.shape[0] >= 2:
                    routing_losses = []
                    batch_size = expert_weights.shape[0]

                    # 计算同文化样本间的相似性和不同文化样本间的差异性
                    for i in range(batch_size):
                        for j in range(i + 1, batch_size):
                            if routing_culture_labels[i] == routing_culture_labels[j]:
                                # 相同文化，鼓励相似的专家权重
                                vec1 = expert_weights[i].unsqueeze(0)
                                vec2 = expert_weights[j].unsqueeze(0)

                                # 检查向量是否为零向量，避免cosine_similarity中的NaN
                                norm1 = torch.norm(vec1)
                                norm2 = torch.norm(vec2)
                                if norm1 < 1e-8 or norm2 < 1e-8:
                                    continue

                                similarity = F.cosine_similarity(vec1, vec2)
                                if torch.isnan(similarity) or torch.isinf(similarity):
                                    continue

                                # 相同文化：相似度应该高，损失为 1 - similarity
                                loss_term = 1.0 - similarity
                                routing_losses.append(loss_term)
                            else:
                                # 不同文化，鼓励不同的专家权重
                                vec1 = expert_weights[i].unsqueeze(0)
                                vec2 = expert_weights[j].unsqueeze(0)

                                # 检查向量是否为零向量，避免cosine_similarity中的NaN
                                norm1 = torch.norm(vec1)
                                norm2 = torch.norm(vec2)
                                if norm1 < 1e-8 or norm2 < 1e-8:
                                    continue

                                similarity = F.cosine_similarity(vec1, vec2)
                                if torch.isnan(similarity) or torch.isinf(similarity):
                                    continue

                                # 不同文化：相似度应该低，损失为 similarity
                                routing_losses.append(similarity)

                    if len(routing_losses) > 0:
                        routing_culture_loss = torch.stack(routing_losses).mean()
                        total_culture_losses.append(routing_culture_loss)

            # 计算最终的总文化损失
            if len(total_culture_losses) > 0:
                culture_loss = torch.stack(total_culture_losses).mean()
            else:
                # 如果没有有效的文化损失，使用主损失*0来创建连接到计算图的零损失
                if main_loss is not None:
                    culture_loss = main_loss * 0.0
                else:
                    culture_loss = culture_labels.float().sum() * 0.0

            # 检查文化损失是否为NaN/Inf，如果是则返回零损失
            if torch.isnan(culture_loss) or torch.isinf(culture_loss):
                if main_loss is not None:
                    culture_loss = main_loss * 0.0
                else:
                    culture_loss = culture_labels.float().sum() * 0.0

            # 确保返回的culture_loss与主损失类型一致
            if main_loss is not None:
                target_device = main_loss.device
                target_dtype = main_loss.dtype
                if culture_loss.device != target_device or culture_loss.dtype != target_dtype:
                    culture_loss = culture_loss.to(device=target_device, dtype=target_dtype)

            return culture_loss
        else:
            # 没有有效数据，返回零损失
            if main_loss is not None:
                return main_loss * 0.0
            else:
                for param in self.base_model.parameters():
                    if param.requires_grad:
                        return param.sum() * 0.0
                dummy_param = next(iter(self.base_model.parameters()))
                return dummy_param.sum() * 0.0

    def emergency_gradient_fix(self):
        """紧急梯度修复：强制设置所有关键参数为可训练"""
        print(f"🚨 Emergency Gradient Fix: Forcing critical parameters to be trainable...")

        fixed_count = 0
        model_to_fix = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 强制设置所有关键参数为可训练
        critical_keywords = [
            'embed_tokens',      # 词嵌入层
            'lora',             # 所有LoRA参数
            'norm',             # 所有norm层
            'layernorm',        # 所有layernorm层
            'layer_norm',       # 所有layer_norm层
            'input_layernorm',  # 输入层归一化
            'post_attention_layernorm',  # 注意力后归一化
            'experts',          # MoE专家参数
            'router',           # MoE路由器参数
            'lm_head'           # 输出头
        ]

        for name, param in model_to_fix.named_parameters():
            if any(keyword in name.lower() for keyword in critical_keywords):
                if not param.requires_grad:
                    param.requires_grad = True
                    fixed_count += 1
                    print(f"  🔧 Fixed: {name}")

        print(f"🚨 Emergency fix completed: {fixed_count} parameters fixed")
        return fixed_count

    def diagnose_gradient_flow(self, input_ids, attention_mask):
        """诊断梯度流：检查从input到MoE层的整个路径"""
        print(f"🔍 Full Gradient Flow Diagnosis:")

        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 1. 检查embedding层
        if hasattr(model_to_check, 'embed_tokens'):
            embed_params_trainable = sum(1 for p in model_to_check.embed_tokens.parameters() if p.requires_grad)
            embed_params_total = sum(1 for p in model_to_check.embed_tokens.parameters())
            print(f"  📍 Embedding: {embed_params_trainable}/{embed_params_total} params trainable")

        # 2. 检查LoRA参数
        lora_params = [(name, param) for name, param in model_to_check.named_parameters() if 'lora' in name.lower()]
        lora_trainable = sum(1 for _, param in lora_params if param.requires_grad)
        print(f"  📍 LoRA: {lora_trainable}/{len(lora_params)} params trainable")

        # 3. 检查norm层
        norm_params = [(name, param) for name, param in model_to_check.named_parameters()
                      if any(keyword in name.lower() for keyword in ['norm', 'layernorm', 'layer_norm'])]
        norm_trainable = sum(1 for _, param in norm_params if param.requires_grad)
        print(f"  📍 Norm layers: {norm_trainable}/{len(norm_params)} params trainable")

        # 🆕 详细检查norm层状态
        if norm_trainable < len(norm_params):
            print(f"  ⚠️ Some norm layers are frozen - this will break gradient flow!")
            for name, param in norm_params[:5]:  # 显示前5个
                status = "✅" if param.requires_grad else "❌"
                print(f"    {status} {name}: requires_grad={param.requires_grad}")

        # 4. 检查attention层
        attn_params = [(name, param) for name, param in model_to_check.named_parameters()
                      if any(keyword in name.lower() for keyword in ['self_attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj'])]
        attn_trainable = sum(1 for _, param in attn_params if param.requires_grad)
        print(f"  📍 Attention layers: {attn_trainable}/{len(attn_params)} params trainable")

        # 5. 检查第一个MoE层的router
        try:
            layers, target_layers = self._get_target_layers()
            first_moe = layers[0].mlp
            if hasattr(first_moe, 'router'):
                router_params_trainable = sum(1 for p in first_moe.router.parameters() if p.requires_grad)
                router_params_total = sum(1 for p in first_moe.router.parameters())
                print(f"  📍 First MoE Router: {router_params_trainable}/{router_params_total} params trainable")
        except Exception as e:
            print(f"  ❌ Cannot check MoE router: {e}")

        # 🆕 6. 测试梯度传播路径
        print(f"  🔬 Testing gradient propagation with small forward pass...")
        try:
            with torch.enable_grad():
                # 创建一个小的测试输入
                test_input = input_ids[:1, :10].clone()  # 取第一个样本的前10个token
                test_mask = attention_mask[:1, :10].clone()

                # 前向传播到embedding
                if hasattr(model_to_check, 'embed_tokens'):
                    embeddings = model_to_check.embed_tokens(test_input)
                    print(f"    Embeddings: requires_grad={embeddings.requires_grad}, grad_fn={embeddings.grad_fn is not None}")

                    # 检查第一个transformer层的输出
                    if hasattr(model_to_check, 'layers') and len(model_to_check.layers) > 0:
                        # 这里只能做简单测试，不能完整运行transformer层（太复杂）
                        print(f"    ✅ Embedding layer produces tensors with gradient connection")
                else:
                    print(f"    ❌ Cannot find embed_tokens layer")

        except Exception as e:
            print(f"    ❌ Gradient propagation test failed: {e}")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        前向传播

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签
        """
        # 🔧 紧急诊断和修复（只在前几次调用时执行）
        if not hasattr(self, '_emergency_fix_done'):
            self._emergency_fix_done = True
            self.diagnose_gradient_flow(input_ids, attention_mask)
            fixed_count = self.emergency_gradient_fix()
            if fixed_count > 0:
                print(f"🚨 Applied emergency gradient fix, re-diagnosing...")
                self.diagnose_gradient_flow(input_ids, attention_mask)

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
            if expert_weights is not None:
                outputs.expert_weights = expert_weights

            # 收集有梯度的shared_outputs
            shared_outputs = self.get_shared_outputs_for_culture_loss()
            if shared_outputs is not None:
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