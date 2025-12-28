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
        # 🔧 Router使用FP32：MoE工业标准做法
        self.gate = nn.Linear(hidden_size, num_experts, bias=False).float()

        # 🔧 安全初始化：防止数值溢出
        with torch.no_grad():
            nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)  # 小幅度初始化

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            gate_logits: [batch_size, seq_len, num_experts] 专家权重logits
            gate_probs: [batch_size, seq_len, num_experts] 专家权重概率
        """
        # 🔍 dtype诊断：确认真正的根因
        if not hasattr(self, "_printed_dtype"):
            print("🔍 Router dtype诊断:")
            print(f"  hidden_states.dtype: {hidden_states.dtype}")
            print(f"  gate.weight.dtype: {self.gate.weight.dtype}")
            self._printed_dtype = True

        # 🔍 NaN诊断：Router输入检查
        if torch.isnan(hidden_states).any():
            print(f"🚨 Router输入包含NaN: {torch.isnan(hidden_states).sum().item()}/{hidden_states.numel()}")

        # 🔧 确保gate权重为FP32（MoE工业标准要求）
        if self.gate.weight.dtype != torch.float32:
            self.gate.weight.data = self.gate.weight.data.float()

        # 🔧 输入转FP32匹配gate权重
        hs = hidden_states.float()          # 输入转FP32
        gate_logits = self.gate(hs)         # gate权重为FP32，dtype匹配

        # 🔧 NaN防护：清理异常值
        gate_logits = torch.nan_to_num(gate_logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # 🔍 NaN诊断：gate_logits检查
        if torch.isnan(gate_logits).any():
            print(f"🚨 Router gate_logits包含NaN: {torch.isnan(gate_logits).sum().item()}/{gate_logits.numel()}")
            print(f"  gate_logits范围: [{gate_logits.min().item():.6f}, {gate_logits.max().item():.6f}]")

        gate_probs = F.softmax(gate_logits, dim=-1)

        # 🔍 NaN诊断：gate_probs检查
        if torch.isnan(gate_probs).any():
            print(f"🚨 Router gate_probs包含NaN: {torch.isnan(gate_probs).sum().item()}/{gate_probs.numel()}")
            print(f"  gate_probs范围: [{gate_probs.min().item():.6f}, {gate_probs.max().item():.6f}]")

        return gate_logits, gate_probs




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
        # 🔍 NaN诊断：LoRA专家输入检查
        if torch.isnan(x).any():
            print(f"🚨 LoRA专家输入包含NaN: {torch.isnan(x).sum().item()}/{x.numel()}")

        # LoRA变换: x @ A @ B
        lora_output = x @ self.lora_A @ self.lora_B

        # 🔍 NaN诊断：LoRA变换检查
        if torch.isnan(lora_output).any():
            print(f"🚨 LoRA变换包含NaN: {torch.isnan(lora_output).sum().item()}/{lora_output.numel()}")
            print(f"  lora_A范围: [{self.lora_A.min().item():.6f}, {self.lora_A.max().item():.6f}]")
            print(f"  lora_B范围: [{self.lora_B.min().item():.6f}, {self.lora_B.max().item():.6f}]")

        scaled_output = lora_output * (self.alpha / self.rank)

        # 🔍 NaN诊断：LoRA专家输出检查
        if torch.isnan(scaled_output).any():
            print(f"🚨 LoRA专家输出包含NaN: {torch.isnan(scaled_output).sum().item()}/{scaled_output.numel()}")
            print(f"  alpha/rank比例: {self.alpha / self.rank:.6f}")

        return scaled_output


class CultureMoEFFN(nn.Module):
    """CultureMoE的FFN层，替换原始FFN"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_routing_experts: int = 4,
        num_activated_experts: int = 2,
        use_shared: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_routing_experts = num_routing_experts
        self.num_activated_experts = num_activated_experts
        self.use_shared = use_shared

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

        # 输出投影层：将intermediate_size映射回hidden_size
        self.output_projection = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 🔧 小尺度初始化：确保MoE作为增量时不过度影响FFN
        with torch.no_grad():
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.01)

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

        # 🔍 NaN诊断：Top-k选择检查
        if torch.isnan(top_k_probs).any() or torch.isnan(top_k_indices).any():
            print(f"🚨 Top-k选择产生NaN:")
            print(f"  top_k_probs NaN数量: {torch.isnan(top_k_probs).sum().item()}")
            print(f"  top_k_indices NaN数量: {torch.isnan(top_k_indices.float()).sum().item()}")

        # 🔧 安全归一化：防除零
        top_k_sum = top_k_probs.sum(dim=-1, keepdim=True)
        denom = top_k_sum.clamp(min=1e-6)  # 确保分母不为0
        top_k_probs = top_k_probs / denom

        # 🔍 NaN诊断：归一化检查
        if torch.isnan(top_k_probs).any():
            print(f"🚨 Top-k归一化产生NaN:")
            print(f"  top_k_sum范围: [{top_k_sum.min().item():.6f}, {top_k_sum.max().item():.6f}]")
            print(f"  归一化后top_k_probs范围: [{top_k_probs.min().item():.6f}, {top_k_probs.max().item():.6f}]")

        # 计算路由专家输出
        expert_outputs = []
        for i, expert in enumerate(self.routing_experts):
            expert_output = expert(hidden_states)  # [batch_size, seq_len, intermediate_size]

            # 🔍 NaN诊断：专家输出检查
            if torch.isnan(expert_output).any():
                print(f"🚨 专家{i}输出包含NaN: {torch.isnan(expert_output).sum().item()}/{expert_output.numel()}")

            expert_outputs.append(expert_output)

        # 堆叠所有专家输出 [num_experts, batch_size, seq_len, intermediate_size]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=0)

        # 🔍 NaN诊断：堆叠专家输出检查
        if torch.isnan(stacked_expert_outputs).any():
            print(f"🚨 堆叠专家输出包含NaN: {torch.isnan(stacked_expert_outputs).sum().item()}/{stacked_expert_outputs.numel()}")

        # 🔧 内存优化：使用更高效的专家选择和加权
        # 直接计算激活专家的加权输出，避免创建大型mask矩阵
        routed_output = torch.zeros(batch_size, seq_len, self.intermediate_size,
                                   device=stacked_expert_outputs.device, dtype=stacked_expert_outputs.dtype)

        # 对每个激活的专家位置进行加权求和
        for expert_pos in range(self.num_activated_experts):
            # 获取该位置的专家索引和权重
            expert_indices = top_k_indices[:, :, expert_pos]  # [batch_size, seq_len]
            expert_weights = top_k_probs[:, :, expert_pos]   # [batch_size, seq_len]

            # 为每个专家累加其贡献
            for expert_id in range(self.num_routing_experts):
                # 找到选择了该专家的位置
                mask = (expert_indices == expert_id)  # [batch_size, seq_len]
                if mask.any():
                    # 获取该专家的输出
                    expert_output = stacked_expert_outputs[expert_id]  # [batch_size, seq_len, intermediate_size]
                    # 应用权重和mask
                    weighted_contribution = expert_output * expert_weights.unsqueeze(-1) * mask.unsqueeze(-1)

                    # 🔍 NaN诊断：专家加权贡献检查
                    if torch.isnan(weighted_contribution).any():
                        print(f"🚨 专家{expert_id}加权贡献包含NaN:")
                        print(f"  expert_weights范围: [{expert_weights.min().item():.6f}, {expert_weights.max().item():.6f}]")
                        print(f"  mask激活数量: {mask.sum().item()}")
                        print(f"  weighted_contribution NaN数量: {torch.isnan(weighted_contribution).sum().item()}")

                    routed_output += weighted_contribution

        # 🔧 按照ChatGPT分析：正确的MoE结构
        # y = Δ_shared(x) + Σ_i p_i(x) · Δ_routed_i(x)

        # 1. 共享专家增量：直接计算，不走router，对所有样本生效
        if self.use_shared:
            shared_delta = self.shared_expert(hidden_states)

            # 🔍 NaN诊断：共享专家输出检查
            if torch.isnan(shared_delta).any():
                print(f"🚨 共享专家增量包含NaN: {torch.isnan(shared_delta).sum().item()}/{shared_delta.numel()}")
        else:
            shared_delta = torch.zeros(batch_size, seq_len, self.intermediate_size,
                                     device=hidden_states.device, dtype=hidden_states.dtype)

        # 🔍 NaN诊断：路由输出检查
        if torch.isnan(routed_output).any():
            print(f"🚨 路由专家增量包含NaN: {torch.isnan(routed_output).sum().item()}/{routed_output.numel()}")

        # 2. 总的MoE增量 = 共享增量 + 路由增量
        intermediate_output = shared_delta + routed_output

        # 🔍 NaN诊断：MoE总增量检查
        if torch.isnan(intermediate_output).any():
            print(f"🚨 MoE总增量包含NaN: {torch.isnan(intermediate_output).sum().item()}/{intermediate_output.numel()}")

        # 🔍 NaN诊断：融合后输出检查
        if torch.isnan(intermediate_output).any():
            print(f"🚨 融合后intermediate_output包含NaN: {torch.isnan(intermediate_output).sum().item()}/{intermediate_output.numel()}")

        # 投影到hidden_size维度
        final_output = self.output_projection(intermediate_output)

        # 🔍 NaN诊断：最终输出检查
        if torch.isnan(final_output).any():
            print(f"🚨 MoE最终输出包含NaN: {torch.isnan(final_output).sum().item()}/{final_output.numel()}")
            print(f"  output_projection权重范围: [{self.output_projection.weight.min().item():.6f}, {self.output_projection.weight.max().item():.6f}]")

        # 收集辅助信息
        aux_info = {
            'gate_probs': gate_probs,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs,
            'expert_outputs': expert_outputs,
            'shared_delta': shared_delta,
            'routed_delta': routed_output
        }

        return final_output, aux_info


class MoELoRAFFN(nn.Module):
    """FFN + LoRA-MoE 融合模块，直接替换原始FFN，无需Hook机制"""

    def __init__(
        self,
        original_ffn,
        hidden_size: int,
        intermediate_size: int,
        num_routing_experts: int = 4,
        num_activated_experts: int = 2,
        use_shared: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        freeze_base_ffn: bool = True
    ):
        super().__init__()

        # 保存配置
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_routing_experts = num_routing_experts
        self.num_activated_experts = num_activated_experts
        self.use_shared = use_shared

        # 🔧 保留原始FFN（可选择性冻结）
        self.base_ffn = original_ffn
        if freeze_base_ffn:
            for param in self.base_ffn.parameters():
                param.requires_grad = False

        # 🔧 MoE LoRA组件
        # Router网络（FP32，数值稳定性）
        self.router = Router(hidden_size, num_routing_experts)

        # 路由专家（LoRA增量）
        self.routing_experts = nn.ModuleList([
            LoRAExpert(hidden_size, intermediate_size, lora_rank, lora_alpha)
            for _ in range(num_routing_experts)
        ])

        # 共享专家（LoRA增量）
        if use_shared:
            self.shared_expert = LoRAExpert(hidden_size, intermediate_size, lora_rank, lora_alpha)

        # 输出投影：intermediate_size -> hidden_size
        self.output_projection = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 🔧 小尺度初始化：确保MoE作为增量时不过度影响FFN
        with torch.no_grad():
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        FFN + LoRA-MoE 前向传播：FFN_out = FFN_base + MoE_LoRA_delta

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # 🔧 原始FFN输出（包含激活函数、正确scale）
        base_output = self.base_ffn(hidden_states)

        # 🔧 计算MoE LoRA增量
        moe_delta, aux_info = self._compute_moe_delta(hidden_states)

        # 🔧 FFN + LoRA-MoE：加法组合，保持residual语义
        final_output = base_output + moe_delta

        # 存储辅助信息以便训练时使用
        if hasattr(self, '_store_aux_info') and self._store_aux_info:
            if not hasattr(self, '_aux_info_list'):
                self._aux_info_list = []
            self._aux_info_list.append(aux_info)

        return final_output

    def _compute_moe_delta(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """计算MoE LoRA增量：按照ChatGPT分析的正确结构"""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 🔧 1. 共享专家增量：直接计算，不走router，对所有样本生效
        if self.use_shared:
            shared_delta = self.shared_expert(hidden_states)
        else:
            shared_delta = torch.zeros(batch_size, seq_len, self.intermediate_size,
                                     device=hidden_states.device, dtype=hidden_states.dtype)

        # 🔧 2. 路由专家增量：通过router选择和加权
        gate_logits, gate_probs = self.router(hidden_states)

        # Top-k选择（只在路由专家内部）
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.num_activated_experts, dim=-1)

        # 安全归一化：只在top-k内部进行
        top_k_sum = top_k_probs.sum(dim=-1, keepdim=True)
        denom = top_k_sum.clamp(min=1e-6)
        top_k_probs = top_k_probs / denom

        # 计算路由专家输出
        expert_outputs = []
        for expert in self.routing_experts:
            expert_output = expert(hidden_states)  # [batch_size, seq_len, intermediate_size]
            expert_outputs.append(expert_output)

        # 堆叠所有专家输出
        stacked_expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, batch_size, seq_len, intermediate_size]

        # 高效的专家选择和加权
        routed_delta = torch.zeros(batch_size, seq_len, self.intermediate_size,
                                  device=stacked_expert_outputs.device, dtype=stacked_expert_outputs.dtype)

        for expert_pos in range(self.num_activated_experts):
            expert_indices = top_k_indices[:, :, expert_pos]  # [batch_size, seq_len]
            expert_weights = top_k_probs[:, :, expert_pos]   # [batch_size, seq_len]

            for expert_id in range(self.num_routing_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_output = stacked_expert_outputs[expert_id]
                    weighted_contribution = expert_output * expert_weights.unsqueeze(-1) * mask.unsqueeze(-1)
                    routed_delta += weighted_contribution

        # 🔧 3. 总的MoE增量 = 共享增量 + 路由增量
        # 数学形式：y = Δ_shared(x) + Σ_i p_i(x) · Δ_routed_i(x)
        intermediate_output = shared_delta + routed_delta

        # 投影到hidden_size维度
        moe_delta = self.output_projection(intermediate_output)

        # 收集辅助信息
        aux_info = {
            'gate_probs': gate_probs,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs,
            'expert_outputs': expert_outputs,
            'shared_delta': shared_delta,
            'routed_delta': routed_delta
        }

        return moe_delta, aux_info

    def enable_aux_info_collection(self):
        """启用辅助信息收集（训练时使用）"""
        self._store_aux_info = True
        self._aux_info_list = []

    def get_aux_info(self):
        """获取收集的辅助信息"""
        if hasattr(self, '_aux_info_list'):
            return self._aux_info_list
        return []

    def clear_aux_info(self):
        """清空辅助信息"""
        if hasattr(self, '_aux_info_list'):
            self._aux_info_list.clear()


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

        # 🔧 直接替换每一层的FFN为MoELoRAFFN，无需Hook机制
        print("🔧 直接替换FFN为MoELoRAFFN...")

        # 获取transformer层
        transformer = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
        layers = getattr(transformer, self.layer_attr)

        print(f"检查基座模型层的设备分布:")
        for layer_idx in range(min(5, len(layers))):  # 只检查前5层
            layer_device = next(layers[layer_idx].parameters()).device
            print(f"  基座模型第{layer_idx}层在: {layer_device}")

        # 直接替换每一层的MLP
        for layer_idx in range(min(self.num_layers, len(layers))):
            layer = layers[layer_idx]
            original_ffn = getattr(layer, self.ffn_attr)

            # 创建MoELoRAFFN，包含原始FFN
            moe_lora_ffn = MoELoRAFFN(
                original_ffn=original_ffn,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_routing_experts=config['num_moe_experts'],
                num_activated_experts=config['num_activated_experts'],
                use_shared=config['use_shared'],
                lora_rank=config['lora_rank'],
                lora_alpha=config['lora_alpha'],
                freeze_base_ffn=True  # 冻结原始FFN
            )

            # 设备和dtype一致性处理
            target_device = next(original_ffn.parameters()).device
            target_dtype = next(original_ffn.parameters()).dtype

            # 移动MoE组件到正确设备
            moe_lora_ffn = moe_lora_ffn.to(device=target_device)

            # 精确的dtype控制：Router保持FP32，其他组件使用目标dtype
            for name, module in moe_lora_ffn.named_modules():
                if 'router' in name and hasattr(module, 'weight'):
                    # Router权重强制保持FP32
                    module.weight.data = module.weight.data.float()
                elif hasattr(module, 'weight') and 'router' not in name and 'base_ffn' not in name:
                    # 非Router、非原始FFN的组件转换为目标dtype
                    module.to(dtype=target_dtype)

            # 🔧 直接替换layer的mlp属性
            setattr(layer, self.ffn_attr, moe_lora_ffn)

            if layer_idx < 3:  # 只打印前3层的替换信息
                print(f"✅ 第{layer_idx}层FFN已替换为MoELoRAFFN，设备: {target_device}, Router: FP32, 其他: {target_dtype}")

        print(f"✅ 成功替换 {min(self.num_layers, len(layers))} 层的FFN为MoELoRAFFN")

        # 冻结基座模型参数
        self._freeze_base_model()

        # 添加注意力层LoRA（如果启用）
        if config.get('apply_to_attention', False):
            self._add_attention_lora()

        # 🔧 无Hook架构：直接替换完成，无需额外设置

    def _freeze_base_model(self):
        """选择性冻结基座模型参数，保留LoRA和MoE层可训练"""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.base_model.named_parameters():
            # 保留LoRA相关参数可训练
            if any(lora_key in name.lower() for lora_key in ['lora_a', 'lora_b', 'lora']):
                param.requires_grad = True
                trainable_count += 1
            else:
                # 冻结非LoRA参数
                param.requires_grad = False
                frozen_count += 1

        print(f"Base model: {frozen_count} params frozen, {trainable_count} LoRA params trainable")

        # 统计参数数量
        frozen_params = sum(p.numel() for name, p in self.base_model.named_parameters()
                          if not p.requires_grad)
        trainable_params = sum(p.numel() for name, p in self.base_model.named_parameters()
                             if p.requires_grad)
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable LoRA parameters: {trainable_params:,}")

        # MoE层参数会在CultureMoEFFN中自动设置为可训练

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

    def _get_transformer(self):
        """获取transformer层，处理PEFT包装的多层嵌套"""
        # 🔧 缓存机制：避免每次都重新解包
        if hasattr(self, '_cached_transformer'):
            return self._cached_transformer

        model = self.base_model
        path_trace = [type(model).__name__]

        # 🔧 智能解包：避免自引用循环
        seen_types = set()
        max_depth = 5  # 减少最大深度

        while hasattr(model, "base_model") and len(seen_types) < max_depth:
            model_type = type(model).__name__

            # 🔧 防止自引用循环：如果下一个model和当前model类型相同，直接跳出
            next_model_type = type(model.base_model).__name__
            if model_type == next_model_type:
                print(f"🔍 检测到自引用循环: {model_type} -> {next_model_type}，停止解包")
                break

            if model_type in seen_types:
                print(f"🔍 检测到类型重复: {model_type}，停止解包")
                break

            seen_types.add(model_type)
            print(f"🔍 解包PEFT层级: {model_type} -> {next_model_type}")
            model = model.base_model
            path_trace.append(next_model_type)

        print(f"🔍 完整路径: {' -> '.join(path_trace)}")

        # 🔧 处理HuggingFace模型结构 (LlamaForCausalLM -> LlamaModel)
        if hasattr(model, "model") and hasattr(model.model, self.layer_attr):
            transformer = model.model
            print(f"✅ 找到transformer层: {type(model).__name__}.model.{self.layer_attr}")
        # 🔧 直接检查是否就是transformer模型
        elif hasattr(model, self.layer_attr):
            transformer = model
            print(f"✅ 找到transformer层: {type(model).__name__}.{self.layer_attr}")
        else:
            # 如果所有路径都失败，提供详细的错误信息
            available_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
            raise AttributeError(
                f"无法找到transformer层。\n"
                f"完整路径: {' -> '.join(path_trace)}\n"
                f"最终model类型: {type(model).__name__}\n"
                f"查找属性: {self.layer_attr}\n"
                f"可用属性: {available_attrs[:10]}..."
            )

        # 🔧 缓存结果避免重复解包
        self._cached_transformer = transformer
        return transformer

    # 🔧 Hook机制已移除：直接替换FFN，无需Hook拦截

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """前向传播 - 无Hook版本，MoE直接集成在FFN中"""

        # 🔧 启用MoE辅助信息收集（训练时需要）
        if self.training:
            self._enable_aux_info_collection()

        # 🔧 直接前向传播，MoE已集成在每层的FFN中
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # 🔧 收集MoE辅助信息
        if self.training:
            moe_aux_info = self._collect_aux_info()
            outputs.moe_aux_info = moe_aux_info
        else:
            outputs.moe_aux_info = []

        return outputs

    def _enable_aux_info_collection(self):
        """启用所有MoE层的辅助信息收集"""
        transformer = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
        layers = getattr(transformer, self.layer_attr)

        for layer in layers:
            ffn = getattr(layer, self.ffn_attr)
            if hasattr(ffn, 'enable_aux_info_collection'):
                ffn.enable_aux_info_collection()

    def _collect_aux_info(self):
        """收集所有MoE层的辅助信息"""
        transformer = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
        layers = getattr(transformer, self.layer_attr)

        all_aux_info = []
        for layer in layers:
            ffn = getattr(layer, self.ffn_attr)
            if hasattr(ffn, 'get_aux_info'):
                aux_info = ffn.get_aux_info()
                all_aux_info.extend(aux_info)
                # 清空辅助信息，为下次前向传播准备
                if hasattr(ffn, 'clear_aux_info'):
                    ffn.clear_aux_info()

        return all_aux_info


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