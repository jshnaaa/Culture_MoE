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

        # 🔧 简化FP32计算：gate权重已经是FP32，直接匹配
        hs = hidden_states.float()          # 输入转FP32
        gate_logits = self.gate(hs)         # gate权重也是FP32，dtype匹配

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
        # 🔍 NaN诊断：Gate输入检查
        if torch.isnan(shared_output).any():
            print(f"🚨 Gate shared_output包含NaN: {torch.isnan(shared_output).sum().item()}/{shared_output.numel()}")
        if torch.isnan(routed_output).any():
            print(f"🚨 Gate routed_output包含NaN: {torch.isnan(routed_output).sum().item()}/{routed_output.numel()}")

        # 🔧 使用可学习的加权平均，参数量从4亿降到2个
        # 归一化权重
        total_weight = torch.abs(self.shared_weight) + torch.abs(self.routed_weight)
        shared_norm_weight = torch.abs(self.shared_weight) / (total_weight + 1e-8)
        routed_norm_weight = torch.abs(self.routed_weight) / (total_weight + 1e-8)

        # 🔍 NaN诊断：权重检查
        if torch.isnan(shared_norm_weight).any() or torch.isnan(routed_norm_weight).any():
            print(f"🚨 Gate权重归一化产生NaN:")
            print(f"  shared_weight: {self.shared_weight.item():.6f}")
            print(f"  routed_weight: {self.routed_weight.item():.6f}")
            print(f"  total_weight: {total_weight.item():.6f}")

        # 加权融合
        fused_output = shared_norm_weight * shared_output + routed_norm_weight * routed_output

        # 🔍 NaN诊断：Gate输出检查
        if torch.isnan(fused_output).any():
            print(f"🚨 Gate fused_output包含NaN: {torch.isnan(fused_output).sum().item()}/{fused_output.numel()}")

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

        # 计算共享专家输出
        if self.use_shared:
            shared_output = self.shared_expert(hidden_states)

            # 🔍 NaN诊断：共享专家输出检查
            if torch.isnan(shared_output).any():
                print(f"🚨 共享专家输出包含NaN: {torch.isnan(shared_output).sum().item()}/{shared_output.numel()}")
        else:
            shared_output = None

        # 🔍 NaN诊断：路由输出检查
        if torch.isnan(routed_output).any():
            print(f"🚨 路由输出包含NaN: {torch.isnan(routed_output).sum().item()}/{routed_output.numel()}")

        # 融合输出
        if self.use_gate and self.use_shared:
            intermediate_output = self.gate(shared_output, routed_output)
        elif self.use_shared:
            # 简单相加
            intermediate_output = shared_output + routed_output

            # 🔍 NaN诊断：相加融合检查
            if torch.isnan(intermediate_output).any():
                print(f"🚨 相加融合输出包含NaN: {torch.isnan(intermediate_output).sum().item()}/{intermediate_output.numel()}")
        else:
            intermediate_output = routed_output

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

        # 🔧 一次性初始化：解包PEFT并注册持久化Hook
        self._setup_persistent_hooks()

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

    def _setup_persistent_hooks(self):
        """一次性设置持久化Hook，避免每次前向传播重复注册"""
        print("🔧 设置持久化MoE Hook...")

        # 获取transformer层（只解包一次）
        transformer = self._get_transformer()
        layers = getattr(transformer, self.layer_attr)

        # 存储Hook引用以便清理
        self._persistent_hooks = []

        # 为每一层注册持久化Hook
        for layer_idx, layer in enumerate(layers):
            ffn = getattr(layer, self.ffn_attr)
            hook = ffn.register_forward_hook(self._create_persistent_hook(layer_idx))
            self._persistent_hooks.append(hook)

        print(f"✅ 成功注册 {len(self._persistent_hooks)} 个持久化Hook")

    def _create_persistent_hook(self, layer_idx):
        """创建持久化Hook函数"""
        def hook_fn(module, input, output):
            # input[0]是FFN的输入hidden_states
            hidden_states = input[0]

            # 确保MoE层和输入在同一设备且数据类型一致
            moe_layer = self.culture_moe_layers[layer_idx]
            moe_device = next(moe_layer.parameters()).device
            moe_dtype = next(moe_layer.parameters()).dtype

            # 检查设备和数据类型一致性
            if hidden_states.device != moe_device or hidden_states.dtype != moe_dtype:
                moe_layer = moe_layer.to(device=hidden_states.device, dtype=hidden_states.dtype)
                self.culture_moe_layers[layer_idx] = moe_layer

            # 🔧 极简梯度连接：只在必要时启用梯度
            if not hidden_states.requires_grad:
                hidden_states = hidden_states.requires_grad_(True)

            # 🔧 NaN早期检测和阻断
            if torch.isnan(hidden_states).any():
                print(f"🚨 Hook Layer {layer_idx}: 输入包含NaN，停止处理")
                return hidden_states  # 直接返回原始输入，避免NaN传播

            # 通过MoE层计算输出
            moe_output, aux_info = moe_layer(hidden_states)

            # 存储辅助信息（如果需要的话）
            if not hasattr(self, '_current_moe_aux_info'):
                self._current_moe_aux_info = []
            self._current_moe_aux_info.append(aux_info)

            return moe_output

        return hook_fn

    def __del__(self):
        """清理持久化Hook"""
        if hasattr(self, '_persistent_hooks'):
            for hook in self._persistent_hooks:
                hook.remove()
            print("🧹 清理持久化Hook")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """前向传播 - 持久化Hook版本，无需重复注册"""

        # 🔧 清空上次的辅助信息
        self._current_moe_aux_info = []

        # 🔧 直接前向传播，持久化Hook会自动拦截FFN
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # 🔧 添加MoE辅助信息
        if hasattr(self, '_current_moe_aux_info'):
            outputs.moe_aux_info = self._current_moe_aux_info
        else:
            outputs.moe_aux_info = []

        return outputs


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