#!/usr/bin/env python3
"""
联合训练脚本：同时训练预训练LoRA适配器 + 新增MoE处理层（共享专家版本）
实现端到端的LoRA+MoE联合优化训练，增加共享专家功能
针对48GB×2卡优化

特点：
1. 同时训练基础模型的LoRA适配器和新增MoE层
2. 增加共享专家：共享专家权重0.1，路由专家权重0.9
3. 共享专家使用LoRA rank=4，学习率2e-5
4. 分层学习率：基础LoRA、共享专家、路由专家使用不同学习率
5. 端到端优化，避免预训练LoRA权重冻结
6. 加权融合机制：共享专家+激活路由专家的加权融合
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    raise ImportError("PEFT library is required. Please install with: pip install peft")

# 复用现有的数据集类
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    load_and_process_data,
    extract_answer_from_text,
    generate_answer,
    dynamic_padding_collate_fn
)

# 导入基础配置类
from src.llamafactory.model.joint_lora_moe_model import JointLoRAMoEConfig


@dataclass
class JointLoRAMoESharedConfig(JointLoRAMoEConfig):
    """联合LoRA+MoE配置（共享专家版本）"""

    # 共享专家配置
    use_shared_expert: bool = True
    shared_lora_rank: int = 4
    shared_expert_weight: float = 0.1
    routed_expert_weight: float = 0.9
    shared_expert_lr: float = 2e-5

    # 共享专家LoRA配置
    shared_lora_alpha: int = 8  # shared_lora_rank * 2
    shared_lora_dropout: float = 0.1

    def __post_init__(self):
        super().__post_init__()

        # 确保权重和为1
        total_weight = self.shared_expert_weight + self.routed_expert_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"⚠️ Warning: shared_expert_weight + routed_expert_weight = {total_weight}, normalizing...")
            self.shared_expert_weight = self.shared_expert_weight / total_weight
            self.routed_expert_weight = self.routed_expert_weight / total_weight

        # 确保shared_lora_alpha与shared_lora_rank匹配
        if self.shared_lora_alpha < self.shared_lora_rank:
            self.shared_lora_alpha = self.shared_lora_rank * 2


class SharedExpert(nn.Module):
    """共享专家层 - 使用小型LoRA适配器实现，变弱且不干扰专家特化"""

    def __init__(self, hidden_dim: int, config: JointLoRAMoESharedConfig, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        self.dtype = dtype

        # 共享专家使用小型LoRA适配器
        self.lora_rank = config.shared_lora_rank
        self.lora_alpha = config.shared_lora_alpha
        self.lora_dropout = config.shared_lora_dropout

        # LoRA矩阵A和B
        self.lora_A = nn.Linear(hidden_dim, self.lora_rank, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(self.lora_rank, hidden_dim, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(self.lora_dropout)

        # LoRA缩放因子 - 比正常小很多
        self.scaling = self.lora_alpha / self.lora_rank

        # 🔧 关键改进1：LayerNorm稳定输出
        self.layer_norm = nn.LayerNorm(hidden_dim, dtype=dtype)

        # 🔧 关键改进2：可学习的极小scale，防止dominate专家输出
        self.learnable_scale = nn.Parameter(torch.tensor(0.02, dtype=dtype))

        # 🔧 关键改进3：输出scale控制，防止幅度过大
        self.output_scale_factor = 0.1  # 额外的固定缩放

        self._init_weights()

    def _init_weights(self):
        """LoRA标准初始化"""
        # A矩阵使用高斯初始化
        nn.init.normal_(self.lora_A.weight, std=1 / self.lora_rank)
        # B矩阵初始化为零
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """
        前向传播 - 变弱且不干扰专家特化的共享专家

        Args:
            x: [B, L, H] 输入隐藏状态

        Returns:
            output: [B, L, H] 共享专家输出（经过多重控制）
            scale_info: dict 包含scale监控信息
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros_like(x), {"shared_scale": 0.0, "input_scale": 0.0}

        try:
            # 🔧 监控输入scale
            input_scale = torch.norm(x).item()

            # LoRA前向传播: x -> A -> dropout -> B -> scale
            lora_output = self.lora_A(x)  # [B, L, rank]
            lora_output = self.dropout(lora_output)
            lora_output = self.lora_B(lora_output)  # [B, L, H]
            lora_output = lora_output * self.scaling

            # 🔧 关键改进1：LayerNorm稳定输出，防止scale失控
            lora_output = self.layer_norm(lora_output)

            # 🔧 关键改进2：应用可学习的极小scale
            # learnable_scale初始化为0.02，确保共享专家输出很小
            lora_output = lora_output * torch.clamp(self.learnable_scale, min=0.001, max=0.1)

            # 🔧 关键改进3：额外的固定缩放，进一步压制输出幅度
            lora_output = lora_output * self.output_scale_factor

            # 🔧 监控最终输出scale
            output_scale = torch.norm(lora_output).item()

            # 检查输出有效性
            if torch.isnan(lora_output).any() or torch.isinf(lora_output).any():
                return torch.zeros_like(x), {"shared_scale": 0.0, "input_scale": input_scale}

            scale_info = {
                "shared_scale": output_scale,
                "input_scale": input_scale,
                "learnable_scale": self.learnable_scale.item()
            }

            return lora_output, scale_info

        except Exception as e:
            print(f"⚠️ SharedExpert forward failed: {e}")
            return torch.zeros_like(x), {"shared_scale": 0.0, "input_scale": 0.0}


class MoESharedLayer(nn.Module):
    """MoE层（共享专家版本）"""

    def __init__(self, config: JointLoRAMoESharedConfig, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.hidden_dim = config.moe_hidden_dim
        self.dtype = dtype

        # 统计计数器
        self.nan_count = 0
        self.total_forward_calls = 0

        # 导入基础组件
        from src.llamafactory.model.joint_lora_moe_model import MoERouter, MoEExpert

        # 创建路由器
        self.router = MoERouter(
            hidden_dim=config.moe_hidden_dim,
            num_experts=config.num_moe_experts,
            dropout=config.dropout,
            dtype=torch.float32  # 路由器强制使用Float32
        )

        # 创建路由专家
        self.routed_experts = nn.ModuleList([
            MoEExpert(
                hidden_dim=config.moe_hidden_dim,
                intermediate_dim=config.moe_intermediate_dim,
                dropout=config.dropout,
                dtype=dtype
            ) for _ in range(config.num_moe_experts)
        ])

        # 创建共享专家
        if config.use_shared_expert:
            self.shared_expert = SharedExpert(
                hidden_dim=config.moe_hidden_dim,
                config=config,
                dtype=dtype
            )

            # 🔧 关键改进：可控融合机制，避免直接线性相加
            # 可学习的融合gate，初始化为很小的值
            self.fusion_gate = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

            # 残差连接的权重，用于平衡shared和routed的贡献
            self.residual_weight = nn.Parameter(torch.tensor(0.95, dtype=torch.float32))

            # 路由器敏感性保护：监控路由分布
            self.router_entropy_history = []
            self.router_variance_history = []

        else:
            self.shared_expert = None
            self.fusion_gate = None
            self.residual_weight = None

        # 确保专家分化
        print("🔧 Initializing experts with different seeds...")
        for i, expert in enumerate(self.routed_experts):
            torch.manual_seed(42 + i * 100)
            expert._init_weights()
            print(f"🔧 Routed Expert {i} initialized with seed {42 + i * 100}")

        if self.shared_expert is not None:
            torch.manual_seed(42 + 1000)  # 共享专家使用不同种子
            self.shared_expert._init_weights()
            print(f"🔧 Shared Expert initialized with seed {42 + 1000}")

    def get_nan_stats(self):
        """获取NaN统计信息"""
        if self.total_forward_calls == 0:
            return 0.0, 0, 0
        nan_rate = self.nan_count / self.total_forward_calls
        return nan_rate, self.nan_count, self.total_forward_calls

    def reset_nan_stats(self):
        """重置NaN统计"""
        self.nan_count = 0
        self.total_forward_calls = 0

    def forward(self, hidden_states):
        """
        前向传播 - 共享专家+路由专家加权融合

        Args:
            hidden_states: [B, L, H] 输入隐藏状态

        Returns:
            output: [B, L, H] 输出隐藏状态
            expert_weights: [B, num_experts] 路由专家权重
            aux_loss: 辅助损失
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        self.total_forward_calls += 1
        current_nan_detected = False

        try:
            # 检查输入有效性
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print("⚠️ NaN/Inf in MoE input, using passthrough")
                expert_weights = torch.ones(batch_size, self.num_experts,
                                           device=hidden_states.device,
                                           dtype=hidden_states.dtype,
                                           requires_grad=True) / self.num_experts
                aux_loss = torch.tensor(0.0, device=hidden_states.device,
                                      dtype=hidden_states.dtype, requires_grad=True)
                return hidden_states, expert_weights, aux_loss

            # 1. 共享专家输出（如果启用）- 使用新的返回格式
            shared_output = None
            shared_scale_info = {}
            if self.config.use_shared_expert and self.shared_expert is not None:
                shared_output, shared_scale_info = self.shared_expert(hidden_states)  # [B, L, H], scale_info

                # 检查共享专家输出
                if torch.isnan(shared_output).any() or torch.isinf(shared_output).any():
                    print("⚠️ Shared expert output invalid, using zeros")
                    shared_output = torch.zeros_like(hidden_states)
                    current_nan_detected = True

            # 2. 路由专家处理
            # 路由决策
            pooled = hidden_states.mean(dim=1)  # [B, H]
            all_expert_weights, router_logits = self.router(pooled, temperature=1.0)

            # 🔧 路由器敏感性监控
            if self.config.use_shared_expert and self.fusion_gate is not None:
                # 计算路由器entropy（分布平均程度）
                router_probs = torch.softmax(router_logits, dim=-1)
                router_entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1).mean().item()

                # 计算路由器logits方差（区分能力）
                router_variance = torch.var(router_logits, dim=-1).mean().item()

                # 记录历史，用于监控路由器健康状态
                self.router_entropy_history.append(router_entropy)
                self.router_variance_history.append(router_variance)

                # 保持历史长度不超过100
                if len(self.router_entropy_history) > 100:
                    self.router_entropy_history.pop(0)
                    self.router_variance_history.pop(0)

            # Top-2激活机制
            top_k_logits, top_k_indices = torch.topk(router_logits, k=2, dim=-1)  # [B, 2]
            top_k_weights = torch.softmax(top_k_logits, dim=-1)  # [B, 2]

            # 创建稀疏权重矩阵
            expert_weights = torch.zeros_like(all_expert_weights)  # [B, num_experts]
            expert_weights.scatter_(1, top_k_indices, top_k_weights)

            # 计算激活的路由专家输出
            expert_outputs = {}
            valid_experts = 0
            activated_experts = torch.unique(top_k_indices.flatten()).cpu().tolist()

            for expert_idx in activated_experts:
                try:
                    expert_output = self.routed_experts[expert_idx](hidden_states)  # [B, L, H]

                    if not (torch.isnan(expert_output).any() or torch.isinf(expert_output).any()):
                        if torch.all(expert_output == 0) and torch.all(hidden_states != 0):
                            current_nan_detected = True
                        expert_outputs[expert_idx] = expert_output
                        valid_experts += 1
                    else:
                        expert_outputs[expert_idx] = torch.zeros_like(hidden_states)
                        current_nan_detected = True

                except Exception as e:
                    print(f"⚠️ Routed Expert {expert_idx} failed: {e}")
                    expert_outputs[expert_idx] = torch.zeros_like(hidden_states)
                    current_nan_detected = True

            # 3. 路由专家加权融合
            routed_output = torch.zeros_like(hidden_states)
            total_routed_weight = 0.0

            if valid_experts > 0:
                for i in range(expert_weights.size(1)):
                    weight_val = expert_weights[:, i]  # [B]
                    if weight_val.sum().item() > 1e-8:
                        if i in expert_outputs:
                            weight = weight_val.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
                            weight = torch.clamp(weight, min=0.0, max=1.0)
                            routed_output += weight * expert_outputs[i]
                            total_routed_weight += weight_val.mean().item()

            # 如果路由专家权重太小，使用输入passthrough
            if total_routed_weight < 0.001:
                print("⚠️ Routed expert weights too small, using input passthrough")
                routed_output = hidden_states
            else:
                routed_output = torch.clamp(routed_output, min=-5.0, max=5.0)

            # 4. 🔧 改进的可控融合机制：避免直接线性相加
            if shared_output is not None and self.fusion_gate is not None:
                # 🔧 方法1：残差连接式融合，shared作为残差分支
                # 限制fusion_gate在合理范围内
                clamped_fusion_gate = torch.clamp(self.fusion_gate, min=0.001, max=0.2)
                clamped_residual_weight = torch.clamp(self.residual_weight, min=0.8, max=0.999)

                # 残差式融合：主干是routed_output，shared作为小的残差修正
                final_output = clamped_residual_weight * routed_output + clamped_fusion_gate * shared_output

                # 🔧 方法2：Scale监控和自适应调整
                if len(shared_scale_info) > 0:
                    shared_scale = shared_scale_info.get("shared_scale", 0.0)
                    routed_scale = torch.norm(routed_output).item()

                    # 如果shared的scale太大，进一步压制
                    if shared_scale > routed_scale * 0.1:  # shared不应该超过routed的10%
                        scale_adjustment = min(0.5, (routed_scale * 0.1) / (shared_scale + 1e-8))
                        final_output = clamped_residual_weight * routed_output + clamped_fusion_gate * shared_output * scale_adjustment
            elif shared_output is not None:
                # 回退到原始加权融合，但使用更保守的权重
                conservative_shared_weight = min(self.config.shared_expert_weight, 0.05)  # 最多5%
                conservative_routed_weight = 1.0 - conservative_shared_weight
                final_output = (conservative_shared_weight * shared_output +
                              conservative_routed_weight * routed_output)
            else:
                # 如果没有共享专家，只使用路由专家输出
                final_output = routed_output

            # 5. 最终检查
            if torch.isnan(final_output).any() or torch.isinf(final_output).any():
                print("⚠️ Final MoE output invalid, using input passthrough")
                final_output = hidden_states

            # 6. 计算辅助损失
            try:
                # 负载均衡损失
                target_uniform = torch.ones_like(expert_weights) / self.num_experts
                balance_loss = F.mse_loss(expert_weights, target_uniform)

                # 路由器正则化损失
                router_reg_loss = torch.mean(router_logits ** 2)

                # 组合辅助损失
                aux_loss = balance_loss * 0.01 + router_reg_loss * 0.001

                if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                    aux_loss = torch.tensor(0.0, device=hidden_states.device,
                                          dtype=hidden_states.dtype, requires_grad=True)

            except Exception as e:
                print(f"⚠️ Aux loss computation failed: {e}")
                aux_loss = torch.tensor(0.0, device=hidden_states.device,
                                      dtype=hidden_states.dtype, requires_grad=True)

            # 更新NaN计数
            if current_nan_detected:
                self.nan_count += 1

            # 🔧 返回监控信息
            monitoring_info = {
                "shared_scale_info": shared_scale_info,
                "router_entropy": self.router_entropy_history[-1] if self.router_entropy_history else 0.0,
                "router_variance": self.router_variance_history[-1] if self.router_variance_history else 0.0,
                "fusion_gate": self.fusion_gate.item() if self.fusion_gate is not None else 0.0
            }

            return final_output, expert_weights, aux_loss, monitoring_info

        except Exception as e:
            print(f"⚠️ MoE shared layer completely failed: {e}")
            # Fallback
            expert_weights = torch.ones(batch_size, self.num_experts,
                                       device=hidden_states.device,
                                       dtype=hidden_states.dtype,
                                       requires_grad=True) / self.num_experts
            aux_loss = torch.tensor(0.01, device=hidden_states.device,
                                  dtype=hidden_states.dtype, requires_grad=True)
            fallback_monitoring = {
                "shared_scale_info": {},
                "router_entropy": 0.0,
                "router_variance": 0.0,
                "fusion_gate": 0.0
            }
            return hidden_states, expert_weights, aux_loss, fallback_monitoring


class JointLoRAMoESharedModel(nn.Module):
    """联合LoRA+MoE模型（共享专家版本）"""

    def __init__(self, base_model, config: JointLoRAMoESharedConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model

        # 确保config中的moe_hidden_dim与模型一致
        if hasattr(base_model.config, 'hidden_size'):
            self.config.moe_hidden_dim = base_model.config.hidden_size
        elif hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
            self.config.moe_hidden_dim = base_model.config.hidden_size

        # 1. 应用LoRA到基础模型
        self._apply_lora()

        # 获取基础模型的设备和数据类型
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype

        # 2. 添加MoE共享层（传递正确的dtype）
        self.moe_layer = MoESharedLayer(config, dtype=dtype)

        # 3. 确保MoE层在正确设备上，但保持路由器为Float32
        self.moe_layer = self.moe_layer.to(device=device)

        # 专家层可以转换为指定dtype，但路由器保持Float32
        for expert in self.moe_layer.routed_experts:
            expert = expert.to(dtype=dtype)

        # 共享专家也转换为指定dtype
        if self.moe_layer.shared_expert is not None:
            self.moe_layer.shared_expert = self.moe_layer.shared_expert.to(dtype=dtype)

        # 确保路由器保持Float32
        self.moe_layer.router = self.moe_layer.router.to(device=device, dtype=torch.float32)

        logging.info(f"Joint LoRA+MoE Shared model initialized with {config.num_moe_experts} experts + 1 shared expert")

    def _apply_lora(self):
        """应用LoRA到基础模型"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        self.base_model = get_peft_model(self.base_model, lora_config)
        logging.info(f"LoRA applied to base model with rank={self.config.lora_rank}")

        # 验证LoRA是否正确应用
        lora_params = 0
        total_params = 0
        for name, param in self.base_model.named_parameters():
            total_params += param.numel()
            if 'lora' in name.lower():
                lora_params += param.numel()
                if lora_params <= 5:  # 只打印前5个LoRA参数
                    print(f"🔍 LoRA参数: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")

        print(f"🔍 LoRA参数统计: {lora_params:,} / {total_params:,} ({100*lora_params/total_params:.2f}%)")

    def forward(self, input_ids=None, attention_mask=None, labels=None, culture_labels=None, **kwargs):
        """
        前向传播（共享专家版本）

        Args:
            input_ids: [B, L] 输入token IDs
            attention_mask: [B, L] 注意力掩码
            labels: [B, L] 标签（用于计算损失）
            culture_labels: [B] 文化标签（用于计算文化损失）

        Returns:
            outputs: 包含loss、logits、expert_weights等的字典
        """
        # 1. 基础模型前向传播（包含LoRA）
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 2. 获取最后一层隐藏状态
        hidden_states = base_outputs.hidden_states[-1]  # [B, L, H]

        # 确保hidden_states与MoE层的hidden_dim匹配
        if hidden_states.size(-1) != self.config.moe_hidden_dim:
            # 如果维度不匹配，需要投影
            if not hasattr(self, 'hidden_proj'):
                self.hidden_proj = nn.Linear(
                    hidden_states.size(-1),
                    self.config.moe_hidden_dim,
                    bias=False,
                    dtype=hidden_states.dtype
                )
                nn.init.normal_(self.hidden_proj.weight, mean=0.0, std=0.001)
                self.hidden_proj = self.hidden_proj.to(hidden_states.device, hidden_states.dtype)
                self.add_module('hidden_proj', self.hidden_proj)
            hidden_states = self.hidden_proj(hidden_states)

        # 3. MoE共享层处理 - 增量架构实现（新增监控信息）
        moe_delta, expert_weights, moe_aux_loss, monitoring_info = self.moe_layer(hidden_states)

        # 4. 增量架构：基础LoRA + 加权MoE增量
        base_hidden_states = base_outputs.hidden_states[-1]  # [B, L, H]

        # 处理维度不匹配问题
        if hasattr(self, 'hidden_proj') and moe_delta.size(-1) != base_hidden_states.size(-1):
            if not hasattr(self, 'hidden_proj_back'):
                self.hidden_proj_back = nn.Linear(
                    moe_delta.size(-1),
                    base_hidden_states.size(-1),
                    bias=False,
                    dtype=moe_delta.dtype
                )
                nn.init.normal_(self.hidden_proj_back.weight, mean=0.0, std=0.001)
                self.hidden_proj_back = self.hidden_proj_back.to(moe_delta.device, moe_delta.dtype)
                self.add_module('hidden_proj_back', self.hidden_proj_back)
            moe_delta = self.hidden_proj_back(moe_delta)

        # MoE增量权重（控制MoE影响程度）
        moe_influence_weight = self.config.moe_influence_weight

        # 组合输出：基础LoRA + 加权MoE增量
        combined_output = base_hidden_states + moe_influence_weight * moe_delta
        final_hidden_states = combined_output

        # 5. 语言模型头 - 使用组合后的隐藏状态
        if hasattr(self.base_model, 'lm_head'):
            logits = self.base_model.lm_head(final_hidden_states)
        elif hasattr(self.base_model, 'base_model') and hasattr(self.base_model.base_model, 'lm_head'):
            logits = self.base_model.base_model.lm_head(final_hidden_states)
        else:
            # 创建临时的lm_head
            vocab_size = self.base_model.config.vocab_size
            if not hasattr(self, 'temp_lm_head'):
                self.temp_lm_head = nn.Linear(
                    final_hidden_states.size(-1), vocab_size, bias=False, dtype=final_hidden_states.dtype
                )
                nn.init.normal_(self.temp_lm_head.weight, mean=0.0, std=0.02)
                self.temp_lm_head = self.temp_lm_head.to(final_hidden_states.device, final_hidden_states.dtype)
                self.add_module('temp_lm_head', self.temp_lm_head)
            logits = self.temp_lm_head(final_hidden_states)

        # 6. 计算损失
        loss = None
        culture_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if labels is not None:
            try:
                # 检查logits是否包含NaN/Inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("⚠️ NaN/Inf detected in logits, using base LoRA only")
                    base_only_logits = self.base_model.lm_head(base_hidden_states)
                    logits = base_only_logits

                # 严格的logits范围限制
                logits = torch.clamp(logits, min=-20.0, max=20.0)

                # 语言模型损失
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0)
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # 检查lm_loss是否为NaN/Inf
                if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                    print("⚠️ NaN/Inf detected in lm_loss, using fallback loss")
                    # 使用模型参数的L2损失作为fallback
                    param_loss = torch.tensor(0.0, device=shift_logits.device,
                                            dtype=shift_logits.dtype, requires_grad=True)
                    param_count = 0
                    for param in self.parameters():
                        if param.requires_grad:
                            param_loss = param_loss + torch.sum(param * param)
                            param_count += 1
                    if param_count > 0:
                        lm_loss = param_loss / param_count * 0.001
                    else:
                        lm_loss = torch.sum(shift_logits * shift_logits) * 0.001

                # 限制辅助损失的影响
                moe_aux_loss = torch.clamp(moe_aux_loss, min=0.0, max=1.0)

                # 计算文化损失（如果启用）
                culture_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype, requires_grad=True)
                if self.config.use_culture_loss and culture_labels is not None and expert_weights is not None:
                    # 导入文化损失计算函数
                    from train_joint_lora_moe import compute_culture_loss

                    # 计算文化损失
                    culture_loss = compute_culture_loss(expert_weights, culture_labels,
                                                      self.config.culture_loss_weight)
                    culture_loss = culture_loss.to(device=lm_loss.device, dtype=lm_loss.dtype)

                # 总损失 = 语言模型损失 + MoE辅助损失 + 文化损失
                loss = lm_loss + 0.01 * moe_aux_loss + culture_loss

                # 最终损失检查
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ NaN/Inf detected in total loss, using lm_loss only")
                    loss = lm_loss

            except Exception as e:
                print(f"⚠️ Loss computation failed: {e}, using fallback loss")
                # 使用模型参数的L2损失作为fallback
                first_param = next(self.parameters())
                param_loss = torch.tensor(0.0, device=first_param.device,
                                        dtype=first_param.dtype, requires_grad=True)
                param_count = 0
                for param in self.parameters():
                    if param.requires_grad:
                        param_loss = param_loss + torch.sum(param * param)
                        param_count += 1
                if param_count > 0:
                    loss = param_loss / param_count * 0.01
                else:
                    loss = torch.tensor(1.0, device=first_param.device,
                                      dtype=first_param.dtype, requires_grad=True)

                # fallback情况下的文化损失
                culture_loss = torch.tensor(0.0, device=first_param.device,
                                          dtype=first_param.dtype, requires_grad=True)

        # 7. 返回结果（包含监控信息）
        return type('Outputs', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': final_hidden_states,
            'expert_weights': expert_weights,
            'moe_aux_loss': moe_aux_loss,
            'culture_loss': culture_loss,
            'base_hidden_states': base_hidden_states,
            'moe_delta': moe_delta,
            'monitoring_info': monitoring_info,  # 🔧 新增监控信息
        })()

    def get_parameter_groups(self, base_lr: float, moe_lr: float, shared_lr: float = None):
        """
        获取分层参数组，用于设置不同的学习率（共享专家版本）

        Args:
            base_lr: 基础模型LoRA参数的学习率
            moe_lr: MoE路由专家的学习率
            shared_lr: 共享专家的学习率，如果为None则使用config中的shared_expert_lr

        Returns:
            param_groups: 参数组列表
        """
        if shared_lr is None:
            shared_lr = self.config.shared_expert_lr

        base_params = []
        routed_expert_params = []
        shared_expert_params = []
        router_params = []

        # 分离不同类型的参数
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'moe_layer.shared_expert' in name:
                    shared_expert_params.append(param)
                elif 'moe_layer.routed_experts' in name:
                    routed_expert_params.append(param)
                elif 'moe_layer.router' in name:
                    router_params.append(param)
                elif 'moe_layer' in name:
                    # 其他MoE相关参数归入路由专家组
                    routed_expert_params.append(param)
                else:
                    base_params.append(param)

        param_groups = []

        # 基础LoRA参数组
        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': base_lr,
                'weight_decay': 0.001,
                'name': 'base_lora'
            })

        # 共享专家参数组
        if shared_expert_params:
            param_groups.append({
                'params': shared_expert_params,
                'lr': shared_lr,
                'weight_decay': 0.0001,  # 共享专家使用较小的权重衰减
                'name': 'shared_expert'
            })

        # 路由专家参数组
        if routed_expert_params:
            param_groups.append({
                'params': routed_expert_params,
                'lr': moe_lr,
                'weight_decay': 0.01,
                'name': 'routed_experts'
            })

        # 路由器参数组
        if router_params:
            param_groups.append({
                'params': router_params,
                'lr': moe_lr * 0.5,  # 路由器使用较小的学习率
                'weight_decay': 0.001,
                'name': 'router'
            })

        logging.info(f"Parameter groups: base_lora={len(base_params)}, "
                    f"shared_expert={len(shared_expert_params)}, "
                    f"routed_experts={len(routed_expert_params)}, "
                    f"router={len(router_params)}")
        return param_groups

    def print_trainable_parameters(self):
        """打印可训练参数统计（共享专家版本）"""
        total_params = 0
        trainable_params = 0
        base_lora_params = 0
        shared_expert_params = 0
        routed_expert_params = 0
        router_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'moe_layer.shared_expert' in name:
                    shared_expert_params += param.numel()
                elif 'moe_layer.routed_experts' in name:
                    routed_expert_params += param.numel()
                elif 'moe_layer.router' in name:
                    router_params += param.numel()
                elif 'moe_layer' in name:
                    routed_expert_params += param.numel()
                else:
                    base_lora_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || "
              f"Total params: {total_params:,} || "
              f"Trainable%: {100 * trainable_params / total_params:.4f}%")

        print(f"  - Base LoRA params: {base_lora_params:,}")
        print(f"  - Shared Expert params: {shared_expert_params:,}")
        print(f"  - Routed Expert params: {routed_expert_params:,}")
        print(f"  - Router params: {router_params:,}")
        print(f"  - Architecture: Joint LoRA + MoE + Shared Expert")
        print(f"  - MoE experts: {self.config.num_moe_experts} + 1 shared")
        print(f"  - Shared expert weight: {self.config.shared_expert_weight}")
        print(f"  - Routed expert weight: {self.config.routed_expert_weight}")

    def save_model(self, save_path: str):
        """保存模型权重（共享专家版本）"""
        import os

        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        # 保存LoRA权重
        if hasattr(self.base_model, 'save_pretrained'):
            lora_path = os.path.join(save_path, 'lora_weights')
            self.base_model.save_pretrained(lora_path)

        # 保存MoE权重（包括共享专家）
        moe_state_dict = {}
        for name, param in self.moe_layer.named_parameters():
            moe_state_dict[name] = param.data

        moe_path = os.path.join(save_path, 'moe_shared_weights.pt')
        torch.save(moe_state_dict, moe_path)

        # 保存配置
        config_path = os.path.join(save_path, 'joint_shared_config.json')
        import json
        config_dict = {
            'lora_rank': self.config.lora_rank,
            'lora_alpha': self.config.lora_alpha,
            'lora_dropout': self.config.lora_dropout,
            'lora_target_modules': self.config.lora_target_modules,
            'num_moe_experts': self.config.num_moe_experts,
            'moe_hidden_dim': self.config.moe_hidden_dim,
            'moe_intermediate_dim': self.config.moe_intermediate_dim,
            'use_culture_loss': self.config.use_culture_loss,
            'culture_loss_weight': self.config.culture_loss_weight,
            'dropout': self.config.dropout,
            # 共享专家配置
            'use_shared_expert': self.config.use_shared_expert,
            'shared_lora_rank': self.config.shared_lora_rank,
            'shared_expert_weight': self.config.shared_expert_weight,
            'routed_expert_weight': self.config.routed_expert_weight,
            'shared_expert_lr': self.config.shared_expert_lr,
            'shared_lora_alpha': self.config.shared_lora_alpha,
            'shared_lora_dropout': self.config.shared_lora_dropout
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ Joint LoRA+MoE+Shared model saved to {save_path}")
        print(f"  - LoRA weights: {lora_path}")
        print(f"  - MoE+Shared weights: {moe_path}")
        print(f"  - Config: {config_path}")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=10,
                 do_sample=False, temperature=0.7, pad_token_id=None, eos_token_id=None, **kwargs):
        """
        生成方法（共享专家版本）
        """
        # 复用基础模型的generate方法，但需要经过共享专家MoE层
        with torch.no_grad():
            batch_size = input_ids.size(0)
            current_ids = input_ids.clone()

            if attention_mask is None:
                attention_mask = torch.ones_like(current_ids)

            # 重复检测计数器
            repeated_count = 0
            max_repeated_allowed = 2
            last_few_tokens = []

            for step in range(max_new_tokens):
                # 使用我们的forward方法（包含共享专家MoE层）
                outputs = self.forward(
                    input_ids=current_ids,
                    attention_mask=attention_mask
                )

                # 获取最后一个位置的logits
                next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

                # 生成下一个token
                if do_sample:
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 数字答案检测和早停机制
                current_token = next_token_id.item()

                # 检查是否生成了有效的数字答案
                if step == 0:
                    if current_token in [16, 17, 18, 19]:  # 数字1-4的token ID
                        current_ids = torch.cat([current_ids, next_token_id], dim=-1)
                        break

                last_few_tokens.append(current_token)
                if len(last_few_tokens) > 5:
                    last_few_tokens.pop(0)

                # 检查重复
                if step > 0:
                    last_token = current_ids[:, -1].item()
                    if current_token == last_token:
                        repeated_count += 1
                        if repeated_count >= max_repeated_allowed:
                            # 惩罚重复token
                            for token in set(last_few_tokens):
                                next_token_logits[:, token] -= 15.0

                            diversity_temp = max(temperature, 1.2) if temperature > 0 else 1.2
                            next_token_logits = next_token_logits / diversity_temp
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token_id = torch.multinomial(probs, num_samples=1)
                            current_token = next_token_id.item()
                            repeated_count = 0
                        else:
                            next_token_logits[:, current_token] -= 8.0
                            if do_sample and temperature > 0:
                                next_token_logits = next_token_logits / temperature
                                probs = torch.softmax(next_token_logits, dim=-1)
                                next_token_id = torch.multinomial(probs, num_samples=1)
                            else:
                                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                            current_token = next_token_id.item()
                    else:
                        repeated_count = 0

                # 添加新token
                current_ids = torch.cat([current_ids, next_token_id], dim=-1)

                # 更新attention_mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=-1)

                # 检查结束token
                if eos_token_id is not None and (next_token_id == eos_token_id).any():
                    break

            return current_ids


# 复用现有训练函数，但修改为使用共享专家模型
from train_joint_lora_moe import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    compute_culture_loss
)


def train_epoch_joint_shared(model, train_loader, optimizer, device, tokenizer,
                           num_accumulation_steps=1, rank=0, use_culture_loss=True, culture_loss_weight=0.01):
    """
    联合训练一个epoch：同时训练LoRA、共享专家和MoE（共享专家版本）
    """
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_moe_loss = 0
    total_culture_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Joint Shared Training", disable=(rank != 0), mininterval=1.0)

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 检查batch是否有有效的训练标签
        total_valid_labels = (labels != -100).sum().item()
        if total_valid_labels == 0:
            if rank == 0:
                print(f"⚠️ Batch {batch_idx}: 所有标签都被mask，跳过此batch")
            continue

        # 获取文化标签
        culture_labels = None
        if 'culture_labels' in batch:
            culture_labels = batch['culture_labels'].to(device)
        elif 'label' in batch:
            if isinstance(batch['label'], list):
                label_ints = [int(label) if label.isdigit() else 0 for label in batch['label']]
                culture_labels = torch.tensor(label_ints, dtype=torch.long, device=device)
            else:
                culture_labels = batch['label'].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            culture_labels=culture_labels,
            return_dict=True
        )

        # 获取各种损失
        lm_loss = outputs.loss
        expert_weights = getattr(outputs, 'expert_weights', None)
        moe_aux_loss = getattr(outputs, 'moe_aux_loss', None)
        if moe_aux_loss is None:
            moe_aux_loss = torch.tensor(0.0, device=device, dtype=lm_loss.dtype, requires_grad=True)

        culture_loss = getattr(outputs, 'culture_loss', torch.tensor(0.0, device=device, dtype=lm_loss.dtype, requires_grad=True))

        # 🔧 获取监控信息
        monitoring_info = getattr(outputs, 'monitoring_info', {})

        # 总损失
        total_batch_loss = outputs.loss

        # 检查 NaN/Inf loss
        if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
            if rank == 0:
                print(f"⚠️ Batch {batch_idx}: NaN/Inf loss detected, 跳过此batch")
            continue

        # 梯度累积
        total_batch_loss = total_batch_loss / num_accumulation_steps
        total_batch_loss.backward()

        total_loss += total_batch_loss.item() * num_accumulation_steps
        total_lm_loss += lm_loss.item()
        total_moe_loss += moe_aux_loss.item()
        total_culture_loss += culture_loss.item()
        num_batches += 1

        # 梯度更新
        if (batch_idx + 1) % num_accumulation_steps == 0:
            # 检查和清理NaN/Inf梯度
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"⚠️ Cleaning NaN/Inf gradient in {name}")
                        param.grad.zero_()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

        # 内存清理
        if (batch_idx + 1) % 1 == 0:
            torch.cuda.empty_cache()

        # 检查内存使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            if memory_allocated > 35:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if rank == 0:
                    print(f"⚠️ High memory usage ({memory_allocated:.1f}GB), forced cleanup")

        # 更新进度条
        postfix = {
            'loss': f"{total_batch_loss.item() * num_accumulation_steps:.6f}",
            'lm': f"{lm_loss.item():.6f}",
            'moe': f"{moe_aux_loss.item():.6f}"
        }
        if use_culture_loss:
            postfix['culture'] = f"{culture_loss.item():.4f}"

        # 🔧 添加监控信息到进度条
        if monitoring_info:
            fusion_gate = monitoring_info.get('fusion_gate', 0.0)
            router_entropy = monitoring_info.get('router_entropy', 0.0)
            postfix['gate'] = f"{fusion_gate:.4f}"
            postfix['entropy'] = f"{router_entropy:.3f}"

        pbar.set_postfix(postfix)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_lm_loss = total_lm_loss / num_batches if num_batches > 0 else 0
    avg_moe_loss = total_moe_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'moe_loss': avg_moe_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def evaluate_joint_shared(model, val_loader, device, tokenizer, rank=0, use_culture_loss=True, culture_loss_weight=0.01):
    """
    联合模型验证（共享专家版本）
    """
    model.eval()
    total_loss = 0
    total_lm_loss = 0
    total_moe_loss = 0
    total_culture_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Evaluating Shared", disable=(rank != 0), mininterval=1.0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 检查batch是否有有效的训练标签
            total_valid_labels = (labels != -100).sum().item()
            if total_valid_labels == 0:
                if rank == 0:
                    print(f"⚠️ Eval Batch {batch_idx}: 所有标签都被mask，跳过此batch")
                continue

            # 处理labels masking
            for i in range(labels.shape[0]):
                instruction = batch['instruction'][i] if isinstance(batch['instruction'], list) else batch['instruction']
                input_text = batch['input'][i] if isinstance(batch['input'], list) else batch['input']

                if input_text:
                    input_part = f"{instruction}\n{input_text}\n"
                else:
                    input_part = f"{instruction}\n"

                input_tokens = tokenizer(input_part, add_special_tokens=False, truncation=False)['input_ids']
                input_length = len(input_tokens)

                if input_length < labels.shape[1]:
                    max_mask_length = min(input_length, labels.shape[1] - 5)
                    labels[i, :max_mask_length] = -100
                else:
                    labels[i, :-10] = -100

            # 获取文化标签
            culture_labels = None
            if 'culture_labels' in batch:
                culture_labels = batch['culture_labels'].to(device)
            elif 'label' in batch:
                if isinstance(batch['label'], list):
                    label_ints = [int(label) if label.isdigit() else 0 for label in batch['label']]
                    culture_labels = torch.tensor(label_ints, dtype=torch.long, device=device)
                else:
                    culture_labels = batch['label'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                culture_labels=culture_labels,
                return_dict=True
            )

            lm_loss = outputs.loss
            expert_weights = getattr(outputs, 'expert_weights', None)
            moe_aux_loss = getattr(outputs, 'moe_aux_loss', torch.tensor(0.0, device=device, dtype=torch.float16))
            culture_loss = getattr(outputs, 'culture_loss', torch.tensor(0.0, device=device, dtype=torch.float16))

            lm_loss = lm_loss.to(dtype=torch.float16)
            total_batch_loss = outputs.loss

            # 检查总损失是否为NaN/Inf
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                continue

            total_loss += total_batch_loss.item()
            total_lm_loss += lm_loss.item()
            total_moe_loss += moe_aux_loss.item()
            total_culture_loss += culture_loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{total_batch_loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_lm_loss = total_lm_loss / num_batches if num_batches > 0 else 0
    avg_moe_loss = total_moe_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'moe_loss': avg_moe_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def generate_and_evaluate_answers_joint_shared(
    model, val_dataset, tokenizer, device, output_dir, epoch=None, rank=0
):
    """
    联合模型生成答案并评估准确率（共享专家版本）
    """
    model.eval()

    correct = 0
    total = 0
    generated_data = []

    for idx in tqdm(range(len(val_dataset)), desc="Generating Shared", disable=(rank != 0), mininterval=1.0):
        # 获取原始数据集
        if hasattr(val_dataset, 'dataset'):
            original_idx = val_dataset.indices[idx]
            sample = val_dataset.dataset[original_idx]
        else:
            sample = val_dataset[idx]

        instruction = sample['instruction']
        input_text = sample['input']
        true_output = sample['output']
        label = sample['label']

        # 生成答案
        full_model = model.module if hasattr(model, 'module') else model
        generated_text = generate_answer(
            full_model, tokenizer, instruction, input_text, device
        )

        # 提取答案
        predicted_answer = extract_answer_from_text(generated_text)

        # 比对答案
        if predicted_answer == true_output:
            correct += 1
        total += 1

        # 保存生成的数据
        generated_data.append({
            'instruction': instruction,
            'input': input_text,
            'true_output': true_output,
            'label': label,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': predicted_answer == true_output
        })

    accuracy = correct / total if total > 0 else 0

    # 保存生成的答案
    with open(os.path.join(output_dir, 'generated_answers_shared.json'), 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    if epoch is not None:
        epoch_answers_file = os.path.join(output_dir, f'generated_answers_shared_epoch_{epoch}.json')
        with open(epoch_answers_file, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # 打印前三条生成的答案
    if rank == 0:
        print("\n📋 生成答案样例 (共享专家版本):")
        for idx in range(min(3, len(generated_data))):
            item = generated_data[idx]
            correct_mark = '✅' if item['correct'] else '❌'
            print(f"  样本{idx+1}: 真实={item['true_output']}, 预测={item['predicted_answer']}, 生成='{item['generated_text'][:20]}...' {correct_mark}")
        print()

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    # 初始化分布式训练
    rank, world_size, local_rank = setup_distributed()

    parser = argparse.ArgumentParser(description="Joint LoRA + MoE + Shared Expert Training")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--data_id", type=str, required=True,
                        help="Data ID for dataset-specific optimizations")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--learning_rate_base", type=float, default=5e-5,
                        help="Learning rate for base LoRA")
    parser.add_argument("--learning_rate_moe", type=float, default=1e-4,
                        help="Learning rate for MoE components")
    parser.add_argument("--learning_rate_shared", type=float, default=2e-5,
                        help="Learning rate for shared expert")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval (every N epochs)")

    # 模型参数
    parser.add_argument("--backbone", type=str, default="qwen", choices=["llama", "qwen"],
                        help="Model backbone type")
    parser.add_argument("--num_moe_experts", type=int, default=4,
                        help="Number of MoE experts")
    parser.add_argument("--use_culture_loss", type=str, default="true",
                        help="Whether to use culture loss")
    parser.add_argument("--culture_loss_weight", type=float, default=0.01,
                        help="Culture loss weight")

    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    # 共享专家参数
    parser.add_argument("--use_shared_expert", type=str, default="true",
                        help="Whether to use shared expert (always true in this script)")
    parser.add_argument("--shared_lora_rank", type=int, default=4,
                        help="Shared expert LoRA rank")
    parser.add_argument("--shared_expert_weight", type=float, default=0.1,
                        help="Shared expert weight in fusion")
    parser.add_argument("--routed_expert_weight", type=float, default=0.9,
                        help="Routed expert weight in fusion")

    parser.add_argument("--memory_efficient", action='store_true',
                        help="Enable memory efficient training")

    args = parser.parse_args()

    # 转换字符串参数
    use_culture_loss = args.use_culture_loss.lower() == 'true'
    use_shared_expert = args.use_shared_expert.lower() == 'true'

    # 共享专家版本强制启用共享专家
    if not use_shared_expert:
        print("⚠️ 警告：在共享专家版本中，强制启用共享专家功能")
        use_shared_expert = True

    # 根据backbone和data_id组合设置MoE影响权重
    if args.backbone == "llama":
        moe_influence_weight = 0.5
        if is_main_process(rank):
            print(f"🔧 LLaMA backbone: MoE影响权重设置为 {moe_influence_weight}")
    elif args.backbone == "qwen":
        if args.data_id == "2":  # CulturalBench
            moe_influence_weight = 0.05
        elif args.data_id in ["4", "1"]:  # CultureLLM或其他短文本数据集
            moe_influence_weight = 0.1
        elif args.data_id == "3":  # Normad
            moe_influence_weight = 0.2
        else:
            moe_influence_weight = 0.2

        if is_main_process(rank):
            dataset_name = {"2": "CulturalBench", "3": "Normad", "4": "CultureLLM", "1": "其他"}.get(args.data_id, "未知")
            print(f"🔧 Qwen backbone + {dataset_name} (DATA_ID={args.data_id}): MoE影响权重设置为 {moe_influence_weight}")
    else:
        moe_influence_weight = 0.5
        if is_main_process(rank):
            print(f"🔧 未知backbone: 使用默认MoE影响权重 {moe_influence_weight}")

    # 设置内存优化
    if world_size > 1:
        args.memory_efficient = True

    if args.memory_efficient:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if world_size > 1:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)

    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        print("\n" + "="*80)
        print("联合训练 LoRA + MoE + 共享专家")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Base model: {args.base_model_path}")
        print(f"Backbone: {args.backbone}")
        print(f"Training data: {args.train_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size} (per GPU)")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"Base LoRA learning rate: {args.learning_rate_base}")
        print(f"MoE learning rate: {args.learning_rate_moe}")
        print(f"Shared expert learning rate: {args.learning_rate_shared}")
        print(f"Max length: {args.max_length}")
        print(f"MoE experts: {args.num_moe_experts} + 1 shared")
        print(f"Shared expert weight: {args.shared_expert_weight}")
        print(f"Routed expert weight: {args.routed_expert_weight}")
        print(f"Use shared expert: {use_shared_expert}")
        print(f"Use culture loss: {use_culture_loss}")
        if use_culture_loss:
            print(f"Culture loss weight: {args.culture_loss_weight}")
        print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")
        print(f"Shared LoRA config: rank={args.shared_lora_rank}")
        print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer (复用原有的tokenizer配置)
    print("Loading tokenizer...")
    print("🚨 使用共享专家版本的train_joint_lora_moe_shared.py")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    # 复用原有的tokenizer修复代码
    print(f"🔧 原始tokenizer状态: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")

    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128009:
        print(f"🔧 检测到Llama 3.1模型，查找官方padding token...")
        official_pad_token = "<|finetune_right_pad_id|>"
        try:
            pad_token_id = tokenizer.convert_tokens_to_ids(official_pad_token)
            if pad_token_id != tokenizer.unk_token_id and pad_token_id is not None:
                tokenizer.pad_token = official_pad_token
                tokenizer.pad_token_id = pad_token_id
                print(f"✅ Llama 3.1: 使用官方padding token: '{official_pad_token}' (id={pad_token_id})")
            else:
                raise ValueError("Official pad token not found or invalid")
        except Exception as e:
            print(f"⚠️ 无法找到官方padding token，使用安全字符...")
            safe_tokens = ['~', '`', '|', '^', '§', '¶']
            for safe_token in safe_tokens:
                try:
                    safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                    if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                        tokenizer.pad_token = safe_token
                        tokenizer.pad_token_id = safe_token_id
                        print(f"🔧 Llama 3.1: 使用安全字符 '{safe_token}' (id={safe_token_id}) 作为padding")
                        break
                except:
                    continue
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"🔧 标准配置: pad_token = eos_token")

    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded and validated")

    # 加载数据
    print("\nLoading and processing data...")
    datasets = load_and_process_data(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        val_split=args.val_split
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    print("✅ Data loaded")

    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    # 创建数据加载器
    def collate_fn(batch):
        return dynamic_padding_collate_fn(batch, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 加载基础模型
    if is_main_process(rank):
        print("\nLoading base model...")

    load_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': None,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True
    }

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **load_kwargs)
    torch.cuda.empty_cache()
    base_model = base_model.to(device)
    torch.cuda.empty_cache()

    # 启用梯度检查点
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
        if is_main_process(rank):
            print("✅ Gradient checkpointing enabled")
    elif hasattr(base_model, 'config'):
        base_model.config.use_cache = False
        if is_main_process(rank):
            print("✅ KV cache disabled for gradient checkpointing")

    if is_main_process(rank):
        print("✅ Base model loaded")

    # 创建联合LoRA+MoE+共享专家配置
    if is_main_process(rank):
        print(f"\nConfiguring Joint LoRA + MoE + Shared Expert...")

    joint_shared_config = JointLoRAMoESharedConfig(
        # LoRA配置
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        # MoE配置
        num_moe_experts=args.num_moe_experts,
        moe_hidden_dim=2048,
        moe_influence_weight=moe_influence_weight,

        # 共享专家配置
        use_shared_expert=use_shared_expert,
        shared_lora_rank=args.shared_lora_rank,
        shared_expert_weight=args.shared_expert_weight,
        routed_expert_weight=args.routed_expert_weight,
        shared_expert_lr=args.learning_rate_shared,

        # 文化损失配置
        use_culture_loss=use_culture_loss,
        culture_loss_weight=args.culture_loss_weight,

        # 其他配置
        dropout=0.1
    )

    # 创建联合模型
    model = JointLoRAMoESharedModel(base_model, joint_shared_config)

    # 确保梯度检查点在联合模型中仍然有效
    if hasattr(model.base_model, 'gradient_checkpointing_enable'):
        model.base_model.gradient_checkpointing_enable()
        if is_main_process(rank):
            print("✅ Gradient checkpointing re-enabled for joint shared model")

    if is_main_process(rank):
        model.print_trainable_parameters()
        print("✅ Joint LoRA + MoE + Shared Expert configured")

    # 确保所有参数在正确设备上
    torch.cuda.empty_cache()

    # 使用DDP包装模型（仅在多GPU时）
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )
        model._set_static_graph()
        if is_main_process(rank):
            print("✅ Model wrapped with DDP with static graph")

    # 设置分层优化器（共享专家版本）
    if hasattr(model, 'module'):
        param_groups = model.module.get_parameter_groups(
            base_lr=args.learning_rate_base,
            moe_lr=args.learning_rate_moe,
            shared_lr=args.learning_rate_shared
        )
    else:
        param_groups = model.get_parameter_groups(
            base_lr=args.learning_rate_base,
            moe_lr=args.learning_rate_moe,
            shared_lr=args.learning_rate_shared
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # 训练循环
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Starting joint shared training...")
        print("="*80 + "\n")

    best_eval_accuracy = 0.0
    best_model_dir = os.path.join(args.output_dir, 'best_joint_shared_model')
    epoch_results = []

    for epoch in range(args.num_epochs):
        # 设置分布式采样器的epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process(rank):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_metrics = train_epoch_joint_shared(
            model, train_loader, optimizer, device, tokenizer,
            num_accumulation_steps=args.gradient_accumulation_steps,
            rank=rank,
            use_culture_loss=use_culture_loss,
            culture_loss_weight=args.culture_loss_weight
        )

        if is_main_process(rank):
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    LM Loss: {train_metrics['lm_loss']:.4f}")
            print(f"    MoE Loss: {train_metrics['moe_loss']:.4f}")
            if use_culture_loss:
                print(f"    Culture Loss: {train_metrics['culture_loss']:.4f}")

            # 打印MoE层NaN统计信息
            try:
                actual_model = model.module if hasattr(model, 'module') else model
                if hasattr(actual_model, 'moe_layer'):
                    nan_rate, nan_count, total_calls = actual_model.moe_layer.get_nan_stats()
                    print(f"    MoE NaN统计: {nan_count}/{total_calls} ({nan_rate:.2%}) - Epoch {epoch + 1}")
                    if nan_count > 0:
                        print(f"      ⚠️ 检测到 {nan_count} 次NaN，占总前向传播的 {nan_rate:.2%}")
                    actual_model.moe_layer.reset_nan_stats()
            except Exception as e:
                print(f"    ⚠️ 无法获取MoE NaN统计: {e}")

        # 每eval_interval个epoch进行一次验证
        if (epoch + 1) % args.eval_interval == 0:
            # 验证
            val_metrics = evaluate_joint_shared(
                model, val_loader, device, tokenizer, rank=rank,
                use_culture_loss=use_culture_loss,
                culture_loss_weight=args.culture_loss_weight
            )

            # 生成答案并评估准确率（只在主进程执行）
            if is_main_process(rank):
                gen_metrics = generate_and_evaluate_answers_joint_shared(
                    model, val_dataset, tokenizer, device, args.output_dir, epoch=epoch+1, rank=rank
                )
            else:
                gen_metrics = {'accuracy': 0.0, 'correct': 0, 'total': 0}

            # 同步所有进程
            if world_size > 1:
                dist.barrier()

            if is_main_process(rank):
                print(f"  Eval Loss: {val_metrics['loss']:.4f}")
                print(f"    LM Loss: {val_metrics['lm_loss']:.4f}")
                print(f"    MoE Loss: {val_metrics['moe_loss']:.4f}")
                if use_culture_loss:
                    print(f"    Culture Loss: {val_metrics['culture_loss']:.4f}")
                print(f"  Eval Accuracy: {gen_metrics['accuracy']:.4f} ({gen_metrics['correct']}/{gen_metrics['total']})")

                # 根据accuracy保存最好的模型
                if gen_metrics['accuracy'] > best_eval_accuracy:
                    best_eval_accuracy = gen_metrics['accuracy']

                    # 删除旧的最好模型
                    if os.path.exists(best_model_dir):
                        import shutil
                        shutil.rmtree(best_model_dir)

                    # 保存新的最好模型
                    os.makedirs(best_model_dir, exist_ok=True)

                    # 保存模型权重
                    if hasattr(model, 'module'):
                        model.module.save_model(best_model_dir)
                    else:
                        model.save_model(best_model_dir)

                    # 保存tokenizer
                    tokenizer.save_pretrained(best_model_dir)

                    print(f"  ✅ Best shared model saved (accuracy: {best_eval_accuracy:.4f})")

            # 记录结果
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_lm_loss': train_metrics['lm_loss'],
                'train_moe_loss': train_metrics['moe_loss'],
                'train_culture_loss': train_metrics.get('culture_loss', 0),
                'eval_loss': val_metrics['loss'],
                'eval_lm_loss': val_metrics['lm_loss'],
                'eval_moe_loss': val_metrics['moe_loss'],
                'eval_culture_loss': val_metrics.get('culture_loss', 0),
                'eval_accuracy': gen_metrics['accuracy'],
                'correct': gen_metrics['correct'],
                'total': gen_metrics['total'],
                'is_best': gen_metrics['accuracy'] == best_eval_accuracy
            })
        else:
            # 不评估的epoch，只记录训练损失
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_lm_loss': train_metrics['lm_loss'],
                'train_moe_loss': train_metrics['moe_loss'],
                'train_culture_loss': train_metrics.get('culture_loss', 0),
                'eval_loss': None,
                'eval_lm_loss': None,
                'eval_moe_loss': None,
                'eval_culture_loss': None,
                'eval_accuracy': None,
                'correct': None,
                'total': None,
                'is_best': False
            })

    # 保存训练结果（只在主进程执行）
    if is_main_process(rank):
        with open(os.path.join(args.output_dir, 'epoch_eval_results_shared.json'), 'w', encoding='utf-8') as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)

        # 保存配置
        config = {
            'base_model': args.base_model_path,
            'backbone': args.backbone,
            'training_mode': 'joint_lora_moe_shared',
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'effective_batch_size': args.batch_size * world_size * args.gradient_accumulation_steps,
            'world_size': world_size,
            'learning_rate_base': args.learning_rate_base,
            'learning_rate_moe': args.learning_rate_moe,
            'learning_rate_shared': args.learning_rate_shared,
            'max_length': args.max_length,
            'num_moe_experts': args.num_moe_experts,
            'shared_expert_config': {
                'shared_lora_rank': args.shared_lora_rank,
                'shared_expert_weight': args.shared_expert_weight,
                'routed_expert_weight': args.routed_expert_weight,
            },
            'use_shared_expert': use_shared_expert,
            'use_culture_loss': use_culture_loss,
            'culture_loss_weight': args.culture_loss_weight,
            'lora_config': {
                'rank': args.lora_rank,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout
            },
            'eval_interval': args.eval_interval,
            'best_eval_accuracy': best_eval_accuracy,
            'architecture': 'joint_lora_moe_shared_expert'
        }

        with open(os.path.join(args.output_dir, 'config_shared.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print("\n" + "="*80)
        print("✅ Joint shared training completed!")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")
        print(f"\nFiles generated:")
        print(f"  - best_joint_shared_model/ (Best model weights)")
        print(f"  - epoch_eval_results_shared.json (Epoch-by-epoch results)")
        print(f"  - generated_answers_shared.json (Generated answers on validation set)")
        print(f"  - config_shared.json (Training configuration)")
        print(f"\nBest validation accuracy: {best_eval_accuracy:.4f}")
        print(f"Architecture: Joint LoRA + MoE + Shared Expert")
        print(f"MoE experts: {args.num_moe_experts} routed + 1 shared")
        print(f"Shared expert weight: {args.shared_expert_weight}")
        print(f"Routed expert weight: {args.routed_expert_weight}")
        print(f"Culture loss: {'enabled' if use_culture_loss else 'disabled'}")
        print("="*80)

    # 清理分布式训练
    cleanup_distributed()


if __name__ == "__main__":
    main()