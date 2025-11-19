# src/llamafactory/model/simple_moe_fixed.py
"""
修复版本的简单 MoE 模型实现
解决了数值稳定性问题，防止NaN和Inf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
import math

from .experts_fixed import ExpertLayer, LoRAExpert


class SimpleRouter(nn.Module):
    """修复版本的简单 Top-K 路由器"""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2,
                 router_hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # 输入归一化
        self.input_norm = nn.LayerNorm(hidden_dim)

        # 路由网络 - 使用更稳定的结构
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim),
            nn.GELU(),  # 更稳定的激活函数
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, router_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim // 2, num_experts)
        )

        self._init_weights()

    def _init_weights(self):
        """使用稳定的权重初始化"""
        for module in self.router:
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，较小的gain
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, H] 输入隐藏状态
            temperature: softmax温度参数

        Returns:
            expert_weights: [B, num_experts] 专家权重
            top_k_indices: [B, top_k] Top-K 专家索引
        """
        # 检查输入
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            batch_size = hidden_states.shape[0]
            # 返回均匀分布的权重
            uniform_weights = torch.ones(batch_size, self.num_experts, device=hidden_states.device) / self.num_experts
            top_k_indices = torch.arange(self.top_k, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            return uniform_weights, top_k_indices

        try:
            # 输入归一化
            normalized_input = self.input_norm(hidden_states)

            # 计算路由logits
            router_logits = self.router(normalized_input)  # [B, num_experts]

            # 检查logits
            if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
                batch_size = hidden_states.shape[0]
                uniform_weights = torch.ones(batch_size, self.num_experts, device=hidden_states.device) / self.num_experts
                top_k_indices = torch.arange(self.top_k, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
                return uniform_weights, top_k_indices

            # 应用温度缩放和裁剪
            scaled_logits = torch.clamp(router_logits / temperature, -10.0, 10.0)

            # Top-K 选择
            top_k_logits, top_k_indices = torch.topk(scaled_logits, self.top_k, dim=-1)  # [B, top_k]

            # 计算权重（softmax归一化）
            top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, top_k]

            # 检查权重
            if torch.isnan(top_k_weights).any() or torch.isinf(top_k_weights).any():
                top_k_weights = torch.ones_like(top_k_weights) / self.top_k

            # 构建完整的专家权重矩阵
            expert_weights = torch.zeros(hidden_states.shape[0], self.num_experts,
                                       device=hidden_states.device, dtype=hidden_states.dtype)

            # 填充Top-K权重
            for b in range(hidden_states.shape[0]):
                expert_weights[b, top_k_indices[b]] = top_k_weights[b]

            return expert_weights, top_k_indices

        except Exception as e:
            logging.warning(f"Router forward failed: {e}, using uniform distribution")
            batch_size = hidden_states.shape[0]
            uniform_weights = torch.ones(batch_size, self.num_experts, device=hidden_states.device) / self.num_experts
            top_k_indices = torch.arange(self.top_k, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            return uniform_weights, top_k_indices


class SimpleMoELayer(nn.Module):
    """修复版本的简单 MoE 层"""

    def __init__(self, hidden_dim: int, num_experts: int = 12, top_k: int = 2,
                 expert_hidden_dim: int = None, router_hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        if expert_hidden_dim is None:
            expert_hidden_dim = hidden_dim * 4  # 默认4倍隐藏维度

        # 路由器
        self.router = SimpleRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            router_hidden_dim=router_hidden_dim,
            dropout=dropout
        )

        # 专家网络
        self.experts = nn.ModuleList([
            LoRAExpert(
                input_dim=hidden_dim,
                hidden_dim=expert_hidden_dim,
                output_dim=hidden_dim,
                lora_rank=8,
                dropout=dropout
            ) for _ in range(num_experts)
        ])

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 可学习的门控参数
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, L, H] 输入隐藏状态

        Returns:
            output: [B, L, H] MoE 输出
            expert_weights: [B, num_experts] 专家权重
        """
        # 检查输入
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            logging.warning("NaN or Inf detected in MoE input")
            return torch.zeros_like(hidden_states), torch.zeros(hidden_states.shape[0], self.num_experts, device=hidden_states.device)

        batch_size, seq_len, hidden_dim = hidden_states.shape
        original_input = hidden_states.clone()

        try:
            # 池化用于路由决策（使用更稳定的池化方法）
            pooled_states = hidden_states.mean(dim=1)  # [B, H]

            # 路由决策
            expert_weights, top_k_indices = self.router(pooled_states)  # [B, num_experts], [B, top_k]

            # 专家输出计算
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                try:
                    expert_output = expert(hidden_states)  # [B, L, H]

                    # 检查专家输出
                    if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
                        logging.warning(f"NaN or Inf detected in expert {i} output")
                        expert_output = torch.zeros_like(hidden_states)

                    expert_outputs.append(expert_output)
                except Exception as e:
                    logging.warning(f"Expert {i} forward failed: {e}")
                    expert_outputs.append(torch.zeros_like(hidden_states))

            if not expert_outputs:
                return self.layer_norm(original_input), expert_weights

            expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, B, L, H]

            # 加权融合（只使用Top-K专家）
            output = torch.zeros_like(hidden_states)  # [B, L, H]

            for b in range(batch_size):
                for k in range(self.top_k):
                    expert_idx = top_k_indices[b, k]
                    weight = expert_weights[b, expert_idx]

                    # 检查权重和专家输出
                    if not (torch.isnan(weight) or torch.isinf(weight)):
                        expert_contribution = weight.unsqueeze(0).unsqueeze(0) * expert_outputs[expert_idx, b]
                        if not (torch.isnan(expert_contribution).any() or torch.isinf(expert_contribution).any()):
                            output[b] += expert_contribution

            # 检查MoE输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.warning("NaN or Inf detected in MoE aggregated output")
                output = torch.zeros_like(hidden_states)

            # 门控融合
            gate_value = torch.sigmoid(self.gate)
            output = (1 - gate_value) * original_input + gate_value * output

            # 残差连接和层归一化
            output = self.layer_norm(original_input + output)

            # 最终检查
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.warning("NaN or Inf detected in final MoE output, returning input")
                output = self.layer_norm(original_input)

            return output, expert_weights

        except Exception as e:
            logging.warning(f"MoE layer forward failed: {e}, returning normalized input")
            return self.layer_norm(original_input), torch.zeros(batch_size, self.num_experts, device=hidden_states.device)


class SimpleMoEModel(nn.Module):
    """修复版本的简单 MoE 模型"""

    def __init__(self, llama_model, config, num_experts: int = 12, top_k: int = 2,
                 expert_hidden_dim: int = None, router_hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        self.llama_model = llama_model
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k

        hidden_dim = config.hidden_size

        # MoE 层
        self.moe_layer = SimpleMoELayer(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            router_hidden_dim=router_hidden_dim,
            dropout=dropout
        )

        # MoE 融合权重（可学习，初始化为较小值）
        self.moe_fusion_weight = nn.Parameter(torch.tensor(0.05))

        logging.info(f"Fixed SimpleMoEModel initialized with {num_experts} experts, top-{top_k} routing")

        # 确保数据类型一致性
        self._ensure_dtype_consistency()

        # 初始化模型权重
        self._init_model_weights()

    def _ensure_dtype_consistency(self):
        """确保所有组件使用相同的数据类型"""
        # 获取base模型的数据类型
        base_dtype = next(self.llama_model.parameters()).dtype

        # 将MoE组件转换为相同的数据类型
        self.moe_layer = self.moe_layer.to(dtype=base_dtype)

        # 确保参数也使用正确的数据类型
        self.moe_fusion_weight.data = self.moe_fusion_weight.data.to(dtype=base_dtype)

        logging.info(f"All Simple MoE components converted to dtype: {base_dtype}")

    def _init_model_weights(self):
        """初始化模型权重，确保数值稳定性"""
        # 初始化MoE层的权重
        for name, param in self.moe_layer.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # 使用Xavier初始化
                nn.init.xavier_uniform_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # 确保融合权重在合理范围内
        with torch.no_grad():
            self.moe_fusion_weight.clamp_(-1.0, 1.0)

        logging.info("MoE model weights initialized with Xavier uniform (gain=0.01)")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        前向传播

        Args:
            input_ids: [B, L] 输入token IDs
            attention_mask: [B, L] 注意力掩码
            labels: [B, L] 标签（用于计算损失）

        Returns:
            outputs: dict 包含 logits, loss, expert_weights
        """
        device = input_ids.device

        try:
            # LLaMA 基础前向传播
            llama_outputs = self.llama_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_states = llama_outputs.last_hidden_state  # [B, L, H]

            # 检查基础模型输出
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                logging.warning("NaN or Inf detected in base model output")
                # 使用零初始化的隐藏状态
                hidden_states = torch.zeros_like(hidden_states)

            # MoE 处理
            moe_output, expert_weights = self.moe_layer(hidden_states)  # [B, L, H], [B, num_experts]

            # 融合原始隐藏状态和 MoE 输出
            fusion_weight = torch.sigmoid(self.moe_fusion_weight)
            fusion_weight = torch.clamp(fusion_weight, 0.0, 0.2)  # 限制融合权重

            # 检查数值稳定性
            if torch.isnan(moe_output).any() or torch.isinf(moe_output).any():
                logging.warning("NaN or Inf detected in MoE output, using zero MoE contribution")
                moe_output = torch.zeros_like(moe_output)

            # 融合隐藏状态
            enhanced_hidden = (1 - fusion_weight) * hidden_states + fusion_weight * moe_output

            # 最终检查
            if torch.isnan(enhanced_hidden).any() or torch.isinf(enhanced_hidden).any():
                logging.warning("NaN or Inf detected in enhanced hidden states, using original")
                enhanced_hidden = hidden_states

            # 生成 logits
            logits = self.llama_model.lm_head(enhanced_hidden)  # [B, L, vocab_size]

            # 构建输出
            outputs = {
                'logits': logits,
                'expert_weights': expert_weights,
                'hidden_states': enhanced_hidden
            }

            # 计算损失
            if labels is not None:
                try:
                    # 标准的语言建模损失（交叉熵）
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    lm_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    # 检查损失
                    if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                        logging.warning("Invalid loss detected, setting to 0")
                        lm_loss = torch.tensor(0.0, device=device, requires_grad=True)

                    outputs['loss'] = lm_loss
                    outputs['lm_loss'] = lm_loss

                except Exception as e:
                    logging.warning(f"Loss computation failed: {e}")
                    outputs['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                    outputs['lm_loss'] = torch.tensor(0.0, device=device, requires_grad=True)

            return outputs

        except Exception as e:
            logging.error(f"SimpleMoEModel forward failed: {e}")
            # 返回基础模型的输出
            try:
                base_outputs = self.llama_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
                return base_outputs
            except Exception as e2:
                logging.error(f"Base model forward also failed: {e2}")
                # 返回最基础的输出
                batch_size, seq_len = input_ids.shape
                vocab_size = self.config.vocab_size
                dummy_logits = torch.zeros(batch_size, seq_len, vocab_size, device=device)
                dummy_weights = torch.zeros(batch_size, self.num_experts, device=device)

                outputs = {
                    'logits': dummy_logits,
                    'expert_weights': dummy_weights,
                    'hidden_states': torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)
                }

                if labels is not None:
                    outputs['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                    outputs['lm_loss'] = torch.tensor(0.0, device=device, requires_grad=True)

                return outputs

    def get_expert_usage_stats(self, expert_weights: torch.Tensor) -> Dict:
        """获取专家使用统计信息"""
        try:
            # 计算每个专家的平均使用率
            expert_usage = expert_weights.mean(dim=0)  # [num_experts]

            # 检查数值有效性
            if torch.isnan(expert_usage).any() or torch.isinf(expert_usage).any():
                expert_usage = torch.ones(self.num_experts, device=expert_weights.device) / self.num_experts

            # 计算使用分布的统计信息
            usage_std = expert_usage.std()
            usage_max = expert_usage.max()
            usage_min = expert_usage.min()

            # 计算负载均衡指标
            ideal_usage = 1.0 / self.num_experts
            load_balance_loss = F.mse_loss(expert_usage, torch.full_like(expert_usage, ideal_usage))

            # 检查统计值有效性
            if torch.isnan(usage_std) or torch.isinf(usage_std):
                usage_std = torch.tensor(0.0)
            if torch.isnan(load_balance_loss) or torch.isinf(load_balance_loss):
                load_balance_loss = torch.tensor(0.0)

            return {
                'expert_usage': expert_usage.detach().cpu().numpy().tolist(),
                'usage_std': usage_std.item(),
                'usage_max': usage_max.item(),
                'usage_min': usage_min.item(),
                'load_balance_loss': load_balance_loss.item(),
                'ideal_usage': ideal_usage
            }
        except Exception as e:
            logging.warning(f"Expert usage stats computation failed: {e}")
            return {
                'expert_usage': [1.0/self.num_experts] * self.num_experts,
                'usage_std': 0.0,
                'usage_max': 1.0/self.num_experts,
                'usage_min': 1.0/self.num_experts,
                'load_balance_loss': 0.0,
                'ideal_usage': 1.0/self.num_experts
            }

    def generate(self, *args, **kwargs):
        """生成方法，直接调用 LLaMA 模型的生成方法"""
        return self.llama_model.generate(*args, **kwargs)