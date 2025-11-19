# src/llamafactory/model/simple_moe.py
"""
简单的 MoE 模型实现
在 LoRA 微调后的完整模型基础上添加简单的 MoE 结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .experts import ExpertLayer, LoRAExpert


class SimpleRouter(nn.Module):
    """简单的 Top-K 路由器"""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2,
                 router_hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, num_experts)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.router:
            if isinstance(module, nn.Linear):
                # 使用更小的标准差进行初始化，避免梯度爆炸
                nn.init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, H] 输入隐藏状态

        Returns:
            expert_weights: [B, num_experts] 专家权重
            top_k_indices: [B, top_k] Top-K 专家索引
        """
        # 确保路由器在正确的设备上
        if next(self.router.parameters()).device != hidden_states.device:
            self.router = self.router.to(hidden_states.device)

        try:
            # 计算路由logits
            router_logits = self.router(hidden_states)  # [B, num_experts]

            # 检查路由logits的数值稳定性
            if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
                logging.warning("Router logits contain NaN/Inf, using uniform distribution")
                router_logits = torch.zeros_like(router_logits)

            # 裁剪logits到合理范围
            router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)

            # Top-K 选择
            top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [B, top_k]

            # 计算权重（softmax归一化，添加温度参数提高稳定性）
            temperature = 2.0  # 增加温度以提高稳定性
            top_k_weights = F.softmax(top_k_logits / temperature, dim=-1)  # [B, top_k]

            # 检查权重的数值稳定性
            if torch.isnan(top_k_weights).any() or torch.isinf(top_k_weights).any():
                logging.warning("Router weights contain NaN/Inf, using uniform weights")
                top_k_weights = torch.ones_like(top_k_weights) / self.top_k

        except Exception as e:
            logging.warning(f"Router forward failed: {e}, using uniform distribution")
            # 创建均匀分布的权重和随机索引
            top_k_weights = torch.ones(hidden_states.shape[0], self.top_k, device=hidden_states.device) / self.top_k
            top_k_indices = torch.randint(0, self.num_experts, (hidden_states.shape[0], self.top_k), device=hidden_states.device)

        # 构建完整的专家权重矩阵
        expert_weights = torch.zeros(hidden_states.shape[0], self.num_experts,
                                   device=hidden_states.device, dtype=hidden_states.dtype)

        # 填充Top-K权重
        for b in range(hidden_states.shape[0]):
            expert_weights[b, top_k_indices[b]] = top_k_weights[b]

        return expert_weights, top_k_indices


class SimpleMoELayer(nn.Module):
    """简单的 MoE 层"""

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

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, L, H] 输入隐藏状态

        Returns:
            output: [B, L, H] MoE 输出
            expert_weights: [B, num_experts] 专家权重
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 池化用于路由决策
        pooled_states = hidden_states.mean(dim=1)  # [B, H]

        # 路由决策
        expert_weights, top_k_indices = self.router(pooled_states)  # [B, num_experts], [B, top_k]

        # 专家输出计算
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 确保专家在正确的设备上
            if next(expert.parameters()).device != hidden_states.device:
                expert = expert.to(hidden_states.device)

            try:
                expert_output = expert(hidden_states)  # [B, L, H]

                # 检查专家输出的数值稳定性
                if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
                    logging.warning(f"Expert {i} output contains NaN/Inf, using zero output")
                    expert_output = torch.zeros_like(hidden_states)

                # 裁剪极值
                expert_output = torch.clamp(expert_output, min=-10.0, max=10.0)
                expert_outputs.append(expert_output)

            except Exception as e:
                logging.warning(f"Expert {i} forward failed: {e}, using zero output")
                expert_outputs.append(torch.zeros_like(hidden_states))

        expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, B, L, H]

        # 加权融合（只使用Top-K专家）
        output = torch.zeros_like(hidden_states)  # [B, L, H]

        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = top_k_indices[b, k]
                weight = expert_weights[b, expert_idx]

                # 检查权重的数值稳定性
                if torch.isnan(weight) or torch.isinf(weight):
                    continue  # 跳过无效权重

                # 裁剪权重到合理范围
                weight = torch.clamp(weight, min=0.0, max=1.0)

                weighted_output = weight.unsqueeze(0) * expert_outputs[expert_idx, b]

                # 检查加权输出
                if torch.isnan(weighted_output).any() or torch.isinf(weighted_output).any():
                    continue  # 跳过无效输出

                output[b] += weighted_output

        # 最终数值稳定性检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            logging.warning("MoE output contains NaN/Inf before layer_norm, using input")
            output = torch.zeros_like(hidden_states)

        # 确保layer_norm在正确的设备上
        if next(self.layer_norm.parameters()).device != hidden_states.device:
            self.layer_norm = self.layer_norm.to(hidden_states.device)

        # 残差连接和层归一化
        residual_input = hidden_states + output

        # 检查残差连接后的结果
        if torch.isnan(residual_input).any() or torch.isinf(residual_input).any():
            logging.warning("Residual connection contains NaN/Inf, using original input")
            residual_input = hidden_states

        try:
            output = self.layer_norm(residual_input)

            # 检查layer_norm输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.warning("Layer norm output contains NaN/Inf, using input")
                output = hidden_states

        except Exception as e:
            logging.warning(f"Layer norm failed: {e}, using input")
            output = hidden_states

        return output, expert_weights


class SimpleMoEModel(nn.Module):
    """简单的 MoE 模型"""

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
        self.moe_fusion_weight = nn.Parameter(torch.tensor(0.1))

        logging.info(f"SimpleMoEModel initialized with {num_experts} experts, top-{top_k} routing")

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
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # 确保融合权重在合理范围内
        with torch.no_grad():
            self.moe_fusion_weight.clamp_(-2.0, 2.0)

        logging.info("MoE model weights initialized with Xavier uniform (gain=0.1)")

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

        # LLaMA 基础前向传播
        llama_outputs = self.llama_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = llama_outputs.last_hidden_state  # [B, L, H]

        # hidden_states现在应该有梯度，因为最后一层是可训练的

        # 确保MoE层在正确的设备上
        if next(self.moe_layer.parameters()).device != hidden_states.device:
            self.moe_layer = self.moe_layer.to(hidden_states.device)

        # MoE 处理
        moe_output, expert_weights = self.moe_layer(hidden_states)  # [B, L, H], [B, num_experts]

        # 融合原始隐藏状态和 MoE 输出
        fusion_weight = torch.sigmoid(self.moe_fusion_weight)

        # 检查数值稳定性
        if torch.isnan(moe_output).any() or torch.isinf(moe_output).any():
            logging.warning("NaN or Inf detected in MoE output, using zero MoE contribution")
            # 使用零张量替代MoE输出，保持梯度连接
            moe_output = torch.zeros_like(moe_output)

        # 融合隐藏状态
        enhanced_hidden = (1 - fusion_weight) * hidden_states + fusion_weight * moe_output

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
            # 标准的语言建模损失（交叉熵）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            outputs['loss'] = lm_loss
            outputs['lm_loss'] = lm_loss

        return outputs

    def get_expert_usage_stats(self, expert_weights: torch.Tensor) -> Dict:
        """获取专家使用统计信息"""
        # 计算每个专家的平均使用率
        expert_usage = expert_weights.mean(dim=0)  # [num_experts]

        # 计算使用分布的统计信息
        usage_std = expert_usage.std()
        usage_max = expert_usage.max()
        usage_min = expert_usage.min()

        # 计算负载均衡指标
        ideal_usage = 1.0 / self.num_experts
        load_balance_loss = F.mse_loss(expert_usage, torch.full_like(expert_usage, ideal_usage))

        return {
            'expert_usage': expert_usage.detach().cpu().numpy().tolist(),
            'usage_std': usage_std.item(),
            'usage_max': usage_max.item(),
            'usage_min': usage_min.item(),
            'load_balance_loss': load_balance_loss.item(),
            'ideal_usage': ideal_usage
        }

    def generate(self, *args, **kwargs):
        """生成方法，直接调用 LLaMA 模型的生成方法"""
        return self.llama_model.generate(*args, **kwargs)