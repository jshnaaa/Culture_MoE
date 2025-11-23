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
        """初始化权重 - 平衡稳定性和有效性"""
        for module in self.router:
            if isinstance(module, nn.Linear):
                # 使用更合理的初始化，保证有足够的信号强度
                nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 增加初始化方差
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # 适度限制初始权重范围，保持有效性
                with torch.no_grad():
                    module.weight.data.clamp_(-0.5, 0.5)  # 扩大权重范围
                    if module.bias is not None:
                        module.bias.data.clamp_(-0.1, 0.1)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, H] 输入隐藏状态

        Returns:
            expert_weights: [B, num_experts] 专家权重
            top_k_indices: [B, top_k] Top-K 专家索引
            load_balance_loss: 负载均衡损失
        """
        # 确保路由器在正确的设备上
        if next(self.router.parameters()).device != hidden_states.device:
            self.router = self.router.to(hidden_states.device)

        # 输入预处理：限制范围防止极端值
        hidden_states = torch.clamp(hidden_states, min=-5.0, max=5.0)

        # 确保路由器权重在合理范围内
        for module in self.router:
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.data.clamp_(-1.0, 1.0)
                    if module.bias is not None:
                        module.bias.data.clamp_(-1.0, 1.0)

        try:
            # 计算路由logits
            router_logits = self.router(hidden_states)  # [B, num_experts]

            # 强制限制router_logits范围
            router_logits = torch.clamp(router_logits, min=-5.0, max=5.0)

            # 检查路由logits的数值稳定性
            if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
                logging.warning("Router logits contain NaN/Inf, using uniform distribution")
                router_logits = torch.zeros_like(router_logits)

            # 数值稳定的Top-K选择
            try:
                top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [B, top_k]
            except:
                # 如果top-k失败，使用前k个专家
                top_k_indices = torch.arange(self.top_k, device=router_logits.device).unsqueeze(0).expand(router_logits.size(0), -1)
                top_k_logits = router_logits[:, :self.top_k]

            # 数值稳定的softmax归一化
            try:
                # 减去最大值提高数值稳定性
                top_k_logits_max = top_k_logits.max(dim=-1, keepdim=True)[0]
                top_k_logits_stable = top_k_logits - top_k_logits_max
                # 使用更高的温度提高稳定性
                temperature = 3.0
                top_k_weights = F.softmax(top_k_logits_stable / temperature, dim=-1)  # [B, top_k]
            except:
                # 如果softmax失败，使用均匀权重
                top_k_weights = torch.ones_like(top_k_logits) / self.top_k

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

        # 计算负载均衡损失
        load_balance_loss = self._compute_load_balance_loss(router_logits, expert_weights)

        return expert_weights, top_k_indices, load_balance_loss

    def _compute_load_balance_loss(self, router_logits: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        计算专家负载均衡损失

        Args:
            router_logits: [B, num_experts] 路由器原始logits
            expert_weights: [B, num_experts] 专家权重矩阵

        Returns:
            load_balance_loss: 负载均衡损失
        """
        try:
            batch_size = router_logits.size(0)

            # 计算每个专家的使用频率 (fraction of tokens routed to each expert)
            expert_usage_freq = expert_weights.sum(dim=0) / batch_size  # [num_experts]
            expert_usage_freq = torch.clamp(expert_usage_freq, min=1e-8, max=1.0)

            # 计算每个专家的平均概率 (average probability assigned to each expert)
            router_probs = F.softmax(router_logits, dim=-1)  # [B, num_experts]
            expert_avg_prob = router_probs.mean(dim=0)  # [num_experts]
            expert_avg_prob = torch.clamp(expert_avg_prob, min=1e-8, max=1.0)

            # 检查数值稳定性
            if (torch.isnan(expert_usage_freq).any() or torch.isinf(expert_usage_freq).any() or
                torch.isnan(expert_avg_prob).any() or torch.isinf(expert_avg_prob).any()):
                # 如果有NaN/Inf，返回零损失
                return torch.zeros(1, device=router_logits.device, dtype=router_logits.dtype, requires_grad=True).sum()

            # 负载均衡损失：鼓励专家使用的均匀分布
            # Loss = num_experts * sum(usage_freq * avg_prob)
            # 当专家使用均匀时，usage_freq ≈ avg_prob ≈ 1/num_experts，损失最小
            balance_product = expert_usage_freq * expert_avg_prob
            balance_product = torch.clamp(balance_product, max=1.0)
            load_balance_loss = self.num_experts * torch.sum(balance_product)

            # 添加方差惩罚，进一步鼓励均匀分布
            usage_variance = torch.var(expert_usage_freq)
            prob_variance = torch.var(expert_avg_prob)
            variance_penalty = usage_variance + prob_variance

            # 组合损失
            total_loss = load_balance_loss + 0.1 * variance_penalty

            # 最终检查和限制
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                return torch.zeros(1, device=router_logits.device, dtype=router_logits.dtype, requires_grad=True).sum()

            total_loss = torch.clamp(total_loss, min=0.0, max=50.0)

            return total_loss

        except Exception as e:
            logging.warning(f"Load balance loss computation failed: {e}, using zero loss")
            return torch.zeros(1, device=router_logits.device, dtype=router_logits.dtype, requires_grad=True).sum()


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

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 - 数值稳定版本

        Args:
            hidden_states: [B, L, H] 输入隐藏状态

        Returns:
            output: [B, L, H] MoE 输出
            expert_weights: [B, num_experts] 专家权重
            load_balance_loss: 负载均衡损失
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 输入预处理：限制范围防止极端值传播
        hidden_states = torch.clamp(hidden_states, min=-5.0, max=5.0)

        # 池化用于路由决策（使用更稳定的池化）
        pooled_states = hidden_states.mean(dim=1)  # [B, H]
        pooled_states = torch.clamp(pooled_states, min=-3.0, max=3.0)

        # 路由决策
        expert_weights, top_k_indices, load_balance_loss = self.router(pooled_states)  # [B, num_experts], [B, top_k], scalar

        # 专家输出计算（使用更严格的数值稳定性控制）
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 确保专家在正确的设备和数据类型上
            if next(expert.parameters()).device != hidden_states.device:
                expert = expert.to(hidden_states.device)
            if next(expert.parameters()).dtype != hidden_states.dtype:
                expert = expert.to(hidden_states.dtype)

            try:
                # 限制专家权重范围，防止权重爆炸
                for param in expert.parameters():
                    if param.requires_grad:
                        with torch.no_grad():
                            param.data.clamp_(-2.0, 2.0)

                expert_output = expert(hidden_states)  # [B, L, H]

                # 强制限制专家输出范围
                expert_output = torch.clamp(expert_output, min=-3.0, max=3.0)

                # 检查专家输出的数值稳定性
                if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
                    logging.warning(f"Expert {i} output contains NaN/Inf, using zero output")
                    expert_output = torch.zeros_like(hidden_states)

                expert_outputs.append(expert_output)

            except Exception as e:
                logging.warning(f"Expert {i} forward failed: {e}, using zero output")
                expert_outputs.append(torch.zeros_like(hidden_states))

        expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, B, L, H]

        # 加权融合（只使用Top-K专家，添加缩放因子）
        output = torch.zeros_like(hidden_states)  # [B, L, H]
        expert_scale = 0.1  # 缩放专家贡献，防止输出过大

        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = top_k_indices[b, k]
                weight = expert_weights[b, expert_idx]

                # 检查权重的数值稳定性
                if torch.isnan(weight) or torch.isinf(weight):
                    continue  # 跳过无效权重

                # 裁剪权重到更保守的范围
                weight = torch.clamp(weight, min=0.0, max=0.5)  # 限制单个专家最大贡献

                # 应用专家缩放
                scaled_weight = weight * expert_scale

                weighted_output = scaled_weight.unsqueeze(0) * expert_outputs[expert_idx, b]

                # 检查加权输出
                if torch.isnan(weighted_output).any() or torch.isinf(weighted_output).any():
                    continue  # 跳过无效输出

                # 再次限制加权输出
                weighted_output = torch.clamp(weighted_output, min=-1.0, max=1.0)
                output[b] += weighted_output

        # 限制MoE输出范围
        output = torch.clamp(output, min=-2.0, max=2.0)

        # 最终数值稳定性检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            logging.warning("MoE output contains NaN/Inf before layer_norm, using zero output")
            output = torch.zeros_like(hidden_states)

        # 确保layer_norm在正确的设备和数据类型上
        if next(self.layer_norm.parameters()).device != hidden_states.device:
            self.layer_norm = self.layer_norm.to(hidden_states.device)
        if next(self.layer_norm.parameters()).dtype != hidden_states.dtype:
            self.layer_norm = self.layer_norm.to(hidden_states.dtype)

        # 保守的残差连接（减少MoE贡献）
        residual_scale = 0.05  # 进一步减少MoE对残差的影响
        scaled_output = output * residual_scale
        residual_input = hidden_states + scaled_output

        # 限制残差连接后的范围
        residual_input = torch.clamp(residual_input, min=-5.0, max=5.0)

        # 检查残差连接后的结果
        if torch.isnan(residual_input).any() or torch.isinf(residual_input).any():
            logging.warning("Residual connection contains NaN/Inf, using original input")
            residual_input = hidden_states

        try:
            # 限制LayerNorm输入范围
            residual_input_norm = torch.clamp(residual_input, min=-3.0, max=3.0)
            output = self.layer_norm(residual_input_norm)

            # 检查layer_norm输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.warning("Layer norm output contains NaN/Inf, using input")
                output = hidden_states
            else:
                # 限制最终输出范围
                output = torch.clamp(output, min=-5.0, max=5.0)

        except Exception as e:
            logging.warning(f"Layer norm failed: {e}, using input")
            output = hidden_states

        return output, expert_weights, load_balance_loss


class SimpleMoEModel(nn.Module):
    """简单的 MoE 模型"""

    def __init__(self, llama_model, config, num_experts: int = 12, top_k: int = 2,
                 expert_hidden_dim: int = None, router_hidden_dim: int = 512,
                 dropout: float = 0.1, load_balance_weight: float = 0.01):
        super().__init__()

        self.llama_model = llama_model
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

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

        # MoE 融合权重（可学习，初始化为适中值平衡稳定性和有效性）
        self.moe_fusion_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))  # 适中的初始值

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
        """初始化模型权重，平衡稳定性和有效性"""
        # 初始化MoE层的权重（使用更合理的初始化）
        for name, param in self.moe_layer.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # 使用更合理的初始化，保证有效的信号传播
                nn.init.normal_(param, mean=0.0, std=0.02)  # 增加初始化方差
                # 适度限制初始权重范围
                with torch.no_grad():
                    param.data.clamp_(-0.5, 0.5)  # 扩大权重范围
            elif 'bias' in name:
                nn.init.zeros_(param)
                # 限制偏置范围
                with torch.no_grad():
                    param.data.clamp_(-0.1, 0.1)

        # 确保融合权重在合理的范围内
        with torch.no_grad():
            self.moe_fusion_weight.data = torch.tensor(0.1, dtype=self.moe_fusion_weight.dtype)  # 适中的初始融合权重
            self.moe_fusion_weight.clamp_(-2.0, 2.0)  # 扩大权重范围

        logging.info("MoE model weights initialized with balanced initialization (std=0.02)")

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

        # 确保MoE融合权重在合理范围内
        with torch.no_grad():
            self.moe_fusion_weight.data.clamp_(-1.0, 1.0)

        # MoE 处理
        moe_output, expert_weights, moe_load_balance_loss = self.moe_layer(hidden_states)  # [B, L, H], [B, num_experts], scalar

        # 融合原始隐藏状态和 MoE 输出（使用更保守的融合策略）
        try:
            fusion_weight = torch.sigmoid(self.moe_fusion_weight)
        except:
            logging.warning("Sigmoid failed, using fixed fusion weight")
            fusion_weight = torch.tensor(0.1, device=hidden_states.device, dtype=hidden_states.dtype)

        # 检查数值稳定性
        if torch.isnan(moe_output).any() or torch.isinf(moe_output).any():
            logging.warning("NaN or Inf detected in MoE output, using zero MoE contribution")
            # 使用零张量替代MoE输出，保持梯度连接
            moe_output = torch.zeros_like(moe_output)

        # 裁剪融合权重到合理的范围
        fusion_weight = torch.clamp(fusion_weight, min=0.0, max=0.3)  # 允许MoE贡献最多30%

        # 检查融合权重的数值稳定性
        if torch.isnan(fusion_weight) or torch.isinf(fusion_weight):
            logging.warning("NaN or Inf detected in fusion weight, using 0.1")
            fusion_weight = torch.tensor(0.1, device=hidden_states.device, dtype=hidden_states.dtype)

        # 保守的融合策略
        try:
            # 限制MoE输出范围
            moe_output_clamped = torch.clamp(moe_output, min=-2.0, max=2.0)

            # 融合隐藏状态
            enhanced_hidden = (1 - fusion_weight) * hidden_states + fusion_weight * moe_output_clamped

            # 限制融合后的输出范围
            enhanced_hidden = torch.clamp(enhanced_hidden, min=-5.0, max=5.0)
        except Exception as e:
            logging.warning(f"Fusion failed: {e}, using original hidden states")
            enhanced_hidden = hidden_states

        # 最终数值稳定性检查
        if torch.isnan(enhanced_hidden).any() or torch.isinf(enhanced_hidden).any():
            logging.warning("NaN or Inf detected in enhanced hidden states, using original hidden states")
            enhanced_hidden = hidden_states

        # 生成 logits
        logits = self.llama_model.lm_head(enhanced_hidden)  # [B, L, vocab_size]

        # 构建输出
        outputs = {
            'logits': logits,
            'expert_weights': expert_weights,
            'hidden_states': enhanced_hidden,
            'load_balance_loss': moe_load_balance_loss
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

            # 总损失 = 语言建模损失 + 负载均衡损失
            total_loss = lm_loss + self.load_balance_weight * moe_load_balance_loss

            outputs['loss'] = total_loss
            outputs['lm_loss'] = lm_loss
            outputs['load_balance_loss'] = moe_load_balance_loss
            outputs['weighted_load_balance_loss'] = self.load_balance_weight * moe_load_balance_loss

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

        # 计算负载均衡质量指标
        usage_balance_score = 1.0 - (usage_std.item() / ideal_usage)  # 越接近1表示越均衡
        usage_efficiency = (expert_usage > 0.01).sum().item() / self.num_experts  # 有效专家比例

        return {
            'expert_usage': expert_usage.detach().cpu().numpy().tolist(),
            'usage_std': usage_std.item(),
            'usage_max': usage_max.item(),
            'usage_min': usage_min.item(),
            'load_balance_loss': load_balance_loss.item(),
            'ideal_usage': ideal_usage,
            'usage_balance_score': usage_balance_score,
            'usage_efficiency': usage_efficiency,
            'load_balance_weight': self.load_balance_weight
        }

    def generate(self, *args, **kwargs):
        """生成方法，直接调用 LLaMA 模型的生成方法"""
        return self.llama_model.generate(*args, **kwargs)