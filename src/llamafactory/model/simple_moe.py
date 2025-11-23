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

        # 梯度监控统计
        self.register_buffer('grad_norm_history', torch.zeros(100))  # 存储最近100次梯度范数
        self.register_buffer('grad_norm_idx', torch.tensor(0))
        self.register_buffer('dead_expert_count', torch.zeros(num_experts))

        self._init_weights()

    def _init_weights(self):
        """初始化权重 - 改进的Xavier初始化"""
        for i, module in enumerate(self.router):
            if isinstance(module, nn.Linear):
                # 使用Xavier uniform初始化，更稳定
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # 减小gain防止梯度爆炸
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

                # 最后一层使用更小的初始化，防止路由坍塌
                if i == len(self.router) - 1:  # 最后一层
                    with torch.no_grad():
                        module.weight.data *= 0.1  # 进一步缩小最后一层权重
                        if module.bias is not None:
                            # 添加小的随机偏置，促进专家多样性
                            module.bias.data.uniform_(-0.01, 0.01)

    def _apply_gentle_weight_constraints(self):
        """温和的权重约束，避免破坏梯度流"""
        for module in self.router:
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    # 使用软约束而非硬裁剪
                    weight_norm = torch.norm(module.weight, dim=1, keepdim=True)
                    # 只对过大的权重进行缩放
                    scale_factor = torch.clamp(2.0 / (weight_norm + 1e-8), max=1.0)
                    module.weight.data *= scale_factor

                    if module.bias is not None:
                        # 偏置使用温和的tanh约束
                        module.bias.data = torch.tanh(module.bias.data) * 0.5

    def _monitor_gradients(self):
        """监控梯度统计信息"""
        if self.training:
            total_grad_norm = 0.0
            param_count = 0

            for param in self.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    param_count += 1

            if param_count > 0:
                total_grad_norm = total_grad_norm ** 0.5

                # 更新梯度历史
                idx = self.grad_norm_idx.item() % 100
                self.grad_norm_history[idx] = total_grad_norm
                self.grad_norm_idx += 1

                # 检查梯度爆炸
                if total_grad_norm > 10.0:
                    logging.warning(f"Router gradient norm too large: {total_grad_norm:.4f}")

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

        # 温和的权重约束（渐进式限制）
        self._apply_gentle_weight_constraints()

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

        # 监控专家死亡情况
        self._monitor_expert_usage(expert_weights)

        # 监控梯度（在训练模式下）
        if self.training:
            self._monitor_gradients()

        # 计算负载均衡损失
        load_balance_loss = self._compute_load_balance_loss(router_logits, expert_weights)

        return expert_weights, top_k_indices, load_balance_loss

    def _monitor_expert_usage(self, expert_weights: torch.Tensor):
        """监控专家使用情况，检测死亡专家"""
        if self.training:
            with torch.no_grad():
                # 计算每个专家的使用频率
                expert_usage = expert_weights.sum(dim=0)  # [num_experts]

                # 更新死亡专家计数
                dead_threshold = 1e-6
                is_dead = expert_usage < dead_threshold
                self.dead_expert_count += is_dead.float()

                # 警告死亡专家
                if self.grad_norm_idx.item() % 100 == 0:  # 每100步检查一次
                    dead_experts = (self.dead_expert_count > 50).nonzero().squeeze(-1)
                    if len(dead_experts) > 0:
                        logging.warning(f"Dead experts detected: {dead_experts.tolist()}")

    def get_gradient_stats(self) -> Dict:
        """获取梯度统计信息"""
        if self.grad_norm_idx.item() == 0:
            return {}

        valid_history = self.grad_norm_history[:min(self.grad_norm_idx.item(), 100)]
        return {
            'avg_grad_norm': valid_history.mean().item(),
            'max_grad_norm': valid_history.max().item(),
            'min_grad_norm': valid_history.min().item(),
            'grad_norm_std': valid_history.std().item(),
            'dead_expert_count': self.dead_expert_count.sum().item(),
            'total_updates': self.grad_norm_idx.item()
        }

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

            # 改进的负载均衡损失：使用更稳定的KL散度
            ideal_prob = torch.ones_like(expert_usage_freq) / self.num_experts

            # KL散度损失：D_KL(uniform || usage_freq) + D_KL(uniform || avg_prob)
            usage_kl = F.kl_div(
                torch.log(expert_usage_freq + 1e-8),
                ideal_prob,
                reduction='sum'
            )
            prob_kl = F.kl_div(
                torch.log(expert_avg_prob + 1e-8),
                ideal_prob,
                reduction='sum'
            )

            # 传统的乘积损失作为辅助
            balance_product = expert_usage_freq * expert_avg_prob
            balance_product = torch.clamp(balance_product, max=1.0)
            product_loss = self.num_experts * torch.sum(balance_product)

            # 方差惩罚（更温和）
            usage_variance = torch.var(expert_usage_freq)
            prob_variance = torch.var(expert_avg_prob)

            # 组合损失（权重调整）
            total_loss = (
                0.5 * product_loss +  # 传统损失
                0.3 * (usage_kl + prob_kl) +  # KL散度损失
                0.2 * (usage_variance + prob_variance)  # 方差惩罚
            )

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

        # 专家激活监控
        self.register_buffer('expert_activation_history', torch.zeros(num_experts, 100))
        self.register_buffer('activation_idx', torch.tensor(0))

    def _apply_expert_weight_constraints(self, expert):
        """对专家应用温和的权重约束"""
        for param in expert.parameters():
            if param.requires_grad:
                with torch.no_grad():
                    # 使用L2范数约束而非硬裁剪
                    param_norm = torch.norm(param.data)
                    if param_norm > 3.0:  # 只对过大的参数进行缩放
                        param.data *= (3.0 / param_norm)

    def _monitor_expert_activations(self, expert_weights: torch.Tensor):
        """监控专家激活模式"""
        if self.training:
            with torch.no_grad():
                # 计算每个专家的平均激活
                expert_activation = expert_weights.mean(dim=0)  # [num_experts]

                # 更新激活历史
                idx = self.activation_idx.item() % 100
                self.expert_activation_history[:, idx] = expert_activation
                self.activation_idx += 1

    def _sanitize_expert_input(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        为专家清理输入，防止NaN传播
        """
        # 检查是否有NaN/Inf
        has_nan = torch.isnan(hidden_states).any()
        has_inf = torch.isinf(hidden_states).any()

        if has_nan or has_inf:
            logging.warning(f"Expert {expert_idx} input contains NaN: {has_nan}, Inf: {has_inf}")

            # 使用逐元素替换，保持梯度连接
            clean_states = torch.where(
                torch.isnan(hidden_states) | torch.isinf(hidden_states),
                torch.zeros_like(hidden_states),
                hidden_states
            )

            # 额外的范围限制
            clean_states = torch.clamp(clean_states, min=-5.0, max=5.0)

            # 最终验证
            if torch.isnan(clean_states).any() or torch.isinf(clean_states).any():
                logging.error(f"Failed to sanitize expert {expert_idx} input, using zeros")
                return torch.zeros_like(hidden_states)

            return clean_states

        # 即使没有NaN/Inf，也应用温和的范围限制
        return torch.clamp(hidden_states, min=-5.0, max=5.0)

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

        # 监控专家激活模式
        self._monitor_expert_activations(expert_weights)

        # 专家输出计算（使用更严格的数值稳定性控制）
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 确保专家在正确的设备和数据类型上
            if next(expert.parameters()).device != hidden_states.device:
                expert = expert.to(hidden_states.device)
            if next(expert.parameters()).dtype != hidden_states.dtype:
                expert = expert.to(hidden_states.dtype)

            try:
                # 温和的专家权重约束（避免破坏梯度）
                self._apply_expert_weight_constraints(expert)

                # 关键修复：在传给专家之前先清理输入
                clean_input = self._sanitize_expert_input(hidden_states, i)
                expert_output = expert(clean_input)  # [B, L, H]

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

        # 梯度监控和自适应裁剪
        self.register_buffer('model_grad_history', torch.zeros(50))
        self.register_buffer('model_grad_idx', torch.tensor(0))
        self.adaptive_grad_clip = True

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

    def _sanitize_hidden_states(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        清理和修复隐藏状态中的NaN/Inf，这是防止级联失败的关键步骤
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 检查是否存在NaN/Inf
        has_nan = torch.isnan(hidden_states).any()
        has_inf = torch.isinf(hidden_states).any()

        if has_nan or has_inf:
            logging.warning(f"Base model output contains NaN: {has_nan}, Inf: {has_inf}. Applying emergency cleanup.")

            # 策略1: 使用embedding层重新计算干净的隐藏状态
            try:
                with torch.no_grad():
                    # 获取embedding层输出作为fallback
                    if hasattr(self.llama_model, 'model') and hasattr(self.llama_model.model, 'embed_tokens'):
                        clean_embeds = self.llama_model.model.embed_tokens(input_ids)

                        # 如果embedding也有问题，使用随机初始化
                        if torch.isnan(clean_embeds).any() or torch.isinf(clean_embeds).any():
                            logging.warning("Embedding layer also corrupted, using random initialization")
                            clean_embeds = torch.randn_like(hidden_states) * 0.02

                        # 逐位置检查并替换损坏的隐藏状态
                        mask_nan = torch.isnan(hidden_states)
                        mask_inf = torch.isinf(hidden_states)
                        mask_bad = mask_nan | mask_inf

                        # 使用embedding扩展到hidden_dim（如果维度不匹配）
                        if clean_embeds.size(-1) != hidden_dim:
                            # 简单的线性投影或填充
                            if clean_embeds.size(-1) < hidden_dim:
                                padding = torch.zeros(batch_size, seq_len, hidden_dim - clean_embeds.size(-1),
                                                    device=clean_embeds.device, dtype=clean_embeds.dtype)
                                clean_embeds = torch.cat([clean_embeds, padding], dim=-1)
                            else:
                                clean_embeds = clean_embeds[:, :, :hidden_dim]

                        # 替换损坏的部分
                        hidden_states = torch.where(mask_bad, clean_embeds, hidden_states)

            except Exception as e:
                logging.warning(f"Emergency embedding fallback failed: {e}")
                # 最后的fallback：使用小的随机值
                hidden_states = torch.where(
                    torch.isnan(hidden_states) | torch.isinf(hidden_states),
                    torch.randn_like(hidden_states) * 0.01,
                    hidden_states
                )

        # 额外的数值范围保护
        hidden_states = torch.clamp(hidden_states, min=-10.0, max=10.0)

        # 最终验证
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            logging.error("Failed to sanitize hidden states, using zero tensor")
            hidden_states = torch.zeros_like(hidden_states)

        return hidden_states

    def _compute_safe_fusion_weight(self, device, dtype) -> torch.Tensor:
        """
        安全地计算融合权重，防止NaN传播
        """
        try:
            # 检查融合权重参数本身
            if torch.isnan(self.moe_fusion_weight).any() or torch.isinf(self.moe_fusion_weight).any():
                logging.warning("Fusion weight parameter contains NaN/Inf, resetting to 0.1")
                with torch.no_grad():
                    self.moe_fusion_weight.data.fill_(0.1)

            # 安全计算sigmoid
            raw_weight = self.moe_fusion_weight.clamp(-10.0, 10.0)  # 防止sigmoid饱和
            fusion_weight = torch.sigmoid(raw_weight)

            # 验证结果
            if torch.isnan(fusion_weight).any() or torch.isinf(fusion_weight).any():
                logging.warning("Sigmoid result contains NaN/Inf, using fallback")
                fusion_weight = torch.tensor(0.1, device=device, dtype=dtype)

            return fusion_weight

        except Exception as e:
            logging.warning(f"Fusion weight computation failed: {e}, using fallback")
            return torch.tensor(0.1, device=device, dtype=dtype)

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

        # 关键修复：立即检查和清理基础模型输出
        hidden_states = self._sanitize_hidden_states(hidden_states, input_ids)

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
        fusion_weight = self._compute_safe_fusion_weight(hidden_states.device, hidden_states.dtype)

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

    def check_model_health(self) -> Dict:
        """
        检查模型健康状态，识别NaN/Inf参数
        """
        health_report = {
            'has_nan_params': False,
            'has_inf_params': False,
            'problematic_modules': [],
            'total_params': 0,
            'nan_param_count': 0,
            'inf_param_count': 0
        }

        for name, param in self.named_parameters():
            if param is not None:
                health_report['total_params'] += param.numel()

                if torch.isnan(param).any():
                    health_report['has_nan_params'] = True
                    health_report['nan_param_count'] += torch.isnan(param).sum().item()
                    health_report['problematic_modules'].append(f"{name} (NaN)")

                if torch.isinf(param).any():
                    health_report['has_inf_params'] = True
                    health_report['inf_param_count'] += torch.isinf(param).sum().item()
                    health_report['problematic_modules'].append(f"{name} (Inf)")

        # 如果发现问题，尝试修复
        if health_report['has_nan_params'] or health_report['has_inf_params']:
            logging.error(f"Model health check failed: {health_report}")
            self._emergency_parameter_reset()

        return health_report

    def _emergency_parameter_reset(self):
        """
        紧急参数重置，修复NaN/Inf参数
        """
        logging.warning("Performing emergency parameter reset due to NaN/Inf detection")

        for name, param in self.named_parameters():
            if param is not None and param.requires_grad:
                with torch.no_grad():
                    # 检查并修复NaN/Inf
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        logging.warning(f"Resetting parameter: {name}")

                        # 根据参数类型选择重置策略
                        if 'weight' in name:
                            if len(param.shape) >= 2:
                                nn.init.xavier_uniform_(param, gain=0.1)
                            else:
                                param.data.normal_(0, 0.01)
                        elif 'bias' in name:
                            param.data.zero_()
                        else:
                            param.data.normal_(0, 0.01)

                        # 应用范围限制
                        param.data.clamp_(-1.0, 1.0)

    def apply_adaptive_gradient_clipping(self, max_norm: float = 1.0):
        """自适应梯度裁剪"""
        if not self.adaptive_grad_clip or not self.training:
            return

        # 计算当前梯度范数
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # 更新梯度历史
        idx = self.model_grad_idx.item() % 50
        self.model_grad_history[idx] = total_norm
        self.model_grad_idx += 1

        # 自适应调整裁剪阈值
        if self.model_grad_idx.item() > 10:
            valid_history = self.model_grad_history[:min(self.model_grad_idx.item(), 50)]
            mean_grad_norm = valid_history.mean().item()
            std_grad_norm = valid_history.std().item()

            # 动态调整阈值：平均值 + 2倍标准差
            adaptive_max_norm = min(max_norm, mean_grad_norm + 2 * std_grad_norm)
            adaptive_max_norm = max(adaptive_max_norm, 0.1)  # 最小阈值
        else:
            adaptive_max_norm = max_norm

        # 执行梯度裁剪
        if total_norm > adaptive_max_norm:
            clip_coef = adaptive_max_norm / (total_norm + 1e-6)
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

            logging.info(f"Gradient clipped: {total_norm:.4f} -> {adaptive_max_norm:.4f}")

    def get_comprehensive_stats(self) -> Dict:
        """获取全面的模型统计信息"""
        stats = {}

        # 基础专家统计
        if hasattr(self, 'moe_layer') and hasattr(self.moe_layer, 'expert_activation_history'):
            if self.moe_layer.activation_idx.item() > 0:
                valid_activations = self.moe_layer.expert_activation_history[:, :min(self.moe_layer.activation_idx.item(), 100)]
                expert_mean_activation = valid_activations.mean(dim=1)
                expert_std_activation = valid_activations.std(dim=1)

                stats['expert_activation_stats'] = {
                    'mean_activations': expert_mean_activation.tolist(),
                    'std_activations': expert_std_activation.tolist(),
                    'activation_balance_score': 1.0 - expert_std_activation.mean().item(),
                    'most_active_expert': expert_mean_activation.argmax().item(),
                    'least_active_expert': expert_mean_activation.argmin().item()
                }

        # 路由器梯度统计
        if hasattr(self.moe_layer, 'router'):
            router_stats = self.moe_layer.router.get_gradient_stats()
            if router_stats:
                stats['router_gradient_stats'] = router_stats

        # 模型级别梯度统计
        if self.model_grad_idx.item() > 0:
            valid_grads = self.model_grad_history[:min(self.model_grad_idx.item(), 50)]
            stats['model_gradient_stats'] = {
                'avg_grad_norm': valid_grads.mean().item(),
                'max_grad_norm': valid_grads.max().item(),
                'grad_norm_std': valid_grads.std().item(),
                'gradient_stability_score': 1.0 / (1.0 + valid_grads.std().item())
            }

        # 融合权重统计
        stats['fusion_weight'] = {
            'current_value': torch.sigmoid(self.moe_fusion_weight).item(),
            'raw_value': self.moe_fusion_weight.item()
        }

        return stats

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