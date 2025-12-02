# src/llamafactory/model/joint_lora_moe_model.py
"""
联合LoRA+MoE模型
同时训练预训练LoRA适配器和新增MoE处理层
实现端到端优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    raise ImportError("PEFT library is required. Please install with: pip install peft")


@dataclass
class JointLoRAMoEConfig:
    """联合LoRA+MoE配置"""

    # LoRA配置
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

    # MoE配置
    num_moe_experts: int = 4
    moe_hidden_dim: int = 4096
    moe_intermediate_dim: int = None  # 默认为 moe_hidden_dim * 4

    # 文化损失配置
    use_culture_loss: bool = True
    culture_loss_weight: float = 0.01

    # 其他配置
    dropout: float = 0.1

    def __post_init__(self):
        if self.lora_target_modules is None:
            # 默认LoRA目标模块（注意力层）
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        if self.moe_intermediate_dim is None:
            self.moe_intermediate_dim = self.moe_hidden_dim * 4


class MoEExpert(nn.Module):
    """MoE专家层 - 极度保守的数值稳定版本"""

    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.1, dtype: torch.dtype = torch.float16):
        super().__init__()
        # 修复：恢复合理的中间维度，确保专家有足够的表达能力
        safe_intermediate_dim = min(intermediate_dim, hidden_dim * 2)  # 恢复为合理的大小

        self.gate_proj = nn.Linear(hidden_dim, safe_intermediate_dim, bias=True, dtype=dtype)  # 添加bias
        self.up_proj = nn.Linear(hidden_dim, safe_intermediate_dim, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(safe_intermediate_dim, hidden_dim, bias=True, dtype=dtype)
        self.act_fn = nn.GELU()  # 使用更稳定的GELU而不是SiLU
        self.dropout = nn.Dropout(dropout)
        # 暂时移除LayerNorm，可能导致数值不稳定
        # self.layer_norm = nn.LayerNorm(safe_intermediate_dim, dtype=dtype)

        # 保守的初始化
        self._init_weights()

    def _init_weights(self):
        """修复权重初始化 - 防NaN版本"""
        # 🔧 根据新的归一化范围[-3,3]调整初始化
        gate_up_std = 0.05   # 增加初始化方差，匹配[-3,3]输入范围
        down_std = 0.02      # 相应增加下游投影初始化

        try:
            for module in [self.gate_proj, self.up_proj]:
                # 清零权重和偏置，防止NaN残留
                with torch.no_grad():
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()

                # 安全的正态分布初始化
                nn.init.normal_(module.weight, mean=0.0, std=gate_up_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

                # 检查初始化后是否有NaN
                if torch.isnan(module.weight).any() or torch.isinf(module.weight).any():
                    print(f"    ⚠️ NaN detected in {module.__class__.__name__} after init, using zeros")
                    module.weight.zero_()

            # down_proj安全初始化
            with torch.no_grad():
                self.down_proj.weight.zero_()
                if self.down_proj.bias is not None:
                    self.down_proj.bias.zero_()

            nn.init.normal_(self.down_proj.weight, mean=0.0, std=down_std)
            if self.down_proj.bias is not None:
                nn.init.constant_(self.down_proj.bias, 0.0)

            if torch.isnan(self.down_proj.weight).any() or torch.isinf(self.down_proj.weight).any():
                print("    ⚠️ NaN detected in down_proj after init, using zeros")
                self.down_proj.weight.zero_()

            # print(f"🔧 Expert safely initialized: gate/up_std={gate_up_std:.4f}, down_std={down_std:.4f}")  # 注释掉频繁日志

        except Exception as e:
            print(f"⚠️ Weight initialization failed: {e}")
            # 最后的安全措施：全部置零
            with torch.no_grad():
                for module in [self.gate_proj, self.up_proj, self.down_proj]:
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, x):
        """前向传播 - 修复版本，减少过度限制"""
        # 添加调试信息：检查模型模式和输入
        # print(f"    🔍 Expert mode: training={self.training}")  # 注释掉详细调试
        # input_mean = x.mean().item()
        # input_std = x.std().item()
        # print(f"    🔍 input: mean={input_mean:.6f}, std={input_std:.6f}")  # 注释掉详细调试

        # 🔧 关键修复：输入归一化，防止Float16溢出
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros_like(x)

        # 🔧 调整归一化策略：减少压缩，保留更多信息
        # 使用更温和的归一化，保留更多原始信号强度
        x_normalized = torch.clamp(x / 5.0, min=-3.0, max=3.0)  # 线性缩放后clip，保留更多动态范围

        try:
            # 第一阶段：gate和up投影 - 使用归一化输入
            gate_output = self.gate_proj(x_normalized)
            up_output = self.up_proj(x_normalized)

            # 添加调试信息：检查投影层权重
            gate_weight_has_nan = torch.isnan(self.gate_proj.weight).any() or torch.isinf(self.gate_proj.weight).any()
            up_weight_has_nan = torch.isnan(self.up_proj.weight).any() or torch.isinf(self.up_proj.weight).any()
            # print(f"    🔍 gate_proj weight NaN: {gate_weight_has_nan}")
            # print(f"    🔍 up_proj weight NaN: {up_weight_has_nan}")

            if gate_weight_has_nan or up_weight_has_nan:
                print("    🚨 专家权重包含NaN，重新初始化并返回零输出!")
                self._init_weights()
                return torch.zeros_like(x)

            # 添加调试信息：检查投影层输出
            gate_mean = gate_output.mean().item()
            gate_std = gate_output.std().item()
            up_mean = up_output.mean().item()
            up_std = up_output.std().item()
            # print(f"    🔍 gate_proj: mean={gate_mean:.6f}, std={gate_std:.6f}")
            # print(f"    🔍 up_proj: mean={up_mean:.6f}, std={up_std:.6f}")

            # 检查第一阶段输出 - 如果是NaN，强制重新初始化
            if torch.isnan(gate_output).any() or torch.isinf(gate_output).any():
                print("    ⚠️ gate_proj output is NaN, reinitializing weights...")
                self._init_weights()
                return torch.zeros_like(x)
            if torch.isnan(up_output).any() or torch.isinf(up_output).any():
                print("    ⚠️ up_proj output is NaN, reinitializing weights...")
                self._init_weights()
                return torch.zeros_like(x)

            # 激活函数 - 移除激活前的限制
            gate_activated = self.act_fn(gate_output)

            # 添加调试信息：检查激活后的输出
            gate_act_mean = gate_activated.mean().item()
            gate_act_std = gate_activated.std().item()
            # print(f"    🔍 gate_activated: mean={gate_act_mean:.6f}, std={gate_act_std:.6f}")

            # 检查激活后的输出
            if torch.isnan(gate_activated).any() or torch.isinf(gate_activated).any():
                return torch.zeros_like(x)

            # 元素乘法 - 移除过度限制
            intermediate = gate_activated * up_output

            # 添加调试信息：检查元素乘法后的输出
            inter_mean = intermediate.mean().item()
            inter_std = intermediate.std().item()
            # print(f"    🔍 intermediate: mean={inter_mean:.6f}, std={inter_std:.6f}")

            # 暂时移除LayerNorm稳定化，可能是导致零输出的原因
            # intermediate = self.layer_norm(intermediate)

            # Dropout - 在推理时不应该有影响
            intermediate = self.dropout(intermediate)

            # 最终投影 - 移除输出限制，让模型自由表达
            output = self.down_proj(intermediate)

            # 🔧 增量架构：MoE作为基础LoRA的小幅调整
            # MoE应该产生小的增量调整，而不是替代基础LoRA
            # 目标：MoE增量约为基础LoRA输出的5-10%
            # 基础LoRA std≈2.2，MoE增量目标 std≈0.2-0.4
            # 原始MoE std≈0.022，需要约10-20倍缩放
            output = output * 2.0  # 小幅缩放，产生增量调整

            # 添加调试信息：检查最终输出
            output_mean = output.mean().item()
            output_std = output.std().item()
            print(f"    🔍 专家最终输出: mean={output_mean:.6f}, std={output_std:.6f}, range=[{output.min().item():.3f}, {output.max().item():.3f}]")

            # 最终检查 - 只检查NaN/Inf，不限制数值范围
            if torch.isnan(output).any() or torch.isinf(output).any():
                return torch.zeros_like(x)

            return output

        except Exception as e:
            print(f"⚠️ MoEExpert forward failed: {e}")
            return torch.zeros_like(x)


class MoERouter(nn.Module):
    """MoE路由器 - 极简数值稳定版本，专为batch_size=1优化"""

    def __init__(self, hidden_dim: int, num_experts: int, dropout: float = 0.1, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # 根本性解决方案：路由器强制使用Float32，避免Float16精度问题
        # 完全忽略传入的dtype参数，强制使用Float32
        self.router = nn.Linear(hidden_dim, num_experts, bias=True, dtype=torch.float32)

        print(f"🔧 MoERouter initialized with Float32, router dtype: {self.router.weight.dtype}")

        # 极保守的初始化 - 确保输出接近uniform
        with torch.no_grad():
            # 权重初始化为极小值，确保初始输出接近uniform
            nn.init.constant_(self.router.weight, 0.0)
            # 偏置设置为小的负值，softmax后趋向uniform
            nn.init.constant_(self.router.bias, -1.0)

        # Float32路由器的保守权重限制
        self.max_weight_value = 0.01   # 更保守的权重限制，防止专家权重爆炸
        self.max_bias_value = 0.1      # 更保守的偏置限制

    def forward(self, x, temperature: float = 1.0):
        """
        极简前向传播 - 专为数值稳定性设计

        Args:
            x: [B, H] 输入隐藏状态
            temperature: 温度参数

        Returns:
            expert_weights: [B, num_experts] 专家权重
            router_logits: [B, num_experts] 原始logits
        """
        batch_size = x.size(0)

        try:
            # 1. 简化的权重保护（Float32下应该不再需要）
            with torch.no_grad():
                # 检查并修复NaN/Inf（Float32下极少发生）
                weight_has_nan = torch.isnan(self.router.weight).any() or torch.isinf(self.router.weight).any()
                bias_has_nan = torch.isnan(self.router.bias).any() or torch.isinf(self.router.bias).any()

                if weight_has_nan:
                    print("⚠️ Resetting router weights due to NaN/Inf")
                    nn.init.constant_(self.router.weight, 0.0)

                if bias_has_nan:
                    print("⚠️ Resetting router bias due to NaN/Inf")
                    nn.init.constant_(self.router.bias, -1.0)

                # 合理的权重限制（Float32下不需要过于严格）
                self.router.weight.data.clamp_(-self.max_weight_value, self.max_weight_value)
                self.router.bias.data.clamp_(-self.max_bias_value, self.max_bias_value)

            # 2. 输入预处理和dtype转换 - 使用与专家相同的归一化
            x_safe = torch.clamp(x / 5.0, min=-3.0, max=3.0)  # 与专家网络保持一致的归一化

            # 转换为Float32进行路由计算，确保数值稳定
            x_float32 = x_safe.float()

            # 调试信息：检查dtype匹配
            # print(f"🔧 Router debug: input dtype={x_float32.dtype}, router weight dtype={self.router.weight.dtype}")

            # 3. 路由计算（在Float32精度下）
            router_logits = self.router(x_float32)  # [B, num_experts] in Float32

            # 4. 限制logits范围（在Float32下更安全）
            router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)

            # 5. 在Float32精度下进行稳定的softmax计算
            safe_temperature = torch.clamp(torch.tensor(temperature, device=x.device, dtype=torch.float32), min=0.5, max=2.0)

            # 温度缩放（Float32精度）
            scaled_logits = router_logits / safe_temperature

            # 数值稳定的softmax（Float32精度）
            max_logits = torch.max(scaled_logits, dim=-1, keepdim=True)[0]
            shifted_logits = scaled_logits - max_logits

            # Float32下的指数计算更稳定
            safe_logits = torch.clamp(shifted_logits, min=-20.0, max=0.0)

            # 计算expert权重（Float32精度）
            exp_logits = torch.exp(safe_logits)
            weight_sum = torch.sum(exp_logits, dim=-1, keepdim=True) + 1e-8
            expert_weights_f32 = exp_logits / weight_sum

            # 6. 转换回原始dtype，但保持梯度连接
            expert_weights = expert_weights_f32.to(x.dtype)
            router_logits = router_logits.to(x.dtype)

            # 7. 最终检查（现在应该极少触发）
            if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
                print("⚠️ Fallback to uniform weights")
                expert_weights = torch.full((batch_size, self.num_experts),
                                           1.0 / self.num_experts,
                                           device=x.device,
                                           dtype=x.dtype,
                                           requires_grad=True)
                router_logits = torch.zeros((batch_size, self.num_experts),
                                          device=x.device,
                                          dtype=x.dtype,
                                          requires_grad=True)

            return expert_weights, router_logits

        except Exception as e:
            print(f"⚠️ Router completely failed: {e}, using uniform fallback")
            # 完全安全的fallback
            expert_weights = torch.full((batch_size, self.num_experts),
                                       1.0 / self.num_experts,
                                       device=x.device,
                                       dtype=x.dtype,
                                       requires_grad=True)
            router_logits = torch.zeros((batch_size, self.num_experts),
                                      device=x.device,
                                      dtype=x.dtype,
                                      requires_grad=True)
            return expert_weights, router_logits


class MoELayer(nn.Module):
    """MoE层"""

    def __init__(self, config: JointLoRAMoEConfig, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.hidden_dim = config.moe_hidden_dim
        self.dtype = dtype

        # 创建路由器 - 强制使用Float32以确保数值稳定
        self.router = MoERouter(
            hidden_dim=config.moe_hidden_dim,
            num_experts=config.num_moe_experts,
            dropout=config.dropout,
            dtype=torch.float32  # 强制Float32，忽略传入的dtype
        )

        # 创建专家
        self.experts = nn.ModuleList([
            MoEExpert(
                hidden_dim=config.moe_hidden_dim,
                intermediate_dim=config.moe_intermediate_dim,
                dropout=config.dropout,
                dtype=dtype
            ) for _ in range(config.num_moe_experts)
        ])

        # 🔧 确保专家分化：每个专家使用不同的初始化
        print("🔧 Force reinitializing all experts with different seeds...")
        for i, expert in enumerate(self.experts):
            # 为每个专家设置不同的随机种子，确保分化
            torch.manual_seed(42 + i * 100)  # 不同的种子
            expert._init_weights()
            print(f"🔧 Expert {i} reinitialized with seed {42 + i * 100}")

        # 简化：移除复杂的门控和共享专家机制，只保留基本的专家混合
        # 不使用共享专家和门控，避免额外的复杂性

    def forward(self, hidden_states):
        """
        前向传播 - 极简版本，只保留基本的专家混合

        Args:
            hidden_states: [B, L, H] 输入隐藏状态

        Returns:
            output: [B, L, H] 输出隐藏状态
            expert_weights: [B, num_experts] 专家权重（用于文化损失）
            aux_loss: 辅助损失
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        try:
            # 移除过度严格的输入限制，只检查NaN/Inf
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print("⚠️ NaN/Inf in MoE input, using passthrough")
                expert_weights = torch.ones(batch_size, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True) / self.num_experts
                aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True)
                return hidden_states, expert_weights, aux_loss

            # 1. 路由决策（极简版）
            # 使用平均池化获取序列表示
            pooled = hidden_states.mean(dim=1)  # [B, H]
            # 移除pooled的数值限制

            # 路由计算 - 使用较低温度增强专家选择的区分度
            all_expert_weights, router_logits = self.router(pooled, temperature=0.5)

            # 🔧 实现Top-2激活机制
            # 1. 选择top-2专家
            top_k_logits, top_k_indices = torch.topk(router_logits, k=2, dim=-1)  # [B, 2]

            # 2. 对top-2专家的logits重新归一化
            top_k_weights = torch.softmax(top_k_logits, dim=-1)  # [B, 2] 归一化权重

            # 3. 创建稀疏权重矩阵（只有激活的专家有权重）
            expert_weights = torch.zeros_like(all_expert_weights)  # [B, num_experts]
            expert_weights.scatter_(1, top_k_indices, top_k_weights)  # 将归一化权重分配给激活专家

            # 添加调试信息：检查Top-2激活和路由器学习
            weights_mean = expert_weights.mean(dim=0)
            router_logits_mean = router_logits.mean(dim=0)
            print(f"🔍 Top-2激活专家权重: {weights_mean.detach().cpu().numpy()}")
            print(f"🔍 路由器原始logits: {router_logits_mean.detach().cpu().numpy()}")
            print(f"🔍 Top-2选择的logits差异: {(top_k_logits[:, 0] - top_k_logits[:, 1]).mean().item():.6f}")

            # 2. 专家计算（Top-2版本）- 只计算激活的专家
            expert_outputs = {}  # 使用字典存储，只计算需要的专家
            valid_experts = 0

            # 获取所有激活的专家索引（去重）
            activated_experts = torch.unique(top_k_indices.flatten()).cpu().tolist()
            print(f"🔍 激活的专家索引: {activated_experts}")

            for expert_idx in activated_experts:
                try:
                    expert_output = self.experts[expert_idx](hidden_states)  # [B, L, H]

                    # 添加调试信息：检查每个专家的输出
                    expert_mean = expert_output.mean().item()
                    expert_std = expert_output.std().item()
                    expert_min = expert_output.min().item()
                    expert_max = expert_output.max().item()
                    print(f"    🔍 专家最终输出: mean={expert_mean:.6f}, std={expert_std:.6f}, range=[{expert_min:.3f}, {expert_max:.3f}]")

                    # 检查专家输出
                    if not (torch.isnan(expert_output).any() or torch.isinf(expert_output).any()):
                        expert_outputs[expert_idx] = expert_output
                        valid_experts += 1
                    else:
                        print(f"🚨 专家{expert_idx}输出无效(NaN/Inf)，使用零输出!")
                        expert_outputs[expert_idx] = torch.zeros_like(hidden_states)

                except Exception as e:
                    print(f"⚠️ Expert {expert_idx} failed: {e}, using zeros")
                    expert_outputs[expert_idx] = torch.zeros_like(hidden_states)

            # 3. Top-2专家输出混合
            if valid_experts == 0:
                print("🚨 所有激活专家都失效，使用输入passthrough!")
                final_output = hidden_states
            else:
                # Top-2加权融合
                final_output = torch.zeros_like(hidden_states)
                total_weight = 0.0

                for i in range(expert_weights.size(1)):  # 遍历所有专家
                    weight_val = expert_weights[:, i]  # [B]
                    if weight_val.sum().item() > 1e-8:  # 只处理有权重的专家
                        if i in expert_outputs:
                            weight = weight_val.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
                            weight = torch.clamp(weight, min=0.0, max=1.0)
                            final_output += weight * expert_outputs[i]
                            total_weight += weight_val.mean().item()

                # 降低总权重阈值，避免过早使用passthrough
                if total_weight < 0.001:  # 从0.01降到0.001
                    print("⚠️ Expert weights too small, using passthrough")
                    final_output = hidden_states
                else:
                    # 🔧 增量架构：限制MoE输出为小的增量调整
                    # 目标：增量约为基础LoRA输出的5-10%，范围约[-5, +5]
                    final_output = torch.clamp(final_output, min=-5.0, max=5.0)
                    # 添加调试信息：检查混合后的输出
                    # final_mean = final_output.mean().item()
                    # final_std = final_output.std().item()
                    # final_min = final_output.min().item()
                    # final_max = final_output.max().item()
                    # print(f"🔍 Mixed output: mean={final_mean:.6f}, std={final_std:.6f}, range=[{final_min:.3f}, {final_max:.3f}], total_weight={total_weight:.6f}")

            # 4. 最终检查
            if torch.isnan(final_output).any() or torch.isinf(final_output).any():
                print("⚠️ Final MoE output invalid, using input passthrough with gradient connection")
                # 使用输入passthrough，确保梯度连接
                final_output = hidden_states

            # 5. MoE辅助损失 - 修复版本，确保始终有梯度连接
            try:
                # 始终计算辅助损失，即使使用了fallback机制
                # 负载均衡损失：鼓励专家使用均匀
                target_uniform = torch.ones_like(expert_weights) / self.num_experts
                balance_loss = F.mse_loss(expert_weights, target_uniform)

                # 路由器正则化损失：防止logits过大
                # 即使是fallback的logits也应该参与损失计算
                router_reg_loss = torch.mean(router_logits ** 2)

                # 组合辅助损失
                aux_loss = balance_loss * 0.01 + router_reg_loss * 0.001

                # 推理模式下不需要梯度连接，直接返回数值
                if not aux_loss.requires_grad:
                    # 在推理模式下(torch.no_grad)，这是正常现象
                    # 不需要打印警告或添加参数连接
                    pass

                # 最终数值检查
                if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                    print("⚠️ NaN/Inf in aux_loss, using parameter-connected fallback")
                    aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True)
                    for param in self.router.parameters():
                        if param.requires_grad:
                            aux_loss = aux_loss + torch.sum(param * param) * 1e-8
                            break

            except Exception as e:
                print(f"⚠️ Aux loss computation failed: {e}")
                # 确保fallback有梯度连接
                aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True)
                for param in self.router.parameters():
                    if param.requires_grad:
                        aux_loss = aux_loss + torch.sum(param * param) * 1e-8
                        break

            return final_output, expert_weights, aux_loss

        except Exception as e:
            print(f"⚠️ MoE layer completely failed: {e}, using fallback transformation")
            # 确保有梯度连接的fallback
            if not hasattr(self, 'fallback_transform'):
                self.fallback_transform = nn.Linear(
                    hidden_states.size(-1), hidden_states.size(-1),
                    bias=False, dtype=hidden_states.dtype
                )
                nn.init.eye_(self.fallback_transform.weight)
                self.fallback_transform = self.fallback_transform.to(hidden_states.device)
                self.add_module('fallback_transform', self.fallback_transform)

            fallback_output = self.fallback_transform(hidden_states)
            expert_weights = torch.ones(batch_size, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True) / self.num_experts
            aux_loss = torch.tensor(0.01, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True)
            return fallback_output, expert_weights, aux_loss


class JointLoRAMoEModel(nn.Module):
    """联合LoRA+MoE模型"""

    def __init__(self, base_model, config: JointLoRAMoEConfig):
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

        # 2. 添加MoE层（传递正确的dtype）
        self.moe_layer = MoELayer(config, dtype=dtype)

        # 3. 确保MoE层在正确设备上，但保持路由器为Float32
        self.moe_layer = self.moe_layer.to(device=device)

        # 专家层可以转换为指定dtype，但路由器保持Float32
        for expert in self.moe_layer.experts:
            expert = expert.to(dtype=dtype)

        # 确保路由器保持Float32
        self.moe_layer.router = self.moe_layer.router.to(device=device, dtype=torch.float32)

        logging.info(f"Joint LoRA+MoE model initialized with {config.num_moe_experts} experts")

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

        # 🔍 验证LoRA是否正确应用
        lora_params = 0
        total_params = 0
        for name, param in self.base_model.named_parameters():
            total_params += param.numel()
            if 'lora' in name.lower():
                lora_params += param.numel()
                if lora_params <= 5:  # 只打印前5个LoRA参数
                    print(f"🔍 LoRA参数: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")

        print(f"🔍 LoRA参数统计: {lora_params:,} / {total_params:,} ({100*lora_params/total_params:.2f}%)")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        前向传播

        Args:
            input_ids: [B, L] 输入token IDs
            attention_mask: [B, L] 注意力掩码
            labels: [B, L] 标签（用于计算损失）

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

        # 🔍 调试基础模型输出
        base_range = f"min={hidden_states.min().item():.3f}, max={hidden_states.max().item():.3f}"
        base_std = hidden_states.std().item()
        print(f"🔍 基础模型输出: {base_range}, std={base_std:.6f}")

        # 确保hidden_states与MoE层的hidden_dim匹配
        if hidden_states.size(-1) != self.config.moe_hidden_dim:
            # 如果维度不匹配，需要投影
            if not hasattr(self, 'hidden_proj'):
                self.hidden_proj = nn.Linear(
                    hidden_states.size(-1),
                    self.config.moe_hidden_dim,
                    bias=False,
                    dtype=hidden_states.dtype  # 确保dtype匹配
                )
                # 初始化权重
                nn.init.normal_(self.hidden_proj.weight, mean=0.0, std=0.001)
                self.hidden_proj = self.hidden_proj.to(hidden_states.device, hidden_states.dtype)
                # 注册为模型参数，避免重复创建
                self.add_module('hidden_proj', self.hidden_proj)
            hidden_states = self.hidden_proj(hidden_states)

        # 3. MoE层处理 - 增量架构实现
        print("🔧 启用增量MoE架构：MoE作为基础LoRA的增量调整")
        moe_delta, expert_weights, moe_aux_loss = self.moe_layer(hidden_states)

        # 关键调试：检查MoE增量输出
        moe_range = f"min={moe_delta.min().item():.3f}, max={moe_delta.max().item():.3f}"
        moe_std = moe_delta.std().item()
        print(f"🔧 MoE增量输出: {moe_range}, std={moe_std:.6f}")

        # 检查MoE增量是否合理
        if moe_std < 1e-6:
            print(f"⚠️ 警告: MoE增量几乎为零 (std={moe_std:.8f})!")
        elif moe_std > 5.0:
            print(f"⚠️ 警告: MoE增量过大 (std={moe_std:.6f})，应该是小的调整!")

        # 🔧 关键修改：实现增量架构
        # 保存原始基础LoRA输出
        base_hidden_states = base_outputs.hidden_states[-1]  # [B, L, H]

        # 🔧 处理维度不匹配问题
        if hasattr(self, 'hidden_proj') and moe_delta.size(-1) != base_hidden_states.size(-1):
            # MoE增量需要反向投影回原始维度才能与base_hidden_states相加
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
        moe_influence_weight = 0.3  # 30%的影响权重，可调

        # 组合输出：基础LoRA + 加权MoE增量
        combined_output = base_hidden_states + moe_influence_weight * moe_delta

        # 调试信息：检查组合后的输出
        combined_range = f"min={combined_output.min().item():.3f}, max={combined_output.max().item():.3f}"
        combined_std = combined_output.std().item()
        print(f"🔧 组合输出(基础+MoE): {combined_range}, std={combined_std:.6f}")
        print(f"🔧 MoE影响权重: {moe_influence_weight}, 实际增量贡献: {(moe_influence_weight * moe_std):.6f}")

        # 使用组合输出作为最终隐藏状态
        final_hidden_states = combined_output

        # 4. 语言模型头 - 使用组合后的隐藏状态
        # 维度已经在上面处理过了，final_hidden_states = combined_output应该与base_hidden_states维度一致

        # 使用基础模型的lm_head计算最终logits
        if hasattr(self.base_model, 'lm_head'):
            print(f"🔧 使用组合隐藏状态(基础LoRA+MoE增量)计算logits")
            logits = self.base_model.lm_head(final_hidden_states)
        elif hasattr(self.base_model, 'base_model') and hasattr(self.base_model.base_model, 'lm_head'):
            print(f"🔧 使用组合隐藏状态(基础LoRA+MoE增量)计算logits")
            logits = self.base_model.base_model.lm_head(final_hidden_states)
        else:
            print(f"🔧 Creating temporary lm_head for combined hidden states")
            # 创建临时的lm_head，但使用合理的初始化
            vocab_size = self.base_model.config.vocab_size
            if not hasattr(self, 'temp_lm_head'):
                self.temp_lm_head = nn.Linear(
                    final_hidden_states.size(-1), vocab_size, bias=False, dtype=final_hidden_states.dtype
                )
                # 使用更合理的初始化，避免全零logits
                nn.init.normal_(self.temp_lm_head.weight, mean=0.0, std=0.02)  # 增大std
                self.temp_lm_head = self.temp_lm_head.to(final_hidden_states.device, final_hidden_states.dtype)
                # 注册为模型参数，避免重复创建
                self.add_module('temp_lm_head', self.temp_lm_head)
                print(f"🔧 Temp lm_head created with std=0.02")
            logits = self.temp_lm_head(final_hidden_states)

        # 关键调试：检查最终logits
        logits_range = f"min={logits.min().item():.3f}, max={logits.max().item():.3f}"
        logits_std = logits.std().item()
        print(f"🔧 最终Logits: {logits_range}, std={logits_std:.6f}")

        # 检查logits是否异常
        if logits_std < 1e-6:
            print(f"⚠️ 警告: Logits几乎为零 (std={logits_std:.8f})!")
        elif torch.isnan(logits).any():
            print(f"⚠️ 警告: Logits包含NaN!")
        elif torch.isinf(logits).any():
            print(f"⚠️ 警告: Logits包含Inf!")

        # 5. 计算损失 - 增量架构版本
        loss = None
        if labels is not None:
            try:
                # 🔧 增量架构：不需要限制组合输出，因为基础LoRA已经稳定
                # final_hidden_states = base_hidden_states + 0.1 * moe_delta
                # 组合输出应该接近基础LoRA的范围，数值稳定

                # 检查logits是否包含NaN/Inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("⚠️ NaN/Inf detected in logits, using base LoRA only")
                    # 如果组合输出有问题，回退到纯基础LoRA
                    base_only_logits = self.base_model.lm_head(base_hidden_states)
                    logits = base_only_logits
                    print("🔧 Fallback to base LoRA logits only")

                # 严格的logits范围限制，防止inf loss
                logits = torch.clamp(logits, min=-20.0, max=20.0)

                # 语言模型损失
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # 计算shift后的有效标签数量（用于检查是否有训练目标）
                shift_valid = (shift_labels.view(-1) != -100).sum().item()
                total_labels = shift_labels.numel()

                # 🔍 重新启用损失调试信息
                print(f"🔍 损失计算分析:")
                print(f"  shift_logits shape: {shift_logits.shape}")
                print(f"  shift_labels shape: {shift_labels.shape}")
                print(f"  有效标签数量: {shift_valid}/{total_labels}")
                print(f"  logits范围: min={shift_logits.min().item():.3f}, max={shift_logits.max().item():.3f}")

                # 检查第一个样本的有效标签
                if shift_valid > 0:
                    valid_positions = (shift_labels.view(-1) != -100).nonzero().flatten()
                    if len(valid_positions) > 0:
                        first_valid_pos = valid_positions[0].item()
                        first_valid_label = shift_labels.view(-1)[first_valid_pos].item()
                        first_valid_logit = shift_logits.view(-1, shift_logits.size(-1))[first_valid_pos]
                        print(f"  第一个有效标签: 位置{first_valid_pos}, 标签{first_valid_label}")
                        print(f"  对应logit范围: min={first_valid_logit.min().item():.3f}, max={first_valid_logit.max().item():.3f}")

                # 检查第一个样本的labels变化（注释掉详细输出）
                # if labels.shape[0] > 0:
                #     sample_original = labels[0]
                #     sample_shifted = shift_labels[0]
                #     orig_valid_pos = (sample_original != -100).nonzero().flatten()
                #     shift_valid_pos = (sample_shifted != -100).nonzero().flatten()

                #     print(f"  Sample 0 original valid positions: {orig_valid_pos.tolist()}")
                #     print(f"  Sample 0 shifted valid positions: {shift_valid_pos.tolist()}")

                #     if len(orig_valid_pos) > 0:
                #         print(f"  Sample 0 original valid tokens: {sample_original[orig_valid_pos].tolist()}")
                #     if len(shift_valid_pos) > 0:
                #         print(f"  Sample 0 shifted valid tokens: {sample_shifted[shift_valid_pos].tolist()}")

                if shift_valid == 0:
                    print(f"⚠️ No valid labels found after shift! All {total_labels} labels are masked (-100)")

                # 暂时移除label smoothing，使用Float32计算loss
                # loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0)

                # 调试：在计算loss前检查输入
                # print(f"🔍 Before CrossEntropyLoss:")
                # print(f"  shift_logits.shape: {shift_logits.shape}")
                # print(f"  shift_labels.shape: {shift_labels.shape}")
                # print(f"  shift_logits contains NaN: {torch.isnan(shift_logits).any()}")
                # print(f"  shift_logits contains Inf: {torch.isinf(shift_logits).any()}")
                # print(f"  shift_labels min: {shift_labels.min().item()}, max: {shift_labels.max().item()}")

                # 直接使用Float16计算，但移除label smoothing避免数值问题
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # print(f"🔍 After CrossEntropyLoss:")
                # print(f"  lm_loss: {lm_loss.item()}")
                # print(f"  lm_loss.dtype: {lm_loss.dtype}")

                # 检查lm_loss是否为NaN/Inf
                is_nan = torch.isnan(lm_loss)
                is_inf = torch.isinf(lm_loss)
                # print(f"🔍 NaN/Inf check: isnan={is_nan}, isinf={is_inf}")

                if is_nan or is_inf:
                    print("⚠️ NaN/Inf detected in lm_loss, using fallback loss")
                    # 使用模型参数的L2损失作为fallback，确保梯度连接
                    param_loss = torch.tensor(0.0, device=shift_logits.device, dtype=shift_logits.dtype, requires_grad=True)
                    param_count = 0
                    for param in self.parameters():
                        if param.requires_grad:
                            param_loss = param_loss + torch.sum(param * param)
                            param_count += 1
                    if param_count > 0:
                        lm_loss = param_loss / param_count * 0.001  # 小的正则化损失
                    else:
                        lm_loss = torch.sum(shift_logits * shift_logits) * 0.001

                # 限制辅助损失的影响
                moe_aux_loss = torch.clamp(moe_aux_loss, min=0.0, max=1.0)

                # 总损失 = 语言模型损失 + MoE辅助损失
                loss = lm_loss + 0.01 * moe_aux_loss  # 增强负载均衡损失权重

                # 最终损失检查
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ NaN/Inf detected in total loss, using lm_loss only")
                    loss = lm_loss

                # 注释掉正则化损失，避免DDP参数重复问题
                # loss = loss + self._get_regularization_loss()

            except Exception as e:
                print(f"⚠️ Loss computation failed: {e}, using fallback loss")
                # 使用模型参数的L2损失作为fallback，确保梯度连接
                # 动态获取设备上第一个参数的dtype，避免硬编码
                first_param = next(self.parameters())
                param_loss = torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype, requires_grad=True)
                param_count = 0
                for param in self.parameters():
                    if param.requires_grad:
                        param_loss = param_loss + torch.sum(param * param)
                        param_count += 1
                if param_count > 0:
                    loss = param_loss / param_count * 0.01
                else:
                    loss = torch.tensor(1.0, device=first_param.device, dtype=first_param.dtype, requires_grad=True)

        # 6. 返回结果 - 增量架构版本
        return type('Outputs', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': final_hidden_states,  # 使用组合后的隐藏状态
            'expert_weights': expert_weights,
            'moe_aux_loss': moe_aux_loss,
            'base_hidden_states': base_hidden_states,  # 额外返回基础LoRA输出用于调试
            'moe_delta': moe_delta,  # 额外返回MoE增量用于调试
        })()

    def _get_regularization_loss(self):
        """
        获取正则化损失，确保所有参数都参与损失计算（DDP要求）
        """
        first_param = next(self.parameters())
        reg_loss = torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype, requires_grad=True)

        # 对所有可训练参数添加极小的L2正则化
        for param in self.parameters():
            if param.requires_grad:
                reg_loss = reg_loss + 1e-8 * torch.sum(param * param)

        return reg_loss

    def get_parameter_groups(self, base_lr: float, moe_lr: float):
        """
        获取分层参数组，用于设置不同的学习率

        Args:
            base_lr: 基础模型LoRA参数的学习率
            moe_lr: MoE组件的学习率

        Returns:
            param_groups: 参数组列表
        """
        base_params = []
        moe_params = []

        # 分离基础模型LoRA参数和MoE参数
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'moe_layer' in name:
                    moe_params.append(param)
                else:
                    base_params.append(param)

        param_groups = []

        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': base_lr,
                'weight_decay': 0.001,
                'name': 'base_lora'
            })

        if moe_params:
            param_groups.append({
                'params': moe_params,
                'lr': moe_lr,
                'weight_decay': 0.01,
                'name': 'moe'
            })

        logging.info(f"Parameter groups: base_lora={len(base_params)}, moe={len(moe_params)}")
        return param_groups

    def print_trainable_parameters(self):
        """打印可训练参数统计"""
        total_params = 0
        trainable_params = 0
        base_lora_params = 0
        moe_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'moe_layer' in name:
                    moe_params += param.numel()
                else:
                    base_lora_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || "
              f"Total params: {total_params:,} || "
              f"Trainable%: {100 * trainable_params / total_params:.4f}%")

        print(f"  - Base LoRA params: {base_lora_params:,}")
        print(f"  - MoE params: {moe_params:,}")
        print(f"  - Architecture: Joint LoRA + MoE")
        print(f"  - MoE experts: {self.config.num_moe_experts}")

    def save_model(self, save_path: str):
        """保存模型权重"""
        import os

        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        # 保存LoRA权重
        if hasattr(self.base_model, 'save_pretrained'):
            lora_path = os.path.join(save_path, 'lora_weights')
            self.base_model.save_pretrained(lora_path)

        # 保存MoE权重
        moe_state_dict = {}
        for name, param in self.moe_layer.named_parameters():
            moe_state_dict[name] = param.data

        moe_path = os.path.join(save_path, 'moe_weights.pt')
        torch.save(moe_state_dict, moe_path)

        # 保存配置
        config_path = os.path.join(save_path, 'joint_config.json')
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
            'dropout': self.config.dropout
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ Joint LoRA+MoE model saved to {save_path}")
        print(f"  - LoRA weights: {lora_path}")
        print(f"  - MoE weights: {moe_path}")
        print(f"  - Config: {config_path}")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=150,
                 do_sample=False, temperature=0.7, pad_token_id=None, eos_token_id=None, **kwargs):
        """
        改进的生成方法 - 确保使用MoE层

        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成token数
            do_sample: 是否采样
            temperature: 温度
            pad_token_id: padding token id
            eos_token_id: 结束token id

        Returns:
            生成的token ids
        """
        # 由于我们需要经过MoE层，不能直接使用base_model.generate
        # 需要实现自定义的生成循环

        with torch.no_grad():
            batch_size = input_ids.size(0)
            current_ids = input_ids.clone()

            if attention_mask is None:
                attention_mask = torch.ones_like(current_ids)

            # 重复检测计数器
            repeated_count = 0
            max_repeated_allowed = 2  # 减少到2次，更早介入
            last_few_tokens = []  # 跟踪最近几个token

            for step in range(max_new_tokens):
                # 使用我们的forward方法（包含MoE层）
                outputs = self.forward(
                    input_ids=current_ids,
                    attention_mask=attention_mask
                )

                # 获取最后一个位置的logits
                next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

                # 关键调试：检查前几步的logits - 注释掉，与tokenizer问题无关
                # if step < 3:
                #     max_logit = next_token_logits.max().item()
                #     min_logit = next_token_logits.min().item()
                #     print(f"🔍 Step {step} - Logits range: max={max_logit:.3f}, min={min_logit:.3f}")

                # 生成下一个token
                if do_sample:
                    # 采样生成
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪心解码
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 关键调试：前几步的token - 注释掉，与tokenizer问题无关
                # if step < 3:
                #     print(f"🔍 Step {step} - Token: {next_token_id.item()}")

                # 改进的重复检测和处理
                current_token = next_token_id.item()
                last_few_tokens.append(current_token)
                if len(last_few_tokens) > 5:  # 只保留最近5个token
                    last_few_tokens.pop(0)

                # 检查是否与前一个token重复
                if step > 0:
                    last_token = current_ids[:, -1].item()
                    if current_token == last_token:
                        repeated_count += 1
                        print(f"⚠️ Repeat {current_token} (#{repeated_count})")

                        # 不切换到base model，而是使用更强的惩罚策略
                        if repeated_count >= max_repeated_allowed:
                            print(f"🚨 Too many repeats, applying diversity boost")

                            # 惩罚最近出现的所有token
                            for token in set(last_few_tokens):
                                next_token_logits[:, token] -= 15.0

                            # 强制增加多样性 - 使用更高的温度
                            diversity_temp = max(temperature, 1.2) if temperature > 0 else 1.2
                            next_token_logits = next_token_logits / diversity_temp
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token_id = torch.multinomial(probs, num_samples=1)
                            current_token = next_token_id.item()

                            print(f"🔍 Diversity token: {current_token}")
                            repeated_count = 0  # 重置计数器

                        else:
                            # 轻度惩罚重复token
                            next_token_logits[:, current_token] -= 8.0

                            # 重新生成
                            if do_sample and temperature > 0:
                                next_token_logits = next_token_logits / temperature
                                probs = torch.softmax(next_token_logits, dim=-1)
                                next_token_id = torch.multinomial(probs, num_samples=1)
                            else:
                                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                            current_token = next_token_id.item()
                            print(f"🔍 New token: {current_token}")
                    else:
                        repeated_count = 0  # 重置计数器

                # 添加新token
                current_ids = torch.cat([current_ids, next_token_id], dim=-1)

                # 更新attention_mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=-1)

                # 检查是否生成了结束token
                if eos_token_id is not None and (next_token_id == eos_token_id).any():
                    break

            return current_ids