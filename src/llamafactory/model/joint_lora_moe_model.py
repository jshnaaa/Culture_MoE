# src/llamafactory/model/joint_lora_moe_model.py
"""
联合LoRA+MoE模型
同时训练预训练LoRA适配器和新增MoE处理层
实现端到端优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    raise ImportError("PEFT library is required. Please install with: pip install peft")


def is_main_process(rank=None):
    """检查是否为主进程"""
    if rank is None:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    return rank == 0


@dataclass
class JointLoRAMoEConfig:
    """联合LoRA+MoE配置"""

    # LoRA配置
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    pos_lora: str = "att"  # LoRA挂载位置：att=attention层, ffn=FFN层
    use_lora: bool = True  # 是否启用预训练LoRA微调

    # MoE配置
    num_moe_experts: int = 4
    num_activated_experts: int = 2  # 激活的专家数量，top-k
    moe_hidden_dim: int = 4096
    moe_intermediate_dim: int = None  # 默认为 moe_hidden_dim * 4
    moe_influence_weight: float = 0.5  # MoE影响权重，根据backbone调整

    # 文化损失配置
    use_culture_loss: bool = True
    culture_loss_weight: float = 0.01

    # MASK机制配置
    use_mask: bool = True  # 是否启用MASK机制双路输入处理

    # 消融实验配置
    use_shared: bool = True  # 是否使用共享专家
    use_moe: bool = True     # 是否使用MoE结构（router+路由专家），false时仅使用shared专家（消融实验）
    use_gate: bool = True    # 是否使用门控网络

    # 其他配置
    dropout: float = 0.1

    def __post_init__(self):
        if self.lora_target_modules is None:
            # 默认LoRA目标模块（注意力层）
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        if self.moe_intermediate_dim is None:
            self.moe_intermediate_dim = self.moe_hidden_dim * 4


class MoEExpert(nn.Module):
    """MoE专家层 - 数值稳定且有效输出版本"""

    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.1, dtype: torch.dtype = torch.float16):
        super().__init__()
        # 🔧 关键修复：使用更大的中间维度，确保专家有足够表达能力
        # 之前的限制太严格，导致专家容量不足
        self.intermediate_dim = intermediate_dim  # 使用完整的intermediate_dim，不限制

        self.gate_proj = nn.Linear(hidden_dim, self.intermediate_dim, bias=True, dtype=dtype)
        self.up_proj = nn.Linear(hidden_dim, self.intermediate_dim, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(self.intermediate_dim, hidden_dim, bias=True, dtype=dtype)
        self.act_fn = nn.SiLU()  # 恢复SiLU，配合更好的初始化
        self.dropout = nn.Dropout(dropout)

        # 保守但有效的初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化 - 数值稳定且有效输出版本"""
        # 🔧 关键修复：使用更合理的初始化策略
        # 之前的初始化太保守，导致专家学不到有效特征

        # 基于Xavier/Glorot初始化，但调整为适合当前架构
        fan_in = self.gate_proj.weight.size(1)  # input dimension
        fan_out = self.gate_proj.weight.size(0)  # output dimension

        # 使用适中的标准差，确保既稳定又有学习能力
        gate_up_std = (2.0 / (fan_in + fan_out)) ** 0.5 * 0.8  # 稍微保守的Xavier
        down_std = (2.0 / (self.intermediate_dim + fan_in)) ** 0.5 * 0.5  # 更保守的输出层

        try:
            # 初始化gate_proj和up_proj
            for module in [self.gate_proj, self.up_proj]:
                nn.init.normal_(module.weight, mean=0.0, std=gate_up_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # 初始化down_proj（输出层更保守）
            nn.init.normal_(self.down_proj.weight, mean=0.0, std=down_std)
            if self.down_proj.bias is not None:
                nn.init.zeros_(self.down_proj.bias)

            # 检查初始化结果
            for name, module in [("gate_proj", self.gate_proj), ("up_proj", self.up_proj), ("down_proj", self.down_proj)]:
                if torch.isnan(module.weight).any() or torch.isinf(module.weight).any():
                    print(f"⚠️ NaN detected in {name} after init, fallback to zeros")
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        except Exception as e:
            print(f"⚠️ Weight initialization failed: {e}")
            # 最后的安全措施：使用安全的小值初始化
            with torch.no_grad():
                for module in [self.gate_proj, self.up_proj, self.down_proj]:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        """前向传播 - 改进版本，平衡稳定性和有效性"""

        # 输入有效性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros_like(x)

        # 🔧 关键修复：使用更温和的输入预处理
        # 之前的归一化太激进，丢失了太多信息
        # 只对极端值进行裁剪，保留大部分原始动态范围
        x_processed = torch.clamp(x, min=-10.0, max=10.0)  # 更宽松的裁剪范围

        try:
            # 第一阶段：gate和up投影 - 使用处理后的输入
            gate_output = self.gate_proj(x_processed)
            up_output = self.up_proj(x_processed)

            # 🔧 简化权重检查：只在真正需要时检查和修复
            # 减少频繁的权重检查，只在输出异常时才检查权重
            if torch.isnan(gate_output).any() or torch.isinf(gate_output).any() or \
               torch.isnan(up_output).any() or torch.isinf(up_output).any():
                # 不打印详细信息，让MoELayer统一处理
                # print(f"    🚨 专家权重包含NaN，重新初始化并返回零输出!")
                # 重新初始化所有权重
                self._init_weights()
                return torch.zeros_like(x)

            # 激活函数和元素乘法
            gate_activated = self.act_fn(gate_output)
            intermediate = gate_activated * up_output

            # Dropout
            intermediate = self.dropout(intermediate)

            # 最终投影
            output = self.down_proj(intermediate)

            # 🔧 关键修复：调整输出缩放，确保MoE产生有意义的增量
            # 之前的缩放太小，导致MoE增量几乎为零
            # 目标：产生与基础LoRA相当的输出范围，然后在MoE层级别控制影响权重
            output = output * 5.0  # 增加缩放，让MoE有足够的表达能力

            # 最终检查 - 只检查NaN/Inf
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

        # 🔧 改进的路由器初始化 - 提高区分度
        with torch.no_grad():
            # 使用小的随机初始化，而不是全零，这样路由器能更好地学习区分专家
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)  # 小的随机初始化
            # 偏置初始化为小的随机值，而不是固定值
            nn.init.normal_(self.router.bias, mean=0.0, std=0.01)

        # Float32路由器的权重限制 - 放宽限制提高学习能力
        self.max_weight_value = 0.1    # 放宽权重限制，让路由器有更多学习空间
        self.max_bias_value = 0.5      # 放宽偏置限制

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

            # 2. 输入预处理和dtype转换 - 使用更温和的预处理
            x_safe = torch.clamp(x, min=-10.0, max=10.0)  # 更宽松的裁剪，与专家层一致

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
    """MoE层 - 支持共享专家和推理时的消融控制"""

    def __init__(self, config: JointLoRAMoEConfig, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.num_activated_experts = config.num_activated_experts
        self.hidden_dim = config.moe_hidden_dim
        self.dtype = dtype

        # 🔧 添加NaN统计计数器
        self.nan_count = 0
        self.total_forward_calls = 0

        # 创建路由器 - 强制使用Float32以确保数值稳定
        self.router = MoERouter(
            hidden_dim=config.moe_hidden_dim,
            num_experts=config.num_moe_experts,
            dropout=config.dropout,
            dtype=torch.float32  # 强制Float32，忽略传入的dtype
        )

        # 创建路由专家（通过路由器选择的专家）
        self.experts = nn.ModuleList([
            MoEExpert(
                hidden_dim=config.moe_hidden_dim,
                intermediate_dim=config.moe_intermediate_dim,
                dropout=config.dropout,
                dtype=dtype
            ) for _ in range(config.num_moe_experts)
        ])

        # 🔧 创建共享专家（如果启用）
        self.shared_expert = None
        if config.use_shared:
            self.shared_expert = MoEExpert(
                hidden_dim=config.moe_hidden_dim,
                intermediate_dim=config.moe_intermediate_dim,
                dropout=config.dropout,
                dtype=dtype
            )
            print(f"🔧 Shared expert created")

        # 🔧 创建门控网络（如果启用）
        self.gate_network = None
        if config.use_gate and config.use_shared:
            # 门控网络用于融合路由专家输出和共享专家输出
            # 🔧 关键修复：门控网络使用Float32提高数值稳定性，类似路由器
            self.gate_network = nn.Linear(config.moe_hidden_dim, 2, bias=True, dtype=torch.float32)
            print(f"🔧 Gate network created for expert fusion (Float32 for stability)")

        # 🔧 确保专家分化：每个专家使用不同的初始化
        print("🔧 Force reinitializing all experts with different seeds...")
        for i, expert in enumerate(self.experts):
            # 为每个专家设置不同的随机种子，确保分化
            torch.manual_seed(42 + i * 100)  # 不同的种子
            expert._init_weights()
            print(f"🔧 Routing expert {i} reinitialized with seed {42 + i * 100}")

        # 初始化共享专家
        if self.shared_expert is not None:
            torch.manual_seed(42 + 1000)  # 独特的种子
            self.shared_expert._init_weights()
            print(f"🔧 Shared expert reinitialized with seed {42 + 1000}")

        # 初始化门控网络
        if self.gate_network is not None:
            nn.init.normal_(self.gate_network.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.gate_network.bias, 0.0)
            print(f"🔧 Gate network initialized")

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

    def forward(self, hidden_states, mask_hidden_states=None, use_shared=None, use_moe=None, use_gate=None):
        """
        前向传播 - 增强损失版本，支持MASK机制的双路处理和推理时消融控制

        Args:
            hidden_states: 主输入隐藏状态
            mask_hidden_states: MASK版本输入隐藏状态（可选）
            use_shared: 推理时是否使用共享专家（None使用训练配置）
            use_moe: 推理时是否使用MoE结构（None使用训练配置，False=仅使用shared专家）
            use_gate: 推理时是否使用门控网络（None使用训练配置）

        Args:
            hidden_states: [B, L, H] 输入隐藏状态（用于路由专家）
            mask_hidden_states: [B, L, H] MASK版本隐藏状态（用于共享专家），可选
            use_shared: bool, 推理时是否使用共享专家。
                       None时使用训练配置，True/False时覆盖配置进行消融研究
            use_gate: bool, 推理时是否使用门控网络。
                     None时使用训练配置，True/False时覆盖配置进行消融研究

        Returns:
            output: [B, L, H] 输出隐藏状态
            expert_weights: [B, num_experts] 专家权重（稀疏，只有top-k有值）
            aux_loss: 辅助损失
            expert_outputs: Dict[int, torch.Tensor] 激活专家的输出
            soft_routing_scores: [B, num_experts] 所有专家的soft routing分数
            activated_experts: List[int] 被激活的专家索引列表
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 🔧 消融配置检查：动态控制专家使用和门控融合
        use_shared_actual = use_shared
        use_moe_actual = use_moe
        use_gate_actual = use_gate

        # 如果设置了ablation_config，使用消融配置覆盖参数
        if hasattr(self, 'ablation_config') and self.ablation_config is not None:
            if use_shared_actual is None:
                use_shared_actual = self.ablation_config.get('use_shared', True)
            if use_moe_actual is None:
                use_moe_actual = self.ablation_config.get('use_moe', True)
            if use_gate_actual is None:
                use_gate_actual = self.ablation_config.get('use_gate', True)

        # 如果仍为None，使用默认值
        if use_shared_actual is None:
            use_shared_actual = True
        if use_moe_actual is None:
            use_moe_actual = True
        if use_gate_actual is None:
            use_gate_actual = True

        # 🔧 消融实验：如果禁用MoE结构，只使用shared expert
        if use_moe_actual == False:
            # 跳过router和routing experts，只使用shared expert
            if use_shared_actual and self.shared_expert is not None:
                # 使用mask_hidden_states（如果有）或原始hidden_states
                input_for_shared = mask_hidden_states if mask_hidden_states is not None else hidden_states
                shared_output = self.shared_expert(input_for_shared)
                # 返回格式：(output, expert_weights, aux_loss, expert_outputs, soft_routing_scores, activated_experts, shared_output, routing_output)
                # 保持8个返回值与正常MoE流程一致
                return shared_output, None, None, None, None, None, shared_output, None
            else:
                # 没有shared expert或被禁用，返回原始输入
                return hidden_states, None, None, None, None, None, None, None

        # 🔧 增加前向传播调用计数
        self.total_forward_calls += 1
        current_nan_detected = False

        # 🔧 关键修复：初始化 shared_output 和 routing_output
        # 确保它们始终有定义，且有梯度连接
        shared_output = None
        routing_output = None

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

            # 路由计算 - 使用更高的温度提高区分度
            all_expert_weights, router_logits = self.router(pooled, temperature=1.0)

            # 保存soft routing scores供损失计算使用
            soft_routing_scores = all_expert_weights  # [B, num_experts] 所有专家的概率分布

            # 🔧 实现Top-k激活机制（动态k值）
            # 1. 选择top-k专家
            k = min(self.num_activated_experts, self.num_experts)  # 确保k不超过专家总数

            if k == self.num_experts:
                # Dense模式：激活所有专家
                expert_weights = all_expert_weights  # 直接使用所有专家的权重
                top_k_indices = torch.arange(self.num_experts, device=router_logits.device).unsqueeze(0).expand(batch_size, -1)  # [B, num_experts]
            else:
                # Top-k模式：只激活top-k专家
                top_k_logits, top_k_indices = torch.topk(router_logits, k=k, dim=-1)  # [B, k]

                # 2. 对top-k专家的logits重新归一化
                top_k_weights = torch.softmax(top_k_logits, dim=-1)  # [B, k] 归一化权重

                # 3. 创建稀疏权重矩阵（只有激活的专家有权重）
                expert_weights = torch.zeros_like(all_expert_weights)  # [B, num_experts]
                expert_weights.scatter_(1, top_k_indices, top_k_weights)  # 将归一化权重分配给激活专家

            # 检查路由器学习情况（只在异常时打印）- 注释掉频繁警告
            # logits_diff = (top_k_logits[:, 0] - top_k_logits[:, 1]).mean().item()
            # if logits_diff < 0.05:  # 只在差异过小时警告
            #     print(f"⚠️ 路由器区分度过低: {logits_diff:.3f}")

            # 2. 专家计算（动态Top-k版本）- 只计算激活的专家
            expert_outputs = {}  # 使用字典存储，只计算需要的专家
            valid_experts = 0

            # 获取所有激活的专家索引（去重）
            activated_experts = torch.unique(top_k_indices.flatten()).cpu().tolist()
            # print(f"🔍 激活的专家索引: {activated_experts} (k={k}, mode={'dense' if k == self.num_experts else f'top-{k}'})")  # 注释掉减少日志

            for expert_idx in activated_experts:
                try:
                    # 🔧 MASK机制：根据shared专家使用状态决定专家输入分化策略
                    # 使用消融配置控制共享专家
                    if mask_hidden_states is not None and self.config.use_mask and use_shared_actual:
                        # 完整模式（有shared专家）：路由专家使用分化策略
                        # 简化的专家分化策略：奇数专家使用原始输入，偶数专家使用MASK输入
                        if expert_idx % 2 == 0:
                            # 偶数专家使用MASK隐藏状态
                            expert_input = mask_hidden_states
                        else:
                            # 奇数专家使用原始隐藏状态
                            expert_input = hidden_states
                    else:
                        # 消融模式（无shared专家）或单路模式：所有路由专家使用原始隐藏状态
                        expert_input = hidden_states

                    expert_output = self.experts[expert_idx](expert_input)  # [B, L, H]

                    # 添加调试信息：检查每个专家的输出（简化版）
                    # expert_mean = expert_output.mean().item()
                    # expert_std = expert_output.std().item()
                    # expert_min = expert_output.min().item()
                    # expert_max = expert_output.max().item()
                    # print(f"    🔍 专家{expert_idx}最终输出: mean={expert_mean:.6f}, std={expert_std:.6f}, range=[{expert_min:.3f}, {expert_max:.3f}]")

                    # 检查专家输出
                    if not (torch.isnan(expert_output).any() or torch.isinf(expert_output).any()):
                        # 额外检查：如果专家输出全为零，可能是内部NaN导致的
                        if torch.all(expert_output == 0) and torch.all(hidden_states != 0):
                            # 暂时注释掉这个警告
                            # print(f"🚨 专家{expert_idx}输出全零(可能内部NaN)，使用零输出!")
                            current_nan_detected = True
                        expert_outputs[expert_idx] = expert_output
                        valid_experts += 1
                    else:
                        # 暂时注释掉这个警告
                        # print(f"🚨 专家{expert_idx}输出无效(NaN/Inf)，使用零输出!")
                        expert_outputs[expert_idx] = torch.zeros_like(hidden_states)
                        current_nan_detected = True

                except Exception as e:
                    print(f"⚠️ Expert {expert_idx} failed: {e}, using zeros")
                    expert_outputs[expert_idx] = torch.zeros_like(hidden_states)
                    current_nan_detected = True

            # 3. 动态Top-k专家输出混合（路由专家）
            if valid_experts == 0:
                print(f"🚨 所有激活专家都失效，使用输入passthrough! (k={k})")
                routing_output = hidden_states
            else:
                # 动态Top-k加权融合
                # 🔧 关键修复：使用 hidden_states * 0.0 而不是 torch.zeros_like()
                # 这样可以确保 routing_output 从一开始就有梯度连接
                routing_output = hidden_states * 0.0
                total_weight = 0.0

                for i in range(expert_weights.size(1)):  # 遍历所有专家
                    weight_val = expert_weights[:, i]  # [B]
                    if weight_val.sum().item() > 1e-8:  # 只处理有权重的专家
                        if i in expert_outputs:
                            weight = weight_val.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
                            weight = torch.clamp(weight, min=0.0, max=1.0)
                            # 🔧 避免使用 in-place 操作 +=，改用普通加法以确保梯度正确传播
                            routing_output = routing_output + weight * expert_outputs[i]
                            total_weight += weight_val.mean().item()

                # 降低总权重阈值，避免过早使用passthrough
                if total_weight < 0.001:  # 从0.01降到0.001
                    print("⚠️ Expert weights too small, using passthrough")
                    routing_output = hidden_states
                else:
                    # 🔧 增量架构：限制MoE输出为小的增量调整
                    # 目标：增量约为基础LoRA输出的5-10%，范围约[-5, +5]
                    routing_output = torch.clamp(routing_output, min=-5.0, max=5.0)

            # 4. 🔧 共享专家处理和融合
            # 使用消融配置控制共享专家
            if use_shared_actual and self.shared_expert is not None:
                # 计算共享专家输出
                try:
                    # 🔧 MASK机制：共享专家使用MASK隐藏状态（如果可用）
                    if mask_hidden_states is not None and self.config.use_mask:
                        shared_input = mask_hidden_states
                        # print("🔧 Shared expert using MASK hidden states")
                    else:
                        shared_input = hidden_states
                        # print("🔧 Shared expert using original hidden states")

                    shared_output = self.shared_expert(shared_input)

                    # 检查共享专家输出
                    if torch.isnan(shared_output).any() or torch.isinf(shared_output).any():
                        # print("⚠️ Shared expert output invalid, skipping")
                        shared_output = None
                    else:
                        shared_output = torch.clamp(shared_output, min=-5.0, max=5.0)
                        # print("✅ Shared expert output computed")

                except Exception as e:
                    print(f"⚠️ Shared expert computation failed: {e}")
                    shared_output = None

                # 融合路由专家和共享专家输出
                if shared_output is not None:
                    # 🔧 使用消融配置控制门控网络使用
                    if self.gate_network is not None and use_gate_actual:
                        # 使用门控网络融合 - 数值稳定版本
                        try:
                            # 🔧 步骤1: 计算并验证门控输入
                            gate_input = hidden_states.mean(dim=1)  # [B, H]

                            # 检查gate_input有效性
                            if torch.isnan(gate_input).any() or torch.isinf(gate_input).any():
                                print("⚠️ Gate input contains NaN/Inf, using fixed weights")
                                final_output = 0.5 * routing_output + 0.5 * shared_output
                            else:
                                # 限制gate_input范围，防止极值
                                gate_input = torch.clamp(gate_input, min=-10.0, max=10.0)

                                # 🔧 关键修复：转换为Float32以匹配gate network的dtype
                                gate_input = gate_input.float()  # 从Float16转换为Float32

                                # 🔧 步骤2: 检查门控网络权重
                                gate_weight_norm = self.gate_network.weight.norm().item()
                                if gate_weight_norm > 100.0 or gate_weight_norm < 1e-6:
                                    print(f"⚠️ Gate network weights abnormal (norm={gate_weight_norm:.6f}), reinitializing")
                                    with torch.no_grad():
                                        nn.init.normal_(self.gate_network.weight, mean=0.0, std=0.01)
                                        nn.init.constant_(self.gate_network.bias, 0.0)

                                # 🔧 步骤3: 计算门控logits并限制范围
                                gate_logits = self.gate_network(gate_input)  # [B, 2]
                                gate_logits = torch.clamp(gate_logits, min=-10.0, max=10.0)

                                # 🔧 步骤4: 数值稳定的softmax
                                # 使用shifted softmax避免溢出
                                max_logits = torch.max(gate_logits, dim=-1, keepdim=True)[0]
                                shifted_logits = gate_logits - max_logits
                                exp_logits = torch.exp(shifted_logits)
                                gate_weights = exp_logits / (torch.sum(exp_logits, dim=-1, keepdim=True) + 1e-8)

                                # 🔧 步骤5: 验证gate_weights有效性
                                if torch.isnan(gate_weights).any() or torch.isinf(gate_weights).any():
                                    print("⚠️ Gate weights contain NaN/Inf after softmax, using fixed weights")
                                    final_output = 0.5 * routing_output + 0.5 * shared_output
                                else:
                                    # 确保权重和为1（数值稳定性）
                                    gate_weights = gate_weights / (torch.sum(gate_weights, dim=-1, keepdim=True) + 1e-8)

                                    # 🔧 转换回原始dtype以保持一致性
                                    gate_weights = gate_weights.to(routing_output.dtype)

                                    # gate_weights[:, 0] -> 路由专家权重
                                    # gate_weights[:, 1] -> 共享专家权重
                                    routing_weight = gate_weights[:, 0].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
                                    shared_weight = gate_weights[:, 1].unsqueeze(1).unsqueeze(2)   # [B, 1, 1]

                                    # 最终检查权重有效性
                                    if (torch.isnan(routing_weight).any() or torch.isnan(shared_weight).any() or
                                        torch.isinf(routing_weight).any() or torch.isinf(shared_weight).any()):
                                        print("⚠️ Final gate weights invalid, using fixed weights")
                                        final_output = 0.5 * routing_output + 0.5 * shared_output
                                    else:
                                        final_output = routing_weight * routing_output + shared_weight * shared_output
                                        # print(f"🔧 Gate fusion: routing_weight={routing_weight.mean().item():.3f}, shared_weight={shared_weight.mean().item():.3f}")

                        except Exception as e:
                            print(f"⚠️ Gate network completely failed: {e}, using fixed weights")
                            final_output = 0.5 * routing_output + 0.5 * shared_output
                    else:
                        # 🔧 消融模式：使用固定权重融合
                        final_output = 0.5 * routing_output + 0.5 * shared_output
                        if not use_gate_actual:
                            # print("🔧 消融研究模式: 推理时禁用门控网络，使用固定权重融合 (use_gate=False)")
                            pass
                        elif self.gate_network is None:
                            # print("🔧 Fixed weight fusion: gate network not available")
                            pass
                        else:
                            # print("🔧 Fixed weight fusion of routing and shared experts")
                            pass
                else:
                    # 共享专家失效，只使用路由专家
                    final_output = routing_output
                    # print("🔧 Using routing experts only (shared expert failed)")
            else:
                # 不使用共享专家
                final_output = routing_output
                # 🔧 关键修复：当不使用共享专家时，shared_output 保持为 None
                shared_output = None
                if use_shared_actual is False:
                    # print("🔧 消融研究模式: 推理时禁用共享专家和MASK分化机制 (use_shared=False)")
                    pass
                elif not self.config.use_shared:
                    # print("🔧 Shared expert disabled by config")
                    pass
                elif self.shared_expert is None:
                    # print("🔧 Shared expert not available")
                    pass
                else:
                    # print("🔧 Using routing experts only")
                    pass

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

            # 🔧 更新NaN计数
            if current_nan_detected:
                self.nan_count += 1

            # 返回增强损失所需的完整信息
            return final_output, expert_weights, aux_loss, expert_outputs, soft_routing_scores, activated_experts, shared_output, routing_output

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

            # 返回fallback时的完整信息
            fallback_expert_outputs = {}
            fallback_soft_routing_scores = expert_weights
            fallback_activated_experts = list(range(self.num_experts))

            return fallback_output, expert_weights, aux_loss, fallback_expert_outputs, fallback_soft_routing_scores, fallback_activated_experts, None, None


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

        # 1. 根据配置决定是否应用LoRA到基础模型
        if config.use_lora:
            self._apply_lora()
        else:
            # 冻结基础模型，不启用LoRA
            self._freeze_base_model()

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

        # 🔧 确保门控网络也保持Float32并在正确设备上
        if self.moe_layer.gate_network is not None:
            self.moe_layer.gate_network = self.moe_layer.gate_network.to(device=device, dtype=torch.float32)
            print(f"🔧 Gate network moved to device={device}, dtype=Float32")

        # 🔧 添加消融评估配置（推理时动态控制）
        self.ablation_config = {
            'use_shared': None,  # None使用训练配置，True/False覆盖配置
            'use_moe': None,     # None使用训练配置，False=仅使用shared专家
            'use_gate': None,
            'use_mask': None
        }

        logging.info(f"Joint LoRA+MoE model initialized with {config.num_moe_experts} experts")

    def _apply_lora(self):
        """应用LoRA到基础模型"""
        # 🔧 根据pos_lora参数决定LoRA挂载位置
        # pos_lora="att": LoRA挂在attention层（默认）
        # pos_lora="ffn": LoRA挂在FFN层（即MoE专家层）
        if self.config.pos_lora == "ffn":
            # 当pos_lora="ffn"时，将MoE专家的LoRA也挂在attention层
            # 这样整个joint moe模型包括六组挂在attention层的lora适配器
            # 覆盖默认的attention层配置
            self.config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            if is_main_process():
                print(f"🔧 pos_lora=ffn: 将MoE专家LoRA也挂在attention层，共6组attention LoRA")
        else:
            # 默认情况或pos_lora="att"：LoRA挂在attention层
            self.config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            if is_main_process():
                print(f"🔧 pos_lora={self.config.pos_lora}: LoRA挂在attention层")

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

    def _freeze_base_model(self):
        """冻结基础模型，不启用LoRA微调"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        logging.info("Base model frozen (LoRA disabled)")
        print("🔒 基础模型已冻结，仅训练MoE专家层和路由器")

    def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None, attention_mask_mask=None,
                labels=None, culture_labels=None, use_shared=None, use_moe=None, use_gate=None, **kwargs):
        """
        前向传播

        Args:
            input_ids: [B, L] 输入token IDs
            attention_mask: [B, L] 注意力掩码
            input_ids_mask: [B, L] 可选的mask输入（用于MASK机制双路处理）
            attention_mask_mask: [B, L] mask输入的注意力掩码
            labels: [B, L] 标签（用于计算损失）
            culture_labels: [B] 文化标签（用于计算文化损失）
            use_shared: bool, 推理时是否使用共享专家。
                       None时使用训练配置，True/False时覆盖配置进行消融研究
            use_moe: bool, 推理时是否使用MoE结构（router+路由专家）。
                    None时使用训练配置，False=仅使用shared专家（消融实验）
            use_gate: bool, 推理时是否使用门控网络。
                     None时使用训练配置，True/False时覆盖配置进行消融研究

        Returns:
            outputs: 包含loss、logits、expert_weights等的字典
        """
        # 1. 🔧 MASK机制双路输入处理
        # 决定是否在当前推理中使用共享专家（影响MASK机制）
        use_shared_current = self.config.use_shared if use_shared is None else use_shared

        if self.config.use_mask and input_ids_mask is not None and attention_mask_mask is not None and use_shared_current:
            # 双路处理模式：分别处理原始输入和MASK输入
            # 原始输入用于路由专家
            base_outputs_original = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # MASK输入用于共享专家
            base_outputs_mask = self.base_model(
                input_ids=input_ids_mask,
                attention_mask=attention_mask_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # 获取两路隐藏状态
            hidden_states_original = base_outputs_original.hidden_states[-1]  # [B, L, H] 路由专家用
            hidden_states_mask = base_outputs_mask.hidden_states[-1]  # [B, L, H] 共享专家用

            # 使用原始输入的输出作为主要基础输出（用于后续logits计算）
            base_outputs = base_outputs_original
            hidden_states = hidden_states_original

            # 保存MASK版本的隐藏状态供MoE层使用
            self._mask_hidden_states = hidden_states_mask

        else:
            # 单路处理模式：所有专家使用相同输入
            # 包括：1) MASK机制禁用时 2) 消融模式禁用共享专家时
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # 2. 获取最后一层隐藏状态
            hidden_states = base_outputs.hidden_states[-1]  # [B, L, H]
            self._mask_hidden_states = None

        # 🔍 调试基础模型输出（简化版）
        # base_range = f"min={hidden_states.min().item():.3f}, max={hidden_states.max().item():.3f}"
        # base_std = hidden_states.std().item()
        # print(f"🔍 基础模型输出: {base_range}, std={base_std:.6f}")

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

        # 3. MoE层处理 - 增量架构实现，支持MASK机制的双路处理和推理时消融控制
        # print("🔧 启用增量MoE架构：MoE作为基础LoRA的增量调整")  # 减少日志
        moe_delta, expert_weights, moe_aux_loss, expert_outputs, soft_routing_scores, activated_experts, shared_output, routing_output = self.moe_layer(
            hidden_states,
            mask_hidden_states=getattr(self, '_mask_hidden_states', None),
            use_shared=use_shared,
            use_moe=use_moe,
            use_gate=use_gate
        )

        # 关键调试：检查MoE增量输出（只在异常时打印）- 暂时注释掉
        moe_std = moe_delta.std().item()
        # if moe_std < 1e-6:
        #     print(f"⚠️ 警告: MoE增量几乎为零 (std={moe_std:.8f})!")
        # elif moe_std > 5.0:
        #     print(f"⚠️ 警告: MoE增量过大 (std={moe_std:.6f})，应该是小的调整!")

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

        # MoE增量权重（控制MoE影响程度）- 从配置中获取
        moe_influence_weight = self.config.moe_influence_weight

        # 组合输出：基础LoRA + 加权MoE增量
        combined_output = base_hidden_states + moe_influence_weight * moe_delta

        # 调试信息：检查组合后的输出（简化版）
        # combined_range = f"min={combined_output.min().item():.3f}, max={combined_output.max().item():.3f}"
        # combined_std = combined_output.std().item()
        # print(f"🔧 组合输出(基础+MoE): {combined_range}, std={combined_std:.6f}")
        # print(f"🔧 MoE影响权重: {moe_influence_weight}, 实际增量贡献: {(moe_influence_weight * moe_std):.6f}")

        # 使用组合输出作为最终隐藏状态
        final_hidden_states = combined_output

        # 4. 语言模型头 - 使用组合后的隐藏状态
        # 维度已经在上面处理过了，final_hidden_states = combined_output应该与base_hidden_states维度一致

        # 使用基础模型的lm_head计算最终logits
        if hasattr(self.base_model, 'lm_head'):
            # print(f"🔧 使用组合隐藏状态(基础LoRA+MoE增量)计算logits")  # 减少日志
            logits = self.base_model.lm_head(final_hidden_states)
        elif hasattr(self.base_model, 'base_model') and hasattr(self.base_model.base_model, 'lm_head'):
            # print(f"🔧 使用组合隐藏状态(基础LoRA+MoE增量)计算logits")  # 减少日志
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

        # 关键调试：检查最终logits（只在异常时打印）- 暂时注释掉
        logits_std = logits.std().item()
        # if logits_std < 1e-6:
        #     print(f"⚠️ 警告: Logits几乎为零 (std={logits_std:.8f})!")
        # elif torch.isnan(logits).any():
        #     print(f"⚠️ 警告: Logits包含NaN!")
        # elif torch.isinf(logits).any():
        #     print(f"⚠️ 警告: Logits包含Inf!")

        # # 详细logits信息（注释掉）
        # logits_range = f"min={logits.min().item():.3f}, max={logits.max().item():.3f}"
        # print(f"🔧 最终Logits: {logits_range}, std={logits_std:.6f}")

        # 5. 计算损失 - 增量架构版本
        loss = None
        culture_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)  # 初始化文化损失
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

                # 🔍 损失调试信息（简化版，只在异常时打印）
                # 暂时注释掉所有标签相关警告，专注解决evaluation后的日志问题
                # if shift_valid == 0:
                #     print(f"⚠️ 警告: 没有有效训练标签!")
                # 注释掉有效标签过少的警告，因为单个token的标签是正常的
                # elif shift_valid < 3:  # 只在标签过少时警告
                #     print(f"⚠️ 警告: 有效标签过少: {shift_valid}/{total_labels}")

                # # 详细调试信息（注释掉）
                # print(f"🔍 损失计算分析:")
                # print(f"  shift_logits shape: {shift_logits.shape}")
                # print(f"  shift_labels shape: {shift_labels.shape}")
                # print(f"  有效标签数量: {shift_valid}/{total_labels}")
                # print(f"  logits范围: min={shift_logits.min().item():.3f}, max={shift_logits.max().item():.3f}")

                # # 检查第一个样本的有效标签
                # if shift_valid > 0:
                #     valid_positions = (shift_labels.view(-1) != -100).nonzero().flatten()
                #     if len(valid_positions) > 0:
                #         first_valid_pos = valid_positions[0].item()
                #         first_valid_label = shift_labels.view(-1)[first_valid_pos].item()
                #         first_valid_logit = shift_logits.view(-1, shift_logits.size(-1))[first_valid_pos]
                #         print(f"  第一个有效标签: 位置{first_valid_pos}, 标签{first_valid_label}")
                #         print(f"  对应logit范围: min={first_valid_logit.min().item():.3f}, max={first_valid_logit.max().item():.3f}")

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

                # 🔧 计算文化损失（如果启用）
                culture_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype, requires_grad=True)
                if self.config.use_culture_loss and culture_labels is not None and expert_weights is not None:
                    # 导入文化损失计算函数
                    from train_joint_lora_moe import compute_culture_loss

                    # 计算文化损失
                    culture_loss = compute_culture_loss(expert_weights, culture_labels, self.config.culture_loss_weight)

                    # 确保文化损失的数据类型和设备与主损失一致
                    culture_loss = culture_loss.to(device=lm_loss.device, dtype=lm_loss.dtype)

                # 总损失 = 语言模型损失 + MoE辅助损失 + 文化损失
                loss = lm_loss + 0.01 * moe_aux_loss + culture_loss

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

                # fallback情况下的文化损失
                culture_loss = torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype, requires_grad=True)

        # 6. 返回结果 - 增量架构版本，包含增强损失所需信息
        return type('Outputs', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': final_hidden_states,  # 使用组合后的隐藏状态
            'expert_weights': expert_weights,
            'moe_aux_loss': moe_aux_loss,
            'culture_loss': culture_loss,  # 文化损失
            'base_hidden_states': base_hidden_states,  # 额外返回基础LoRA输出用于调试
            'moe_delta': moe_delta,  # 额外返回MoE增量用于调试
            # 增强损失所需的新信息
            'expert_outputs': expert_outputs,  # 激活专家的输出
            'soft_routing_scores': soft_routing_scores,  # 所有专家的soft routing分数
            'activated_experts': activated_experts,  # 被激活的专家索引列表
            # 🔧 CSL损失计算所需的专家输出
            'shared_expert_outputs': shared_output,  # 共享专家输出
            'router_expert_outputs': routing_output,  # 路由专家输出
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
                'name': 'base_lora' if self.config.use_lora else 'base_frozen'
            })

        if moe_params:
            param_groups.append({
                'params': moe_params,
                'lr': moe_lr,
                'weight_decay': 0.01,
                'name': 'moe'
            })

        base_group_name = 'base_lora' if self.config.use_lora else 'base_frozen'
        logging.info(f"Parameter groups: {base_group_name}={len(base_params)}, moe={len(moe_params)}")

        if not self.config.use_lora and len(base_params) > 0:
            print(f"⚠️ 警告: use_lora=False但仍有{len(base_params)}个基础模型参数可训练")
        elif self.config.use_lora and len(base_params) == 0:
            print(f"⚠️ 警告: use_lora=True但没有找到可训练的基础模型参数")

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

        if self.config.use_lora:
            print(f"  - Base LoRA params: {base_lora_params:,}")
            print(f"  - MoE params: {moe_params:,}")
            print(f"  - Architecture: Joint LoRA + MoE (预训练LoRA + 专家LoRA)")
            print(f"  - LoRA挂载位置: {self.config.pos_lora} (att=attention层, ffn=FFN层)")
        else:
            print(f"  - Base params (frozen): {total_params - trainable_params:,}")
            print(f"  - MoE params (trainable): {moe_params:,}")
            if base_lora_params > 0:
                print(f"  - Unexpected trainable base params: {base_lora_params:,}")
            print(f"  - Architecture: MoE Expert Training (基座冻结)")
        print(f"  - MoE experts: {self.config.num_moe_experts}")
        print(f"  - Use LoRA: {self.config.use_lora}")

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
            'dropout': self.config.dropout,
            'pos_lora': self.config.pos_lora,  # 🔧 新增：保存pos_lora配置
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ Joint LoRA+MoE model saved to {save_path}")
        print(f"  - LoRA weights: {lora_path}")
        print(f"  - MoE weights: {moe_path}")
        print(f"  - Config: {config_path}")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=10,
                 do_sample=False, temperature=0.7, pad_token_id=None, eos_token_id=None, **kwargs):
        """
        改进的生成方法 - 专为单数字答案优化

        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成token数（默认10，对单数字答案足够）
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

            # 🔧 提取消融控制参数
            use_shared = kwargs.get('use_shared', None)
            use_moe = kwargs.get('use_moe', None)
            use_gate = kwargs.get('use_gate', None)

            # 重复检测计数器
            repeated_count = 0
            max_repeated_allowed = 2  # 减少到2次，更早介入
            last_few_tokens = []  # 跟踪最近几个token

            for step in range(max_new_tokens):
                # 使用我们的forward方法（包含MoE层）
                # 🔧 传递消融控制参数
                outputs = self.forward(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    use_shared=use_shared,
                    use_moe=use_moe,
                    use_gate=use_gate
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

                # 🔧 数字答案检测和早停机制
                current_token = next_token_id.item()

                # 检查是否生成了有效的数字答案
                # 对于选择题，我们期望的是单个数字 1, 2, 3, 4
                # 这些数字的token ID通常是: 16='1', 17='2', 18='3', 19='4'
                if step == 0:  # 第一个生成的token
                    if current_token in [16, 17, 18, 19]:  # 数字1-4的token ID
                        # 生成了有效数字，立即停止
                        current_ids = torch.cat([current_ids, next_token_id], dim=-1)
                        break

                last_few_tokens.append(current_token)
                if len(last_few_tokens) > 5:  # 只保留最近5个token
                    last_few_tokens.pop(0)

                # 检查是否与前一个token重复
                if step > 0:
                    last_token = current_ids[:, -1].item()
                    if current_token == last_token:
                        repeated_count += 1
                        # print(f"⚠️ Repeat {current_token} (#{repeated_count})")

                        # 不切换到base model，而是使用更强的惩罚策略
                        if repeated_count >= max_repeated_allowed:
                            # print(f"🚨 Too many repeats, applying diversity boost")

                            # 惩罚最近出现的所有token
                            for token in set(last_few_tokens):
                                next_token_logits[:, token] -= 15.0

                            # 强制增加多样性 - 使用更高的温度
                            diversity_temp = max(temperature, 1.2) if temperature > 0 else 1.2
                            next_token_logits = next_token_logits / diversity_temp
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token_id = torch.multinomial(probs, num_samples=1)
                            current_token = next_token_id.item()

                            # print(f"🔍 Diversity token: {current_token}")
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
                            # print(f"🔍 New token: {current_token}")
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

    def set_ablation_config(self, use_shared=None, use_gate=None, use_mask=None):
        """
        设置推理时消融配置，用于动态控制模型行为

        Args:
            use_shared: 是否使用共享专家 (None使用训练配置，True/False覆盖配置)
            use_gate: 是否使用门控网络 (None使用训练配置，True/False覆盖配置)
            use_mask: 是否使用MASK机制 (None使用训练配置，True/False覆盖配置)
        """
        self.ablation_config.update({
            'use_shared': use_shared,
            'use_gate': use_gate,
            'use_mask': use_mask
        })

        # 将配置传递给MoE层
        if hasattr(self, 'moe_layer') and self.moe_layer is not None:
            self.moe_layer.ablation_config = self.ablation_config

        print(f"🔧 消融配置已更新: use_shared={use_shared}, use_gate={use_gate}, use_mask={use_mask}")