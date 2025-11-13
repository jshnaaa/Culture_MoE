# src/llamafactory/model/CultureMoE.py
import torch
import torch.nn as nn

from .experts import ExpertLayer
from .moe_args import ModelArgs
from .router import ExpertRouter


# class CrossAttentionBlock(nn.Module):
#     """ Shared 层输出作为 Q，专家+shared 输出作为 KV """
#
#     def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
#         )
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )
#         self.norm2 = nn.LayerNorm(hidden_dim)
#
#     def forward(self, query, key_value):
#         # Q = shared_out, KV = shared + experts
#         attn_out, _ = self.attn(query, key_value, key_value)
#         x = self.norm1(query + attn_out)
#
#         ffn_out = self.ffn(x)
#         out = self.norm2(x + ffn_out)
#         return out


class LlamaSharedRouterExpertsModel(nn.Module):
    def __init__(self, llama_model, config, args: ModelArgs):
        super().__init__()

        self.llama_model = llama_model
        self.args = args
        self.config = llama_model.config

        # ✅ 用于存储最近一次前向传播的专家权重
        self._last_expert_weights = None

        # ✅ MoE 预热相关
        self.moe_warmup_steps = 0
        self.moe_warmup_total_steps = 0
        self.moe_warmup_enabled = False

        # ✅ generation_config 从 llama_model 继承
        if hasattr(llama_model, 'generation_config'):
            self.generation_config = llama_model.generation_config
        else:
            from transformers import GenerationConfig
            self.generation_config = GenerationConfig()

        hidden_dim = self.config.hidden_size

        # ✅ 可学习的 MoE 融合系数（替代拼接方式）
        # h_final = h_shared + alpha * h_moe
        # 初始值 0.05，让 MoE 初期影响较小，逐步学习增加影响力
        self.moe_fusion_alpha = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        # 2. Shared 层 - 改进初始化
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, args.shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.shared_hidden_dim, hidden_dim)
        )

        # ✅ 改进 Shared 层初始化
        self._init_shared_layer()

        # 3. Router
        self.router = ExpertRouter(
            router_input_dim=hidden_dim,
            num_experts=args.num_experts,
            router_hidden_dim=args.router_hidden_dim,
            dropout=args.dropout
        )

        # 4. Experts 层
        self.experts_layer = ExpertLayer(
            experts_input_dim=hidden_dim,
            experts_hidden_dim=args.experts_hidden_dim,
            experts_output_dim=hidden_dim,
            num_experts=args.num_experts,
            lora_rank=args.lora_rank,
            dropout=args.dropout,
            return_all=True
        )

        # ✅ 5. 移除分类头，使用 LLaMA 的 lm_head 进行生成
        # 不需要额外的分类头，直接使用 llama_model.lm_head

    # ✅ 添加梯度检查点相关方法
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        启用梯度检查点
        """
        if hasattr(self.llama_model, 'gradient_checkpointing_enable'):
            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            # 兼容旧版本
            if hasattr(self.llama_model, 'enable_input_require_grads'):
                self.llama_model.enable_input_require_grads()
            if hasattr(self.llama_model, 'gradient_checkpointing'):
                self.llama_model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """
        禁用梯度检查点
        """
        if hasattr(self.llama_model, 'gradient_checkpointing_disable'):
            self.llama_model.gradient_checkpointing_disable()
        else:
            if hasattr(self.llama_model, 'gradient_checkpointing'):
                self.llama_model.gradient_checkpointing = False

    def is_gradient_checkpointing(self):
        """
        检查是否启用了梯度检查点
        """
        if hasattr(self.llama_model, 'is_gradient_checkpointing'):
            return self.llama_model.is_gradient_checkpointing
        elif hasattr(self.llama_model, 'gradient_checkpointing'):
            return self.llama_model.gradient_checkpointing
        return False

    def _init_shared_layer(self):
        """
        ✅ 改进 Shared 层初始化
        使用小的初始化确保 MoE 层不会过度改变 LLaMA 的输出
        """
        for i, module in enumerate(self.shared):
            if isinstance(module, nn.Linear):
                # 第一层：从 hidden_dim 到 shared_hidden_dim
                if i == 0:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                # 最后一层：从 shared_hidden_dim 回到 hidden_dim
                # 使用小的初始化，使输出接近 0（接近恒等映射）
                elif i == len(self.shared) - 1:
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def set_moe_warmup(self, total_steps: int, enabled: bool = True):
        """
        ✅ 设置 MoE 预热参数

        Args:
            total_steps: 总训练步数
            enabled: 是否启用预热
        """
        self.moe_warmup_enabled = enabled
        self.moe_warmup_total_steps = total_steps
        self.moe_warmup_steps = 0

    def get_moe_warmup_weight(self) -> float:
        """
        ✅ 获取当前的 MoE 预热权重

        Returns:
            float: 0.0 到 1.0 之间的权重
        """
        if not self.moe_warmup_enabled or self.moe_warmup_total_steps == 0:
            return 1.0

        # 前 20% 的步数用于预热
        warmup_steps = int(self.moe_warmup_total_steps * 0.2)

        if self.moe_warmup_steps < warmup_steps:
            # 线性预热
            return self.moe_warmup_steps / warmup_steps
        else:
            return 1.0

    def step_moe_warmup(self):
        """
        ✅ 更新 MoE 预热步数
        """
        if self.moe_warmup_enabled:
            self.moe_warmup_steps += 1

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.llama_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.llama_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None, attention_mask_mask=None,
                labels=None, culture_labels=None, use_culture_loss=False, culture_loss_lambda=0.5,
                culture_loss_alpha=2.0, culture_loss_beta=1.0,
                use_shared_experts=True, **kwargs):
        """
        ✅ 生成式前向传播

        Args:
            input_ids: [B, L] 输入 token IDs
            attention_mask: [B, L] 注意力掩码
            input_ids_mask: [B, L] 可选的 mask 输入（用于文化损失）
            attention_mask_mask: [B, L] mask 输入的注意力掩码
            labels: [B, L] 生成式标签（shifted input_ids）
            culture_labels: [B] 文化标签
            use_culture_loss: 是否使用文化损失
            culture_loss_lambda: 文化损失权重
            use_shared_experts: 是否使用共享专家层（消融实验）

        Returns:
            outputs: dict with keys:
                - logits: [B, L, vocab_size] 生成 logits
                - loss: 总损失（如果提供了 labels）
                - generation_loss: 生成损失
                - culture_loss: 文化损失
        """
        # ✅ 获取输入的设备和数据类型
        device = input_ids.device
        dtype = self.shared[0].weight.dtype

        # ✅ Step 1: LLaMA forward for h_all (instruction + input)
        outputs_all = self.llama_model.model(  # ✅ 使用 model 而不是整个 llama_model
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # ✅ 获取最后一层的 hidden states
        hidden_all = outputs_all.last_hidden_state  # [B, L, H]
        if hidden_all.device != device:
            hidden_all = hidden_all.to(device)
        if hidden_all.dtype != dtype:
            hidden_all = hidden_all.to(dtype)

        h_all = hidden_all.clone()

        # ✅ Step 2: LLaMA forward for h_no (instruction_mask + input)
        # ✅ 如果不使用共享专家，则不需要处理 instruction_mask
        if use_shared_experts and input_ids_mask is not None:
            outputs_no = self.llama_model.model(
                input_ids_mask,
                attention_mask=attention_mask_mask,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_no = outputs_no.last_hidden_state
            if hidden_no.device != device:
                hidden_no = hidden_no.to(device)
            if hidden_no.dtype != dtype:
                hidden_no = hidden_no.to(dtype)

            h_no = hidden_no.clone()
        else:
            h_no = h_all.clone()

        # Step 3: Shared 层
        # ✅ 如果不使用共享专家，则跳过 shared 层
        if use_shared_experts:
            h_no = h_no.to(device=device, dtype=dtype)
            shared_out = self.shared(h_no)  # [B, L, H]
        else:
            # ✅ 消融实验：不使用共享专家，shared_out 为零
            shared_out = torch.zeros_like(h_all)

        # Step 4: Router（基于 pooled representation）
        # ✅ 如果使用共享专家，基于 shared_out 计算路由；否则基于 h_all 计算
        if use_shared_experts:
            pooled = shared_out.mean(dim=1)  # [B, H]
        else:
            pooled = h_all.mean(dim=1)  # [B, H]
        expert_weights, router_logits = self.router(pooled)  # [B, E]

        # ✅ 保存专家权重
        self._last_expert_weights = expert_weights.detach()

        # Step 5: Experts 层
        h_all = h_all.to(device=device, dtype=dtype)
        expert_outs = self.experts_layer(h_all)  # list of [B, L, H]

        # ✅ Step 6: 获取 MoE 预热权重
        moe_warmup_weight = self.get_moe_warmup_weight()

        # Step 7: 权重缩放专家输出
        weighted_expert_outs = [
            expert_outs[i] * expert_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            for i in range(len(expert_outs))
        ]

        # Step 8: 融合 shared + 加权专家输出
        # ✅ 使用加权融合 + 残差，替代拼接方式
        # h_final = h_shared + alpha * h_moe
        # 这样可以：
        # 1. 保持输出维度不变（不改变分类头输入维度）
        # 2. 初期 MoE 影响较小，逐步学习增加影响力
        # 3. 梯度流均衡，MoE 能学到有意义的文化分化

        expert_sum = torch.stack(weighted_expert_outs, dim=0).sum(dim=0)  # [B, L_all, H]

        # ✅ 处理序列长度不匹配的情况
        # shared_out 来自 h_no (instruction_mask + input)
        # expert_sum 来自 h_all (instruction + input)
        # 它们的长度可能不同

        if shared_out.size(1) != expert_sum.size(1):
            # 取较短的长度
            min_len = min(shared_out.size(1), expert_sum.size(1))
            shared_out = shared_out[:, :min_len, :]
            expert_sum = expert_sum[:, :min_len, :]

        # ✅ 应用可学习的 MoE 融合系数
        # 初期 alpha 很小（0.05），MoE 影响较小，不会破坏原模型输出分布
        # 随着训练，alpha 会自动调节，学习 MoE 在文化差异显著样本上的影响力
        # 同时应用 MoE 预热权重（在预热阶段逐步增加 MoE 的影响）
        moe_contribution = self.moe_fusion_alpha * expert_sum  # [B, L, H]
        enhanced_hidden = shared_out + moe_warmup_weight * moe_contribution  # [B, L, H]

        # ✅ Step 8: 使用 LLaMA 的 lm_head 生成 logits
        # 确保 enhanced_hidden 与 lm_head 的数据类型一致
        lm_head_dtype = self.llama_model.lm_head.weight.dtype
        if enhanced_hidden.dtype != lm_head_dtype:
            enhanced_hidden = enhanced_hidden.to(lm_head_dtype)

        logits = self.llama_model.lm_head(enhanced_hidden)  # [B, L, vocab_size]

        # ✅ 裁剪 logits（防止数值溢出导致异常高的损失）
        logits = torch.clamp(logits, min=-100, max=100)

        # ✅ Step 9: 计算损失（如果提供了 labels）
        outputs = {'logits': logits, 'expert_weights': expert_weights, 'router_logits': router_logits}

        if labels is not None:
            # ✅ 检查 labels 的维度
            if labels.dim() == 1:
                # 如果 labels 是 1D [B]，说明是分类标签，不是生成式标签
                raise ValueError(
                    f"Expected labels to be 2D [batch_size, seq_len] for generative training, "
                    f"but got 1D [batch_size]. This suggests the data collator is not correctly "
                    f"preparing labels for generative training. "
                    f"Labels shape: {labels.shape}, Logits shape: {logits.shape}"
                )

            # ✅ 如果 labels 的长度与 logits 不匹配，截断 labels
            if labels.size(1) != logits.size(1):
                min_len = min(labels.size(1), logits.size(1))
                labels = labels[:, :min_len]
                logits = logits[:, :min_len, :]

            # ✅ 生成式损失（CrossEntropyLoss）
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # ✅ 确保 shift_labels 与 shift_logits 在同一设备上
            if shift_labels.device != shift_logits.device:
                shift_labels = shift_labels.to(shift_logits.device)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            generation_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs['generation_loss'] = generation_loss

            # 文化损失（如果启用）
            if use_culture_loss and culture_labels is not None:
                # ✅ 新的文化专注性损失（包含 specialization 和 diversity）
                culture_loss = self.compute_culture_loss(
                    expert_weights,
                    culture_labels,
                    alpha=culture_loss_alpha,  # specialization 权重
                    beta=culture_loss_beta,   # diversity 权重
                    eps=1e-8,
                    min_samples=2
                )

                # ✅ 确保 culture_loss 与 generation_loss 在同一设备上
                if culture_loss.device != generation_loss.device:
                    culture_loss = culture_loss.to(generation_loss.device)

                outputs['culture_loss'] = culture_loss

                # ✅ 计算 specialization 和 diversity 的分量（用于监控）
                # 注意：这里只是为了监控，不影响梯度
                with torch.no_grad():
                    spec_loss, div_loss = self.compute_culture_loss_components(
                        expert_weights, culture_labels
                    )
                    outputs['specialization_loss'] = spec_loss
                    outputs['diversity_loss'] = div_loss

                # ✅ 总损失：生成损失 + 文化损失
                # 注意：不再添加 router_entropy_loss，因为它与 specialization 相反
                total_loss = generation_loss + culture_loss_lambda * culture_loss
            else:
                # ✅ 确保 culture_loss 与 generation_loss 在同一设备上
                outputs['culture_loss'] = torch.tensor(0.0, device=generation_loss.device)
                outputs['specialization_loss'] = torch.tensor(0.0, device=generation_loss.device)
                outputs['diversity_loss'] = torch.tensor(0.0, device=generation_loss.device)
                # ✅ 总损失：只有生成损失
                total_loss = generation_loss

            outputs['loss'] = total_loss

        return outputs

    def compute_router_entropy_loss(self, expert_weights):
        """
        计算 Router 熵正则化损失，防止 Router 塌陷

        Args:
            expert_weights: [B, E] 专家权重

        Returns:
            entropy_loss: 标量（负熵，最小化负熵 = 最大化熵）
        """
        # 计算每个样本的熵
        entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-8), dim=-1)

        # 返回负熵（我们希望最大化熵，即最小化负熵）
        entropy_loss = -entropy.mean()

        return entropy_loss

    def compute_culture_loss(self, expert_weights, culture_labels, alpha=1.0, beta=1.0, eps=1e-8, min_samples=2):
        """
        ✅ 新的文化专注性损失：鼓励每个文化专注于少数专家，且不同专家专注于不同文化

        Args:
            expert_weights: [B, E] 专家权重（router 输出的软权重）
            culture_labels: [B] 文化标签
            alpha: specialization 损失权重（文化内稀疏化）
            beta: diversity 损失权重（文化间互异化）
            eps: 数值稳定性常数
            min_samples: 最小样本数阈值（只对样本数 >= min_samples 的文化计算损失）

        Returns:
            culture_loss: 标量
        """
        device = expert_weights.device
        dtype = expert_weights.dtype

        # ✅ 将 culture_labels 转换为 tensor
        if not isinstance(culture_labels, torch.Tensor):
            culture_labels = torch.tensor(culture_labels, device=device, dtype=torch.long)
        else:
            culture_labels = culture_labels.to(device=device, dtype=torch.long)

        # ✅ 确保 culture_labels 是 1D [B]
        if culture_labels.dim() > 1:
            culture_labels = culture_labels.squeeze()
        if culture_labels.dim() == 0:
            culture_labels = culture_labels.unsqueeze(0)

        batch_size, num_experts = expert_weights.shape

        # ✅ 验证 batch size 一致性
        if culture_labels.size(0) != batch_size:
            raise ValueError(f"culture_labels size ({culture_labels.size(0)}) does not match batch size ({batch_size})")

        # ✅ 获取 batch 中的所有文化
        unique_cultures = torch.unique(culture_labels)
        num_cultures = len(unique_cultures)

        if num_cultures == 1:
            # 如果 batch 中只有一个文化，返回 0
            return torch.tensor(0.0, device=device, dtype=dtype)

        # ============================================================
        # 步骤 1: 构建每文化的专家分配向量 p_k
        # ============================================================
        # 初始化累加矩阵 S [C, E] 和计数向量 nu [C]
        max_culture_id = culture_labels.max().item() + 1
        S = torch.zeros(max_culture_id, num_experts, device=device, dtype=dtype)
        nu = torch.zeros(max_culture_id, device=device, dtype=dtype)

        # 对每个样本累加
        for b in range(batch_size):
            k = culture_labels[b].item()
            S[k, :] += expert_weights[b, :]
            nu[k] += 1

        # 计算每文化的平均权重 p_k = S_k / nu_k
        # 只对有样本的文化计算（nu_k > 0）
        p_k = torch.zeros_like(S)
        valid_cultures = []

        for k in unique_cultures:
            k_idx = k.item()
            if nu[k_idx] >= min_samples:  # ✅ 只对样本数 >= min_samples 的文化计算
                p_k[k_idx, :] = S[k_idx, :] / (nu[k_idx] + eps)
                # 归一化：先加 eps 再归一化
                p_k[k_idx, :] = (p_k[k_idx, :] + eps) / (p_k[k_idx, :].sum() + num_experts * eps)
                valid_cultures.append(k_idx)

        if len(valid_cultures) == 0:
            # 没有有效的文化（样本数都太少）
            return torch.tensor(0.0, device=device, dtype=dtype)

        # ============================================================
        # 步骤 2: 文化内稀疏化 - Specialization Loss（熵最小化）
        # ============================================================
        # 对每个文化 k，计算熵 H(p_k) = -sum(p_k[i] * log(p_k[i]))
        specialization_loss = 0.0

        for k_idx in valid_cultures:
            # 计算熵
            H_k = -torch.sum(p_k[k_idx, :] * torch.log(p_k[k_idx, :] + eps))
            specialization_loss += H_k

        # 平均
        specialization_loss = specialization_loss / len(valid_cultures)

        # ============================================================
        # 步骤 3: 文化间互异化 - Diversity Loss（专家向量相似度最小化）
        # ============================================================
        # 构建专家向量 v_i [num_experts, num_valid_cultures]
        # v_i[j] = p_{valid_cultures[j]}[i]
        v = torch.zeros(num_experts, len(valid_cultures), device=device, dtype=dtype)
        for i, k_idx in enumerate(valid_cultures):
            v[:, i] = p_k[k_idx, :]

        # L2 归一化
        v_norm = torch.nn.functional.normalize(v, p=2, dim=1)  # [num_experts, num_valid_cultures]

        # 计算余弦相似度矩阵 sim(i, j) = v_i^T * v_j
        sim_matrix = torch.mm(v_norm, v_norm.t())  # [num_experts, num_experts]

        # Diversity Loss: 最小化非对角元素的平方相似度
        # L_div = (2 / (n * (n-1))) * sum_{i<j} sim(i, j)^2
        diversity_loss = 0.0
        count = 0

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                diversity_loss += sim_matrix[i, j] ** 2
                count += 1

        if count > 0:
            diversity_loss = diversity_loss / count

        # ============================================================
        # 步骤 4: 合并为文化专注性损失
        # ============================================================
        culture_loss = alpha * specialization_loss + beta * diversity_loss

        return culture_loss

    def compute_culture_loss_components(self, expert_weights, culture_labels, eps=1e-8, min_samples=2):
        """
        计算 culture loss 的各个分量（用于监控）

        Returns:
            specialization_loss, diversity_loss
        """
        device = expert_weights.device
        dtype = expert_weights.dtype

        # 转换 culture_labels
        if not isinstance(culture_labels, torch.Tensor):
            culture_labels = torch.tensor(culture_labels, device=device, dtype=torch.long)
        else:
            culture_labels = culture_labels.to(device=device, dtype=torch.long)

        if culture_labels.dim() > 1:
            culture_labels = culture_labels.squeeze()
        if culture_labels.dim() == 0:
            culture_labels = culture_labels.unsqueeze(0)

        batch_size, num_experts = expert_weights.shape

        unique_cultures = torch.unique(culture_labels)
        if len(unique_cultures) == 1:
            return torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)

        # 构建 p_k
        max_culture_id = culture_labels.max().item() + 1
        S = torch.zeros(max_culture_id, num_experts, device=device, dtype=dtype)
        nu = torch.zeros(max_culture_id, device=device, dtype=dtype)

        for b in range(batch_size):
            k = culture_labels[b].item()
            S[k, :] += expert_weights[b, :]
            nu[k] += 1

        p_k = torch.zeros_like(S)
        valid_cultures = []

        for k in unique_cultures:
            k_idx = k.item()
            if nu[k_idx] >= min_samples:
                p_k[k_idx, :] = S[k_idx, :] / (nu[k_idx] + eps)
                p_k[k_idx, :] = (p_k[k_idx, :] + eps) / (p_k[k_idx, :].sum() + num_experts * eps)
                valid_cultures.append(k_idx)

        if len(valid_cultures) == 0:
            return torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)

        # Specialization Loss
        specialization_loss = 0.0
        for k_idx in valid_cultures:
            H_k = -torch.sum(p_k[k_idx, :] * torch.log(p_k[k_idx, :] + eps))
            specialization_loss += H_k
        specialization_loss = specialization_loss / len(valid_cultures)

        # Diversity Loss
        v = torch.zeros(num_experts, len(valid_cultures), device=device, dtype=dtype)
        for i, k_idx in enumerate(valid_cultures):
            v[:, i] = p_k[k_idx, :]

        v_norm = torch.nn.functional.normalize(v, p=2, dim=1)
        sim_matrix = torch.mm(v_norm, v_norm.t())

        diversity_loss = 0.0
        count = 0
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                diversity_loss += sim_matrix[i, j] ** 2
                count += 1

        if count > 0:
            diversity_loss = diversity_loss / count

        return specialization_loss, diversity_loss

    def get_expert_weights(self):
        """
        获取最近一次前向传播的专家权重

        Returns:
            expert_weights: [B, E]
        """
        return self._last_expert_weights
