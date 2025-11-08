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

        # ✅ generation_config 从 llama_model 继承
        if hasattr(llama_model, 'generation_config'):
            self.generation_config = llama_model.generation_config
        else:
            from transformers import GenerationConfig
            self.generation_config = GenerationConfig()

        hidden_dim = self.config.hidden_size

        # 2. Shared 层
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, args.shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.shared_hidden_dim, hidden_dim)
        )

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

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, args.classification_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.classification_hidden_dim, args.num_classes)
        )

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

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.llama_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.llama_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, input_ids_mask=None, attention_mask_mask=None,
                labels=None, culture_labels=None, use_culture_loss=False, culture_loss_lambda=0.5, **kwargs):
        # ✅ 获取输入的设备和数据类型
        device = input_ids.device
        # 使用 shared 层的第一个 Linear 的 dtype（MoE 层总是有参数的）
        dtype = self.shared[0].weight.dtype

        # ✅ Step 1: LLaMA forward for h_all (instruction + input)
        outputs_all = self.llama_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # ✅ 确保 hidden states 在正确的设备上
        hidden_all = outputs_all.hidden_states[-1]
        if hidden_all.device != device:
            hidden_all = hidden_all.to(device)
        if hidden_all.dtype != dtype:
            hidden_all = hidden_all.to(dtype)

        # ✅ 关键修复：立即克隆 h_all，避免后续重复使用
        h_all = hidden_all.clone()

        # ✅ Step 2: LLaMA forward for h_no (instruction_mask + input)
        # 如果提供了 mask 输入，使用它；否则使用相同的输入
        if input_ids_mask is not None:
            outputs_no = self.llama_model(
                input_ids_mask,
                attention_mask=attention_mask_mask,
                output_hidden_states=True
            )

            hidden_no = outputs_no.hidden_states[-1]
            if hidden_no.device != device:
                hidden_no = hidden_no.to(device)
            if hidden_no.dtype != dtype:
                hidden_no = hidden_no.to(dtype)

            h_no = hidden_no.clone()
        else:
            # ✅ 如果没有提供 mask 输入，再次克隆以创建独立副本
            h_no = h_all.clone()

        # Step 2: Shared 层
        # ✅ 确保 h_no 在正确的设备和数据类型上
        h_no = h_no.to(device=device, dtype=dtype)
        shared_out = self.shared(h_no)  # [B, L, H]

        # Step 3: Router
        pooled = shared_out.mean(dim=1)  # [B, H]
        expert_weights, router_logits = self.router(pooled)  # [B, E]

        # ✅ 保存专家权重（用于损失计算）
        self._last_expert_weights = expert_weights.detach()

        # Step 4: Experts 层
        # ✅ 确保 h_all 在正确的设备和数据类型上
        h_all = h_all.to(device=device, dtype=dtype)
        expert_outs = self.experts_layer(h_all)  # list of [B, L, H]

        # Step 5: 权重缩放专家输出
        weighted_expert_outs = [
            expert_outs[i] * expert_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            for i in range(len(expert_outs))
        ]

        # Step 6: 拼接 shared + 加权专家输出
        kv = torch.cat([shared_out] + weighted_expert_outs, dim=1)  # [B, (E+1)*L, H]

        # Step 7: 分类
        final_repr = kv
        logits = self.classifier(final_repr)

        # Step 8: 对 logits 进行平均
        logits_avg = logits.mean(dim=1)  # [B, num_classes]

        # ✅ Step 9: 计算损失（如果提供了 labels）
        outputs = {'logits': logits_avg}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits_avg, labels)
            outputs['classification_loss'] = classification_loss

            # 文化损失（如果启用）
            if use_culture_loss and culture_labels is not None:
                # 文化损失：鼓励相同文化的样本使用相似的专家
                culture_loss = self.compute_culture_loss(expert_weights, culture_labels)
                outputs['culture_loss'] = culture_loss

                # 总损失
                total_loss = classification_loss + culture_loss_lambda * culture_loss
            else:
                outputs['culture_loss'] = torch.tensor(0.0, device=device)
                total_loss = classification_loss

            outputs['loss'] = total_loss

        return outputs

    def compute_culture_loss(self, expert_weights, culture_labels):
        """
        计算文化损失：鼓励相同文化的样本使用相似的专家

        Args:
            expert_weights: [B, E] 专家权重
            culture_labels: [B] 文化标签（可以是 list 或 tensor）

        Returns:
            culture_loss: 标量
        """
        device = expert_weights.device
        dtype = expert_weights.dtype

        # ✅ 将 culture_labels 转换为 tensor（如果还不是）
        if not isinstance(culture_labels, torch.Tensor):
            culture_labels = torch.tensor(culture_labels, device=device, dtype=torch.long)
        else:
            # 确保在正确的设备上
            culture_labels = culture_labels.to(device=device, dtype=torch.long)

        # ✅ 确保 culture_labels 是 1D [B]
        if culture_labels.dim() > 1:
            culture_labels = culture_labels.squeeze()

        # ✅ 如果 culture_labels 是 0D（标量），转换为 1D
        if culture_labels.dim() == 0:
            culture_labels = culture_labels.unsqueeze(0)

        batch_size = expert_weights.size(0)

        # ✅ 验证 batch size 一致性
        if culture_labels.size(0) != batch_size:
            raise ValueError(f"culture_labels size ({culture_labels.size(0)}) does not match batch size ({batch_size})")

        # 计算样本对之间的文化相似度（相同文化为1，不同文化为0）
        # ✅ 使用 unsqueeze 和 transpose 而不是 .t()
        culture_labels_1 = culture_labels.unsqueeze(1)  # [B, 1]
        culture_labels_2 = culture_labels.unsqueeze(0)  # [1, B]
        culture_similarity = (culture_labels_1 == culture_labels_2).float()  # [B, B]

        # 计算专家权重之间的余弦相似度
        expert_weights_norm = torch.nn.functional.normalize(expert_weights, p=2, dim=1)
        expert_similarity = torch.mm(expert_weights_norm, expert_weights_norm.t())  # [B, B]

        # 文化损失：相同文化的样本应该有相似的专家权重
        # 使用 MSE 损失
        culture_loss = torch.nn.functional.mse_loss(
            expert_similarity * culture_similarity,
            culture_similarity
        )

        return culture_loss

    def get_expert_weights(self):
        """
        获取最近一次前向传播的专家权重

        Returns:
            expert_weights: [B, E]
        """
        return self._last_expert_weights
