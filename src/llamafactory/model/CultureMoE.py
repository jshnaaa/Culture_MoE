# src/llamafactory/model/CultureMoE.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .experts import ExpertLayer
from .router import ExpertRouter
from .moe_args import ModelArgs


class CrossAttentionBlock(nn.Module):
    """ Shared 层输出作为 Q，专家+shared 输出作为 KV """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value):
        # Q = shared_out, KV = shared + experts
        attn_out, _ = self.attn(query, key_value, key_value)
        x = self.norm1(query + attn_out)

        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        return out


class LlamaSharedRouterExpertsModel(nn.Module):
    def __init__(self, llama_model, config, args: ModelArgs):
        super().__init__()

        self.llama_model = llama_model
        self.args = args
        self.config = llama_model.config

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

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.llama_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.llama_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # ✅ 获取输入的设备和数据类型
        device = input_ids.device
        dtype = next(self.parameters()).dtype  # 使用模型参数的 dtype

        # Step 1: LLaMA forward
        outputs = self.llama_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # ✅ 确保 hidden states 在正确的设备上
        hidden = outputs.hidden_states[-1]
        if hidden.device != device:
            hidden = hidden.to(device)
        if hidden.dtype != dtype:
            hidden = hidden.to(dtype)

        h_no = hidden
        h_all = hidden

        # Step 2: Shared 层
        shared_out = self.shared(h_no)  # [B, L, H]

        # Step 3: Router
        pooled = shared_out.mean(dim=1)  # [B, H]
        expert_weights, _ = self.router(pooled)  # [B, E]

        # Step 4: Experts 层
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

        return logits_avg