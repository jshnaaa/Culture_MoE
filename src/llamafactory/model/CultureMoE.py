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

        # 1. 预训练 LLaMA 模型 (冻结参数)
        # self.llama_model = AutoModelForCausalLM.from_pretrained(
        #     args.llama_model_path, output_hidden_states=True
        # )
        # for param in self.llama_model.parameters():
        #     param.requires_grad = False

        self.llama_model = llama_model
        self.args = args

        self.config = llama_model.config  # ✅ 继承 Llama 的配置

        # ✅ generation_config 从 llama_model 继承，保持对象类型
        if hasattr(llama_model, 'generation_config'):
            self.generation_config = llama_model.generation_config
        else:
            from transformers import GenerationConfig
            self.generation_config = GenerationConfig()

        hidden_dim = self.config.hidden_size  # ✅ 用 LLaMA 的 hidden size

        # hidden_dim = self.llama_model.config.hidden_size
        # hidden_dim = 4096

        # 2. Shared 层
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, args.shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.shared_hidden_dim, hidden_dim)
        )

        # 3. Experts 层 (6 个 LoRA 专家，返回每个专家单独输出)
        # 3. Router
        self.router = ExpertRouter(
            router_input_dim=hidden_dim,
            num_experts=args.num_experts,
            router_hidden_dim=args.router_hidden_dim,
            dropout=args.dropout
        )

        self.experts_layer = ExpertLayer(
            experts_input_dim=hidden_dim,
            experts_hidden_dim=args.experts_hidden_dim,
            experts_output_dim=hidden_dim,
            num_experts=args.num_experts,   # 设为 6
            lora_rank=args.lora_rank,
            dropout=args.dropout,
            return_all=True                 # ⚠️ 需要在 ExpertLayer 内部加个 flag
        )

        # 4. Self-Attention block (统一融合 shared + experts)
        # self.cross_attn = CrossAttentionBlock(
        #     hidden_dim, num_heads=args.num_heads, dropout=args.dropout
        # )

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, args.classification_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.classification_hidden_dim, args.num_classes)
        )

    # 添加 generate 方法
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.llama_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Implement logic to prepare inputs for generation
        return self.llama_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 确保所有张量都在相同的设备上
        device = input_ids.device  # 获取输入张量的设备
        dtype = torch.float16

        # step 1：llama 正常 forward
        outputs = self.llama_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].to(device).to(dtype)  # Ensure hidden states are on the same device
        h_no = hidden  # 保存处理过的 hidden 状态
        h_all = hidden  # 保存所有的 hidden 状态

        # Step 2: Shared 层 (对 h_no 进行处理)
        shared_out = self.shared(h_no.to(dtype))  # [B, L, H]

        # Step 3: Router (基于 pooled shared 表示)
        pooled = shared_out.mean(dim=1)  # [B, H]
        expert_weights, _ = self.router(pooled)  # [B, E]

        # Step 4: Experts 层 (返回 list，每个是 [B, L, H])
        expert_outs = self.experts_layer(h_all.to(dtype))  # list of len(E)

        # Step 5: 权重缩放专家输出
        weighted_expert_outs = [
            expert_outs[i] * expert_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            for i in range(len(expert_outs))
        ]  # list of [B, L, H]

        # Step 6: 拼接 shared + 加权专家输出 → KV
        kv = torch.cat([shared_out] + weighted_expert_outs, dim=1)  # [B, (E+1)*L, H]

        # Step 8: 分类 (取最后 token)
        final_repr = kv
        logits = self.classifier(final_repr)

        # Step 9: 对 logits 进行平均，得到每个样本的平均 logits
        logits_avg = logits.mean(dim=1)  # [B, num_classes]，对所有位置的 logits 进行平均
        return logits_avg


        # # Step 10: 应用 softmax 得到每个类别的概率
        # probs = torch.softmax(logits_avg, dim=-1)  # [B, num_classes]
        # print("probs.shape: ", probs.shape)  # 打印preds的形状
        # print("probs: ", probs)  # 打印preds

        # # Step 11: 通过 argmax 获得最终的分类结果
        # preds = torch.argmax(probs, dim=-1)  # [B]
        # print("preds.shape", preds.shape)  # 打印preds的形状
        # print("preds: ", preds)  # 打印preds

        # # Step 12: 映射到 "yes", "neutral", "no"
        # output = [self.output_map[pred.item()] for pred in preds]
        # return output