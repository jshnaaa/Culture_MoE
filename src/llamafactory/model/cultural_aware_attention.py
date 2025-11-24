# src/llamafactory/model/cultural_aware_attention.py
"""
文化感知注意力机制

实现多层次的文化感知注意力，包括：
1. 文化条件QKV生成
2. 文化偏置注意力权重
3. 文化上下文融合
4. 自适应文化权重调制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from .lora_enhanced_culturemoe import LoRALinear, LoRACultureMoEConfig


class CulturallyAwareAttention(nn.Module):
    """文化感知注意力机制"""

    def __init__(self, config, layer_idx: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_cultures = lora_config.num_cultures
        self.culture_dim = lora_config.culture_dim

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim ** -0.5

        # 1. 文化嵌入层
        self.culture_embeddings = nn.Embedding(self.num_cultures, self.culture_dim)

        # 2. 文化感知QKV生成器
        self.cultural_qkv_generator = CulturalQKVGenerator(
            self.hidden_size, self.culture_dim, self.num_heads, lora_config
        )

        # 3. 文化偏置生成器
        self.cultural_bias_generator = CulturalBiasGenerator(
            self.culture_dim, self.num_heads, self.head_dim, lora_config
        )

        # 4. 文化上下文生成器
        self.cultural_context_generator = CulturalContextGenerator(
            self.culture_dim, self.hidden_size, lora_config
        )

        # 5. 自适应融合网络
        self.adaptive_fusion = AdaptiveCulturalFusion(
            self.hidden_size, self.culture_dim, lora_config
        )

        # 6. 输出投影（LoRA增强）
        base_o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = LoRALinear(
            base_o_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

        # 7. Rotary Position Embedding（如果需要）
        if hasattr(config, 'rope_theta'):
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        culture_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, seq_len, _ = hidden_states.size()

        # 如果没有提供culture_ids，使用默认值
        if culture_ids is None:
            culture_ids = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)

        # 1. 获取文化嵌入
        culture_emb = self.culture_embeddings(culture_ids)  # [B, culture_dim]

        # 2. 生成文化上下文
        cultural_context = self.cultural_context_generator(culture_emb, hidden_states)  # [B, H]

        # 3. 文化感知QKV生成
        query_states, key_states, value_states = self.cultural_qkv_generator(
            hidden_states, culture_emb
        )  # [B, num_heads, L, head_dim]

        # 4. 应用Rotary Position Embedding（如果有）
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 5. 处理past_key_value（KV缓存）
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 6. 计算基础注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # 7. 生成并应用文化偏置
        cultural_bias = self.cultural_bias_generator(
            cultural_context, seq_len, batch_size
        )  # [B, num_heads, L, L]

        # 应用文化偏置
        attn_weights = attn_weights + cultural_bias

        # 8. 应用注意力掩码
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # 9. 计算注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 10. 应用注意力到值
        attn_output = torch.matmul(attn_weights, value_states)  # [B, num_heads, L, head_dim]

        # 11. 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # 12. 自适应文化融合
        attn_output = self.adaptive_fusion(attn_output, cultural_context, culture_ids)

        # 13. 输出投影
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CulturalQKVGenerator(nn.Module):
    """文化感知的Query、Key、Value生成器"""

    def __init__(self, hidden_size: int, culture_dim: int, num_heads: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.culture_dim = culture_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 文化条件的QKV投影（LoRA增强）
        # Q投影：结合原始hidden_states和文化信息
        base_q_proj = nn.Linear(hidden_size + culture_dim, hidden_size, bias=False)
        self.q_proj = LoRALinear(
            base_q_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

        # K投影：主要基于原始hidden_states，轻微文化调制
        base_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = LoRALinear(
            base_k_proj,
            rank=lora_config.lora_rank // 2,  # 较小的rank
            alpha=lora_config.lora_alpha * 0.5,
            dropout=lora_config.lora_dropout
        )

        # V投影：文化感知的值生成
        base_v_proj = nn.Linear(hidden_size + culture_dim // 2, hidden_size, bias=False)
        self.v_proj = LoRALinear(
            base_v_proj,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

        # 文化特征投影
        self.culture_projector = nn.Sequential(
            nn.Linear(culture_dim, culture_dim // 2),
            nn.ReLU(),
            nn.Linear(culture_dim // 2, culture_dim // 2)
        )

    def forward(self, hidden_states: torch.Tensor, culture_emb: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 1. 扩展文化嵌入到序列维度
        culture_emb_expanded = culture_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, culture_dim]

        # 2. 生成Query（文化条件）
        q_input = torch.cat([hidden_states, culture_emb_expanded], dim=-1)  # [B, L, H + culture_dim]
        query_states = self.q_proj(q_input)

        # 3. 生成Key（轻微文化调制）
        key_states = self.k_proj(hidden_states)

        # 4. 生成Value（文化感知）
        projected_culture = self.culture_projector(culture_emb_expanded)  # [B, L, culture_dim//2]
        v_input = torch.cat([hidden_states, projected_culture], dim=-1)  # [B, L, H + culture_dim//2]
        value_states = self.v_proj(v_input)

        # 5. 重塑为多头格式
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        return query_states, key_states, value_states


class CulturalBiasGenerator(nn.Module):
    """文化偏置生成器 - 在注意力权重中注入文化偏置"""

    def __init__(self, culture_dim: int, num_heads: int, head_dim: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.culture_dim = culture_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 文化偏置网络
        self.bias_network = nn.Sequential(
            nn.Linear(culture_dim, culture_dim // 2),
            nn.ReLU(),
            nn.Linear(culture_dim // 2, num_heads),
            nn.Tanh()  # 限制偏置范围
        )

        # 位置感知的文化偏置
        self.position_bias_network = nn.Sequential(
            nn.Linear(culture_dim, num_heads * 4),
            nn.ReLU(),
            nn.Linear(num_heads * 4, num_heads)
        )

        # 可学习的文化注意力模式
        self.cultural_patterns = nn.Parameter(
            torch.randn(6, num_heads, 8, 8) * 0.02  # 6种文化，每种有特定的注意力模式
        )

    def forward(self, cultural_context: torch.Tensor, seq_len: int, batch_size: int):
        # cultural_context: [B, H]

        # 1. 生成全局文化偏置
        global_bias = self.bias_network(cultural_context)  # [B, num_heads]
        global_bias = global_bias.unsqueeze(-1).unsqueeze(-1)  # [B, num_heads, 1, 1]

        # 2. 生成位置感知偏置
        position_bias = self.position_bias_network(cultural_context)  # [B, num_heads]

        # 3. 创建位置偏置矩阵
        # 简化版本：对角线和近邻位置有不同的偏置
        device = cultural_context.device
        pos_bias_matrix = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=device)

        # 对角线偏置（自注意力）
        diag_indices = torch.arange(seq_len, device=device)
        pos_bias_matrix[:, :, diag_indices, diag_indices] = position_bias.unsqueeze(-1) * 0.1

        # 近邻偏置
        for i in range(1, min(4, seq_len)):
            if seq_len > i:
                upper_diag = torch.arange(seq_len - i, device=device)
                lower_diag = torch.arange(seq_len - i, device=device)
                pos_bias_matrix[:, :, upper_diag, upper_diag + i] = position_bias.unsqueeze(-1) * (0.05 / i)
                pos_bias_matrix[:, :, lower_diag + i, lower_diag] = position_bias.unsqueeze(-1) * (0.05 / i)

        # 4. 组合偏置
        total_bias = global_bias + pos_bias_matrix

        return total_bias


class CulturalContextGenerator(nn.Module):
    """文化上下文生成器 - 生成丰富的文化上下文表示"""

    def __init__(self, culture_dim: int, hidden_size: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.culture_dim = culture_dim
        self.hidden_size = hidden_size

        # 文化知识库（可学习的文化原型）
        self.cultural_prototypes = nn.Parameter(
            torch.randn(6, culture_dim, hidden_size // 4) * 0.02
        )

        # 动态文化权重生成
        self.cultural_weight_generator = nn.Sequential(
            nn.Linear(hidden_size, culture_dim),
            nn.ReLU(),
            nn.Linear(culture_dim, 6),  # 6种文化的权重
            nn.Softmax(dim=-1)
        )

        # 文化上下文融合网络（LoRA增强）
        base_context_fusion = nn.Sequential(
            nn.Linear(culture_dim + hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        # LoRA增强最后一层
        base_final = base_context_fusion[-1]
        lora_final = LoRALinear(
            base_final,
            rank=lora_config.lora_rank // 2,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

        self.context_fusion = nn.Sequential(
            base_context_fusion[0],
            base_context_fusion[1],
            lora_final
        )

    def forward(self, culture_emb: torch.Tensor, hidden_states: torch.Tensor):
        batch_size = culture_emb.shape[0]

        # 1. 生成序列摘要用于动态权重计算
        sequence_summary = hidden_states.mean(dim=1)  # [B, H]

        # 2. 计算动态文化权重
        cultural_weights = self.cultural_weight_generator(sequence_summary)  # [B, 6]

        # 3. 加权组合文化原型
        # cultural_prototypes: [6, culture_dim, hidden_size//4]
        # cultural_weights: [B, 6]
        weighted_prototypes = torch.einsum('bc,ckd->bkd', cultural_weights, self.cultural_prototypes)  # [B, culture_dim, H//4]
        weighted_prototypes = weighted_prototypes.mean(dim=1)  # [B, H//4]

        # 4. 文化上下文融合
        context_input = torch.cat([culture_emb, weighted_prototypes], dim=-1)  # [B, culture_dim + H//4]
        cultural_context = self.context_fusion(context_input)  # [B, H]

        return cultural_context


class AdaptiveCulturalFusion(nn.Module):
    """自适应文化融合 - 动态调节文化影响的强度"""

    def __init__(self, hidden_size: int, culture_dim: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.culture_dim = culture_dim

        # 文化影响强度网络
        self.cultural_strength_network = nn.Sequential(
            nn.Linear(hidden_size + culture_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出0-1之间的强度
        )

        # 文化特征融合网络（LoRA增强）
        base_fusion = nn.Linear(hidden_size + culture_dim, hidden_size)
        self.cultural_fusion = LoRALinear(
            base_fusion,
            rank=lora_config.lora_rank,
            alpha=lora_config.lora_alpha * 0.5,  # 较小的alpha，避免过度影响
            dropout=lora_config.lora_dropout
        )

        # 残差门控
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, attn_output: torch.Tensor, cultural_context: torch.Tensor, culture_ids: torch.Tensor):
        batch_size, seq_len, hidden_size = attn_output.shape

        # 1. 扩展文化上下文到序列维度
        cultural_context_expanded = cultural_context.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, H]

        # 2. 计算文化影响强度
        strength_input = torch.cat([attn_output, cultural_context_expanded], dim=-1)  # [B, L, H + H]
        cultural_strength = self.cultural_strength_network(strength_input)  # [B, L, 1]

        # 3. 文化特征融合
        fusion_input = torch.cat([attn_output, cultural_context_expanded], dim=-1)  # [B, L, 2H]
        cultural_features = self.cultural_fusion(fusion_input)  # [B, L, H]

        # 4. 自适应融合
        # 原始注意力输出 + 文化强度 * 文化特征
        enhanced_output = attn_output + cultural_strength * cultural_features

        # 5. 残差门控
        gate_weights = self.residual_gate(enhanced_output)
        final_output = gate_weights * enhanced_output + (1 - gate_weights) * attn_output

        return final_output


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """应用旋转位置编码"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """旋转输入的一半维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)