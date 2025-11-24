# src/llamafactory/model/lora_enhanced_culturemoe.py
"""
LoRA增强的CultureMoE模型 - FFN集成版本
在Attention和MoE专家层中添加LoRA适配器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

from transformers import LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm
)

from .experts import LoRA
from .moe_args import ModelArgs


@dataclass
class LoRACultureMoEConfig:
    """LoRA增强CultureMoE配置"""
    # MoE配置
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25

    # 文化配置
    num_cultures: int = 6
    culture_dim: int = 256

    # LoRA配置
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1

    # Attention LoRA目标
    attention_lora_targets: List[str] = None

    # Expert LoRA目标
    expert_lora_targets: List[str] = None

    # 损失权重
    load_balance_weight: float = 0.01
    entropy_weight: float = 0.1
    culture_loss_weight: float = 0.05

    # 训练配置
    moe_fusion_alpha_init: float = 0.3
    noise_epsilon: float = 1e-2
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 1000

    def __post_init__(self):
        if self.attention_lora_targets is None:
            self.attention_lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if self.expert_lora_targets is None:
            self.expert_lora_targets = ["gate_proj", "up_proj", "down_proj"]


class LoRALinear(nn.Module):
    """
    LoRA适配器线性层
    可以包装任何现有的线性层
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 冻结原始权重
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # LoRA参数
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        base_output = self.base_layer(x)

        # LoRA输出
        lora_output = self.lora_B(self.dropout(self.lora_A(x)))

        return base_output + lora_output * self.scaling


class LoRAEnhancedAttention(nn.Module):
    """
    LoRA增强的Attention模块
    在Q、K、V、O投影层添加LoRA适配器
    """
    def __init__(self, original_attention: LlamaAttention, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.config = original_attention.config
        self.layer_idx = original_attention.layer_idx
        self.attention_dropout = original_attention.attention_dropout
        self.hidden_size = original_attention.hidden_size
        self.num_heads = original_attention.num_heads
        self.head_dim = original_attention.head_dim
        self.num_key_value_heads = original_attention.num_key_value_heads
        self.num_key_value_groups = original_attention.num_key_value_groups
        self.max_position_embeddings = original_attention.max_position_embeddings
        self.rope_theta = original_attention.rope_theta
        self.is_causal = original_attention.is_causal

        # 复制rotary embedding
        self.rotary_emb = original_attention.rotary_emb

        # 用LoRA包装投影层
        if "q_proj" in lora_config.attention_lora_targets:
            self.q_proj = LoRALinear(
                original_attention.q_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
        else:
            self.q_proj = original_attention.q_proj

        if "k_proj" in lora_config.attention_lora_targets:
            self.k_proj = LoRALinear(
                original_attention.k_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
        else:
            self.k_proj = original_attention.k_proj

        if "v_proj" in lora_config.attention_lora_targets:
            self.v_proj = LoRALinear(
                original_attention.v_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
        else:
            self.v_proj = original_attention.v_proj

        if "o_proj" in lora_config.attention_lora_targets:
            self.o_proj = LoRALinear(
                original_attention.o_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
        else:
            self.o_proj = original_attention.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播 - 使用与原始LlamaAttention相同的逻辑
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim**0.5)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CulturalInjectorWithLoRA(nn.Module):
    """
    带LoRA的文化信息注入器
    """
    def __init__(self, hidden_dim: int, culture_dim: int, layer_idx: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.culture_dim = culture_dim
        self.layer_idx = layer_idx

        # 层特定的文化嵌入
        self.layer_culture_embedding = nn.Embedding(6, culture_dim)

        # FiLM调制网络（低秩分解）
        self.film_rank = min(culture_dim, hidden_dim // 4)

        # Gamma网络（缩放因子）- 使用LoRA
        gamma_down_base = nn.Linear(culture_dim, self.film_rank, bias=False)
        gamma_up_base = nn.Linear(self.film_rank, hidden_dim, bias=False)

        self.gamma_down = LoRALinear(
            gamma_down_base,
            rank=lora_config.lora_rank // 2,  # 使用较小的rank
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )
        self.gamma_up = LoRALinear(
            gamma_up_base,
            rank=lora_config.lora_rank // 2,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

        # Beta网络（偏移因子）- 使用LoRA
        beta_down_base = nn.Linear(culture_dim, self.film_rank, bias=False)
        beta_up_base = nn.Linear(self.film_rank, hidden_dim, bias=True)

        self.beta_down = LoRALinear(
            beta_down_base,
            rank=lora_config.lora_rank // 2,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )
        self.beta_up = LoRALinear(
            beta_up_base,
            rank=lora_config.lora_rank // 2,
            alpha=lora_config.lora_alpha,
            dropout=lora_config.lora_dropout
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 门控机制
        self.culture_gate = nn.Sequential(
            nn.Linear(culture_dim, culture_dim // 2),
            nn.ReLU(),
            nn.Linear(culture_dim // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.layer_culture_embedding.weight, mean=0, std=0.02)

        # 门控网络初始化
        for module in self.culture_gate:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor) -> torch.Tensor:
        """FiLM调制前向传播"""
        batch_size, seq_len = hidden_states.shape[:2]

        # 获取层特定的文化嵌入
        culture_emb = self.layer_culture_embedding(culture_ids)

        # 计算FiLM参数（通过LoRA增强）
        gamma = self.gamma_up(torch.relu(self.gamma_down(culture_emb)))
        gamma = 1.0 + gamma  # 确保初始时接近恒等变换

        beta = self.beta_up(torch.relu(self.beta_down(culture_emb)))

        # 计算文化影响强度
        culture_strength = self.culture_gate(culture_emb)

        # 扩展到序列维度
        gamma = gamma.unsqueeze(1).expand(-1, seq_len, -1)
        beta = beta.unsqueeze(1).expand(-1, seq_len, -1)
        culture_strength = culture_strength.unsqueeze(1).expand(-1, seq_len, 1)

        # FiLM调制
        modulated = gamma * hidden_states + beta

        # 应用文化强度门控和层归一化
        output = self.layer_norm(
            hidden_states + culture_strength * (modulated - hidden_states)
        )

        return output


class LoRAExpertFFN(nn.Module):
    """
    带LoRA的专家FFN层
    在gate_proj, up_proj, down_proj上添加LoRA适配器
    """
    def __init__(self, expert_id: int, primary_culture_ids: List[int],
                 hidden_size: int, intermediate_size: int, activation: str,
                 culture_dim: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.expert_id = expert_id
        self.primary_culture_ids = primary_culture_ids
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # 创建基础FFN层
        base_gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        base_up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        base_down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 用LoRA包装FFN层
        if "gate_proj" in lora_config.expert_lora_targets:
            self.gate_proj = LoRALinear(
                base_gate_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
        else:
            self.gate_proj = base_gate_proj

        if "up_proj" in lora_config.expert_lora_targets:
            self.up_proj = LoRALinear(
                base_up_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
        else:
            self.up_proj = base_up_proj

        if "down_proj" in lora_config.expert_lora_targets:
            self.down_proj = LoRALinear(
                base_down_proj,
                rank=lora_config.lora_rank,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
        else:
            self.down_proj = base_down_proj

        # 激活函数
        if activation == "silu":
            self.act_fn = F.silu
        elif activation == "relu":
            self.act_fn = F.relu
        elif activation == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 文化条件向量
        if len(primary_culture_ids) > 0:
            self.culture_prompt = nn.Parameter(
                torch.randn(len(primary_culture_ids), culture_dim) * 0.01
            )
        else:
            self.culture_prompt = nn.Parameter(
                torch.randn(1, culture_dim) * 0.01
            )

        # 文化条件层
        self.culture_condition = nn.Linear(culture_dim, hidden_size)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.culture_condition.weight, mean=0, std=0.001)
        if self.culture_condition.bias is not None:
            nn.init.zeros_(self.culture_condition.bias)

    def forward(self, hidden_states: torch.Tensor, culture_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, L, H]
            culture_ids: [B]

        Returns:
            expert_output: [B, L, H]
            culture_relevance: [B]
        """
        # 获取文化提示
        culture_prompt = self._get_culture_prompt(culture_ids)
        culture_condition = self.culture_condition(culture_prompt).unsqueeze(1)

        # 文化条件的输入
        conditioned_input = hidden_states + culture_condition

        # 标准SwiGLU FFN前向传播（通过LoRA增强）
        gate_output = self.act_fn(self.gate_proj(conditioned_input))
        up_output = self.up_proj(conditioned_input)
        ffn_output = self.down_proj(gate_output * up_output)

        # 计算文化相关性
        culture_relevance = self._compute_culture_relevance(culture_ids)

        return ffn_output, culture_relevance

    def _get_culture_prompt(self, culture_ids: torch.Tensor) -> torch.Tensor:
        """获取文化提示向量"""
        prompts = []
        for culture_id in culture_ids:
            if len(self.primary_culture_ids) == 0:
                prompt = self.culture_prompt[0]
            elif culture_id.item() in self.primary_culture_ids:
                idx = self.primary_culture_ids.index(culture_id.item())
                prompt = self.culture_prompt[idx]
            else:
                prompt = self.culture_prompt.mean(dim=0)
            prompts.append(prompt)

        return torch.stack(prompts)

    def _compute_culture_relevance(self, culture_ids: torch.Tensor) -> torch.Tensor:
        """计算文化相关性"""
        relevance_scores = []
        for culture_id in culture_ids:
            if len(self.primary_culture_ids) == 0:
                relevance_scores.append(0.5)
            elif culture_id.item() in self.primary_culture_ids:
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.1)

        return torch.tensor(relevance_scores, device=culture_ids.device, dtype=torch.float32)


class VectorizedLoRAExpertLayer(nn.Module):
    """
    向量化的LoRA专家层
    """
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int,
                 activation: str, culture_assignments: List[List[int]],
                 culture_dim: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.culture_assignments = culture_assignments

        # 创建LoRA增强的专家
        self.experts = nn.ModuleList([
            LoRAExpertFFN(
                expert_id=i,
                primary_culture_ids=culture_assignments[i],
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                culture_dim=culture_dim,
                lora_config=lora_config
            ) for i in range(num_experts)
        ])

    def forward_with_dispatch(self, hidden_states: torch.Tensor, expert_weights: torch.Tensor,
                            culture_ids: torch.Tensor, top_k: int, capacity_factor: float) -> Tuple[torch.Tensor, Dict]:
        """
        向量化的专家调度和处理
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Top-K专家选择
        top_k_weights, top_k_indices = torch.topk(expert_weights, k=top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # 简化的专家处理（可以后续优化为完全向量化）
        expert_outputs = []
        culture_relevances = []

        for i in range(top_k):
            batch_expert_output = torch.zeros_like(hidden_states)
            batch_relevance = torch.zeros(batch_size, device=hidden_states.device)

            for b in range(batch_size):
                expert_idx = top_k_indices[b, i].item()
                weight = top_k_weights[b, i]

                expert = self.experts[expert_idx]
                expert_out, relevance = expert(hidden_states[b:b+1], culture_ids[b:b+1])

                batch_expert_output[b] = expert_out[0] * weight
                batch_relevance[b] = relevance[0]

            expert_outputs.append(batch_expert_output)
            culture_relevances.append(batch_relevance)

        # 专家输出融合
        final_output = sum(expert_outputs)

        dispatch_info = {
            'top_k_indices': top_k_indices,
            'top_k_weights': top_k_weights,
            'culture_relevances': torch.stack(culture_relevances, dim=1)
        }

        return final_output, dispatch_info


# 导入必要的辅助函数（从transformers复制）
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary positional embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)