# src/llamafactory/model/lora_culturemoe_model.py
"""
完整的LoRA增强CultureMoE模型
集成Attention LoRA和MoE Expert LoRA
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging
from copy import deepcopy

from transformers import LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    BaseModelOutputWithPast
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from .lora_enhanced_culturemoe import (
    LoRACultureMoEConfig,
    LoRAEnhancedAttention
)
from .lora_vectorized_culturemoe_ffn import VectorizedCultureMoE_FFN_WithLoRA


class LoRACultureMoELlamaDecoderLayer(nn.Module):
    """
    LoRA增强的CultureMoE Decoder层
    在Attention和FFN层都添加LoRA适配器
    """
    def __init__(self, config: LlamaConfig, layer_idx: int, lora_config: LoRACultureMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # 创建原始attention（用于获取权重）
        original_attention = LlamaAttention(config=config, layer_idx=layer_idx)

        # LoRA增强的Attention
        self.self_attn = LoRAEnhancedAttention(original_attention, lora_config)

        # LoRA增强的MoE FFN
        self.mlp = VectorizedCultureMoE_FFN_WithLoRA(
            config=config,
            layer_idx=layer_idx,
            lora_config=lora_config
        )

        # Layer Normalization（保持原样）
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 冻结LayerNorm参数
        for param in self.input_layernorm.parameters():
            param.requires_grad = False
        for param in self.post_attention_layernorm.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        culture_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # LoRA增强的Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # LoRA增强的MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, aux_info = self.mlp(hidden_states, culture_ids)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # 添加MoE辅助信息
        if aux_info is not None:
            outputs += (aux_info,)

        return outputs


class LoRACultureMoELlamaModel(LlamaModel):
    """
    LoRA增强的CultureMoE LLaMA模型
    """
    def __init__(self, config: LlamaConfig, lora_config: LoRACultureMoEConfig):
        # 不调用父类__init__，手动初始化
        nn.Module.__init__(self)
        self.config = config
        self.lora_config = lora_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding层（冻结）
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        # LoRA增强的Decoder层
        self.layers = nn.ModuleList([
            LoRACultureMoELlamaDecoderLayer(config, layer_idx, lora_config)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # 最终LayerNorm（冻结）
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for param in self.norm.parameters():
            param.requires_grad = False

        # 初始化权重
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        culture_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # 如果没有提供culture_ids，使用默认值
        if culture_ids is None:
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            culture_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # 收集所有MoE辅助信息
        all_aux_info = []

        # 使用父类的大部分逻辑，但传递culture_ids
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Llama. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        # 处理attention mask
        if getattr(self, '_attn_implementation', None) == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif getattr(self, '_attn_implementation', None) == "sdpa" and not output_attentions:
            # 简化版本，实际需要导入相关函数
            attention_mask = attention_mask
        else:
            # 简化版本，实际需要导入相关函数
            attention_mask = attention_mask

        # 主要前向传播
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    culture_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    culture_ids=culture_ids,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # 收集MoE辅助信息
            if len(layer_outputs) > 3:
                aux_info = layer_outputs[3]
                all_aux_info.append(aux_info)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_aux_info] if v is not None)

        # 创建返回结果
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        # 将MoE辅助信息添加到结果中
        if all_aux_info:
            result.moe_aux_info = all_aux_info

        return result

    @property
    def device(self) -> torch.device:
        """获取模型设备"""
        return next(self.parameters()).device

    def get_lora_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """获取所有LoRA参数，按类型分组"""
        attention_lora_params = []
        expert_lora_params = []
        cultural_lora_params = []

        for layer in self.layers:
            # Attention LoRA参数
            for name, module in layer.self_attn.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    attention_lora_params.extend([module.lora_A.weight, module.lora_B.weight])

            # Expert LoRA参数
            for name, module in layer.mlp.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    if 'expert' in name:
                        expert_lora_params.extend([module.lora_A.weight, module.lora_B.weight])
                    else:
                        cultural_lora_params.extend([module.lora_A.weight, module.lora_B.weight])

        return {
            'attention_lora': attention_lora_params,
            'expert_lora': expert_lora_params,
            'cultural_lora': cultural_lora_params
        }

    def freeze_base_parameters(self):
        """冻结所有基础参数，只训练LoRA"""
        # 冻结embedding和norm
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False

        # 冻结每层的基础参数
        for layer in self.layers:
            layer.mlp.freeze_base_parameters()

            # 冻结attention的基础参数
            for name, module in layer.self_attn.named_modules():
                if hasattr(module, 'base_layer'):
                    for param in module.base_layer.parameters():
                        param.requires_grad = False

            # 冻结LayerNorm
            for param in layer.input_layernorm.parameters():
                param.requires_grad = False
            for param in layer.post_attention_layernorm.parameters():
                param.requires_grad = False

    def unfreeze_all_parameters(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

    def print_parameter_stats(self):
        """打印参数统计信息"""
        lora_params_dict = self.get_lora_parameters()
        total_params = sum(p.numel() for p in self.parameters())

        attention_lora_count = sum(p.numel() for p in lora_params_dict['attention_lora'])
        expert_lora_count = sum(p.numel() for p in lora_params_dict['expert_lora'])
        cultural_lora_count = sum(p.numel() for p in lora_params_dict['cultural_lora'])
        total_lora_count = attention_lora_count + expert_lora_count + cultural_lora_count

        print("=" * 60)
        print("📊 LoRA Enhanced CultureMoE Parameter Statistics")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"LoRA parameters: {total_lora_count:,}")
        print(f"LoRA percentage: {100 * total_lora_count / total_params:.4f}%")
        print("")
        print("LoRA breakdown:")
        print(f"  Attention LoRA: {attention_lora_count:,}")
        print(f"  Expert LoRA: {expert_lora_count:,}")
        print(f"  Cultural LoRA: {cultural_lora_count:,}")
        print("")
        print(f"Model layers: {len(self.layers)}")
        print(f"Experts per layer: {self.lora_config.num_experts}")
        print(f"LoRA rank: {self.lora_config.lora_rank}")
        print(f"LoRA alpha: {self.lora_config.lora_alpha}")
        print("=" * 60)


def create_lora_culturemoe_model(
    base_model_path: str,
    lora_config: LoRACultureMoEConfig
) -> LoRACultureMoELlamaModel:
    """
    创建LoRA增强的CultureMoE模型

    Args:
        base_model_path: 基础LLaMA模型路径
        lora_config: LoRA配置

    Returns:
        LoRA增强的CultureMoE模型
    """
    # 加载基础模型配置
    base_config = LlamaConfig.from_pretrained(base_model_path)

    # 创建LoRA增强模型
    model = LoRACultureMoELlamaModel(base_config, lora_config)

    # 加载基础模型权重到非LoRA部分
    try:
        base_model = LlamaModel.from_pretrained(base_model_path)
        _load_base_weights_to_lora_model(model, base_model)
        del base_model  # 释放内存
    except Exception as e:
        logging.warning(f"Failed to load base model weights: {e}")
        logging.warning("Model will be initialized with random weights")

    return model


def _load_base_weights_to_lora_model(lora_model: LoRACultureMoELlamaModel, base_model: LlamaModel):
    """将基础模型权重加载到LoRA模型的非LoRA部分"""
    with torch.no_grad():
        # 复制embedding权重
        lora_model.embed_tokens.weight.copy_(base_model.embed_tokens.weight)

        # 复制norm权重
        lora_model.norm.weight.copy_(base_model.norm.weight)

        # 复制每层的权重
        for lora_layer, base_layer in zip(lora_model.layers, base_model.layers):
            # 复制LayerNorm权重
            lora_layer.input_layernorm.weight.copy_(base_layer.input_layernorm.weight)
            lora_layer.post_attention_layernorm.weight.copy_(base_layer.post_attention_layernorm.weight)

            # 复制attention权重到LoRA的base_layer
            _copy_attention_weights(lora_layer.self_attn, base_layer.self_attn)

            # 复制FFN权重到共享专家和专家的base_layer
            _copy_ffn_weights_to_experts(lora_layer.mlp, base_layer.mlp)


def _copy_attention_weights(lora_attn: LoRAEnhancedAttention, base_attn: LlamaAttention):
    """复制attention权重"""
    # 复制rotary embedding
    if hasattr(base_attn, 'rotary_emb'):
        lora_attn.rotary_emb = base_attn.rotary_emb

    # 复制投影层权重到base_layer
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        lora_proj = getattr(lora_attn, proj_name)
        base_proj = getattr(base_attn, proj_name)

        if hasattr(lora_proj, 'base_layer'):
            lora_proj.base_layer.weight.copy_(base_proj.weight)
            if base_proj.bias is not None:
                lora_proj.base_layer.bias.copy_(base_proj.bias)
        else:
            lora_proj.weight.copy_(base_proj.weight)
            if base_proj.bias is not None:
                lora_proj.bias.copy_(base_proj.bias)


def _copy_ffn_weights_to_experts(lora_mlp, base_mlp: LlamaMLP):
    """复制FFN权重到专家层"""
    # 复制到共享专家
    shared_expert = lora_mlp.shared_expert
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        shared_proj = getattr(shared_expert, proj_name)
        base_proj = getattr(base_mlp, proj_name)

        if hasattr(shared_proj, 'base_layer'):
            shared_proj.base_layer.weight.copy_(base_proj.weight)
        else:
            shared_proj.weight.copy_(base_proj.weight)

    # 复制到所有专家（初始化为相同权重）
    for expert in lora_mlp.expert_layer.experts:
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            expert_proj = getattr(expert, proj_name)
            base_proj = getattr(base_mlp, proj_name)

            if hasattr(expert_proj, 'base_layer'):
                expert_proj.base_layer.weight.copy_(base_proj.weight)
            else:
                expert_proj.weight.copy_(base_proj.weight)