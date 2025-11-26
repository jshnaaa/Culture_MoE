# src/llamafactory/model/simplified_culturemoe.py
"""
简化版CultureMoE配置
基于MixLoRA实现，只在最后2层使用MoE，添加文化损失
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SimplifiedCultureMoEConfig:
    """简化版CultureMoE配置"""

    # LoRA配置
    lora_rank: int = 16
    lora_alpha: int = 8
    lora_dropout: float = 0.1

    # MoE配置
    num_routing_experts: int = 2  # 路由专家数量
    top_k: int = 1  # Top-K路由，设为1减少计算
    aux_loss_coef: float = 0.001  # 辅助损失系数

    # 指定哪些层使用MoE (0-indexed)
    moe_layers: List[int] = None  # 例如 [26, 27] for Qwen, [30, 31] for LLaMA

    # FFN目标模块
    ffn_target_modules: List[str] = None

    # 文化损失配置
    use_culture_loss: bool = True
    culture_loss_weight: float = 0.01

    # 其他配置
    apply_to_attention: bool = False  # 暂时禁用注意力层LoRA

    def __post_init__(self):
        if self.moe_layers is None:
            # 默认为Qwen的最后2层
            self.moe_layers = [26, 27]

        if self.ffn_target_modules is None:
            self.ffn_target_modules = ['gate_proj', 'up_proj', 'down_proj']

    def to_dict(self):
        """转换为字典格式"""
        return {
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'num_routing_experts': self.num_routing_experts,
            'top_k': self.top_k,
            'aux_loss_coef': self.aux_loss_coef,
            'moe_layers': self.moe_layers,
            'ffn_target_modules': self.ffn_target_modules,
            'use_culture_loss': self.use_culture_loss,
            'culture_loss_weight': self.culture_loss_weight,
            'apply_to_attention': self.apply_to_attention
        }