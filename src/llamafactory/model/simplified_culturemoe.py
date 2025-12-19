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
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # MoE配置 - 支持joint版本参数名
    num_moe_experts: int = 4  # MoE专家数量（与joint版本保持一致）
    num_activated_experts: int = 2  # 激活的专家数量（与joint版本保持一致）
    aux_loss_coef: float = 0.001  # 辅助损失系数

    # 向后兼容的旧参数名（将被新参数覆盖）
    num_routing_experts: int = None  # 将被num_moe_experts覆盖
    top_k: int = None  # 将被num_activated_experts覆盖

    # 占位符参数（与joint版本保持一致）
    use_shared: bool = False  # 是否使用共享专家（占位符）
    use_gate: bool = False  # 是否使用MoE门控（占位符）

    # 指定哪些层使用MoE (0-indexed) - None表示最后两层
    moe_layers: List[int] = None  # None表示最后两层FFN替换为MoE

    # FFN目标模块
    ffn_target_modules: List[str] = None

    # 文化损失配置
    use_culture_loss: str = "new"  # 文化损失模式：ori/new/kl/false
    culture_loss_weight: float = 0.01

    # LoRA相关配置
    use_lora: bool = True  # 是否启用LoRA微调

    # 其他配置
    apply_to_attention: bool = False  # 暂时禁用注意力层LoRA

    def __post_init__(self):
        # 处理参数映射和向后兼容性

        # 如果提供了旧参数名，使用新参数名覆盖
        if self.num_routing_experts is None:
            self.num_routing_experts = self.num_moe_experts
        if self.top_k is None:
            self.top_k = self.num_activated_experts

        # moe_layers为None表示最后两层FFN替换为MoE
        # 具体的层数将在适配器中根据模型动态确定

        if self.ffn_target_modules is None:
            self.ffn_target_modules = ['gate_proj', 'up_proj', 'down_proj']

    def to_dict(self):
        """转换为字典格式"""
        return {
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'num_moe_experts': self.num_moe_experts,
            'num_activated_experts': self.num_activated_experts,
            'num_routing_experts': self.num_routing_experts,
            'top_k': self.top_k,
            'aux_loss_coef': self.aux_loss_coef,
            'use_shared': self.use_shared,
            'use_gate': self.use_gate,
            'moe_layers': self.moe_layers,
            'ffn_target_modules': self.ffn_target_modules,
            'use_culture_loss': self.use_culture_loss,
            'culture_loss_weight': self.culture_loss_weight,
            'use_lora': self.use_lora,
            'apply_to_attention': self.apply_to_attention
        }