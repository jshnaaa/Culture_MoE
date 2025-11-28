# src/llamafactory/model/joint_lora_moe_model.py
"""
联合LoRA+MoE模型
同时训练预训练LoRA适配器和新增MoE处理层
实现端到端优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class JointLoRAMoEConfig:
    """联合LoRA+MoE配置"""

    # LoRA配置
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

    # MoE配置
    num_moe_experts: int = 4
    moe_hidden_dim: int = 4096
    moe_intermediate_dim: int = None  # 默认为 moe_hidden_dim * 4

    # 文化损失配置
    use_culture_loss: bool = True
    culture_loss_weight: float = 0.01

    # 其他配置
    dropout: float = 0.1

    def __post_init__(self):
        if self.lora_target_modules is None:
            # 默认LoRA目标模块（注意力层）
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        if self.moe_intermediate_dim is None:
            self.moe_intermediate_dim = self.moe_hidden_dim * 4


class MoEExpert(nn.Module):
    """MoE专家层"""

    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = nn.SiLU()  # 使用SiLU激活函数（与LLaMA一致）
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """前向传播"""
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate = gate_output * up_output
        intermediate = self.dropout(intermediate)
        output = self.down_proj(intermediate)
        return output


class MoERouter(nn.Module):
    """MoE路由器"""

    def __init__(self, hidden_dim: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def forward(self, x, temperature: float = 1.0):
        """
        前向传播

        Args:
            x: [B, H] 输入隐藏状态
            temperature: 温度参数，控制路由决策的确定性

        Returns:
            expert_weights: [B, num_experts] 专家权重
            router_logits: [B, num_experts] 原始logits
        """
        # 计算路由logits
        router_logits = self.router(x)  # [B, num_experts]

        # 应用温度缩放
        router_logits = router_logits / temperature

        # 计算softmax权重
        expert_weights = F.softmax(router_logits, dim=-1)

        return expert_weights, router_logits


class MoELayer(nn.Module):
    """MoE层"""

    def __init__(self, config: JointLoRAMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts

        # 创建路由器
        self.router = MoERouter(
            hidden_dim=config.moe_hidden_dim,
            num_experts=config.num_moe_experts,
            dropout=config.dropout
        )

        # 创建专家
        self.experts = nn.ModuleList([
            MoEExpert(
                hidden_dim=config.moe_hidden_dim,
                intermediate_dim=config.moe_intermediate_dim,
                dropout=config.dropout
            ) for _ in range(config.num_moe_experts)
        ])

        # 共享专家（可选）
        self.shared_expert = MoEExpert(
            hidden_dim=config.moe_hidden_dim,
            intermediate_dim=config.moe_hidden_dim,  # 共享专家使用较小的维度
            dropout=config.dropout
        )

        # 门控机制
        self.gate = nn.Linear(config.moe_hidden_dim, config.moe_hidden_dim)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.gate.bias, -2.0)  # 初期倾向于使用共享专家

    def forward(self, hidden_states):
        """
        前向传播

        Args:
            hidden_states: [B, L, H] 输入隐藏状态

        Returns:
            output: [B, L, H] 输出隐藏状态
            expert_weights: [B, num_experts] 专家权重（用于文化损失）
            aux_loss: 辅助损失
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. 共享专家处理
        shared_output = self.shared_expert(hidden_states)  # [B, L, H]

        # 2. 路由决策（基于pooled representation）
        pooled = hidden_states.mean(dim=1)  # [B, H]
        expert_weights, router_logits = self.router(pooled)  # [B, num_experts]

        # 3. 专家处理
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states)  # [B, L, H]
            expert_outputs.append(expert_output)

        # 4. 加权融合专家输出
        weighted_expert_output = torch.zeros_like(hidden_states)
        for i, expert_output in enumerate(expert_outputs):
            weight = expert_weights[:, i].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            weighted_expert_output += weight * expert_output

        # 5. 门控融合共享专家和MoE专家输出
        gate_weights = torch.sigmoid(self.gate(hidden_states))  # [B, L, H]
        final_output = shared_output + gate_weights * weighted_expert_output

        # 6. 计算辅助损失（负载均衡）
        # 简单的均匀分布损失
        uniform_dist = torch.ones_like(expert_weights) / self.num_experts
        aux_loss = F.kl_div(
            F.log_softmax(router_logits, dim=-1),
            uniform_dist,
            reduction='batchmean'
        )

        return final_output, expert_weights, aux_loss


class JointLoRAMoEModel(nn.Module):
    """联合LoRA+MoE模型"""

    def __init__(self, base_model, config: JointLoRAMoEConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model

        # 1. 应用LoRA到基础模型
        self._apply_lora()

        # 2. 添加MoE层
        self.moe_layer = MoELayer(config)

        # 3. 确保MoE层在正确设备上
        if hasattr(base_model, 'device'):
            self.moe_layer = self.moe_layer.to(base_model.device)

        logging.info(f"Joint LoRA+MoE model initialized with {config.num_moe_experts} experts")

    def _apply_lora(self):
        """应用LoRA到基础模型"""
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        前向传播

        Args:
            input_ids: [B, L] 输入token IDs
            attention_mask: [B, L] 注意力掩码
            labels: [B, L] 标签（用于计算损失）

        Returns:
            outputs: 包含loss、logits、expert_weights等的字典
        """
        # 1. 基础模型前向传播（包含LoRA）
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 2. 获取最后一层隐藏状态
        hidden_states = base_outputs.hidden_states[-1]  # [B, L, H]

        # 3. MoE层处理
        moe_output, expert_weights, moe_aux_loss = self.moe_layer(hidden_states)

        # 4. 语言模型头
        if hasattr(self.base_model, 'lm_head'):
            logits = self.base_model.lm_head(moe_output)
        else:
            # 如果没有lm_head，使用基础模型的
            logits = self.base_model.base_model.lm_head(moe_output)

        # 5. 计算损失
        loss = None
        if labels is not None:
            # 语言模型损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # 总损失 = 语言模型损失 + MoE辅助损失
            loss = lm_loss + 0.01 * moe_aux_loss

        # 6. 返回结果
        return type('Outputs', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': moe_output,
            'expert_weights': expert_weights,
            'moe_aux_loss': moe_aux_loss,
        })()

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
                'name': 'base_lora'
            })

        if moe_params:
            param_groups.append({
                'params': moe_params,
                'lr': moe_lr,
                'weight_decay': 0.01,
                'name': 'moe'
            })

        logging.info(f"Parameter groups: base_lora={len(base_params)}, moe={len(moe_params)}")
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

        print(f"  - Base LoRA params: {base_lora_params:,}")
        print(f"  - MoE params: {moe_params:,}")
        print(f"  - Architecture: Joint LoRA + MoE")
        print(f"  - MoE experts: {self.config.num_moe_experts}")

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
            'dropout': self.config.dropout
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ Joint LoRA+MoE model saved to {save_path}")
        print(f"  - LoRA weights: {lora_path}")
        print(f"  - MoE weights: {moe_path}")
        print(f"  - Config: {config_path}")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=150,
                 do_sample=True, temperature=0.7, **kwargs):
        """生成方法"""
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs
        )