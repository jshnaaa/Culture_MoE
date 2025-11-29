# src/llamafactory/model/only_moe_model.py
"""
只训练最后两层MoE模型
将最后两层的FFN替换为MoE专家（LoRA适配器），冻结其他所有参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class OnlyMoEConfig:
    """只训练MoE配置"""

    # LoRA配置
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # MoE配置
    num_moe_experts: int = 4
    top_k: int = 2
    moe_layers: List[int] = None  # 需要替换为MoE的层，例如[26, 27]

    # 辅助损失配置
    aux_loss_coef: float = 0.01

    # 其他配置
    dropout: float = 0.1

    def __post_init__(self):
        if self.moe_layers is None:
            # 默认最后两层
            self.moe_layers = []


class LoRAFFNExpert(nn.Module):
    """LoRA FFN专家 - 数值稳定版本"""

    def __init__(self, hidden_size: int, intermediate_size: int, act_fn, lora_rank: int, lora_alpha: int, lora_dropout: float = 0.1, dtype: torch.dtype = torch.float16):
        super().__init__()

        # 保存维度和激活函数
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn
        self.dtype = dtype

        # LoRA适配器
        self.gate_proj_lora = self._create_lora_layer(self.hidden_size, self.intermediate_size, lora_rank, lora_alpha, lora_dropout)
        self.up_proj_lora = self._create_lora_layer(self.hidden_size, self.intermediate_size, lora_rank, lora_alpha, lora_dropout)
        self.down_proj_lora = self._create_lora_layer(self.intermediate_size, self.hidden_size, lora_rank, lora_alpha, lora_dropout)

    def _create_lora_layer(self, in_features: int, out_features: int, rank: int, alpha: int, dropout: float):
        """创建LoRA层"""
        lora = nn.Module()
        lora.lora_A = nn.Linear(in_features, rank, bias=False, dtype=self.dtype)
        lora.lora_B = nn.Linear(rank, out_features, bias=False, dtype=self.dtype)
        lora.dropout = nn.Dropout(dropout)
        lora.scaling = alpha / rank

        # 数值稳定的初始化
        nn.init.normal_(lora.lora_A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(lora.lora_B.weight)

        # 移动到正确设备
        lora = lora.to(device=self.device, dtype=self.dtype)

        return lora

    def _apply_lora(self, lora_layer, x):
        """应用LoRA计算"""
        # 限制输入范围
        x = torch.clamp(x, min=-5.0, max=5.0)

        # LoRA计算
        lora_out = lora_layer.lora_A(x)
        lora_out = torch.clamp(lora_out, min=-3.0, max=3.0)
        lora_out = lora_layer.dropout(lora_out)
        lora_out = lora_layer.lora_B(lora_out)

        # 应用缩放
        lora_out = lora_out * lora_layer.scaling

        # 限制输出范围
        lora_out = torch.clamp(lora_out, min=-5.0, max=5.0)

        # 检查NaN/Inf
        if torch.isnan(lora_out).any() or torch.isinf(lora_out).any():
            lora_out = torch.zeros_like(lora_out)

        return lora_out

    def forward(self, hidden_states):
        """前向传播"""
        try:
            # 输入预处理
            hidden_states = torch.clamp(hidden_states, min=-5.0, max=5.0)

            # 检查输入
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                return torch.zeros_like(hidden_states)

            # LoRA FFN计算
            gate_out = self._apply_lora(self.gate_proj_lora, hidden_states)
            up_out = self._apply_lora(self.up_proj_lora, hidden_states)

            # 激活函数
            gate_out = self.act_fn(gate_out)
            gate_out = torch.clamp(gate_out, min=-5.0, max=5.0)

            # 组合
            intermediate = gate_out * up_out
            intermediate = torch.clamp(intermediate, min=-5.0, max=5.0)

            # 下投影
            output = self._apply_lora(self.down_proj_lora, intermediate)

            # 最终检查
            if torch.isnan(output).any() or torch.isinf(output).any():
                return torch.zeros_like(hidden_states)

            return output

        except Exception as e:
            print(f"⚠️ LoRA FFN Expert forward failed: {e}")
            return torch.zeros_like(hidden_states)


class MoERouter(nn.Module):
    """MoE路由器 - 数值稳定版本"""

    def __init__(self, hidden_dim: int, num_experts: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_experts = num_experts

        # 路由器
        self.router = nn.Linear(hidden_dim, num_experts, bias=False, dtype=dtype)
        self.layer_norm = nn.LayerNorm(hidden_dim, dtype=dtype)

        # 保守的初始化
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)

    def forward(self, x):
        """前向传播"""
        try:
            # 输入预处理
            x = torch.clamp(x, min=-5.0, max=5.0)
            x = self.layer_norm(x)

            # 计算路由logits
            router_logits = self.router(x)
            router_logits = torch.clamp(router_logits, min=-5.0, max=5.0)

            # 检查异常值
            if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
                # 使用均匀分布作为fallback
                expert_weights = torch.ones(x.size(0), self.num_experts, device=x.device, dtype=x.dtype) / self.num_experts
                return expert_weights, router_logits

            # 数值稳定的softmax
            router_logits_max = router_logits.max(dim=-1, keepdim=True)[0]
            router_logits_stable = router_logits - router_logits_max
            router_logits_stable = torch.clamp(router_logits_stable, min=-10.0, max=0.0)

            expert_weights = F.softmax(router_logits_stable, dim=-1)

            # 最终检查
            if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
                expert_weights = torch.ones_like(expert_weights) / self.num_experts

            # 确保权重和为1
            expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-8)

            return expert_weights, router_logits

        except Exception as e:
            print(f"⚠️ Router forward failed: {e}")
            expert_weights = torch.ones(x.size(0), self.num_experts, device=x.device, dtype=x.dtype) / self.num_experts
            router_logits = torch.zeros_like(expert_weights)
            return expert_weights, router_logits


class MoEFFNLayer(nn.Module):
    """MoE FFN层 - 替换原始FFN"""

    def __init__(self, original_mlp, config: OnlyMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.top_k = config.top_k

        # 获取维度信息
        self.hidden_size = original_mlp.gate_proj.in_features
        self.intermediate_size = original_mlp.gate_proj.out_features
        self.device = original_mlp.gate_proj.weight.device
        self.dtype = original_mlp.gate_proj.weight.dtype
        self.act_fn = original_mlp.act_fn

        # 保存原始MLP权重的副本（避免参数共享冲突）
        with torch.no_grad():
            self.shared_gate_weight = original_mlp.gate_proj.weight.clone().detach()
            self.shared_up_weight = original_mlp.up_proj.weight.clone().detach()
            self.shared_down_weight = original_mlp.down_proj.weight.clone().detach()

            # 处理bias（如果存在）
            if hasattr(original_mlp.gate_proj, 'bias') and original_mlp.gate_proj.bias is not None:
                self.shared_gate_bias = original_mlp.gate_proj.bias.clone().detach()
            else:
                self.shared_gate_bias = None

            if hasattr(original_mlp.up_proj, 'bias') and original_mlp.up_proj.bias is not None:
                self.shared_up_bias = original_mlp.up_proj.bias.clone().detach()
            else:
                self.shared_up_bias = None

            if hasattr(original_mlp.down_proj, 'bias') and original_mlp.down_proj.bias is not None:
                self.shared_down_bias = original_mlp.down_proj.bias.clone().detach()
            else:
                self.shared_down_bias = None

        # 创建路由器
        self.router = MoERouter(self.hidden_size, self.num_experts, self.dtype)

        # 创建LoRA专家（传入维度信息而不是原始MLP）
        self.lora_experts = nn.ModuleList([
            LoRAFFNExpert(
                self.hidden_size,
                self.intermediate_size,
                self.act_fn,
                config.lora_rank,
                config.lora_alpha,
                config.lora_dropout,
                self.dtype
            ) for _ in range(self.num_experts)
        ])

        # 移动到正确设备 - 使用to()方法确保所有子模块都移动
        self.to(device=self.device, dtype=self.dtype)

        # 初始化辅助损失
        self._last_aux_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def forward(self, hidden_states):
        """前向传播"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        try:
            # 输入预处理
            hidden_states = torch.clamp(hidden_states, min=-5.0, max=5.0)

            # 检查输入
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print("⚠️ NaN/Inf in MoE input, using shared FFN only")
                return self._compute_shared_ffn(hidden_states), torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

            # 1. 共享FFN计算
            shared_output = self._compute_shared_ffn(hidden_states)

            # 2. 路由决策（使用平均池化获取序列表示）
            pooled = hidden_states.mean(dim=1)  # [B, H]
            expert_weights, router_logits = self.router(pooled)

            # 3. Top-K选择
            try:
                top_k_weights, selected_experts = torch.topk(expert_weights, self.top_k, dim=-1)
                top_k_weights = F.softmax(top_k_weights, dim=-1)
            except Exception as e:
                print(f"⚠️ TopK selection failed: {e}, using uniform")
                selected_experts = torch.arange(self.top_k, device=expert_weights.device).unsqueeze(0).expand(batch_size, -1)
                top_k_weights = torch.ones(batch_size, self.top_k, device=expert_weights.device, dtype=expert_weights.dtype) / self.top_k

            # 4. 稀疏专家计算
            expert_output = self._compute_sparse_experts(hidden_states, selected_experts, top_k_weights)

            # 5. 组合输出
            final_output = shared_output + expert_output
            final_output = torch.clamp(final_output, min=-10.0, max=10.0)

            # 6. 辅助损失
            aux_loss = self._compute_aux_loss(expert_weights)

            # 保存aux_loss用于后续收集
            self._last_aux_loss = aux_loss

            # 最终检查
            if torch.isnan(final_output).any() or torch.isinf(final_output).any():
                print("⚠️ NaN/Inf in final MoE output, using shared FFN only")
                final_output = shared_output

            # 注意：这里只返回final_output，aux_loss通过_last_aux_loss属性保存
            return final_output

        except Exception as e:
            print(f"⚠️ MoE layer failed: {e}, using shared FFN only")
            shared_output = self._compute_shared_ffn(hidden_states)
            aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
            self._last_aux_loss = aux_loss
            return shared_output

    def _compute_shared_ffn(self, hidden_states):
        """计算共享FFN输出"""
        try:
            with torch.no_grad():
                # 使用复制的权重进行计算，避免参数共享
                gate_out = F.linear(hidden_states, self.shared_gate_weight, self.shared_gate_bias)
                gate_out = self.act_fn(gate_out)

                up_out = F.linear(hidden_states, self.shared_up_weight, self.shared_up_bias)
                intermediate = gate_out * up_out

                output = F.linear(intermediate, self.shared_down_weight, self.shared_down_bias)
                output = torch.clamp(output, min=-10.0, max=10.0)
                return output
        except Exception as e:
            print(f"⚠️ Shared FFN failed: {e}, using zeros")
            return torch.zeros_like(hidden_states)

    def _compute_sparse_experts(self, hidden_states, selected_experts, expert_weights):
        """稀疏专家计算"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        expert_output = torch.zeros_like(hidden_states)

        try:
            for k in range(self.top_k):
                for expert_idx in range(self.num_experts):
                    # 找到选择了当前专家的batch
                    expert_mask = (selected_experts[:, k] == expert_idx)

                    if expert_mask.any():
                        # 获取选中的hidden states
                        selected_hidden = hidden_states[expert_mask]  # [num_selected_batch, seq_len, hidden_dim]

                        if selected_hidden.numel() == 0:
                            continue

                        try:
                            # 专家计算
                            expert_out = self.lora_experts[expert_idx](selected_hidden)

                            # 应用权重
                            weight = expert_weights[:, k][expert_mask].unsqueeze(-1).unsqueeze(-1)  # [num_selected_batch, 1, 1]
                            weighted_output = expert_out * weight

                            # 累加到对应位置
                            expert_output[expert_mask] += weighted_output

                        except Exception as e:
                            print(f"⚠️ Expert {expert_idx} computation failed: {e}")
                            continue

            # 限制专家输出
            expert_output = torch.clamp(expert_output, min=-5.0, max=5.0)

            return expert_output

        except Exception as e:
            print(f"⚠️ Sparse expert computation failed: {e}")
            return torch.zeros_like(hidden_states)

    def _compute_aux_loss(self, expert_weights):
        """计算辅助损失"""
        try:
            # 简单的负载均衡损失
            target_uniform = torch.ones_like(expert_weights) / self.num_experts
            aux_loss = F.mse_loss(expert_weights, target_uniform) * self.config.aux_loss_coef

            if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                aux_loss = torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)

            return aux_loss

        except Exception:
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)


class OnlyMoEModel(nn.Module):
    """只训练最后两层MoE的模型"""

    def __init__(self, base_model, config: OnlyMoEConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model

        # 获取模型层
        self._get_model_layers()

        # 替换指定层的FFN为MoE
        self._replace_ffn_with_moe()

        # 冻结非MoE参数
        self._freeze_non_moe_parameters()

        logging.info(f"OnlyMoE model initialized with MoE on layers {config.moe_layers}")

    def _get_model_layers(self):
        """获取模型层"""
        # 检查base_model是否存在
        if not hasattr(self, 'base_model') or self.base_model is None:
            raise RuntimeError("OnlyMoEModel: base_model not properly initialized in _get_model_layers")

        # 处理DDP包装的模型
        model_to_modify = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_modify, 'model'):
            # LlamaForCausalLM
            self.layers = model_to_modify.model.layers
        else:
            # LlamaModel
            self.layers = model_to_modify.layers

    def _replace_ffn_with_moe(self):
        """替换指定层的FFN为MoE"""
        print(f"🔄 Replacing FFN with MoE on layers {self.config.moe_layers}")

        for layer_idx in self.config.moe_layers:
            if layer_idx < len(self.layers):
                original_mlp = self.layers[layer_idx].mlp
                moe_layer = MoEFFNLayer(original_mlp, self.config)

                # 完全替换MLP，确保原始MLP不再是模型的一部分
                self.layers[layer_idx].mlp = moe_layer

                # 显式删除对原始MLP的引用，确保它不在计算图中
                del original_mlp

                print(f"✅ Replaced layer {layer_idx} FFN with MoE")
            else:
                print(f"⚠️ Layer {layer_idx} does not exist, skipping")

    def _freeze_non_moe_parameters(self):
        """冻结非MoE参数"""
        if not hasattr(self, 'base_model') or self.base_model is None:
            raise RuntimeError("OnlyMoEModel: base_model not properly initialized in _freeze_non_moe_parameters")

        model_to_freeze = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        for name, param in model_to_freeze.named_parameters():
            # 只训练MoE相关参数
            if any(f'layers.{layer_idx}.mlp' in name for layer_idx in self.config.moe_layers) and \
               ('lora_' in name or 'router' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """前向传播"""
        # 检查base_model是否存在
        if not hasattr(self, 'base_model') or self.base_model is None:
            raise RuntimeError("OnlyMoEModel: base_model not properly initialized")

        # 基础模型前向传播
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 收集MoE辅助损失
        moe_aux_loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float16)
        moe_layer_count = 0

        for layer_idx in self.config.moe_layers:
            if layer_idx < len(self.layers):
                moe_layer = self.layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLayer) and hasattr(moe_layer, '_last_aux_loss'):
                    layer_aux_loss = getattr(moe_layer, '_last_aux_loss', 0.0)
                    if isinstance(layer_aux_loss, torch.Tensor):
                        layer_aux_loss = layer_aux_loss.to(dtype=torch.float16)
                        moe_aux_loss += layer_aux_loss
                        moe_layer_count += 1

        if moe_layer_count > 0:
            moe_aux_loss = moe_aux_loss / moe_layer_count

        # 计算损失
        loss = None
        if labels is not None:
            try:
                # 语言模型损失
                logits = outputs.logits
                logits = torch.clamp(logits, min=-10.0, max=10.0)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # 检查损失
                if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                    print("⚠️ NaN/Inf detected in lm_loss, using fallback")
                    lm_loss = F.mse_loss(shift_logits, torch.zeros_like(shift_logits)) * 1.0

                # 总损失
                loss = lm_loss + moe_aux_loss

                # 最终检查
                if torch.isnan(loss) or torch.isinf(loss):
                    loss = lm_loss

            except Exception as e:
                print(f"⚠️ Loss computation failed: {e}")
                loss = torch.tensor(1.0, device=input_ids.device, dtype=torch.float16, requires_grad=True)

        # 返回结果
        return type('Outputs', (), {
            'loss': loss,
            'logits': outputs.logits,
            'moe_aux_loss': moe_aux_loss,
        })()

    def save_model(self, save_path: str):
        """保存MoE权重"""
        if not hasattr(self, 'base_model') or self.base_model is None:
            raise RuntimeError("OnlyMoEModel: base_model not properly initialized, cannot save model")

        import os

        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        # 只保存MoE参数
        moe_state_dict = {}
        model_to_save = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        for name, param in model_to_save.named_parameters():
            if param.requires_grad and ('lora_' in name or 'router' in name):
                moe_state_dict[name] = param.data

        # 保存MoE权重
        moe_path = os.path.join(save_path, 'only_moe_weights.pt')
        torch.save(moe_state_dict, moe_path)

        # 保存配置
        config_path = os.path.join(save_path, 'only_moe_config.json')
        import json
        config_dict = {
            'lora_rank': self.config.lora_rank,
            'lora_alpha': self.config.lora_alpha,
            'lora_dropout': self.config.lora_dropout,
            'num_moe_experts': self.config.num_moe_experts,
            'top_k': self.config.top_k,
            'moe_layers': self.config.moe_layers,
            'aux_loss_coef': self.config.aux_loss_coef,
            'dropout': self.config.dropout
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ OnlyMoE model saved to {save_path}")
        print(f"  - MoE weights: {moe_path}")
        print(f"  - Config: {config_path}")

    def print_trainable_parameters(self):
        """打印可训练参数统计"""
        if not hasattr(self, 'base_model') or self.base_model is None:
            print("⚠️ OnlyMoEModel: base_model not properly initialized, cannot print parameters")
            return

        total_params = 0
        trainable_params = 0
        moe_params = 0

        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        for name, param in model_to_check.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'lora_' in name or 'router' in name:
                    moe_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || "
              f"Total params: {total_params:,} || "
              f"Trainable%: {100 * trainable_params / total_params:.4f}%")

        print(f"  - MoE params: {moe_params:,}")
        print(f"  - Architecture: Only MoE (last {len(self.config.moe_layers)} layers)")
        print(f"  - MoE layers: {self.config.moe_layers}")
        print(f"  - Experts per layer: {self.config.num_moe_experts}")

    def generate(self, *args, **kwargs):
        """生成方法 - 委托给基础模型"""
        if not hasattr(self, 'base_model') or self.base_model is None:
            raise RuntimeError("OnlyMoEModel: base_model not properly initialized")
        return self.base_model.generate(*args, **kwargs)

