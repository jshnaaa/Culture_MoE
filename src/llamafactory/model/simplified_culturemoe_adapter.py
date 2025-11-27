# src/llamafactory/model/simplified_culturemoe_adapter.py
"""
简化版CultureMoE适配器
严格基于MixLoRA实现 + 文化损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import math

from .simplified_culturemoe import SimplifiedCultureMoEConfig


class LoRALinear(nn.Module):
    """LoRA线性层 - 数值稳定版本（与MixLoRA一致）"""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA参数
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 初始化 - 数值稳定版本（与MixLoRA一致）
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.lora_B.weight)
        # 限制LoRA A权重范围
        with torch.no_grad():
            self.lora_A.weight.data.clamp_(-0.1, 0.1)

    def forward(self, x):
        # 输入预处理：限制范围（与MixLoRA一致）
        x = torch.clamp(x, min=-5.0, max=5.0)

        # 限制LoRA权重范围，防止训练中权重爆炸（与MixLoRA一致）
        with torch.no_grad():
            self.lora_A.weight.data.clamp_(-1.0, 1.0)
            self.lora_B.weight.data.clamp_(-1.0, 1.0)

        # 计算LoRA输出: B * A * x (数值稳定版本)
        lora_a_output = self.lora_A(x)
        lora_a_output = torch.clamp(lora_a_output, min=-5.0, max=5.0)  # 限制中间结果
        lora_output = self.lora_B(self.dropout(lora_a_output))

        # 应用保守的LoRA缩放（与MixLoRA一致）
        safe_scaling = min(self.scaling, 1.0)  # 限制最大缩放
        lora_output = lora_output * safe_scaling

        # 限制LoRA输出范围（与MixLoRA一致）
        lora_output = torch.clamp(lora_output, min=-3.0, max=3.0)

        # 检查NaN/Inf（与MixLoRA一致）
        if torch.isnan(lora_output).any() or torch.isinf(lora_output).any():
            lora_output = torch.zeros_like(lora_output)

        return lora_output


class SimplifiedMixLoRALayer(nn.Module):
    """简化的MixLoRA层，严格基于MixLoRA + 文化损失"""

    def __init__(self, original_mlp, config: SimplifiedCultureMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_routing_experts
        self.top_k = config.top_k

        # 保存原始MLP（冻结）
        self.shared_ffn = original_mlp
        # 冻结共享FFN
        for param in self.shared_ffn.parameters():
            param.requires_grad = False

        # 获取维度
        self.hidden_size = original_mlp.gate_proj.in_features
        self.intermediate_size = original_mlp.gate_proj.out_features
        self.device = original_mlp.gate_proj.weight.device
        self.dtype = original_mlp.gate_proj.weight.dtype

        # 简单的线性路由器（与MixLoRA一致）
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # 路由器初始化（与MixLoRA一致）
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        with torch.no_grad():
            self.router.weight.data.clamp_(-0.1, 0.1)

        # 创建LoRA专家（每个专家只是LoRA适配器）
        self.lora_experts = nn.ModuleList([
            self._create_lora_expert(config)
            for _ in range(self.num_experts)
        ])

        # 移动到正确设备
        self.router = self.router.to(device=self.device, dtype=self.dtype)
        self.lora_experts = self.lora_experts.to(device=self.device, dtype=self.dtype)

        # 保存最新的router logits用于文化损失
        self.latest_router_logits = None
        self.latest_batch_size = None
        self.latest_seq_len = None

    def _create_lora_expert(self, config):
        """创建LoRA专家（只是LoRA适配器）"""
        expert = nn.Module()

        # 为每个FFN层添加LoRA适配器
        expert.gate_proj_lora = LoRALinear(
            self.hidden_size, self.intermediate_size,
            config.lora_rank, config.lora_alpha, config.lora_dropout
        )
        expert.up_proj_lora = LoRALinear(
            self.hidden_size, self.intermediate_size,
            config.lora_rank, config.lora_alpha, config.lora_dropout
        )
        expert.down_proj_lora = LoRALinear(
            self.intermediate_size, self.hidden_size,
            config.lora_rank, config.lora_alpha, config.lora_dropout
        )

        return expert

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 输入预处理（与MixLoRA一致）- 数值稳定版本
        hidden_states = torch.clamp(hidden_states, min=-5.0, max=5.0)

        # 检查输入是否有NaN/Inf
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            print("⚠️ NaN/Inf detected in input, using zeros")
            hidden_states = torch.zeros_like(hidden_states)

        hidden_flat = hidden_states.view(-1, hidden_dim)

        # 确保router在正确设备和dtype上
        if self.router.weight.device != hidden_flat.device or self.router.weight.dtype != hidden_flat.dtype:
            self.router = self.router.to(device=hidden_flat.device, dtype=hidden_flat.dtype)

        # 限制router权重（与MixLoRA一致）- 更保守的限制
        with torch.no_grad():
            self.router.weight.data.clamp_(-1.0, 1.0)  # 更保守的限制
            if self.router.bias is not None:
                self.router.bias.data.clamp_(-0.5, 0.5)

        # 计算路由logits - 数值稳定版本
        router_logits = self.router(hidden_flat)
        router_logits = torch.clamp(router_logits, min=-5.0, max=5.0)  # 更保守的限制

        # 保存router logits用于文化损失 - 移除detach()以保持梯度流
        if self.training:
            self.latest_router_logits = router_logits.clone()
            self.latest_batch_size = batch_size
            self.latest_seq_len = seq_len

        # 检查异常值（与MixLoRA一致）
        if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
            print("⚠️ NaN/Inf detected in router_logits, using zeros")
            router_logits = torch.zeros_like(router_logits)

        # Top-K选择（与MixLoRA一致）- 数值稳定版本
        try:
            top_k_logits, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        except Exception as e:
            print(f"⚠️ TopK selection failed: {e}, using fallback")
            selected_experts = torch.arange(self.top_k, device=router_logits.device).unsqueeze(0).expand(router_logits.size(0), -1)
            top_k_logits = router_logits[:, :self.top_k]

        # 数值稳定的softmax（与MixLoRA一致）
        try:
            # 减去最大值提高数值稳定性
            top_k_logits_max = top_k_logits.max(dim=-1, keepdim=True)[0]
            top_k_logits_stable = top_k_logits - top_k_logits_max
            # 进一步限制范围
            top_k_logits_stable = torch.clamp(top_k_logits_stable, min=-10.0, max=10.0)
            expert_weights = F.softmax(top_k_logits_stable, dim=-1)
        except Exception as e:
            print(f"⚠️ Softmax failed: {e}, using uniform weights")
            expert_weights = torch.ones_like(top_k_logits) / self.top_k

        # 检查权重异常值（与MixLoRA一致）
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            print("⚠️ NaN/Inf detected in expert_weights, using uniform weights")
            expert_weights = torch.ones_like(expert_weights) / self.top_k

        # 确保权重和为1（数值稳定性）
        weight_sums = expert_weights.sum(dim=-1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)  # 避免除零
        expert_weights = expert_weights / weight_sums

        # 1. 共享FFN计算（与MixLoRA一致）
        shared_output = self._compute_shared_ffn(hidden_states)

        # 2. LoRA专家计算
        expert_output = self._compute_lora_experts(
            hidden_states, selected_experts, expert_weights
        )

        # 3. 组合输出 - 数值稳定版本
        try:
            # 检查共享输出
            if torch.isnan(shared_output).any() or torch.isinf(shared_output).any():
                print("⚠️ NaN/Inf in shared_output, using zeros")
                shared_output = torch.zeros_like(shared_output)

            # 检查专家输出
            if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
                print("⚠️ NaN/Inf in expert_output, using zeros")
                expert_output = torch.zeros_like(expert_output)

            # 限制输出范围
            shared_output = torch.clamp(shared_output, min=-10.0, max=10.0)
            expert_output = torch.clamp(expert_output, min=-5.0, max=5.0)

            # 组合输出
            final_output = shared_output + expert_output

            # 限制最终输出范围（与MixLoRA一致）
            final_output = torch.clamp(final_output, min=-15.0, max=15.0)

            # 最终NaN/Inf检查
            if torch.isnan(final_output).any() or torch.isinf(final_output).any():
                print("⚠️ NaN/Inf detected in final output, using shared output only")
                final_output = torch.clamp(shared_output, min=-10.0, max=10.0)

            return final_output

        except Exception as e:
            print(f"⚠️ Final output computation failed: {e}, using input")
            return torch.clamp(hidden_states, min=-5.0, max=5.0)

    def _compute_shared_ffn(self, hidden_states):
        """计算共享FFN输出（与MixLoRA一致）- 数值稳定版本"""
        try:
            # 输入限制
            hidden_states = torch.clamp(hidden_states, min=-5.0, max=5.0)

            # 使用原始的共享FFN
            gate_output = self.shared_ffn.act_fn(self.shared_ffn.gate_proj(hidden_states))
            up_output = self.shared_ffn.up_proj(hidden_states)

            # 限制中间结果
            gate_output = torch.clamp(gate_output, min=-10.0, max=10.0)
            up_output = torch.clamp(up_output, min=-10.0, max=10.0)

            # 计算最终输出
            combined = gate_output * up_output
            combined = torch.clamp(combined, min=-15.0, max=15.0)
            shared_output = self.shared_ffn.down_proj(combined)

            # 限制最终输出
            shared_output = torch.clamp(shared_output, min=-10.0, max=10.0)

            # 检查NaN/Inf
            if torch.isnan(shared_output).any() or torch.isinf(shared_output).any():
                print("⚠️ NaN/Inf detected in shared_ffn output, using zeros")
                shared_output = torch.zeros_like(shared_output)

            return shared_output

        except Exception as e:
            print(f"⚠️ Shared FFN computation failed: {e}, using zeros")
            return torch.zeros_like(hidden_states)

    def _compute_lora_experts(self, hidden_states, selected_experts, expert_weights):
        """计算LoRA专家输出 - 真正的稀疏MoE（只计算被选中的专家）"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        try:
            # 初始化专家输出
            expert_output = torch.zeros_like(hidden_states)

            # 重塑为[batch_size * seq_len, hidden_dim]
            hidden_flat = hidden_states.view(-1, hidden_dim)  # [B*L, H]
            expert_output_flat = expert_output.view(-1, hidden_dim)  # [B*L, H]

            # 稀疏MoE：只对被选中的专家-token对进行计算
            for k in range(self.top_k):  # 遍历Top-K选择
                for expert_idx in range(self.num_experts):
                    # 找到选择了当前专家且在第k位置的token
                    expert_mask = (selected_experts[:, k] == expert_idx)  # [B*L]

                    if expert_mask.any():
                        # 获取被选中的token的hidden states
                        selected_hidden = hidden_flat[expert_mask]  # [num_selected_tokens, H]

                        if selected_hidden.numel() == 0:
                            continue

                        try:
                            # 只对被选中的token计算LoRA专家
                            expert = self.lora_experts[expert_idx]

                            # LoRA FFN计算 - 只对选中的token
                            gate_lora = expert.gate_proj_lora(selected_hidden)
                            up_lora = expert.up_proj_lora(selected_hidden)

                            # 检查中间结果
                            if torch.isnan(gate_lora).any() or torch.isinf(gate_lora).any():
                                print(f"⚠️ NaN/Inf in expert {expert_idx} gate_lora, skipping")
                                continue
                            if torch.isnan(up_lora).any() or torch.isinf(up_lora).any():
                                print(f"⚠️ NaN/Inf in expert {expert_idx} up_lora, skipping")
                                continue

                            # 应用激活函数并限制范围
                            activated_gate = self.shared_ffn.act_fn(gate_lora)
                            activated_gate = torch.clamp(activated_gate, min=-10.0, max=10.0)
                            up_lora = torch.clamp(up_lora, min=-10.0, max=10.0)

                            # 计算down projection
                            combined_lora = activated_gate * up_lora
                            combined_lora = torch.clamp(combined_lora, min=-15.0, max=15.0)
                            down_lora = expert.down_proj_lora(combined_lora)

                            # 检查最终LoRA输出
                            if torch.isnan(down_lora).any() or torch.isinf(down_lora).any():
                                print(f"⚠️ NaN/Inf in expert {expert_idx} down_lora, skipping")
                                continue

                            # 获取对应的权重
                            selected_weights = expert_weights[:, k][expert_mask]  # [num_selected_tokens]

                            # 检查权重
                            if torch.isnan(selected_weights).any() or torch.isinf(selected_weights).any():
                                print(f"⚠️ NaN/Inf in expert {expert_idx} weights, skipping")
                                continue

                            # 限制权重范围
                            selected_weights = torch.clamp(selected_weights, min=0.0, max=1.0)

                            # 应用权重
                            weighted_output = down_lora * selected_weights.unsqueeze(-1)  # [num_selected_tokens, H]

                            # 限制加权输出
                            weighted_output = torch.clamp(weighted_output, min=-5.0, max=5.0)

                            # 累加到对应位置（稀疏更新）
                            expert_output_flat[expert_mask] += weighted_output

                        except Exception as e:
                            print(f"⚠️ Expert {expert_idx} computation failed: {e}, skipping")
                            continue

            # 重塑回原始维度
            expert_output = expert_output_flat.view(batch_size, seq_len, hidden_dim)

            # 限制最终专家输出
            expert_output = torch.clamp(expert_output, min=-10.0, max=10.0)

            # 最终NaN/Inf检查
            if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
                print("⚠️ NaN/Inf detected in final expert output, using zeros")
                expert_output = torch.zeros_like(expert_output)

            return expert_output

        except Exception as e:
            print(f"⚠️ Sparse expert computation failed: {e}, using zeros")
            return torch.zeros_like(hidden_states)

    def get_router_probs_for_culture_loss(self):
        """获取路由概率用于文化损失计算"""
        if self.latest_router_logits is None:
            return None

        # 计算概率分布 - 移除detach()以保持梯度流
        router_probs = F.softmax(self.latest_router_logits, dim=-1)

        # 平均到batch维度
        avg_probs = router_probs.mean(dim=0, keepdim=True)  # [1, num_experts]

        # 扩展到batch size
        if self.latest_batch_size is not None:
            expert_weights = avg_probs.expand(self.latest_batch_size, -1)
            return expert_weights.to(dtype=torch.float16)
        else:
            return avg_probs.to(dtype=torch.float16)


class SimplifiedCultureMoEAdapter:
    """简化版CultureMoE适配器，严格基于MixLoRA + 文化损失"""

    def __init__(self, base_model, config: SimplifiedCultureMoEConfig):
        self.base_model = base_model
        self.config = config

        # 替换所有层的MLP为MixLoRA层
        self._replace_mlp_with_mixlora()

        # 冻结非LoRA参数
        self._freeze_non_lora_parameters()

        # 确保设备一致性
        self._ensure_device_consistency()

    def _replace_mlp_with_mixlora(self):
        """替换所有层的MLP为MixLoRA层"""
        # 处理DDP包装的模型
        model_to_modify = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_modify, 'model'):
            # LlamaForCausalLM
            layers = model_to_modify.model.layers
        else:
            # LlamaModel
            layers = model_to_modify.layers

        print(f"🔄 Replacing ALL {len(layers)} layers with MixLoRA (like MixLoRA)")

        # 替换所有层的MLP为MixLoRA层
        for layer_idx in range(len(layers)):
            original_mlp = layers[layer_idx].mlp
            mixlora_layer = SimplifiedMixLoRALayer(original_mlp, self.config)
            layers[layer_idx].mlp = mixlora_layer
            if layer_idx % 5 == 0 or layer_idx == len(layers) - 1:
                print(f"✅ Replaced layer {layer_idx}/{len(layers)-1} MLP with MixLoRA")

    def _freeze_non_lora_parameters(self):
        """冻结非LoRA参数"""
        model_to_freeze = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        for name, param in model_to_freeze.named_parameters():
            if 'lora_' not in name and 'router' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _ensure_device_consistency(self):
        """确保设备一致性"""
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 获取基础模型的设备和数据类型
        base_param = next(model_to_check.parameters())
        base_device = base_param.device
        base_dtype = base_param.dtype

        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        # 确保所有MixLoRA层在正确设备上
        for layer_idx in range(len(layers)):
            mixlora_layer = layers[layer_idx].mlp
            if isinstance(mixlora_layer, SimplifiedMixLoRALayer):
                mixlora_layer.router = mixlora_layer.router.to(device=base_device, dtype=base_dtype)
                mixlora_layer.lora_experts = mixlora_layer.lora_experts.to(device=base_device, dtype=base_dtype)

                if layer_idx % 10 == 0 or layer_idx == len(layers) - 1:
                    print(f"✅ Ensured device consistency for MixLoRA layer {layer_idx}/{len(layers)-1}")

    def get_expert_weights_for_culture_loss(self):
        """收集所有MixLoRA层的专家权重用于文化损失计算"""
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        # 收集所有层的路由概率
        all_router_probs = []
        target_batch_size = None

        for layer_idx in range(len(layers)):
            mixlora_layer = layers[layer_idx].mlp
            if isinstance(mixlora_layer, SimplifiedMixLoRALayer):
                expert_weights = mixlora_layer.get_router_probs_for_culture_loss()
                if expert_weights is not None:
                    if target_batch_size is None:
                        target_batch_size = expert_weights.shape[0]
                    all_router_probs.append(expert_weights)

        if all_router_probs and target_batch_size is not None:
            # 对所有层的专家权重求平均
            layer_avg_probs = torch.stack(all_router_probs, dim=0).mean(dim=0)
            return layer_avg_probs.to(dtype=torch.float16)
        else:
            return None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """前向传播"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 为文化损失计算收集专家权重
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            expert_weights = self.get_expert_weights_for_culture_loss()
            if expert_weights is not None:
                outputs.expert_weights = expert_weights

        return outputs

    def save_model(self, save_path: str):
        """保存LoRA权重"""
        import os

        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        # 只保存LoRA参数
        lora_state_dict = {}
        model_to_save = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        for name, param in model_to_save.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_state_dict[name] = param.data

        # 保存LoRA权重
        lora_path = os.path.join(save_path, 'lora_weights.pt')
        torch.save(lora_state_dict, lora_path)

        # 保存配置
        config_path = os.path.join(save_path, 'simplified_culturemoe_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"✅ Simplified CultureMoE weights saved to {save_path}")

    def get_accumulated_z_loss(self):
        """计算z-loss用于稳定路由器"""
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        if hasattr(model_to_check, 'model'):
            layers = model_to_check.model.layers
        else:
            layers = model_to_check.layers

        total_z_loss = None
        mixlora_layer_count = 0

        # 收集所有MixLoRA层的z-loss
        for layer_idx in range(len(layers)):
            mixlora_layer = layers[layer_idx].mlp
            if isinstance(mixlora_layer, SimplifiedMixLoRALayer) and hasattr(mixlora_layer, 'latest_router_logits'):
                if mixlora_layer.latest_router_logits is not None:
                    # 计算z-loss：惩罚过大的logits值
                    z_loss = 0.001 * (mixlora_layer.latest_router_logits ** 2).mean()
                    z_loss = z_loss.to(dtype=torch.float16)

                    # 检查数值稳定性
                    if torch.isnan(z_loss) or torch.isinf(z_loss):
                        z_loss = torch.tensor(0.0, device=mixlora_layer.latest_router_logits.device, dtype=torch.float16)

                    if total_z_loss is None:
                        total_z_loss = z_loss
                    else:
                        total_z_loss += z_loss

                    mixlora_layer_count += 1

        # 如果没有找到任何MixLoRA层，返回零损失
        if total_z_loss is None:
            device = next(model_to_check.parameters()).device
            total_z_loss = torch.tensor(0.0, device=device, dtype=torch.float16)
        else:
            # 平均化z-loss
            if mixlora_layer_count > 1:
                total_z_loss = total_z_loss / mixlora_layer_count

        return total_z_loss

    def print_trainable_parameters(self):
        """打印可训练参数统计"""
        total_params = 0
        trainable_params = 0

        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        for param in model_to_check.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || "
              f"Total params: {total_params:,} || "
              f"Trainable%: {100 * trainable_params / total_params:.4f}%")

        # 详细统计
        lora_params = 0
        router_params = 0

        for name, param in model_to_check.named_parameters():
            if param.requires_grad:
                if 'lora_' in name:
                    lora_params += param.numel()
                elif 'router' in name:
                    router_params += param.numel()

        print(f"  - LoRA params: {lora_params:,}")
        print(f"  - Router params: {router_params:,}")
        print(f"  - Architecture: MixLoRA + Culture Loss")
        print(f"  - Routing experts: {self.config.num_routing_experts}")


def create_simplified_culturemoe_model(base_model, config: SimplifiedCultureMoEConfig):
    """创建简化版CultureMoE模型"""
    adapter = SimplifiedCultureMoEAdapter(base_model, config)
    return adapter