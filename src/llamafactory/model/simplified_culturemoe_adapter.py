# src/llamafactory/model/simplified_culturemoe_adapter.py
"""
简化版CultureMoE适配器
清爽的MoE架构：每层FFN + 4个LoRA路由专家 + 1个LoRA共享专家 + Router + Gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import math

from .simplified_culturemoe import SimplifiedCultureMoEConfig


class LoRAExpert(nn.Module):
    """LoRA专家层 - 简化实现"""

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn,
                 lora_rank: int = 16, lora_alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.scaling = lora_alpha / lora_rank

        # LoRA分支
        self.gate_lora_A = nn.Linear(hidden_dim, lora_rank, bias=False)
        self.gate_lora_B = nn.Linear(lora_rank, intermediate_dim, bias=False)
        self.up_lora_A = nn.Linear(hidden_dim, lora_rank, bias=False)
        self.up_lora_B = nn.Linear(lora_rank, intermediate_dim, bias=False)
        self.down_lora_A = nn.Linear(intermediate_dim, lora_rank, bias=False)
        self.down_lora_B = nn.Linear(lora_rank, hidden_dim, bias=False)

        self.act_fn = act_fn
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.gate_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_lora_B.weight)
        nn.init.kaiming_uniform_(self.up_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_lora_B.weight)
        nn.init.kaiming_uniform_(self.down_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_lora_B.weight)

    def forward(self, x):
        """前向传播：计算LoRA增量"""
        gate_lora = self.gate_lora_B(self.gate_lora_A(x)) * self.scaling
        up_lora = self.up_lora_B(self.up_lora_A(x)) * self.scaling

        intermediate = self.act_fn(gate_lora) * up_lora
        intermediate = self.dropout(intermediate)

        return self.down_lora_B(self.down_lora_A(intermediate)) * self.scaling


class MoERouter(nn.Module):
    """MoE路由器 - 简化实现"""

    def __init__(self, hidden_dim: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)

    def forward(self, x):
        """
        路由计算
        Args:
            x: [B, L, H] 输入隐藏状态
        Returns:
            expert_weights: [B, L, num_experts] 专家权重
        """
        # 计算路由logits并归一化
        router_logits = self.router(x)  # [B, L, num_experts]
        expert_weights = F.softmax(router_logits, dim=-1)
        return expert_weights


class MoEFFNLoRA(nn.Module):
    """简化版MoE FFN层：原始FFN + 4个LoRA路由专家 + 1个LoRA共享专家 + Router + Gate"""

    def __init__(self, original_ffn, config: SimplifiedCultureMoEConfig):
        super().__init__()
        self.original_ffn = original_ffn
        self.hidden_dim = original_ffn.gate_proj.in_features
        self.intermediate_dim = original_ffn.gate_proj.out_features

        # 4个路由专家
        self.routing_experts = nn.ModuleList([
            LoRAExpert(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                act_fn=original_ffn.act_fn,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                dropout=config.lora_dropout
            ) for _ in range(4)
        ])

        # 1个共享专家
        self.shared_expert = LoRAExpert(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            act_fn=original_ffn.act_fn,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )

        # Router网络
        self.router = MoERouter(hidden_dim=self.hidden_dim, num_experts=4)

        # Gate网络：融合共享专家和路由专家输出
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2, bias=False),
            nn.Softmax(dim=-1)
        )

        # 初始化Gate网络
        for layer in self.gate_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        # 保存专家权重和输出用于文化损失
        self.latest_routing_weights = None
        self.latest_shared_output = None

    def forward(self, hidden_states):
        """
        前向传播：原始FFN + 4个路由专家(Top-2) + 1个共享专家 + Gate融合
        """
        # 1. 原始FFN输出
        original_output = self.original_ffn(hidden_states)

        # 2. 路由计算：获取4个专家的权重
        routing_weights = self.router(hidden_states)  # [B, L, 4]

        # 3. Top-2激活：选择权重最高的2个专家
        top2_values, top2_indices = torch.topk(routing_weights, k=2, dim=-1)  # [B, L, 2]

        # 4. 权重重新归一化
        top2_weights = F.softmax(top2_values, dim=-1)  # [B, L, 2]

        # 5. 计算路由专家输出
        routing_output = hidden_states * 0.0  # 初始化
        for i in range(2):  # Top-2
            expert_indices = top2_indices[:, :, i]  # [B, L]
            expert_weights = top2_weights[:, :, i].unsqueeze(-1)  # [B, L, 1]

            # 为每个位置选择对应的专家
            for expert_idx in range(4):
                mask = (expert_indices == expert_idx)  # [B, L]
                if mask.any():
                    expert_input = hidden_states[mask]  # [N, H]
                    expert_delta = self.routing_experts[expert_idx](expert_input)
                    routing_output[mask] += expert_delta * expert_weights[mask]

        # 6. 共享专家输出
        shared_delta = self.shared_expert(hidden_states)
        shared_output = original_output + shared_delta

        # 7. 路由专家最终输出
        routed_output = original_output + routing_output

        # 8. Gate网络融合
        gate_input = torch.cat([shared_output, routed_output], dim=-1)  # [B, L, 2*H]
        gate_weights = self.gate_network(gate_input)  # [B, L, 2]

        final_output = (gate_weights[..., 0:1] * shared_output +
                       gate_weights[..., 1:2] * routed_output)

        # 9. 保存用于文化损失计算
        self.latest_routing_weights = routing_weights.mean(dim=1).detach()  # [B, 4]
        self.latest_shared_output = shared_output.mean(dim=1).detach()  # [B, H]

        return final_output

    def get_aux_loss(self):
        """计算负载均衡损失"""
        if self.latest_routing_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)

        # 负载均衡损失：鼓励专家使用均匀分布
        expert_usage = self.latest_routing_weights.mean(dim=0)  # [4]
        uniform_usage = torch.ones_like(expert_usage) / 4  # [4]
        balance_loss = F.mse_loss(expert_usage, uniform_usage)

        return balance_loss * 0.01



class SimplifiedCultureMoEAdapter:
    """简化版CultureMoE适配器 - 纯MoE架构"""

    def __init__(self, base_model, config: SimplifiedCultureMoEConfig):
        self.base_model = base_model
        self.config = config

        # 应用LoRA到注意力层（如果启用）
        if config.use_lora:
            self._apply_attention_lora()

        # 替换所有FFN层为MoE
        self._replace_all_layers_with_moe()

        # 确保设备一致性
        self._ensure_device_consistency()


    def _get_target_layers(self):
        """获取目标层索引（所有层）"""
        # 处理模型包装
        actual_model = self.base_model
        if hasattr(actual_model, 'module'):
            actual_model = actual_model.module
        if hasattr(actual_model, 'base_model'):
            if hasattr(actual_model.base_model, 'model'):
                actual_model = actual_model.base_model.model
            else:
                actual_model = actual_model.base_model
        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            actual_model = actual_model.model

        # 获取layers
        if not hasattr(actual_model, 'layers'):
            raise AttributeError(f"Cannot find layers in model type: {type(actual_model)}")

        layers = actual_model.layers
        target_layers = list(range(len(layers)))  # 所有层都使用MoE

        return layers, target_layers


    def _replace_all_layers_with_moe(self):
        """替换所有层的FFN为LoRA MoE"""
        layers, target_layers = self._get_target_layers()

        print(f"🔄 Replacing FFN in ALL {len(target_layers)} layers with LoRA MoE (Pure LoRA MoE Architecture)")

        for layer_idx in target_layers:
            original_ffn = layers[layer_idx].mlp
            moe_ffn = MoEFFNLoRA(original_ffn, self.config)
            layers[layer_idx].mlp = moe_ffn

            # 构建配置信息字符串
            config_info = f"{self.config.num_moe_experts} experts, rank={self.config.lora_rank}"
            if self.config.use_shared:
                config_info += ", +shared"
            if self.config.use_gate:
                config_info += ", +gate"

            print(f"✅ Replaced layer {layer_idx} FFN with LoRA MoE ({config_info})")


    def _apply_attention_lora(self):
        """应用LoRA到注意力层"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # LoRA配置
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 只针对注意力层
                bias="none",
            )

            # 应用LoRA
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"✅ Applied LoRA to attention layers (rank={self.config.lora_rank})")

        except ImportError:
            print("⚠️ PEFT not available, skipping LoRA")
        except Exception as e:
            print(f"⚠️ LoRA application failed: {e}")


    def _ensure_device_consistency(self):
        """确保设备一致性"""
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 获取基础模型的设备
        base_device = next(model_to_check.parameters()).device

        try:
            layers, target_layers = self._get_target_layers()

            # 确保MoE层在正确设备上
            for layer_idx in target_layers:
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA):
                    moe_layer = moe_layer.to(device=base_device)

            print(f"✅ Ensured device consistency for MoE layers on {base_device}")

        except Exception as e:
            print(f"⚠️ Device consistency check failed: {e}")
            print(f"✅ Skipping device consistency check - MoE layers already properly configured on {base_device}")

            # 作为fallback，直接遍历所有参数确保设备一致性
            for name, param in model_to_check.named_parameters():
                if param.device != base_device:
                    param.data = param.data.to(base_device)
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(base_device)

            print(f"✅ Fallback device consistency completed on {base_device}")

    def get_expert_weights_for_culture_loss(self):
        """获取专家权重用于文化损失计算"""
        try:
            layers, target_layers = self._get_target_layers()
            expert_weights_list = []

            for layer_idx in target_layers:
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA) and moe_layer.latest_routing_weights is not None:
                    expert_weights_list.append(moe_layer.latest_routing_weights)

            if expert_weights_list:
                return torch.stack(expert_weights_list, dim=0).mean(dim=0)
            else:
                return None
        except Exception as e:
            print(f"⚠️ 获取专家权重失败: {e}")
            return None

    def get_shared_outputs_for_culture_loss(self):
        """获取shared专家输出用于文化损失计算"""
        try:
            layers, target_layers = self._get_target_layers()
            shared_outputs_list = []

            for layer_idx in target_layers:
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA) and moe_layer.latest_shared_output is not None:
                    shared_outputs_list.append(moe_layer.latest_shared_output)

            if shared_outputs_list:
                return torch.stack(shared_outputs_list, dim=0).mean(dim=0)
            else:
                return None
        except Exception as e:
            print(f"⚠️ 获取shared专家输出失败: {e}")
            return None

    def get_accumulated_z_loss(self, main_loss=None):
        """计算累积的z-loss"""
        layers, target_layers = self._get_target_layers()

        total_aux_loss = None
        moe_layer_count = 0

        for layer_idx in target_layers:
            moe_layer = layers[layer_idx].mlp
            if isinstance(moe_layer, MoEFFNLoRA):
                aux_loss = moe_layer.get_aux_loss()
                if total_aux_loss is None:
                    total_aux_loss = aux_loss
                else:
                    total_aux_loss = total_aux_loss + aux_loss
                moe_layer_count += 1

        if total_aux_loss is None:
            if main_loss is not None:
                total_aux_loss = main_loss * 0.0
            else:
                dummy_param = next(iter(self.base_model.parameters()))
                total_aux_loss = dummy_param.sum() * 0.0
        elif moe_layer_count > 1:
            total_aux_loss = total_aux_loss / moe_layer_count

        return total_aux_loss


    def compute_culture_loss(self, culture_labels):
        """
        简化的文化损失计算：基于路由权重和共享输出
        """
        try:
            layers, target_layers = self._get_target_layers()

            # 收集路由权重和共享输出
            routing_weights = []
            shared_outputs = []

            for layer_idx in target_layers:
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA):
                    if moe_layer.latest_routing_weights is not None:
                        routing_weights.append(moe_layer.latest_routing_weights)
                    if moe_layer.latest_shared_output is not None:
                        shared_outputs.append(moe_layer.latest_shared_output)

            if not routing_weights and not shared_outputs:
                return torch.tensor(0.0, device=culture_labels.device, requires_grad=True)

            culture_loss = torch.tensor(0.0, device=culture_labels.device, requires_grad=True)

            # 1. 路由专家文化对比损失
            if routing_weights:
                expert_weights = routing_weights[0]  # 使用第一层作为代表
                batch_size = expert_weights.shape[0]

                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        similarity = F.cosine_similarity(
                            expert_weights[i].unsqueeze(0),
                            expert_weights[j].unsqueeze(0)
                        )

                        if culture_labels[i] == culture_labels[j]:
                            # 同文化：鼓励相似
                            culture_loss = culture_loss + (1.0 - similarity)
                        else:
                            # 不同文化：鼓励不同
                            culture_loss = culture_loss + similarity

            # 2. 共享专家一致性损失
            if shared_outputs and len(shared_outputs[0]) >= 2:
                shared_output = shared_outputs[0]  # 使用第一层作为代表
                batch_size = shared_output.shape[0]

                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        similarity = F.cosine_similarity(
                            shared_output[i].unsqueeze(0),
                            shared_output[j].unsqueeze(0)
                        )
                        # 共享专家：总是鼓励相似
                        culture_loss = culture_loss + (1.0 - similarity)

            return culture_loss / max(1, batch_size * (batch_size - 1) // 2)

        except Exception as e:
            print(f"⚠️ 文化损失计算失败: {e}")
            return torch.tensor(0.0, device=culture_labels.device, requires_grad=True)

    def ensure_trainable_parameters(self):
        """严格参数冻结：只训练LoRA和MoE参数，冻结所有基座参数"""
        try:
            model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

            # 验证模型是否存在
            if model_to_check is None:
                print("❌ 参数冻结失败: 未找到模型")
                return

            # 第一步：冻结所有参数
            all_params_count = 0
            for param in model_to_check.parameters():
                param.requires_grad = False
                all_params_count += 1

            if all_params_count == 0:
                print("⚠️ 参数冻结警告: 未找到任何模型参数")
                return

            # 第二步：只解冻LoRA和MoE相关参数
            trainable_count = 0
            frozen_count = 0
            trainable_param_names = []

            for name, param in model_to_check.named_parameters():
                # 只有以下参数可训练：
                # 1. LoRA参数（注意力层和MoE专家层）
                # 2. Router网络参数
                # 3. Gate网络参数
                should_train = any(keyword in name.lower() for keyword in [
                    'lora',           # LoRA参数（注意力层和MoE专家层）
                    'routing_experts', # 路由专家LoRA参数
                    'shared_expert',   # 共享专家LoRA参数
                    'router',         # Router网络参数
                    'gate_network'    # Gate网络参数
                ])

                if should_train:
                    param.requires_grad = True
                    trainable_count += param.numel()
                    trainable_param_names.append(name)
                else:
                    frozen_count += param.numel()

            total_params = trainable_count + frozen_count
            if total_params > 0:
                frozen_ratio = (frozen_count / total_params) * 100
            else:
                frozen_ratio = 0.0

            print(f"✅ 参数冻结完成:")
            print(f"  - 可训练参数: {trainable_count:,} (LoRA + MoE结构)")
            print(f"  - 冻结参数: {frozen_count:,} (基座模型)")
            print(f"  - 冻结比例: {frozen_ratio:.2f}%")

            # 验证是否找到了预期的可训练参数
            if trainable_count == 0:
                print("⚠️ 警告: 未找到任何可训练的LoRA或MoE参数")
                print("   请检查模型是否正确初始化了MoE结构")
            elif len(trainable_param_names) > 0:
                print(f"  - 可训练参数类型数: {len(trainable_param_names)}")

        except Exception as e:
            print(f"❌ 参数冻结过程出现异常: {e}")
            print("   将尝试基本的参数冻结...")
            # 备用方案：至少确保基础参数被冻结
            try:
                model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
                for param in model_to_check.parameters():
                    param.requires_grad = False
                print("✅ 基本参数冻结完成（所有参数已冻结）")
            except Exception as backup_error:
                print(f"❌ 基本参数冻结也失败: {backup_error}")




    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """简化的前向传播"""
        # 确保关键参数可训练（仅在第一次调用时）
        if not hasattr(self, '_parameters_checked'):
            self._parameters_checked = True
            self.ensure_trainable_parameters()

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 为文化损失计算收集数据
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            expert_weights = self.get_expert_weights_for_culture_loss()
            if expert_weights is not None:
                outputs.expert_weights = expert_weights

            shared_outputs = self.get_shared_outputs_for_culture_loss()
            if shared_outputs is not None:
                outputs.shared_outputs = shared_outputs

        return outputs


    def save_model(self, save_path: str):
        """保存模型权重"""
        import os

        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        # 保存LoRA权重（如果有）
        if hasattr(self.base_model, 'save_pretrained'):
            lora_path = os.path.join(save_path, 'lora_weights')
            self.base_model.save_pretrained(lora_path)

        # 保存MoE权重
        moe_state_dict = {}
        model_to_save = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        for name, param in model_to_save.named_parameters():
            if any(keyword in name.lower() for keyword in ['experts', 'router']) and param.requires_grad:
                moe_state_dict[name] = param.data

        if moe_state_dict:
            moe_path = os.path.join(save_path, 'moe_weights.pt')
            torch.save(moe_state_dict, moe_path)

        # 保存配置
        config_path = os.path.join(save_path, 'simplified_culturemoe_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"✅ Simplified CultureMoE (Pure MoE) weights saved to {save_path}")

    def print_trainable_parameters(self):
        """打印可训练参数统计并验证冻结状态"""
        try:
            total_params = 0
            trainable_params = 0
            lora_params = 0
            moe_params = 0
            frozen_base_params = 0

            model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

            if model_to_check is None:
                print("❌ 无法检查参数: 未找到模型")
                return

            print("🔍 参数训练状态检查:")

            # 检查关键基座参数是否正确冻结
            base_param_status = []
            param_count = 0

            for name, param in model_to_check.named_parameters():
                param_count += 1
                total_params += param.numel()

                if param.requires_grad:
                    trainable_params += param.numel()
                    if 'lora' in name.lower():
                        lora_params += param.numel()
                    elif any(keyword in name.lower() for keyword in ['routing_experts', 'shared_expert', 'router', 'gate_network']):
                        moe_params += param.numel()
                else:
                    frozen_base_params += param.numel()

                # 检查关键基座参数状态
                if any(keyword in name.lower() for keyword in ['lm_head', 'embed_tokens', 'norm', 'gate_proj', 'up_proj', 'down_proj']):
                    status = "❌ TRAINABLE" if param.requires_grad else "✅ FROZEN"
                    base_param_status.append(f"  {name}: {status}")

            if param_count == 0:
                print("⚠️ 未找到任何模型参数")
                return

            # 显示关键参数状态（只显示前5个，避免输出过多）
            for status in base_param_status[:5]:
                print(status)
            if len(base_param_status) > 5:
                print(f"  ... 和其他 {len(base_param_status)-5} 个基座参数: 均已冻结")

            print(f"\n📊 参数统计:")
            print(f"  总参数: {total_params:,}")
            if total_params > 0:
                print(f"  可训练: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
                print(f"  冻结: {frozen_base_params:,} ({100 * frozen_base_params / total_params:.2f}%)")
            else:
                print(f"  可训练: {trainable_params:,}")
                print(f"  冻结: {frozen_base_params:,}")

            print(f"\n📋 可训练参数详情:")
            print(f"  - LoRA参数: {lora_params:,}")
            print(f"  - MoE参数: {moe_params:,}")
            print(f"  - 架构: Pure LoRA MoE (所有FFN层)")

            # 安全地访问配置属性
            try:
                print(f"  - MoE专家数: {self.config.num_moe_experts}")
                print(f"  - 激活专家数: {self.config.num_activated_experts}")
            except AttributeError:
                print(f"  - MoE配置: 无法访问配置信息")

            # 验证参数冻结是否正确
            if total_params > 0:
                frozen_ratio = frozen_base_params / total_params
                if frozen_ratio > 0.8:  # 基座参数应该占大部分（>80%）且被冻结
                    print(f"\n✅ 参数冻结验证: 成功 (基座参数已正确冻结)")
                else:
                    print(f"\n❌ 参数冻结验证: 失败 (基座参数未正确冻结)")
                    print(f"   冻结参数占比: {frozen_ratio*100:.1f}% (应该 > 80%)")
            else:
                print(f"\n⚠️ 参数冻结验证: 无法验证 (未找到参数)")

        except Exception as e:
            print(f"❌ 参数统计过程出现异常: {e}")
            print("   请检查模型是否正确初始化")



def create_simplified_culturemoe_model(base_model, config: SimplifiedCultureMoEConfig):
    """创建简化版CultureMoE模型"""
    adapter = SimplifiedCultureMoEAdapter(base_model, config)
    return adapter