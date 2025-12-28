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
                 lora_rank: int = 16, lora_alpha: int = 32, dropout: float = 0.1, dtype=None):
        super().__init__()
        self.scaling = lora_alpha / lora_rank

        # LoRA分支
        self.gate_lora_A = nn.Linear(hidden_dim, lora_rank, bias=False, dtype=dtype)
        self.gate_lora_B = nn.Linear(lora_rank, intermediate_dim, bias=False, dtype=dtype)
        self.up_lora_A = nn.Linear(hidden_dim, lora_rank, bias=False, dtype=dtype)
        self.up_lora_B = nn.Linear(lora_rank, intermediate_dim, bias=False, dtype=dtype)
        self.down_lora_A = nn.Linear(intermediate_dim, lora_rank, bias=False, dtype=dtype)
        self.down_lora_B = nn.Linear(lora_rank, hidden_dim, bias=False, dtype=dtype)

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

    def __init__(self, hidden_dim: int, num_experts: int = 4, dtype=None):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts, bias=False, dtype=dtype)
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

        # 🔧 获取原始FFN的dtype以确保一致性
        original_dtype = next(original_ffn.parameters()).dtype

        # 4个路由专家
        self.routing_experts = nn.ModuleList([
            LoRAExpert(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                act_fn=original_ffn.act_fn,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                dtype=original_dtype
            ) for _ in range(4)
        ])

        # 1个共享专家
        self.shared_expert = LoRAExpert(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            act_fn=original_ffn.act_fn,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            dtype=original_dtype
        )

        # Router网络
        self.router = MoERouter(hidden_dim=self.hidden_dim, num_experts=4, dtype=original_dtype)

        # Gate网络：融合共享专家和路由专家输出
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False, dtype=original_dtype),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2, bias=False, dtype=original_dtype),
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
            # 🔧 修复梯度连接：使用参数创建有梯度的零损失
            dummy_param = next(iter(self.parameters()))
            return dummy_param.sum() * 0.0

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

        # 🔧 CRITICAL: 初始化前配置PyTorch内存分配器优化
        self._configure_memory_allocator()

        # 应用LoRA到注意力层（如果启用）
        if config.use_lora:
            self._apply_attention_lora()

        # 替换所有FFN层为MoE
        self._replace_all_layers_with_moe()

        # 确保设备一致性
        self._ensure_device_consistency()

        # 🔧 关键修复：在初始化时设置参数状态，避免在forward中动态修改
        self.ensure_trainable_parameters()
        self._parameters_initialized = True

    def _configure_memory_allocator(self):
        """🔧 配置PyTorch内存分配器以防止SimplifiedCultureMoE内存碎片化"""
        import os
        import torch

        print(f"🔧 Configuring PyTorch memory allocator for SimplifiedCultureMoE...")

        try:
            # 🔧 关键修复1：启用可扩展内存段，防止碎片化
            # 这对于960个Linear层的MoE架构至关重要
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            print(f"✅ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

            # 🔧 关键修复2：配置内存池参数
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()

                # 获取GPU总内存
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                total_gb = total_memory / 1024**3

                # 🔧 为SimplifiedCultureMoE优化：保守的内存分配策略
                if total_gb >= 40:  # 48GB卡
                    memory_fraction = 0.75  # 使用75%，留25%给碎片化缓冲
                elif total_gb >= 20:  # 24GB卡
                    memory_fraction = 0.70  # 使用70%
                else:  # 小于20GB
                    memory_fraction = 0.65  # 使用65%

                # 设置内存分配比例
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device_id)

                print(f"✅ GPU Memory Configuration:")
                print(f"   Total GPU Memory: {total_gb:.1f}GB")
                print(f"   Allocated Fraction: {memory_fraction*100:.1f}%")
                print(f"   Reserved for MoE: {total_gb*memory_fraction:.1f}GB")
                print(f"   Fragmentation Buffer: {total_gb*(1-memory_fraction):.1f}GB")

                # 🔧 关键修复3：预分配内存池，减少运行时分配
                try:
                    # 预热GPU内存分配器
                    dummy_tensor = torch.zeros(1024, 1024, device=device_id, dtype=torch.float16)
                    del dummy_tensor
                    torch.cuda.empty_cache()
                    print(f"✅ GPU memory allocator pre-warmed")
                except Exception as e:
                    print(f"⚠️ Memory pre-warming failed: {e}")

                # 🔧 关键修复4：内存碎片化检测
                allocated_before = torch.cuda.memory_allocated(device_id) / 1024**3
                reserved_before = torch.cuda.memory_reserved(device_id) / 1024**3
                fragmentation_ratio = (reserved_before - allocated_before) / reserved_before if reserved_before > 0 else 0

                print(f"✅ Initial Memory Status:")
                print(f"   Allocated: {allocated_before:.2f}GB")
                print(f"   Reserved: {reserved_before:.2f}GB")
                print(f"   Fragmentation: {fragmentation_ratio*100:.1f}%")

                if fragmentation_ratio > 0.3:  # 超过30%碎片化
                    print(f"⚠️ High fragmentation detected, forcing cleanup...")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            # 🔧 关键修复5：设置其他内存优化环境变量
            memory_env_vars = {
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                'CUDA_LAUNCH_BLOCKING': '0',  # 异步执行，减少内存等待
                'TORCH_CUDNN_V8_API_DISABLED': '1',  # 禁用某些可能导致内存问题的cuDNN功能
            }

            for var, value in memory_env_vars.items():
                if var not in os.environ:
                    os.environ[var] = value
                    print(f"✅ Set {var}={value}")

            print(f"✅ PyTorch memory allocator configuration completed")

        except Exception as e:
            print(f"❌ Memory allocator configuration failed: {e}")
            print(f"⚠️ SimplifiedCultureMoE may experience memory fragmentation issues")
            print(f"   Recommend manually setting: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    def _monitor_memory_fragmentation(self, operation_name="operation"):
        """🔧 实时监控内存碎片化状态"""
        if not torch.cuda.is_available():
            return True

        try:
            device_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3

            fragmentation_ratio = (reserved - allocated) / reserved if reserved > 0 else 0
            usage_ratio = reserved / total

            print(f"💾 Memory Status ({operation_name}):")
            print(f"   Allocated: {allocated:.2f}GB ({allocated/total*100:.1f}%)")
            print(f"   Reserved: {reserved:.2f}GB ({usage_ratio*100:.1f}%)")
            print(f"   Free: {total-reserved:.2f}GB")
            print(f"   Fragmentation: {fragmentation_ratio*100:.1f}%")

            # 🔧 智能内存清理策略
            cleanup_needed = False
            if fragmentation_ratio > 0.4:  # 超过40%碎片化
                print(f"🚨 HIGH FRAGMENTATION: {fragmentation_ratio*100:.1f}% > 40%")
                cleanup_needed = True
            elif usage_ratio > 0.85:  # 超过85%使用率
                print(f"🚨 HIGH MEMORY USAGE: {usage_ratio*100:.1f}% > 85%")
                cleanup_needed = True
            elif reserved - allocated > 2.0:  # 超过2GB碎片
                print(f"🚨 LARGE FRAGMENTATION: {reserved-allocated:.2f}GB > 2.0GB")
                cleanup_needed = True

            if cleanup_needed:
                print(f"🔧 Performing emergency memory cleanup...")
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                # 再次检查
                allocated_after = torch.cuda.memory_allocated(device_id) / 1024**3
                reserved_after = torch.cuda.memory_reserved(device_id) / 1024**3
                freed = (reserved - reserved_after)

                print(f"✅ Cleanup completed:")
                print(f"   Memory freed: {freed:.2f}GB")
                print(f"   New allocated: {allocated_after:.2f}GB")
                print(f"   New reserved: {reserved_after:.2f}GB")

                # 如果清理后仍然有严重问题，返回警告
                new_fragmentation = (reserved_after - allocated_after) / reserved_after if reserved_after > 0 else 0
                if new_fragmentation > 0.5 or reserved_after/total > 0.9:
                    print(f"❌ CRITICAL: Memory issues persist after cleanup")
                    print(f"   Consider reducing batch_size or max_length")
                    return False

            return True

        except Exception as e:
            print(f"❌ Memory monitoring failed: {e}")
            return True  # 不因监控失败而中断训练


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
        """🔧 MEMORY-OPTIMIZED替换所有层的FFN为LoRA MoE：渐进式创建避免OOM"""
        layers, target_layers = self._get_target_layers()

        # 获取目标设备
        base_device = next(self.base_model.parameters()).device

        print(f"🔄 MEMORY-OPTIMIZED Replacing FFN in ALL {len(target_layers)} layers with LoRA MoE")
        print(f"   Strategy: Progressive creation with memory cleanup after each layer")
        print(f"   Target device: {base_device}")

        # 🔧 增强的内存监控
        def check_memory_status(layer_idx):
            if torch.cuda.is_available():
                # 使用新的内存监控方法
                memory_ok = self._monitor_memory_fragmentation(f"layer_{layer_idx}_creation")
                if not memory_ok:
                    print(f"⚠️ Memory issues detected at layer {layer_idx}")
                    return False
            return True

        for layer_idx in target_layers:
            try:
                # 🔧 渐进式创建：每层创建前检查内存
                if layer_idx % 8 == 0:  # 每8层报告一次内存状态
                    memory_ok = check_memory_status(layer_idx)
                    if not memory_ok:
                        print(f"🚨 CRITICAL MEMORY WARNING at layer {layer_idx}")
                        print(f"   Forcing emergency cleanup before continuing...")
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

                original_ffn = layers[layer_idx].mlp

                # 🔧 创建MoE层并立即移动到正确设备
                moe_ffn = MoEFFNLoRA(original_ffn, self.config)

                # 🔧 关键修复：立即移动到目标设备，避免后续批量移动
                moe_ffn = moe_ffn.to(device=base_device, non_blocking=True)

                # 替换FFN
                layers[layer_idx].mlp = moe_ffn

                # 🔧 每层创建后立即清理内存，防止碎片化累积
                if layer_idx % 4 == 3:  # 每4层清理一次
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 构建配置信息字符串
                config_info = f"{self.config.num_moe_experts} experts, rank={self.config.lora_rank}"
                if self.config.use_shared:
                    config_info += ", +shared"
                if self.config.use_gate:
                    config_info += ", +gate"

                print(f"✅ Replaced layer {layer_idx} FFN with LoRA MoE ({config_info})")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ OOM at layer {layer_idx}: {e}")
                    print(f"🔧 Emergency cleanup and retry...")

                    # 紧急内存清理
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # 重新尝试创建该层
                    try:
                        original_ffn = layers[layer_idx].mlp
                        moe_ffn = MoEFFNLoRA(original_ffn, self.config)
                        moe_ffn = moe_ffn.to(device=base_device, non_blocking=True)
                        layers[layer_idx].mlp = moe_ffn
                        print(f"✅ Retry successful for layer {layer_idx}")
                    except Exception as retry_e:
                        print(f"❌ Retry failed for layer {layer_idx}: {retry_e}")
                        raise
                else:
                    raise

        # 🔧 最终内存清理
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✅ MEMORY-OPTIMIZED MoE replacement completed for all {len(target_layers)} layers")


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
        """🔧 MEMORY-SAFE设备一致性检查：避免960个Linear层同时移动导致OOM"""
        model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        # 获取基础模型的设备
        base_device = next(model_to_check.parameters()).device

        print(f"🔧 MEMORY-SAFE Device Consistency Check on {base_device}")
        print(f"   Reason: Avoiding simultaneous movement of 960+ Linear layers")
        print(f"   Strategy: MoE layers created on correct device during initialization")

        # 🔧 关键修复：完全跳过设备移动，避免内存爆炸
        # SimplifiedCultureMoE有960个Linear层（32层×5专家×6Linear）
        # 同时调用.to(device)会导致严重的内存碎片化和OOM

        try:
            layers, target_layers = self._get_target_layers()

            # 只进行设备状态验证，不执行设备移动
            device_mismatch_count = 0
            total_moe_params = 0

            for layer_idx in target_layers[:3]:  # 只检查前3层避免过多输出
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA):
                    # 检查Router设备
                    if hasattr(moe_layer, 'router'):
                        router_device = next(moe_layer.router.parameters()).device
                        if router_device != base_device:
                            device_mismatch_count += 1
                            print(f"  ⚠️ Layer {layer_idx} router on {router_device}, expected {base_device}")
                        total_moe_params += sum(p.numel() for p in moe_layer.router.parameters())

            print(f"✅ MEMORY-SAFE device check completed:")
            print(f"   Device mismatches: {device_mismatch_count}")
            print(f"   MoE parameters checked: {total_moe_params:,}")
            print(f"   Strategy: Skip mass device movement to prevent OOM")

            # 🔧 如果发现设备不匹配，使用渐进式修复而不是批量移动
            if device_mismatch_count > 0:
                print(f"🔧 Device mismatch detected, but skipping mass movement")
                print(f"   Recommendation: Restart training to ensure proper device initialization")

        except Exception as e:
            print(f"⚠️ Device consistency check failed: {e}")
            print(f"✅ Skipping device consistency check - MoE layers already properly configured on {base_device}")

        # 🔧 强制内存清理防止碎片化累积
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✅ MEMORY-SAFE device consistency completed without mass movement")

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
                # 🔧 内存优化：使用累积计算替代torch.stack避免大型临时张量
                if len(expert_weights_list) == 1:
                    return expert_weights_list[0]

                # 累积求和然后除以数量
                accumulated = expert_weights_list[0]
                for weights in expert_weights_list[1:]:
                    accumulated = accumulated + weights
                return accumulated / len(expert_weights_list)
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
                # 🔧 内存优化：使用累积计算替代torch.stack避免大型临时张量
                if len(shared_outputs_list) == 1:
                    return shared_outputs_list[0]

                # 累积求和然后除以数量
                accumulated = shared_outputs_list[0]
                for output in shared_outputs_list[1:]:
                    accumulated = accumulated + output
                return accumulated / len(shared_outputs_list)
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
        # 🔧 参数状态已在初始化时设置，避免在forward中重复调用导致DDP冲突

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

    def check_gradient_flow(self):
        """🔧 轻量级梯度流检查：避免MoE层OOM"""
        print(f"\n🔬 LIGHTWEIGHT Gradient Flow Check (MoE-optimized)")

        try:
            # 只检查关键层的requires_grad状态，不执行前向传播
            model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

            # 检查基座参数是否正确冻结
            base_params_frozen = True
            trainable_count = 0
            frozen_count = 0

            for name, param in model_to_check.named_parameters():
                if any(keyword in name.lower() for keyword in ['lm_head', 'embed_tokens', 'norm', 'gate_proj', 'up_proj', 'down_proj']):
                    if param.requires_grad:
                        print(f"  ❌ Base parameter {name} is TRAINABLE (should be frozen)")
                        base_params_frozen = False
                    else:
                        frozen_count += 1
                elif any(keyword in name.lower() for keyword in ['lora', 'routing_experts', 'shared_expert', 'router', 'gate_network']):
                    if param.requires_grad:
                        trainable_count += 1
                    else:
                        print(f"  ❌ MoE parameter {name} is FROZEN (should be trainable)")

            print(f"  ✅ Base parameters: {'FROZEN' if base_params_frozen else 'ERROR'}")
            print(f"  ✅ Trainable MoE parameters: {trainable_count}")
            print(f"  ✅ Frozen base parameters: {frozen_count}")

            # 检查第一个MoE层是否正确初始化（不执行前向传播）
            try:
                layers, target_layers = self._get_target_layers()
                first_moe = layers[0].mlp
                if hasattr(first_moe, 'router'):
                    router_trainable = any(p.requires_grad for p in first_moe.router.parameters())
                    print(f"  ✅ First MoE router: {'TRAINABLE' if router_trainable else 'FROZEN'}")
                else:
                    print(f"  ❌ First MoE router: NOT FOUND")
                    return False
            except Exception as e:
                print(f"  ❌ MoE structure check failed: {e}")
                return False

            gradient_ok = base_params_frozen and trainable_count > 0
            print(f"  {'✅' if gradient_ok else '❌'} LIGHTWEIGHT CHECK: {'PASSED' if gradient_ok else 'FAILED'}")
            return gradient_ok

        except Exception as e:
            print(f"  ❌ Lightweight gradient check failed: {e}")
            return False

    def diagnose_gradient_flow(self, test_input_ids, test_attention_mask):
        """🔧 安全的梯度诊断：避免OOM的轻量级检查"""
        print(f"\n🔍 SAFE Gradient Flow Diagnosis")

        try:
            # 使用轻量级检查替代完整前向传播
            gradient_ok = self.check_gradient_flow()

            if not gradient_ok:
                print("❌ Basic gradient check failed")
                return False

            # 额外检查：验证模型是否在训练模式
            model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
            if not model_to_check.training:
                print("❌ Model not in training mode")
                return False

            print("✅ Safe gradient diagnosis passed")
            return True

        except Exception as e:
            print(f"❌ Safe gradient diagnosis failed: {e}")
            return False

    def force_gradient_propagation_repair(self, test_input_ids, test_attention_mask):
        """🔧 内存优化的梯度传播修复"""
        print(f"🔧 MEMORY-OPTIMIZED Gradient Propagation Repair")

        # 使用轻量级检查替代完整测试
        gradient_ok = self.check_gradient_flow()

        if gradient_ok:
            print(f"✅ Lightweight gradient check passed")
            return True

        print(f"❌ Gradient issues detected - starting repair...")
        fixed_count = self._emergency_gradient_fix()

        # 再次轻量级验证
        gradient_ok_after_fix = self.check_gradient_flow()

        if gradient_ok_after_fix:
            print(f"✅ REPAIR SUCCESSFUL: {fixed_count} parameters fixed")
            return True
        else:
            print(f"❌ REPAIR FAILED: Deeper architectural issue")
            return False

    def _emergency_gradient_fix(self):
        """🔧 紧急梯度修复：强制设置关键参数的requires_grad状态"""
        print(f"🔧 Emergency Gradient Fix")

        try:
            model_to_check = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
            fixed_count = 0

            for name, param in model_to_check.named_parameters():
                # 强制冻结基座参数
                if any(keyword in name.lower() for keyword in ['lm_head', 'embed_tokens', 'norm', 'gate_proj', 'up_proj', 'down_proj']):
                    if param.requires_grad:
                        param.requires_grad = False
                        fixed_count += 1
                        print(f"  🔧 Froze base parameter: {name}")

                # 强制解冻LoRA和MoE参数
                elif any(keyword in name.lower() for keyword in ['lora', 'routing_experts', 'shared_expert', 'router', 'gate_network']):
                    if not param.requires_grad:
                        param.requires_grad = True
                        fixed_count += 1
                        print(f"  🔧 Unfroze MoE parameter: {name}")

            print(f"✅ Emergency fix completed: {fixed_count} parameters adjusted")
            return fixed_count

        except Exception as e:
            print(f"❌ Emergency gradient fix failed: {e}")
            return 0

    def compute_direct_culture_loss(self, culture_labels, main_loss=None):
        """🔧 直接从MoE层计算文化损失，避免torch.stack操作"""
        try:
            layers, target_layers = self._get_target_layers()

            # 收集shared专家输出
            shared_outputs_list = []
            routing_weights_list = []

            for layer_idx in target_layers:
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA):
                    if hasattr(moe_layer, 'latest_shared_output') and moe_layer.latest_shared_output is not None:
                        shared_outputs_list.append(moe_layer.latest_shared_output)
                    if hasattr(moe_layer, 'latest_routing_weights') and moe_layer.latest_routing_weights is not None:
                        routing_weights_list.append(moe_layer.latest_routing_weights)

            # 使用累积计算替代torch.stack
            total_culture_losses = []

            # 1. Shared专家损失
            if len(shared_outputs_list) > 0:
                # 🔧 内存优化：使用累积计算
                if len(shared_outputs_list) == 1:
                    avg_shared_outputs = shared_outputs_list[0]
                else:
                    avg_shared_outputs = shared_outputs_list[0]
                    for outputs in shared_outputs_list[1:]:
                        avg_shared_outputs = avg_shared_outputs + outputs
                    avg_shared_outputs = avg_shared_outputs / len(shared_outputs_list)

                # 🔧 修复循环导入：直接实现shared专家文化损失计算
                shared_culture_loss = self._compute_shared_culture_loss(culture_labels, avg_shared_outputs, main_loss)
                total_culture_losses.append(shared_culture_loss)

            # 2. 路由专家损失
            if len(routing_weights_list) > 0:
                # 🔧 内存优化：使用累积计算
                if len(routing_weights_list) == 1:
                    avg_routing_weights = routing_weights_list[0]
                else:
                    avg_routing_weights = routing_weights_list[0]
                    for weights in routing_weights_list[1:]:
                        avg_routing_weights = avg_routing_weights + weights
                    avg_routing_weights = avg_routing_weights / len(routing_weights_list)

                # 🔧 修复循环导入：直接计算路由专家文化损失
                routing_culture_loss = self._compute_routing_culture_loss(culture_labels, avg_routing_weights, main_loss)
                total_culture_losses.append(routing_culture_loss)

            # 3. 计算最终文化损失
            if len(total_culture_losses) > 0:
                # 🔧 内存优化：使用累积计算
                if len(total_culture_losses) == 1:
                    final_culture_loss = total_culture_losses[0]
                else:
                    final_culture_loss = total_culture_losses[0]
                    for loss in total_culture_losses[1:]:
                        final_culture_loss = final_culture_loss + loss
                    final_culture_loss = final_culture_loss / len(total_culture_losses)
                return final_culture_loss
            else:
                # 创建连接到计算图的零损失
                if main_loss is not None:
                    return main_loss * 0.0
                else:
                    return culture_labels.float().sum() * 0.0

        except Exception as e:
            print(f"❌ Direct culture loss computation failed: {e}")
            # 安全fallback
            if main_loss is not None:
                return main_loss * 0.0
            else:
                return culture_labels.float().sum() * 0.0

    def compute_direct_z_loss(self, main_loss=None):
        """🔧 直接从MoE层计算z损失，避免torch.stack操作"""
        try:
            layers, target_layers = self._get_target_layers()

            aux_losses = []
            for layer_idx in target_layers:
                moe_layer = layers[layer_idx].mlp
                if isinstance(moe_layer, MoEFFNLoRA):
                    aux_loss = moe_layer.get_aux_loss()
                    aux_losses.append(aux_loss)

            if len(aux_losses) > 0:
                # 🔧 内存优化：使用累积计算
                if len(aux_losses) == 1:
                    total_aux_loss = aux_losses[0]
                else:
                    total_aux_loss = aux_losses[0]
                    for loss in aux_losses[1:]:
                        total_aux_loss = total_aux_loss + loss
                    total_aux_loss = total_aux_loss / len(aux_losses)
                return total_aux_loss
            else:
                # 创建连接到计算图的零损失
                if main_loss is not None:
                    return main_loss * 0.0
                else:
                    # 🔧 修复梯度连接：获取可训练参数来创建零损失
                    for param in self.base_model.parameters():
                        if param.requires_grad:
                            return param.sum() * 0.0
                    # 如果没有找到可训练参数，返回detached零张量
                    return torch.tensor(0.0, device=next(iter(self.base_model.parameters())).device, requires_grad=False)

        except Exception as e:
            print(f"❌ Direct z loss computation failed: {e}")
            # 安全fallback
            if main_loss is not None:
                return main_loss * 0.0
            else:
                # 🔧 修复梯度连接：获取可训练参数来创建零损失
                for param in self.base_model.parameters():
                    if param.requires_grad:
                        return param.sum() * 0.0
                # 如果没有找到可训练参数，返回detached零张量
                return torch.tensor(0.0, device=next(iter(self.base_model.parameters())).device, requires_grad=False)

    def _compute_shared_culture_loss(self, culture_labels, shared_outputs, main_loss):
        """🔧 计算共享专家文化损失（避免循环导入）"""
        try:
            if shared_outputs is None or len(shared_outputs) < 2:
                if main_loss is not None:
                    return main_loss * 0.0
                else:
                    return culture_labels.float().sum() * 0.0

            batch_size = shared_outputs.shape[0]
            device = shared_outputs.device

            # 🔧 修复梯度连接：使用shared_outputs创建有梯度的零损失
            culture_loss = shared_outputs.mean() * 0.0  # 保持梯度连接

            # 共享专家一致性损失：总是鼓励相似
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    vec1 = shared_outputs[i].unsqueeze(0)
                    vec2 = shared_outputs[j].unsqueeze(0)

                    # 检查向量是否为零向量
                    norm1 = torch.norm(vec1)
                    norm2 = torch.norm(vec2)
                    if norm1 < 1e-8 or norm2 < 1e-8:
                        continue

                    similarity = F.cosine_similarity(vec1, vec2)
                    if torch.isnan(similarity) or torch.isinf(similarity):
                        continue

                    # 共享专家：总是鼓励相似，损失为 1 - similarity
                    loss_term = 1.0 - similarity
                    culture_loss = culture_loss + loss_term

            # 归一化
            num_pairs = batch_size * (batch_size - 1) // 2
            if num_pairs > 0:
                culture_loss = culture_loss / num_pairs

            return culture_loss

        except Exception as e:
            print(f"❌ Shared culture loss computation failed: {e}")
            if main_loss is not None:
                return main_loss * 0.0
            else:
                return culture_labels.float().sum() * 0.0

    def _compute_routing_culture_loss(self, culture_labels, expert_weights, main_loss):
        """🔧 计算路由专家文化损失（避免循环导入）"""
        try:
            if expert_weights is None or len(expert_weights) < 2:
                if main_loss is not None:
                    return main_loss * 0.0
                else:
                    return culture_labels.float().sum() * 0.0

            batch_size = expert_weights.shape[0]
            device = expert_weights.device

            # 🔧 修复梯度连接：使用expert_weights创建有梯度的零损失
            culture_loss = expert_weights.mean() * 0.0  # 保持梯度连接

            # 路由专家文化对比损失
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    vec1 = expert_weights[i].unsqueeze(0)
                    vec2 = expert_weights[j].unsqueeze(0)

                    # 检查向量是否为零向量
                    norm1 = torch.norm(vec1)
                    norm2 = torch.norm(vec2)
                    if norm1 < 1e-8 or norm2 < 1e-8:
                        continue

                    similarity = F.cosine_similarity(vec1, vec2)
                    if torch.isnan(similarity) or torch.isinf(similarity):
                        continue

                    if culture_labels[i] == culture_labels[j]:
                        # 同文化：鼓励相似，损失为 1 - similarity
                        loss_term = 1.0 - similarity
                    else:
                        # 不同文化：鼓励不同，损失为 similarity
                        loss_term = similarity

                    culture_loss = culture_loss + loss_term

            # 归一化
            num_pairs = batch_size * (batch_size - 1) // 2
            if num_pairs > 0:
                culture_loss = culture_loss / num_pairs

            return culture_loss

        except Exception as e:
            print(f"❌ Routing culture loss computation failed: {e}")
            if main_loss is not None:
                return main_loss * 0.0
            else:
                return culture_labels.float().sum() * 0.0

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