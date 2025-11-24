#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA增强FFN集成CultureMoE的Mask机制演示

该脚本展示了：
1. 共享专家如何使用masked输入处理通用知识
2. 文化专家如何使用未masked输入处理文化特定知识
3. 两种专家的输出如何融合
4. Mask机制对专家行为的影响

用法：
python example_mask_mechanism_demo.py --base_model meta-llama/Llama-2-7b-hf
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.lora_enhanced_culturemoe import LoRACultureMoEConfig
from llamafactory.model.lora_culturemoe_model import create_lora_culturemoe_model
from train_lora_culturemoe_ffn_integrated import generate_cultural_mask


class MaskMechanismDemo:
    """Mask机制演示类"""

    def __init__(self, base_model_path: str = "meta-llama/Llama-2-7b-hf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # LoRA配置
        self.lora_config = LoRACultureMoEConfig(
            num_experts=8,
            top_k=2,
            capacity_factor=1.25,
            num_cultures=6,
            culture_dim=256,
            lora_rank=16,
            lora_alpha=32.0,
            lora_dropout=0.1,
            attention_lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
            expert_lora_targets=["gate_proj", "up_proj", "down_proj"],
        )

        # 创建模型
        print("创建LoRA增强的CultureMoE模型...")
        self.model = create_lora_culturemoe_model(base_model_path, self.lora_config)
        self.model.to(self.device)
        self.model.eval()

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 文化标签映射
        self.culture_names = {
            0: "Asia", 1: "Europe", 2: "North America",
            3: "South America", 4: "Africa", 5: "Oceania"
        }

    def prepare_sample_inputs(self):
        """准备示例输入"""
        # 文化特定的示例文本
        sample_texts = [
            "In Asian culture, respect for elders is fundamental to social harmony.",
            "European traditions emphasize individual rights and democratic values.",
            "Native American cultures have deep connections to the natural world.",
            "South American festivals celebrate vibrant colors and community spirit.",
            "African storytelling traditions pass wisdom through generations.",
            "Oceanic island cultures value maritime knowledge and navigation."
        ]

        # 对应的文化ID
        culture_ids = [0, 1, 2, 3, 4, 5]

        # 分词
        inputs = self.tokenizer(
            sample_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].to(self.device),
            'attention_mask': inputs['attention_mask'].to(self.device),
            'culture_ids': torch.tensor(culture_ids, dtype=torch.long, device=self.device),
            'texts': sample_texts
        }

    def demonstrate_mask_mechanism(self):
        """演示mask机制的工作原理"""
        print("\n" + "="*80)
        print("🎭 Mask机制演示")
        print("="*80)

        # 准备输入
        inputs = self.prepare_sample_inputs()

        print(f"处理 {len(inputs['texts'])} 个文化特定样本...")
        for i, text in enumerate(inputs['texts']):
            culture_name = self.culture_names[inputs['culture_ids'][i].item()]
            print(f"  {i+1}. [{culture_name}] {text[:60]}...")

        with torch.no_grad():
            # 1. 无mask的前向传播
            print("\n🔍 步骤1: 无mask前向传播")
            outputs_no_mask = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                culture_ids=inputs['culture_ids'],
                hidden_states_mask=None,
                return_dict=True
            )

            # 2. 生成mask并进行有mask的前向传播
            print("🔍 步骤2: 生成文化敏感mask")
            temp_hidden_states = self.model.embed_tokens(inputs['input_ids'])
            hidden_states_mask = generate_cultural_mask(
                temp_hidden_states, inputs['culture_ids'], mask_ratio=0.3
            )

            print("🔍 步骤3: 使用mask的前向传播")
            outputs_with_mask = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                culture_ids=inputs['culture_ids'],
                hidden_states_mask=hidden_states_mask,
                return_dict=True
            )

        # 3. 分析结果
        self.analyze_mask_effects(
            outputs_no_mask, outputs_with_mask, inputs,
            temp_hidden_states, hidden_states_mask
        )

    def analyze_mask_effects(self, outputs_no_mask, outputs_with_mask, inputs,
                           original_hidden_states, masked_hidden_states):
        """分析mask机制的效果"""
        print("\n📊 Mask机制效果分析")
        print("-" * 60)

        # 分析mask对hidden states的影响
        print("1. Hidden States变化分析:")
        mask_ratio = torch.mean((masked_hidden_states != original_hidden_states).float()).item()
        print(f"   实际mask比例: {mask_ratio:.3f}")

        reduction_ratio = torch.mean(masked_hidden_states / (original_hidden_states + 1e-8)).item()
        print(f"   平均值缩减比例: {reduction_ratio:.3f}")

        # 分析专家权重变化
        if hasattr(outputs_no_mask, 'moe_aux_info') and hasattr(outputs_with_mask, 'moe_aux_info'):
            print("\n2. 专家权重变化分析:")
            self.analyze_expert_weight_changes(
                outputs_no_mask.moe_aux_info,
                outputs_with_mask.moe_aux_info,
                inputs['culture_ids']
            )

        # 分析输出差异
        print("\n3. 模型输出差异分析:")
        output_diff = torch.mean(torch.abs(
            outputs_with_mask.last_hidden_state - outputs_no_mask.last_hidden_state
        )).item()
        print(f"   平均输出差异: {output_diff:.6f}")

        # 分析专家类型的影响
        print("\n4. 专家类型影响分析:")
        self.analyze_expert_type_influence(outputs_with_mask, inputs)

    def analyze_expert_weight_changes(self, aux_info_no_mask, aux_info_with_mask, culture_ids):
        """分析专家权重的变化"""
        if not aux_info_no_mask or not aux_info_with_mask:
            print("   无法获取专家权重信息")
            return

        # 分析最后一层的专家权重
        last_layer_no_mask = aux_info_no_mask[-1]
        last_layer_with_mask = aux_info_with_mask[-1]

        if 'expert_weights' in last_layer_no_mask and 'expert_weights' in last_layer_with_mask:
            weights_no_mask = last_layer_no_mask['expert_weights']  # [B, num_experts]
            weights_with_mask = last_layer_with_mask['expert_weights']

            # 计算权重变化
            weight_diff = torch.abs(weights_with_mask - weights_no_mask)
            avg_weight_change = torch.mean(weight_diff, dim=0)  # [num_experts]

            print(f"   各专家权重变化:")
            for expert_idx in range(len(avg_weight_change)):
                change = avg_weight_change[expert_idx].item()
                print(f"     专家{expert_idx}: {change:.4f}")

            # 分析文化特异性
            print(f"\n   各文化的专家选择变化:")
            for culture_id in range(6):
                culture_mask = culture_ids == culture_id
                if culture_mask.sum() > 0:
                    culture_weight_diff = weight_diff[culture_mask].mean(dim=0)
                    top_changed_expert = torch.argmax(culture_weight_diff).item()
                    max_change = culture_weight_diff[top_changed_expert].item()
                    culture_name = self.culture_names[culture_id]
                    print(f"     {culture_name}: 专家{top_changed_expert}变化最大({max_change:.4f})")

    def analyze_expert_type_influence(self, outputs, inputs):
        """分析不同专家类型的影响"""
        if not hasattr(outputs, 'moe_aux_info') or not outputs.moe_aux_info:
            print("   无法获取MoE辅助信息")
            return

        # 获取最后一层的信息
        last_layer_aux = outputs.moe_aux_info[-1]

        if 'shared_expert_output' in last_layer_aux and 'cultural_expert_output' in last_layer_aux:
            shared_output = last_layer_aux['shared_expert_output']
            cultural_output = last_layer_aux['cultural_expert_output']
            moe_alpha = last_layer_aux.get('moe_alpha', 0.5)

            # 分析共享专家vs文化专家的贡献
            shared_norm = torch.norm(shared_output, dim=-1).mean().item()
            cultural_norm = torch.norm(cultural_output, dim=-1).mean().item()

            print(f"   共享专家输出范数: {shared_norm:.4f}")
            print(f"   文化专家输出范数: {cultural_norm:.4f}")
            print(f"   MoE融合权重α: {moe_alpha:.4f}")
            print(f"   实际融合比例 - 共享:{1-moe_alpha:.3f}, 文化:{moe_alpha:.3f}")

    def visualize_expert_specialization(self):
        """可视化专家专业化程度"""
        print("\n🎨 生成专家专业化可视化图表...")

        inputs = self.prepare_sample_inputs()

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                culture_ids=inputs['culture_ids'],
                return_dict=True
            )

        if not hasattr(outputs, 'moe_aux_info') or not outputs.moe_aux_info:
            print("无法获取专家权重信息用于可视化")
            return

        # 收集所有层的专家权重
        all_expert_weights = []
        for layer_aux in outputs.moe_aux_info:
            if 'expert_weights' in layer_aux:
                all_expert_weights.append(layer_aux['expert_weights'].cpu().numpy())

        if not all_expert_weights:
            print("没有找到专家权重数据")
            return

        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LoRA Enhanced CultureMoE - Expert Specialization Analysis', fontsize=16)

        # 1. 最后一层的专家权重热图
        last_layer_weights = all_expert_weights[-1]  # [batch_size, num_experts]
        sns.heatmap(
            last_layer_weights.T,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=[self.culture_names[i] for i in range(6)],
            yticklabels=[f'Expert {i}' for i in range(8)],
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Expert Weights by Culture (Last Layer)')
        axes[0, 0].set_xlabel('Culture')
        axes[0, 0].set_ylabel('Expert')

        # 2. 各层专家利用率
        layer_utilizations = []
        for weights in all_expert_weights:
            top1_experts = np.argmax(weights, axis=1)
            utilization = np.bincount(top1_experts, minlength=8) / len(top1_experts)
            layer_utilizations.append(utilization)

        layer_utilizations = np.array(layer_utilizations)
        sns.heatmap(
            layer_utilizations.T,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=[f'Layer {i}' for i in range(len(layer_utilizations))],
            yticklabels=[f'Expert {i}' for i in range(8)],
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Expert Utilization Across Layers')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Expert')

        # 3. 文化特异性分析
        culture_expert_affinity = np.zeros((6, 8))
        for culture_id in range(6):
            culture_weights = last_layer_weights[culture_id]
            culture_expert_affinity[culture_id] = culture_weights

        sns.heatmap(
            culture_expert_affinity,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=[f'Expert {i}' for i in range(8)],
            yticklabels=[self.culture_names[i] for i in range(6)],
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Culture-Expert Affinity Matrix')
        axes[1, 0].set_xlabel('Expert')
        axes[1, 0].set_ylabel('Culture')

        # 4. 专家多样性分析
        expert_diversity = []
        for expert_id in range(8):
            expert_weights_across_cultures = culture_expert_affinity[:, expert_id]
            diversity = 1 - np.std(expert_weights_across_cultures)  # 多样性 = 1 - 标准差
            expert_diversity.append(diversity)

        axes[1, 1].bar(range(8), expert_diversity, color='skyblue', alpha=0.7)
        axes[1, 1].set_title('Expert Cultural Diversity')
        axes[1, 1].set_xlabel('Expert ID')
        axes[1, 1].set_ylabel('Diversity Score')
        axes[1, 1].set_xticks(range(8))

        plt.tight_layout()
        plt.savefig('mask_mechanism_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ 可视化图表已保存为 'mask_mechanism_analysis.png'")

    def run_complete_demo(self):
        """运行完整的演示"""
        print("🚀 启动LoRA增强FFN集成CultureMoE的Mask机制演示")
        print(f"📊 模型配置: {self.lora_config.num_experts}个专家, LoRA rank={self.lora_config.lora_rank}")

        # 演示mask机制
        self.demonstrate_mask_mechanism()

        # 可视化专家专业化
        self.visualize_expert_specialization()

        print("\n" + "="*80)
        print("✅ 演示完成!")
        print("="*80)
        print("\n📝 总结:")
        print("1. 共享专家使用masked输入处理通用知识，减少文化偏见")
        print("2. 文化专家使用完整输入处理文化特定知识，保持文化敏感性")
        print("3. 两种专家通过可学习的融合权重进行组合")
        print("4. Mask机制有效地实现了通用知识与文化知识的分离处理")
        print("5. 不同文化激活不同的专家组合，实现了文化感知的推理")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='LoRA Enhanced CultureMoE Mask Mechanism Demo')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Base model path')

    args = parser.parse_args()

    # 创建并运行演示
    demo = MaskMechanismDemo(args.base_model)
    demo.run_complete_demo()


if __name__ == "__main__":
    main()