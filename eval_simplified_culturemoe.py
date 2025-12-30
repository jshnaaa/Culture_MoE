#!/usr/bin/env python3
"""
CultureMoE模型评估脚本
从训练输出目录加载模型并在测试集上进行评估

特点：
1. 完整模型重建：基座模型 + LoRA权重 + MoE权重
2. 测试集评估：使用训练时的8:1:1数据集划分
3. 全面指标：准确率、精确率、召回率、F1分数
4. 内存优化：支持大模型评估
"""

import argparse
import json
import os
import sys
import pickle
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.simplified_culturemoe import SimplifiedCultureMoEConfig
from src.llamafactory.model.simplified_culturemoe_adapter import create_simplified_culturemoe_model

# 复用现有的数据集类
from ft_lora_only_gen import (
    CultureLLMNewFormatDataset,
    extract_answer_from_text,
    generate_answer,
    dynamic_padding_collate_fn
)


class CultureMoEEvaluator:
    """CultureMoE模型评估器"""

    def __init__(self, model_dir: str, base_model_path: str, device: str = "cuda:0"):
        """
        初始化评估器

        Args:
            model_dir: 训练好的模型目录
            base_model_path: 基座模型路径
            device: 使用的设备
        """
        self.model_dir = model_dir
        self.base_model_path = base_model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None

        print(f"初始化CultureMoE评估器...")
        print(f"  模型目录: {model_dir}")
        print(f"  基座模型: {base_model_path}")
        print(f"  设备: {device}")

    def load_config(self) -> SimplifiedCultureMoEConfig:
        """加载模型配置"""
        config_path = os.path.join(self.model_dir, 'simplified_culturemoe_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置文件不存在: {config_path}")

        print(f"加载模型配置: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        self.config = SimplifiedCultureMoEConfig.from_dict(config_dict)
        print(f"✅ 模型配置加载成功")
        print(f"  MoE专家数: {self.config.num_moe_experts}")
        print(f"  激活专家数: {self.config.num_activated_experts}")
        print(f"  共享专家: {self.config.use_shared}")

        return self.config

    def load_base_model_and_tokenizer(self):
        """加载基座模型和分词器"""
        print(f"加载基座模型: {self.base_model_path}")

        # 检测模型类型
        if 'llama' in self.base_model_path.lower():
            backbone = 'llama'
        elif 'qwen' in self.base_model_path.lower():
            backbone = 'qwen'
        else:
            # 尝试从config.json检测
            config_path = os.path.join(self.base_model_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                    model_type = model_config.get('model_type', '').lower()
                    if 'llama' in model_type:
                        backbone = 'llama'
                    elif 'qwen' in model_type:
                        backbone = 'qwen'
                    else:
                        backbone = 'llama'  # 默认
            else:
                backbone = 'llama'  # 默认

        print(f"  检测到模型类型: {backbone}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side='left'
        )

        if self.tokenizer.pad_token is None:
            if backbone == 'qwen':
                self.tokenizer.pad_token = '<|endoftext|>'
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32,
            trust_remote_code=True,
            device_map=None  # 手动管理设备
        )

        print(f"✅ 基座模型和分词器加载成功")
        return base_model, backbone

    def reconstruct_culturemoe_model(self):
        """重建完整的CultureMoE模型"""
        print("重建CultureMoE模型...")

        # 加载配置
        if self.config is None:
            self.load_config()

        # 加载基座模型
        base_model, backbone = self.load_base_model_and_tokenizer()

        # 创建CultureMoE模型适配器
        print("创建CultureMoE适配器...")
        model_adapter = create_simplified_culturemoe_model(
            base_model=base_model,
            config=self.config
        )

        # 🔧 修复：加载LoRA权重（支持多种格式）
        lora_loaded = False

        # 方法1：尝试加载标准PEFT格式
        lora_weights_dir = os.path.join(self.model_dir, 'lora_weights')
        if os.path.exists(lora_weights_dir):
            print(f"🔍 尝试加载标准LoRA权重: {lora_weights_dir}")
            try:
                from peft import PeftModel
                if hasattr(model_adapter.base_model, 'load_adapter'):
                    model_adapter.base_model.load_adapter(lora_weights_dir)
                else:
                    lora_config_path = os.path.join(lora_weights_dir, 'adapter_config.json')
                    if os.path.exists(lora_config_path):
                        model_adapter.base_model = PeftModel.from_pretrained(
                            model_adapter.base_model,
                            lora_weights_dir
                        )
                print("✅ 标准LoRA权重加载成功")
                lora_loaded = True
            except Exception as e:
                print(f"⚠️ 标准LoRA权重加载失败: {e}")

        # 方法2：尝试加载手动保存的LoRA权重
        if not lora_loaded:
            lora_manual_path = os.path.join(self.model_dir, 'lora_weights_manual.pt')
            if os.path.exists(lora_manual_path):
                print(f"🔍 尝试加载手动保存的LoRA权重: {lora_manual_path}")
                try:
                    lora_state_dict = torch.load(lora_manual_path, map_location='cpu')

                    # 加载权重到模型
                    model_to_load = model_adapter.base_model.module if hasattr(model_adapter.base_model, 'module') else model_adapter.base_model

                    loaded_count = 0
                    missing_count = 0

                    for name, param_data in lora_state_dict.items():
                        try:
                            # 查找对应的参数
                            target_param = model_to_load
                            for attr in name.split('.'):
                                target_param = getattr(target_param, attr)

                            if hasattr(target_param, 'data'):
                                target_param.data.copy_(param_data)
                                loaded_count += 1
                                print(f"  ✅ 加载LoRA参数: {name}, dtype: {param_data.dtype}")
                            else:
                                missing_count += 1
                                print(f"  ❌ 找不到参数: {name}")
                        except AttributeError:
                            missing_count += 1
                            print(f"  ❌ 参数路径错误: {name}")

                    print(f"✅ 手动LoRA权重加载完成: 成功{loaded_count}个, 失败{missing_count}个")
                    if loaded_count > 0:
                        lora_loaded = True

                except Exception as e:
                    print(f"⚠️ 手动LoRA权重加载失败: {e}")

        if not lora_loaded:
            print("⚠️ 没有找到任何LoRA权重文件，将只使用MoE权重")

        # 加载MoE权重
        moe_weights_path = os.path.join(self.model_dir, 'moe_weights.pt')
        if os.path.exists(moe_weights_path):
            print(f"加载MoE权重: {moe_weights_path}")
            try:
                moe_state_dict = torch.load(moe_weights_path, map_location='cpu')

                # 加载MoE权重到模型
                model_to_load = model_adapter.base_model.module if hasattr(model_adapter.base_model, 'module') else model_adapter.base_model

                missing_keys = []

                for name, param in moe_state_dict.items():
                    try:
                        # 获取对应的参数
                        target_param = model_to_load
                        for attr in name.split('.'):
                            target_param = getattr(target_param, attr)
                        target_param.data.copy_(param.data)
                    except AttributeError:
                        missing_keys.append(name)

                if missing_keys:
                    print(f"  ⚠️  缺失键: {len(missing_keys)} 个")

                print("✅ MoE权重加载成功")
            except Exception as e:
                print(f"⚠️  MoE权重加载失败: {e}")

        # 移动到指定设备
        model_adapter.to(self.device)
        model_adapter.eval()

        self.model = model_adapter
        print(f"✅ CultureMoE模型重建完成")

        # 打印模型信息
        if hasattr(model_adapter, 'print_trainable_parameters'):
            model_adapter.print_trainable_parameters()

        return model_adapter

    def load_test_dataset(self, data_split_file: str, train_config_file: Optional[str] = None):
        """加载测试数据集"""
        print(f"加载测试数据集: {data_split_file}")

        # 加载数据集划分信息
        with open(data_split_file, 'rb') as f:
            split_info = pickle.load(f)

        test_indices = split_info['test']
        original_data_path = split_info['original_data_path']
        max_length = split_info.get('max_length', 512)

        print(f"  原始数据路径: {original_data_path}")
        print(f"  测试集样本数: {len(test_indices)}")
        print(f"  最大序列长度: {max_length}")

        # 检查原始数据文件是否存在
        if not os.path.exists(original_data_path):
            raise FileNotFoundError(f"原始数据文件不存在: {original_data_path}")

        # 创建完整数据集
        full_dataset = CultureLLMNewFormatDataset(
            data_path=original_data_path,
            tokenizer=self.tokenizer,
            max_length=max_length
        )

        # 创建测试子集
        test_dataset = Subset(full_dataset, test_indices)

        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # 评估时使用小批次
            shuffle=False,
            collate_fn=dynamic_padding_collate_fn,
            pin_memory=True if 'cuda' in self.device else False
        )

        print(f"✅ 测试数据集加载成功")
        return test_loader, len(test_indices)

    @torch.no_grad()
    def evaluate_classification(self, test_loader: DataLoader) -> Dict[str, Any]:
        """评估分类任务"""
        print("开始分类任务评估...")

        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(test_loader, desc="评估进度")

        for batch in progress_bar:
            try:
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # 计算损失
                loss = outputs.loss if hasattr(outputs, 'loss') else 0.0
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else 0.0

                # 获取预测结果
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                # 只取标签位置的预测（避免padding影响）
                label_mask = labels != -100
                valid_predictions = predictions[label_mask]
                valid_labels = labels[label_mask]

                # 如果是生成任务，需要提取答案
                if len(valid_predictions) > 1:  # 生成任务
                    # 解码预测文本
                    pred_text = self.tokenizer.decode(valid_predictions, skip_special_tokens=True)
                    label_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)

                    # 提取答案
                    pred_answer = extract_answer_from_text(pred_text)
                    true_answer = extract_answer_from_text(label_text)

                    all_predictions.append(pred_answer)
                    all_labels.append(true_answer)
                else:  # 分类任务
                    all_predictions.extend(valid_predictions.cpu().numpy())
                    all_labels.extend(valid_labels.cpu().numpy())

                num_batches += 1

                # 更新进度条
                if num_batches % 10 == 0:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

                # 内存清理
                if num_batches % 50 == 0:
                    torch.cuda.empty_cache() if 'cuda' in self.device else None

            except Exception as e:
                print(f"⚠️  批次处理失败: {e}")
                continue

        # 计算指标
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # 处理字符串答案（生成任务）
        if all_predictions and isinstance(all_predictions[0], str):
            # 将字符串答案转换为数字标签
            unique_labels = list(set(all_labels + all_predictions))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

            numeric_predictions = [label_to_idx.get(pred, 0) for pred in all_predictions]
            numeric_labels = [label_to_idx.get(label, 0) for label in all_labels]

            all_predictions = numeric_predictions
            all_labels = numeric_labels

        # 计算分类指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'loss': float(avg_loss),
            'num_samples': len(all_labels),
            'num_batches': num_batches
        }

        # 生成分类报告
        try:
            class_report = classification_report(
                all_labels, all_predictions,
                output_dict=True,
                zero_division=0
            )
            metrics['classification_report'] = class_report
        except Exception as e:
            print(f"⚠️  无法生成分类报告: {e}")

        print(f"✅ 评估完成")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  平均损失: {avg_loss:.4f}")

        return metrics

    def save_results(self, metrics: Dict[str, Any], output_file: str,
                    train_config_file: Optional[str] = None):
        """保存评估结果"""
        print(f"保存评估结果: {output_file}")

        results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'model_dir': self.model_dir,
                'base_model_path': self.base_model_path,
                'device': self.device,
                'evaluator_version': '1.0.0'
            },
            'model_info': {
                'backbone': self.config.backbone if self.config else 'unknown',
                'num_moe_experts': self.config.num_moe_experts if self.config else 'unknown',
                'num_activated_experts': self.config.num_activated_experts if self.config else 'unknown',
                'use_shared': self.config.use_shared if self.config else 'unknown',
                'use_lora': True,  # 假设都使用LoRA
            },
            'test_metrics': metrics,
            'test_samples': metrics.get('num_samples', 0)
        }

        # 添加训练配置信息（如果有）
        if train_config_file and os.path.exists(train_config_file):
            try:
                with open(train_config_file, 'r') as f:
                    train_config = json.load(f)
                results['training_config'] = train_config
            except Exception as e:
                print(f"⚠️  无法加载训练配置: {e}")

        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ 评估结果已保存")


def main():
    parser = argparse.ArgumentParser(description='CultureMoE模型评估脚本')

    # 必需参数
    parser.add_argument('--model_dir', type=str, required=True,
                       help='训练好的模型目录')
    parser.add_argument('--base_model_path', type=str, required=True,
                       help='基座模型路径')
    parser.add_argument('--data_split_file', type=str, required=True,
                       help='数据集划分文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                       help='评估结果输出文件路径')

    # 可选参数
    parser.add_argument('--train_config_file', type=str,
                       help='训练配置文件路径')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='评估批次大小')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='使用的设备')
    parser.add_argument('--memory_efficient', action='store_true',
                       help='启用内存优化模式')

    args = parser.parse_args()

    # 设置内存优化
    if args.memory_efficient:
        print("启用内存优化模式...")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if 'cuda' in args.device:
            torch.cuda.empty_cache()

    try:
        # 创建评估器
        evaluator = CultureMoEEvaluator(
            model_dir=args.model_dir,
            base_model_path=args.base_model_path,
            device=args.device
        )

        # 重建模型
        evaluator.reconstruct_culturemoe_model()

        # 加载测试数据集
        test_loader, num_test_samples = evaluator.load_test_dataset(
            data_split_file=args.data_split_file,
            train_config_file=args.train_config_file
        )

        # 执行评估
        metrics = evaluator.evaluate_classification(test_loader)

        # 保存结果
        evaluator.save_results(
            metrics=metrics,
            output_file=args.output_file,
            train_config_file=args.train_config_file
        )

        print("\n🎉 评估完成！")

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()