#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced CultureMoE 模型评估脚本
从 Base 模型 + LoRA 权重 + Enhanced MoE 权重还原完整模型并评估

功能：
1. 从三个组件还原完整的 Enhanced CultureMoE 模型
2. 在测试集上进行评估
3. 输出详细的评估结果和生成答案
4. 保存模型配置和评估指标

使用方法：
    python eval_enhanced_culturemoe_from_components.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --moe_weights_path /path/to/enhanced_moe_weights \
        --test_file /path/to/test_data.json \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.enhanced_culturemoe import EnhancedCultureMoE
from llamafactory.model.moe_args import ModelArgs
from peft import PeftModel


class EnhancedCultureMoEEvaluator:
    """Enhanced CultureMoE 模型评估器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 设置随机种子
        set_seed(42)

        # 设置日志
        self.setup_logging()

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 保存配置
        self.save_config()

    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.args.output_dir, 'evaluation.log')
        logging.basicConfig(
            level=logging.INFO,  # 恢复INFO级别，减少详细输出
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info("=" * 80)
        logging.info("Enhanced CultureMoE Evaluation Started")
        logging.info("=" * 80)
        logging.info(f"Arguments: {vars(self.args)}")

    def save_config(self):
        """保存评估配置"""
        config = {
            'evaluation_time': datetime.now().isoformat(),
            'base_model_path': self.args.base_model_path,
            'lora_weights_path': self.args.lora_weights_path,
            'moe_weights_path': self.args.moe_weights_path,
            'test_file': self.args.test_file,
            'backbone': self.args.backbone,
            'num_experts': self.args.num_experts,
            'moe_fusion': self.args.moe_fusion,
            'culture_loss_lambda': self.args.culture_loss_lambda,
            'device': str(self.device),
            'use_multi_gpu': self.args.use_multi_gpu
        }

        config_file = os.path.join(self.args.output_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logging.info(f"Configuration saved to {config_file}")

    def load_enhanced_culturemoe_from_components(self) -> Tuple[EnhancedCultureMoE, AutoTokenizer]:
        """
        从三个组件加载完整的 Enhanced CultureMoE 模型

        Returns:
            model: 完整的 Enhanced CultureMoE 模型
            tokenizer: Tokenizer
        """
        logging.info("=" * 80)
        logging.info("Loading Enhanced CultureMoE Model from Components")
        logging.info("=" * 80)
        logging.info(f"Base model: {self.args.base_model_path}")
        logging.info(f"LoRA weights: {self.args.lora_weights_path}")
        logging.info(f"Enhanced MoE weights: {self.args.moe_weights_path}")
        logging.info("=" * 80)

        # 1. 加载 Tokenizer
        logging.info("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.args.lora_weights_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logging.info("   ✅ Tokenizer loaded")

        # 2. 加载 Base 模型
        logging.info("2. Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None  # 手动管理设备
        )
        logging.info("   ✅ Base model loaded")
        logging.info(f"   Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")

        # 3. 加载 LoRA 权重并合并
        logging.info("3. Loading and merging LoRA weights...")
        model_with_lora = PeftModel.from_pretrained(
            base_model,
            self.args.lora_weights_path,
            is_trainable=False
        )
        logging.info("   ✅ LoRA weights loaded")

        # 合并 LoRA 权重到 base 模型
        logging.info("   Merging LoRA weights into base model...")
        merged_model = model_with_lora.merge_and_unload()
        logging.info("   ✅ LoRA weights merged")

        # 4. 创建 Enhanced CultureMoE 模型结构
        logging.info("4. Creating Enhanced CultureMoE model structure...")

        # 创建MoE参数（基于启动脚本的参数）
        moe_args = ModelArgs(
            num_experts=self.args.num_experts,
            shared_hidden_dim=4096,
            router_hidden_dim=2048,
            experts_hidden_dim=4096,
            lora_rank=32,
            dropout=0.05
        )

        # 创建 Enhanced CultureMoE 模型
        enhanced_culturemoe_model = EnhancedCultureMoE(
            llama_model=merged_model,
            config=merged_model.config,
            args=moe_args,
            culture_loss_lambda=self.args.culture_loss_lambda,
            moe_fusion=self.args.moe_fusion,
            num_cultures=6,   # 支持6个大洲
            culture_dim=256   # 文化嵌入维度
        )
        logging.info("   ✅ Enhanced CultureMoE structure created")

        # 5. 加载 Enhanced MoE 权重
        logging.info("5. Loading Enhanced MoE weights...")
        moe_weights_file = os.path.join(self.args.moe_weights_path, "moe_weights.pth")

        if not os.path.exists(moe_weights_file):
            raise FileNotFoundError(f"Enhanced MoE weights not found: {moe_weights_file}")

        # 加载MoE权重
        moe_state_dict = torch.load(moe_weights_file, map_location="cpu")

        # 确保权重类型匹配
        model_dtype = next(merged_model.parameters()).dtype
        for key in moe_state_dict:
            if isinstance(moe_state_dict[key], torch.Tensor):
                moe_state_dict[key] = moe_state_dict[key].to(model_dtype)

        # 加载MoE权重到模型中（只加载MoE相关的权重）
        missing_keys, unexpected_keys = enhanced_culturemoe_model.load_state_dict(moe_state_dict, strict=False)

        # 统计加载的权重
        moe_param_count = sum(p.numel() for name, p in enhanced_culturemoe_model.named_parameters()
                             if any(keyword in name for keyword in [
                                 'cultural_experts', 'router', 'shared', 'cultural_gate',
                                 'cultural_embedding', 'cultural_context', 'culture_loss'
                             ]))

        logging.info(f"   ✅ Enhanced MoE weights loaded ({moe_param_count:,} parameters)")
        if missing_keys:
            logging.info(f"   ⚠️  Missing keys: {len(missing_keys)} (expected for base model)")
        if unexpected_keys:
            logging.info(f"   ⚠️  Unexpected keys: {len(unexpected_keys)}")

        # 6. 强制使用单GPU评估（避免DataParallel兼容性问题）
        logging.info("6. Using single GPU for evaluation...")
        self.use_dataparallel = False
        logging.info("   ✅ Single GPU mode enabled (DataParallel disabled for evaluation)")

        # 7. 移动到设备
        logging.info(f"7. Moving model to {self.device}...")
        enhanced_culturemoe_model = enhanced_culturemoe_model.to(self.device)
        enhanced_culturemoe_model.eval()
        logging.info("   ✅ Model ready for evaluation")

        # 打印模型信息
        total_params = sum(p.numel() for p in enhanced_culturemoe_model.parameters())
        logging.info(f"\n📊 Enhanced CultureMoE Model Info:")
        logging.info(f"   Total parameters: {total_params:,}")
        logging.info(f"   Enhanced MoE parameters: {moe_param_count:,}")
        logging.info(f"   Device: {self.device}")
        logging.info(f"   Multi-GPU: {self.use_dataparallel}")

        logging.info("\n" + "=" * 80)
        logging.info("✅ Enhanced CultureMoE Model Loaded Successfully!")
        logging.info("=" * 80)

        return enhanced_culturemoe_model, tokenizer

    def load_test_data(self, tokenizer) -> List[Dict]:
        """加载测试数据"""
        logging.info(f"Loading test data from: {self.args.test_file}")

        with open(self.args.test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"Loaded {len(data)} test samples")

        # 处理数据格式
        processed_data = []
        for item in data:
            instruction = item.get('instruction', '')
            instruction_mask = item.get('instruction_mask', instruction)
            input_text = item.get('input', '')
            output = item.get('output', '')
            label = item.get('label', '0')

            # 构建完整输入文本
            if input_text and input_text.strip():
                full_text = f"{instruction}\n{input_text}"
                full_text_mask = f"{instruction_mask}\n{input_text}"
            else:
                full_text = instruction
                full_text_mask = instruction_mask

            # 解析大洲标签
            continent_id, continent_ids_multi = self._parse_continent_label(label)

            processed_data.append({
                'instruction': instruction,
                'instruction_mask': instruction_mask,
                'input': input_text,
                'full_text': full_text,
                'full_text_mask': full_text_mask,
                'output': output,
                'label': label,
                'continent_id': continent_id,
                'continent_ids_multi': continent_ids_multi
            })

        logging.info(f"Processed {len(processed_data)} samples for evaluation")
        return processed_data

    def _parse_continent_label(self, label: str) -> Tuple[int, List[int]]:
        """解析大洲标签"""
        # 大洲映射
        continent_map = {
            '0': 0,  # 亚洲 (Asia)
            '1': 1,  # 欧洲 (Europe)
            '2': 2,  # 北美洲 (North America)
            '3': 3,  # 南美洲 (South America)
            '4': 4,  # 非洲 (Africa)
            '5': 5,  # 大洋洲 (Oceania)
        }

        label = str(label).strip()
        continent_ids = []

        # 处理多大洲情况
        if ',' in label:
            continents = [c.strip() for c in label.split(',')]
            for continent in continents:
                if continent in continent_map:
                    continent_ids.append(continent_map[continent])
        else:
            if label in continent_map:
                continent_ids.append(continent_map[label])

        # 如果没有找到有效的大洲，使用默认值
        if not continent_ids:
            continent_ids.append(0)  # 默认为亚洲

        # 主要大洲ID（取第一个）
        primary_continent_id = continent_ids[0]

        return primary_continent_id, continent_ids

    def _get_continent_name(self, continent_id: int) -> str:
        """获取大洲名称"""
        continent_names = {
            0: "亚洲 (Asia)",
            1: "欧洲 (Europe)",
            2: "北美洲 (North America)",
            3: "南美洲 (South America)",
            4: "非洲 (Africa)",
            5: "大洋洲 (Oceania)"
        }
        return continent_names.get(continent_id, f"未知大洲 (Unknown-{continent_id})")

    def _is_answer_correct(self, predicted_answer: str, true_answer: str) -> bool:
        """
        检查预测答案是否正确，支持逗号分隔的多个正确答案

        Args:
            predicted_answer: 模型预测的答案
            true_answer: 真实答案，可能包含逗号分隔的多个正确答案

        Returns:
            bool: 预测答案是否正确

        Examples:
            _is_answer_correct("1", "1") -> True
            _is_answer_correct("1", "1,2") -> True
            _is_answer_correct("2", "1,2") -> True
            _is_answer_correct("3", "1,2") -> False
        """
        predicted = predicted_answer.strip()
        true_answer = true_answer.strip()

        # 如果true_answer包含逗号，说明有多个正确答案
        if ',' in true_answer:
            valid_answers = [ans.strip() for ans in true_answer.split(',')]
            return predicted in valid_answers
        else:
            # 单一答案的情况
            return predicted == true_answer

    def evaluate_model(self, model, tokenizer, test_data: List[Dict]) -> Dict[str, Any]:
        """评估模型"""
        logging.info("Starting model evaluation...")

        model.eval()
        predictions = []
        true_labels = []
        generated_answers = []

        # 评估参数 - 优化为简洁回答
        max_length = 512
        max_new_tokens = 5  # 减少到5个token，只够回答数字

        with torch.no_grad():
            for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
                try:
                    # 构建输入 - 使用更简单的格式，避免复杂的LLaMA格式
                    prompt = f"{item['full_text']}"
                    prompt_mask = f"{item['full_text_mask']}"

                    # 静默处理，不输出调试信息

                    # 分词
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
                    inputs_mask = tokenizer(prompt_mask, return_tensors="pt", max_length=max_length, truncation=True)

                    # 移动到设备
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    input_ids_mask = inputs_mask['input_ids'].to(self.device)
                    attention_mask_mask = inputs_mask['attention_mask'].to(self.device)

                    # 文化ID
                    culture_ids = torch.tensor([item['continent_id']], dtype=torch.long, device=self.device)

                    # 设置停止词，让模型在生成数字后停止
                    stop_tokens = []
                    if hasattr(tokenizer, 'encode'):
                        # 添加常见的停止标记
                        stop_sequences = ['\n', '.', '。', ' ', '\t']
                        for seq in stop_sequences:
                            try:
                                token_ids = tokenizer.encode(seq, add_special_tokens=False)
                                if token_ids:
                                    stop_tokens.extend(token_ids)
                            except:
                                pass

                    # 生成回答
                    expert_weights = None
                    with torch.amp.autocast('cuda'):
                        # 尝试获取专家权重（简化版本，避免DataParallel问题）
                        try:
                            # 直接调用模型进行前向传播
                            model_outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                culture_ids=culture_ids,
                                use_culture_loss=False
                            )
                            # 简化专家权重提取
                            if isinstance(model_outputs, dict) and 'expert_weights' in model_outputs:
                                expert_weights = model_outputs['expert_weights']
                                if isinstance(expert_weights, torch.Tensor):
                                    expert_weights = expert_weights.detach().cpu().numpy()
                                    # 确保是列表格式
                                    if expert_weights.ndim > 1:
                                        expert_weights = expert_weights[0].tolist()  # 取第一个样本
                                    else:
                                        expert_weights = expert_weights.tolist()
                        except Exception:
                            expert_weights = None

                        # 生成文本 - 优化为简洁数字回答
                        generate_kwargs = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'max_new_tokens': max_new_tokens,
                            'do_sample': False,  # 使用贪婪解码，更确定性
                            'temperature': 1.0,  # 降低随机性
                            'pad_token_id': tokenizer.eos_token_id,
                            'eos_token_id': tokenizer.eos_token_id,
                            'repetition_penalty': 1.0,  # 减少惩罚，避免影响简短回答
                            'length_penalty': 0.0,      # 不惩罚长度，让模型自然停止
                            'early_stopping': True,     # 遇到停止token就停止
                        }

                        # 如果有停止token，添加到生成参数中
                        if stop_tokens:
                            # 去重并限制数量
                            unique_stop_tokens = list(set(stop_tokens))[:10]
                            try:
                                generate_kwargs['bad_words_ids'] = None  # 不使用bad_words，让模型自然生成
                                # 注：stop_tokens在某些版本中可能不支持，所以我们依赖max_new_tokens和early_stopping
                            except:
                                pass

                        outputs = model.generate(**generate_kwargs)

                    # 解码生成的文本
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # 提取生成的答案部分 - 优化为简洁数字提取
                    try:
                        prompt_end_marker = "### Answer: "
                        if prompt_end_marker in generated_text:
                            # 找到"### Answer: "之后的内容
                            answer_part = generated_text.split(prompt_end_marker)[-1].strip()
                            # 只提取第一个数字或字母，忽略后续解释
                            import re
                            number_match = re.search(r'^(\d+)', answer_part)
                            if number_match:
                                generated_answer = number_match.group(1)
                            else:
                                # 如果没有数字，查找字母
                                letter_match = re.search(r'^([A-D])', answer_part, re.IGNORECASE)
                                if letter_match:
                                    generated_answer = letter_match.group(1).upper()
                                else:
                                    generated_answer = answer_part.split()[0] if answer_part.split() else answer_part
                        elif len(generated_text) > len(prompt):
                            # 如果生成的文本比prompt长，提取新生成的部分
                            new_content = generated_text[len(prompt):].strip()
                            # 只提取第一个有效的答案字符
                            import re
                            number_match = re.search(r'(\d+)', new_content)
                            if number_match:
                                generated_answer = number_match.group(1)
                            else:
                                letter_match = re.search(r'([A-D])', new_content, re.IGNORECASE)
                                if letter_match:
                                    generated_answer = letter_match.group(1).upper()
                                else:
                                    generated_answer = new_content.split()[0] if new_content.split() else new_content
                        else:
                            # 如果生成的文本长度和prompt一样或更短，可能没有生成新内容
                            generated_answer = ""
                    except Exception:
                        generated_answer = generated_text.strip()

                    # 简单的答案匹配（提取首字母作为预测）
                    pred_answer = self._extract_answer(generated_answer)
                    true_answer = item['output'].strip()

                    # 使用多答案支持的正确性检查
                    is_correct = self._is_answer_correct(pred_answer, true_answer)

                    predictions.append(pred_answer)
                    true_labels.append(true_answer)

                    # 保存所有生成答案（不再限制数量）
                    answer_data = {
                        'index': i,
                        'question': item['full_text'],
                        'true_answer': true_answer,
                        'generated_answer': generated_answer,
                        'predicted_answer': pred_answer,
                        'continent_id': item['continent_id'],
                        'continent_label': item['label'],
                        'is_correct': is_correct,
                        'culture_context': {
                            'instruction': item['instruction'],
                            'input_text': item['input'],
                            'continent_name': self._get_continent_name(item['continent_id'])
                        }
                    }

                    # 添加专家权重信息（如果可用）
                    if expert_weights is not None and isinstance(expert_weights, list) and len(expert_weights) > 0:
                        answer_data['expert_weights'] = expert_weights
                        # 计算专家权重统计信息
                        try:
                            # 确保是一维列表
                            weights = expert_weights
                            if isinstance(weights[0], list):
                                weights = weights[0]  # 如果是二维，取第一个样本

                            # 确保所有权重都是数值
                            weights = [float(w) for w in weights if isinstance(w, (int, float))]

                            if len(weights) > 0:
                                answer_data['expert_analysis'] = {
                                    'dominant_expert': int(weights.index(max(weights))),
                                    'max_weight': float(max(weights)),
                                    'weight_distribution': 'concentrated' if max(weights) > 0.3 else 'distributed',
                                    'num_active_experts': sum(1 for w in weights if w > 0.05)
                                }
                        except Exception:
                            pass

                    generated_answers.append(answer_data)

                    # 内存清理
                    if i % 100 == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    predictions.append("A")  # 默认答案
                    true_labels.append(item['output'].strip())

                    # 即使出错也保存基本信息
                    generated_answers.append({
                        'index': i,
                        'question': item['full_text'],
                        'true_answer': item['output'].strip(),
                        'generated_answer': f"Error: {str(e)}",
                        'predicted_answer': "A",
                        'continent_id': item['continent_id'],
                        'continent_label': item['label'],
                        'is_correct': False,
                        'culture_context': {
                            'instruction': item['instruction'],
                            'input_text': item['input'],
                            'continent_name': self._get_continent_name(item['continent_id'])
                        },
                        'error': str(e)
                    })
                    continue

        # 计算评估指标 - 使用自定义的多答案正确性检查
        correct_count = sum(1 for answer in generated_answers if answer['is_correct'])
        total_count = len(generated_answers)
        accuracy = correct_count / total_count if total_count > 0 else 0

        # 为了兼容现有的评估指标，我们仍然计算precision, recall, f1
        # 但这些指标在多答案情况下的意义有所不同
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='weighted', zero_division=0
            )
        except Exception:
            precision = recall = f1 = 0.0

        # 计算每个答案选项的准确率（针对多答案进行调整）
        unique_labels = sorted(list(set(predictions)))  # 只考虑预测的标签
        per_class_accuracy = {}

        for label in unique_labels:
            # 找到预测为该标签的样本
            label_samples = [answer for answer in generated_answers if answer['predicted_answer'] == label]
            if label_samples:
                label_correct = sum(1 for sample in label_samples if sample['is_correct'])
                per_class_accuracy[label] = label_correct / len(label_samples)

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'per_class_accuracy': per_class_accuracy,
            'total_samples': len(test_data),
            'correct_predictions': int(correct_count),
            'evaluation_time': datetime.now().isoformat()
        }

        # 计算按大洲的准确率统计
        continent_stats = {}
        for answer in generated_answers:
            continent_name = answer['culture_context']['continent_name']
            if continent_name not in continent_stats:
                continent_stats[continent_name] = {'correct': 0, 'total': 0}
            continent_stats[continent_name]['total'] += 1
            if answer['is_correct']:
                continent_stats[continent_name]['correct'] += 1

        # 计算每个大洲的准确率
        for continent, stats in continent_stats.items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

        # 添加详细统计到结果中
        results['continent_statistics'] = continent_stats
        results['answer_distribution'] = {}

        # 统计答案分布（支持数字答案）
        all_true_answers = set()
        all_pred_answers = set()
        multi_answer_count = 0

        for answer in generated_answers:
            # 收集所有真实答案（包括多答案的拆分）
            true_ans = answer['true_answer']
            if ',' in true_ans:
                multi_answer_count += 1
                all_true_answers.update(ans.strip() for ans in true_ans.split(','))
            else:
                all_true_answers.add(true_ans)

            # 收集预测答案
            all_pred_answers.add(answer['predicted_answer'])

        # 生成答案分布统计
        for label in sorted(all_true_answers.union(all_pred_answers)):
            true_count = 0
            for answer in generated_answers:
                true_ans = answer['true_answer']
                if ',' in true_ans:
                    if label in [ans.strip() for ans in true_ans.split(',')]:
                        true_count += 1
                else:
                    if label == true_ans:
                        true_count += 1

            pred_count = sum(1 for answer in generated_answers if answer['predicted_answer'] == label)
            results['answer_distribution'][label] = {
                'true_count': true_count,
                'predicted_count': pred_count
            }

        # 添加多答案统计信息
        results['multi_answer_statistics'] = {
            'total_samples': len(generated_answers),
            'multi_answer_samples': multi_answer_count,
            'multi_answer_ratio': multi_answer_count / len(generated_answers) if generated_answers else 0
        }

        logging.info(f"Evaluation completed:")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  Total samples: {len(test_data)}")
        logging.info(f"  Correct predictions: {correct_count}")
        if multi_answer_count > 0:
            logging.info(f"  Multi-answer samples: {multi_answer_count} ({multi_answer_count / len(generated_answers) * 100:.1f}%)")

        return results, generated_answers

    def _extract_answer(self, text: str) -> str:
        """从生成的文本中提取答案"""
        text = text.strip()

        # 查找常见的答案模式
        import re

        # 优先查找数字答案（1-10）
        number_patterns = [
            r'答案[是：:]\s*(\d+)',
            r'选择\s*(\d+)',
            r'应该选择\s*(\d+)',
            r'正确答案[是：:]\s*(\d+)',
            r'^(\d+)[\.。]',
            r'\b(\d+)\b'
        ]

        for pattern in number_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(1)
                # 确保数字在合理范围内（1-10）
                if number.isdigit() and 1 <= int(number) <= 10:
                    return number

        # 查找字母答案（A-D）作为备用
        letter_patterns = [
            r'答案[是：:]\s*([A-D])',
            r'选择\s*([A-D])',
            r'应该选择\s*([A-D])',
            r'正确答案[是：:]\s*([A-D])',
            r'^([A-D])[\.。]',
            r'\b([A-D])\b'
        ]

        for pattern in letter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 如果没有找到明确的答案，优先查找单独的数字
        for char in text:
            if char.isdigit() and 1 <= int(char) <= 10:
                return char

        # 查找单独的字母作为备用
        for char in text:
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()

        # 默认返回A（如果完全没有找到）
        return 'A'

    def save_results(self, evaluation_results: Dict[str, Any], generated_answers: List[Dict]):
        """保存评估结果"""
        # 保存详细评估结果
        results_file = os.path.join(self.args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        # 保存生成的答案
        answers_file = os.path.join(self.args.output_dir, 'generated_answers.json')
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(generated_answers, f, indent=2, ensure_ascii=False)

        # 保存评估摘要
        summary = {
            'model_type': 'Enhanced CultureMoE',
            'backbone': self.args.backbone,
            'num_experts': self.args.num_experts,
            'accuracy': evaluation_results['accuracy'],
            'precision': evaluation_results['precision'],
            'recall': evaluation_results['recall'],
            'f1_score': evaluation_results['f1_score'],
            'total_samples': evaluation_results['total_samples'],
            'correct_predictions': evaluation_results['correct_predictions'],
            'evaluation_date': evaluation_results['evaluation_time']
        }

        summary_file = os.path.join(self.args.output_dir, 'evaluation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logging.info(f"Results saved to:")
        logging.info(f"  Detailed results: {results_file}")
        logging.info(f"  Generated answers: {answers_file}")
        logging.info(f"  Summary: {summary_file}")

    def run_evaluation(self):
        """运行完整的评估流程"""
        start_time = time.time()

        try:
            # 1. 加载模型
            model, tokenizer = self.load_enhanced_culturemoe_from_components()

            # 2. 加载测试数据
            test_data = self.load_test_data(tokenizer)

            # 3. 评估模型
            evaluation_results, generated_answers = self.evaluate_model(model, tokenizer, test_data)

            # 4. 保存结果
            self.save_results(evaluation_results, generated_answers)

            end_time = time.time()
            evaluation_time = end_time - start_time

            logging.info(f"\n" + "=" * 80)
            logging.info("✅ Enhanced CultureMoE Evaluation Completed Successfully!")
            logging.info("=" * 80)
            logging.info(f"Total evaluation time: {evaluation_time:.2f} seconds")
            logging.info(f"Final accuracy: {evaluation_results['accuracy']:.4f}")
            logging.info("=" * 80)

            return True

        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Enhanced CultureMoE Evaluation from Components")

    # 必需参数
    parser.add_argument('--base_model_path', type=str, required=True, help='基础模型路径')
    parser.add_argument('--lora_weights_path', type=str, required=True, help='LoRA权重路径')
    parser.add_argument('--moe_weights_path', type=str, required=True, help='Enhanced MoE权重路径')
    parser.add_argument('--test_file', type=str, required=True, help='测试数据文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')

    # 模型参数
    parser.add_argument('--backbone', type=str, default='llama', help='基座模型类型')
    parser.add_argument('--num_experts', type=int, default=12, help='专家数量')
    parser.add_argument('--moe_fusion', type=float, default=0.4, help='MoE融合系数')
    parser.add_argument('--culture_loss_lambda', type=float, default=0.5, help='文化损失权重')

    # 评估参数
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--use_multi_gpu', action='store_true', help='使用多GPU')

    args = parser.parse_args()

    # 创建评估器并运行
    evaluator = EnhancedCultureMoEEvaluator(args)
    success = evaluator.run_evaluation()

    if success:
        print("\n🎉 Evaluation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()