#!/usr/bin/env python3
"""
联合LoRA+MoE模型评估脚本
从基础模型和联合训练的权重还原完整模型并在测试集上评估

功能：
1. 加载基础模型
2. 加载联合训练的LoRA权重和MoE权重
3. 还原完整的联合LoRA+MoE模型
4. 在测试集上进行评估
5. 计算准确率、精确率、召回率、F1等指标
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
import re
import time

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.joint_lora_moe_model import JointLoRAMoEModel, JointLoRAMoEConfig
from ft_lora_only_gen import CultureLLMNewFormatDataset, extract_answer_from_text


class JointLoRAMoEEvaluator:
    """联合LoRA+MoE模型评估器"""

    def __init__(self, base_model_path: str, joint_model_path: str, backbone: str, device: str = 'cuda'):
        """
        初始化评估器

        Args:
            base_model_path: 基础模型路径
            joint_model_path: 联合训练模型路径
            backbone: 模型骨干 (llama/qwen)
            device: 设备
        """
        self.base_model_path = base_model_path
        self.joint_model_path = joint_model_path
        self.backbone = backbone
        self.device = device

        # 加载tokenizer
        print("Loading tokenizer...")
        self.tokenizer = self._load_tokenizer()

        # 加载并还原完整模型
        print("Loading and restoring joint LoRA+MoE model...")
        self.model = self._load_joint_model()

    def _load_tokenizer(self):
        """加载tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)

        # 修复tokenizer配置（与训练时保持一致）
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128009:
            # Llama 3.1: 使用官方的finetune_right_pad_id
            try:
                official_pad_token = "<|finetune_right_pad_id|>"
                pad_token_id = tokenizer.convert_tokens_to_ids(official_pad_token)
                if pad_token_id != tokenizer.unk_token_id and pad_token_id is not None:
                    tokenizer.pad_token = official_pad_token
                    tokenizer.pad_token_id = pad_token_id
                else:
                    raise ValueError("Official pad token not found")
            except:
                # 使用安全的低频字符作为fallback
                safe_tokens = ['~', '`', '|', '^']
                for safe_token in safe_tokens:
                    try:
                        safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                        if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                            tokenizer.pad_token = safe_token
                            tokenizer.pad_token_id = safe_token_id
                            break
                    except:
                        continue
        elif tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"
        return tokenizer

    def _load_joint_model(self):
        """加载并还原联合LoRA+MoE模型"""
        # 1. 加载基础模型
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map=None,  # 先不映射到设备
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # 2. 加载联合配置
        joint_config_path = os.path.join(self.joint_model_path, 'joint_config.json')
        with open(joint_config_path, 'r') as f:
            config_dict = json.load(f)

        print("Joint model configuration:")
        for key, value in config_dict.items():
            print(f"  {key}: {value}")

        # 3. 创建联合配置对象
        joint_config = JointLoRAMoEConfig(
            lora_rank=config_dict['lora_rank'],
            lora_alpha=config_dict['lora_alpha'],
            lora_dropout=config_dict['lora_dropout'],
            lora_target_modules=config_dict['lora_target_modules'],
            num_moe_experts=config_dict['num_moe_experts'],
            moe_hidden_dim=config_dict['moe_hidden_dim'],
            moe_intermediate_dim=config_dict['moe_intermediate_dim'],
            use_culture_loss=config_dict['use_culture_loss'],
            culture_loss_weight=config_dict['culture_loss_weight'],
            dropout=config_dict['dropout']
        )

        # 4. 创建联合模型
        print("Creating joint LoRA+MoE model...")
        model = JointLoRAMoEModel(base_model, joint_config)

        # 5. 加载LoRA权重
        lora_weights_path = os.path.join(self.joint_model_path, 'lora_weights')
        print(f"Loading LoRA weights from: {lora_weights_path}")
        model.base_model.load_adapter(lora_weights_path, adapter_name="default")

        # 6. 加载MoE权重
        moe_weights_path = os.path.join(self.joint_model_path, 'moe_weights.pt')
        print(f"Loading MoE weights from: {moe_weights_path}")
        moe_state_dict = torch.load(moe_weights_path, map_location='cpu')

        # 加载MoE权重到模型
        for name, param in model.moe_layer.named_parameters():
            if name in moe_state_dict:
                param.data.copy_(moe_state_dict[name])
                print(f"  Loaded MoE parameter: {name}")
            else:
                print(f"  Warning: MoE parameter not found in saved weights: {name}")

        # 7. 移动到设备
        model = model.to(self.device)
        model.eval()

        print("✅ Joint LoRA+MoE model loaded successfully")
        return model

    def generate_answer(self, instruction: str, input_text: str, max_new_tokens: int = 5) -> str:
        """
        生成答案

        Args:
            instruction: 指令
            input_text: 输入文本
            max_new_tokens: 最大生成token数

        Returns:
            生成的文本
        """
        # 构建输入（与训练时格式保持一致）
        if input_text:
            full_input = f"{instruction}\\n{input_text}"
        else:
            full_input = instruction

        # 确保以"### Answer: "结尾
        if not full_input.endswith("### Answer: "):
            if "### Answer:" in full_input:
                full_input = full_input.split("### Answer:")[0].strip() + " ### Answer: "
            else:
                full_input = f"{full_input.rstrip()} ### Answer: "

        # Tokenize
        inputs = self.tokenizer(full_input, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # 使用联合模型的generate方法
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,  # 贪心解码
                temperature=0.7
            )

        # 解码生成的部分
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return generated_text

    def evaluate_on_dataset(self, test_file: str, num_classes: int) -> Dict:
        """
        在测试集上评估模型

        Args:
            test_file: 测试文件路径
            num_classes: 类别数量

        Returns:
            评估结果字典
        """
        print(f"Loading test dataset: {test_file}")

        # 加载测试数据
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        print(f"Test dataset size: {len(test_data)}")

        # 评估
        predictions = []
        true_labels = []
        generated_answers = []

        print("Generating answers...")
        for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            true_output = item.get('output', '')

            # 生成答案
            try:
                generated_text = self.generate_answer(instruction, input_text)
                predicted_answer = extract_answer_from_text(generated_text)

                # 记录结果
                is_correct = predicted_answer == true_output
                predictions.append(predicted_answer)
                true_labels.append(true_output)

                generated_answers.append({
                    'index': idx,
                    'instruction': instruction,
                    'input': input_text,
                    'true_output': true_output,
                    'generated_text': generated_text,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct
                })

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                # 记录失败的情况
                predictions.append("")
                true_labels.append(true_output)
                generated_answers.append({
                    'index': idx,
                    'instruction': instruction,
                    'input': input_text,
                    'true_output': true_output,
                    'generated_text': "",
                    'predicted_answer': "",
                    'is_correct': False,
                    'error': str(e)
                })

        # 计算指标
        print("Computing metrics...")

        # 转换为数字标签（用于sklearn计算）
        unique_labels = sorted(list(set(true_labels + predictions)))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

        true_ids = [label_to_id.get(label, -1) for label in true_labels]
        pred_ids = [label_to_id.get(pred, -1) for pred in predictions]

        # 过滤掉无效的预测
        valid_indices = [i for i, (true_id, pred_id) in enumerate(zip(true_ids, pred_ids))
                        if true_id != -1 and pred_id != -1]

        if valid_indices:
            valid_true = [true_ids[i] for i in valid_indices]
            valid_pred = [pred_ids[i] for i in valid_indices]

            # 计算指标
            accuracy = accuracy_score(valid_true, valid_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                valid_true, valid_pred, average='weighted', zero_division=0
            )

            # 计算混淆矩阵
            cm = confusion_matrix(valid_true, valid_pred)

        else:
            accuracy = precision = recall = f1 = 0.0
            cm = np.array([])

        # 统计结果
        correct_count = sum(1 for item in generated_answers if item['is_correct'])
        total_count = len(generated_answers)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'correct_count': correct_count,
            'total_count': total_count,
            'valid_predictions': len(valid_indices),
            'confusion_matrix': cm.tolist() if cm.size > 0 else [],
            'label_mapping': label_to_id,
            'unique_labels': unique_labels
        }

        return results, generated_answers


def main():
    parser = argparse.ArgumentParser(description="Evaluate Joint LoRA+MoE Model")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--joint_model_path", type=str, required=True,
                        help="Path to joint trained model")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test data file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--backbone", type=str, required=True, choices=["llama", "qwen"],
                        help="Model backbone")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes in the dataset")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    args = parser.parse_args()

    print("="*80)
    print("Joint LoRA+MoE Model Evaluation")
    print("="*80)
    print(f"Base model: {args.base_model_path}")
    print(f"Joint model: {args.joint_model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Backbone: {args.backbone}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    print()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # 初始化评估器
        evaluator = JointLoRAMoEEvaluator(
            base_model_path=args.base_model_path,
            joint_model_path=args.joint_model_path,
            backbone=args.backbone,
            device=args.device
        )

        # 评估
        start_time = time.time()
        results, generated_answers = evaluator.evaluate_on_dataset(
            test_file=args.test_file,
            num_classes=args.num_classes
        )
        end_time = time.time()

        # 保存结果
        print("Saving results...")

        # 保存详细评估结果
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存生成的答案
        answers_file = os.path.join(args.output_dir, 'generated_answers.json')
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(generated_answers, f, indent=2, ensure_ascii=False)

        # 保存配置信息
        config = {
            'base_model_path': args.base_model_path,
            'joint_model_path': args.joint_model_path,
            'test_file': args.test_file,
            'backbone': args.backbone,
            'num_classes': args.num_classes,
            'device': args.device,
            'evaluation_time': end_time - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        config_file = os.path.join(args.output_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 保存评估摘要
        summary = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'correct_count': results['correct_count'],
            'total_count': results['total_count'],
            'valid_predictions': results['valid_predictions']
        }

        summary_file = os.path.join(args.output_dir, 'evaluation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 打印结果
        print()
        print("="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Correct: {results['correct_count']}/{results['total_count']}")
        print(f"Valid predictions: {results['valid_predictions']}")
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        print("="*80)

        # 显示前几个样例
        print()
        print("Sample predictions:")
        for i, item in enumerate(generated_answers[:5]):
            status = "✅" if item['is_correct'] else "❌"
            print(f"{i+1}. {status} True: {item['true_output']}, Predicted: {item['predicted_answer']}")
            print(f"   Generated: '{item['generated_text']}'")
            print()

        print(f"Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()