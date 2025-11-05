#!/usr/bin/env python3
"""
从 Base 模型 + LoRA 权重 + MoE 权重还原完整的 CultureMoE 模型

使用方法：
    python load_culturemoe_gen.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --moe_weights_path /path/to/moe_weights \
        --test_file /path/to/test_data.json \
        --output_file /path/to/results.json \
        --num_classes 5
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


def load_model_from_components(
    base_model_path: str,
    lora_weights_path: str,
    moe_weights_path: str,
    device: str = "cuda"
):
    """
    从三个组件加载完整的 CultureMoE 模型

    Args:
        base_model_path: Base 模型路径
        lora_weights_path: LoRA 权重路径
        moe_weights_path: MoE 权重路径
        device: 设备

    Returns:
        model: 完整的 CultureMoE 模型
        tokenizer: Tokenizer
    """
    print("\n" + "="*80)
    print("Loading CultureMoE Model from Components")
    print("="*80)
    print(f"Base model: {base_model_path}")
    print(f"LoRA weights: {lora_weights_path}")
    print(f"MoE weights: {moe_weights_path}")
    print("="*80)
    print("")

    # 1. 加载 Tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载 Base 模型
    print("\n2. Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("   ✅ Base model loaded")

    # 3. 加载 LoRA 权重并合并
    print("\n3. Loading and merging LoRA weights...")
    from peft import PeftModel

    # 加载 LoRA
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        is_trainable=False
    )

    # 合并 LoRA 权重到 base 模型
    print("   Merging LoRA weights into base model...")
    merged_model = model_with_lora.merge_and_unload()
    print("   ✅ LoRA weights merged")

    # 4. 加载 MoE 配置
    print("\n4. Loading MoE configuration...")
    moe_config_path = os.path.join(moe_weights_path, "moe_config.json")
    if not os.path.exists(moe_config_path):
        raise FileNotFoundError(f"MoE config not found: {moe_config_path}")

    with open(moe_config_path, 'r') as f:
        moe_config = json.load(f)

    print(f"   Num experts: {moe_config['num_experts']}")
    print(f"   Num classes: {moe_config['num_classes']}")
    print("   ✅ MoE config loaded")

    # 5. 创建 MoE 模型结构
    print("\n5. Creating CultureMoE model structure...")
    moe_args = ModelArgs(
        num_experts=moe_config['num_experts'],
        shared_hidden_dim=moe_config['shared_hidden_dim'],
        router_hidden_dim=moe_config['router_hidden_dim'],
        experts_hidden_dim=moe_config['experts_hidden_dim'],
        lora_rank=moe_config['moe_lora_rank'],
        num_classes=moe_config['num_classes'],
        classification_hidden_dim=moe_config['classification_hidden_dim'],
        dropout=moe_config['dropout'],
        num_heads=moe_config['num_heads']
    )

    # 创建 CultureMoE 模型（使用合并后的模型）
    culturemoe_model = LlamaSharedRouterExpertsModel(
        llama_model=merged_model,
        config=merged_model.config,
        args=moe_args
    )
    print("   ✅ CultureMoE structure created")

    # 6. 加载 MoE 权重
    print("\n6. Loading MoE weights...")
    moe_state_dict_path = os.path.join(moe_weights_path, "moe_state_dict.pt")
    if not os.path.exists(moe_state_dict_path):
        raise FileNotFoundError(f"MoE weights not found: {moe_state_dict_path}")

    moe_state_dict = torch.load(moe_state_dict_path, map_location="cpu")

    # 加载 MoE 部分的权重
    missing_keys, unexpected_keys = culturemoe_model.load_state_dict(moe_state_dict, strict=False)

    if missing_keys:
        print(f"   ⚠️  Missing keys: {len(missing_keys)}")
        # LLaMA 的权重是预期缺失的（因为已经从 merged_model 加载）
    if unexpected_keys:
        print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)}")

    print("   ✅ MoE weights loaded")

    # 7. 移动到设备
    print(f"\n7. Moving model to {device}...")
    culturemoe_model = culturemoe_model.to(device)
    culturemoe_model.eval()
    print("   ✅ Model ready")

    # 打印模型信息
    total_params = sum(p.numel() for p in culturemoe_model.parameters())
    print(f"\n📊 Model Info:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Device: {device}")

    print("\n" + "="*80)
    print("✅ CultureMoE Model Loaded Successfully!")
    print("="*80)
    print("")

    return culturemoe_model, tokenizer


def load_test_data(data_path: str, tokenizer, max_length: int = 512):
    """加载测试数据"""
    print(f"Loading test data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} test samples")

    processed_data = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item['output']

        # 组合文本
        if input_text and input_text.strip():
            full_text = f"{instruction}\n{input_text}"
        else:
            full_text = instruction

        # Tokenize
        encoded = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        processed_data.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': output
        })

    return processed_data


def evaluate_model(model, tokenizer, test_dataset, num_classes: int, batch_size: int = 8, device: str = "cuda"):
    """评估模型"""
    print(f"\nEvaluating on {len(test_dataset)} samples...")

    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch = test_dataset[i:i+batch_size]

            # 准备批次数据
            input_ids = [item['input_ids'] for item in batch]
            attention_mask = [item['attention_mask'] for item in batch]
            labels = [item['label'] for item in batch]

            # Padding
            max_len = max(len(ids) for ids in input_ids)
            input_ids_padded = []
            attention_mask_padded = []

            for ids, mask in zip(input_ids, attention_mask):
                padding_length = max_len - len(ids)
                input_ids_padded.append(ids + [tokenizer.pad_token_id] * padding_length)
                attention_mask_padded.append(mask + [0] * padding_length)

            # 转换为 tensor
            input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long).to(device)
            attention_mask_tensor = torch.tensor(attention_mask_padded, dtype=torch.long).to(device)

            # 前向传播（双路输入相同）
            outputs = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                input_ids_mask=input_ids_tensor,  # 使用相同的输入
                attention_mask_mask=attention_mask_tensor
            )

            # 获取预测
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    # 计算指标
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    accuracy = accuracy_score(labels, predictions)

    # 根据类别数选择平均方式
    if num_classes == 2:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', pos_label=1, zero_division=0
        )
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )

    # 混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(labels),
        'num_classes': num_classes
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="从组件加载 CultureMoE 模型并评估")

    # 模型路径
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Base 模型路径")
    parser.add_argument("--lora_weights_path", type=str, required=True,
                        help="LoRA 权重路径（best_lora 目录）")
    parser.add_argument("--moe_weights_path", type=str, required=True,
                        help="MoE 权重路径")

    # 数据和评估
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件")
    parser.add_argument("--output_file", type=str, required=True,
                        help="评估结果输出文件")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="分类数量")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")

    args = parser.parse_args()

    # 检查路径
    if not os.path.exists(args.base_model_path):
        print(f"❌ Error: Base model not found: {args.base_model_path}")
        sys.exit(1)

    if not os.path.exists(args.lora_weights_path):
        print(f"❌ Error: LoRA weights not found: {args.lora_weights_path}")
        sys.exit(1)

    if not os.path.exists(args.moe_weights_path):
        print(f"❌ Error: MoE weights not found: {args.moe_weights_path}")
        sys.exit(1)

    if not os.path.exists(args.test_file):
        print(f"❌ Error: Test file not found: {args.test_file}")
        sys.exit(1)

    print("\n" + "="*80)
    print("CultureMoE Evaluation (From Components)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.lora_weights_path}")
    print(f"MoE weights: {args.moe_weights_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output file: {args.output_file}")
    print("="*80)
    print("")

    # 加载模型
    model, tokenizer = load_model_from_components(
        args.base_model_path,
        args.lora_weights_path,
        args.moe_weights_path,
        args.device
    )

    # 加载测试数据
    test_dataset = load_test_data(args.test_file, tokenizer, args.max_length)

    # 评估
    results = evaluate_model(
        model, tokenizer, test_dataset,
        args.num_classes, args.batch_size, args.device
    )

    # 打印结果
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("="*80)

    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    output_data = {
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_model_path': args.base_model_path,
        'lora_weights_path': args.lora_weights_path,
        'moe_weights_path': args.moe_weights_path,
        'test_file': args.test_file,
        'num_classes': args.num_classes,
        'results': results
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {args.output_file}")
    print("")


if __name__ == "__main__":
    main()

