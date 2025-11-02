#!/usr/bin/env python3
"""
加载 CultureMoE 模型（从 MoE 权重 + 合并后的 LLM）并在测试集上评估

使用方法：
    python load_culturemoe.py \
        --moe_weights_path /path/to/moe_weights \
        --test_file /path/to/test_data.json \
        --output_dir /path/to/output

示例：
    python load_culturemoe.py \
        --moe_weights_path /root/autodl-tmp/CultureMoE/Culture_Alignment/model_moe_llama_2 \
        --test_file /root/autodl-fs/wvs_2_merged \
        --output_dir /root/autodl-tmp/CultureMoE/Culture_Alignment/eval_results
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs
from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data
from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator


def load_culturemoe_model(moe_weights_path: str, device: str = "cuda"):
    """
    加载 CultureMoE 模型

    Args:
        moe_weights_path: MoE 权重目录路径
        device: 设备

    Returns:
        model: CultureMoE 模型
        tokenizer: Tokenizer
        config: MoE 配置
    """
    print("\n" + "="*80)
    print("Loading CultureMoE Model")
    print("="*80)
    print(f"MoE weights path: {moe_weights_path}")
    print("="*80)
    print("")

    # 1. 加载 MoE 配置
    print("1. Loading MoE configuration...")
    config_path = os.path.join(moe_weights_path, "moe_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"MoE config not found: {config_path}")

    with open(config_path, 'r') as f:
        moe_config = json.load(f)

    print(f"   ✅ MoE config loaded")
    print(f"      Model type: {moe_config['model_type']}")
    print(f"      Architecture: {moe_config['architecture']}")
    print(f"      Num experts: {moe_config['num_experts']}")
    print(f"      Num classes: {moe_config['num_classes']}")
    print(f"      Best accuracy: {moe_config['best_accuracy']:.4f}")

    # 2. 加载 Tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(moe_weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 3. 加载合并后的 LLM
    merged_llm_path = moe_config['merged_llm_path']
    print(f"\n3. Loading merged LLM from {merged_llm_path}...")

    if not os.path.exists(merged_llm_path):
        raise FileNotFoundError(
            f"Merged LLM not found: {merged_llm_path}\n"
            f"Please ensure the merged LLM is available at the specified path."
        )

    llama_model = AutoModelForCausalLM.from_pretrained(
        merged_llm_path,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # 冻结 LLM
    for param in llama_model.parameters():
        param.requires_grad = False

    print("   ✅ Merged LLM loaded and frozen")
    print(f"      LLM parameters: {sum(p.numel() for p in llama_model.parameters()):,}")

    # 4. 创建 CultureMoE 模型架构
    print("\n4. Creating CultureMoE architecture...")
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

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    )

    print("   ✅ CultureMoE architecture created")

    # 5. 加载 MoE 权重
    print("\n5. Loading MoE weights...")
    weights_path = os.path.join(moe_weights_path, "moe_weights.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"MoE weights not found: {weights_path}")

    moe_state_dict = torch.load(weights_path, map_location="cpu")

    # 加载权重（strict=False 因为我们只加载 MoE 部分）
    missing_keys, unexpected_keys = model.load_state_dict(moe_state_dict, strict=False)

    # 验证加载
    llama_missing = [k for k in missing_keys if k.startswith('llama_model.')]
    non_llama_missing = [k for k in missing_keys if not k.startswith('llama_model.')]

    if non_llama_missing:
        print(f"   ⚠️  Warning: Missing non-LLM keys: {len(non_llama_missing)}")
        print(f"      {non_llama_missing[:5]}...")

    print(f"   ✅ MoE weights loaded")
    print(f"      Missing LLM keys: {len(llama_missing)} (expected)")
    print(f"      Missing MoE keys: {len(non_llama_missing)} (should be 0)")
    print(f"      Unexpected keys: {len(unexpected_keys)}")

    # 6. 将 MoE 层移到与 LLM 相同的设备
    print("\n6. Moving MoE layers to correct device...")
    llm_device = next(llama_model.parameters()).device

    # 只移动 MoE 层（不移动 llama_model）
    if hasattr(model, 'shared'):
        model.shared = model.shared.to(llm_device)
    if hasattr(model, 'router'):
        model.router = model.router.to(llm_device)
    if hasattr(model, 'experts_layer'):
        model.experts_layer = model.experts_layer.to(llm_device)
    if hasattr(model, 'classifier'):
        model.classifier = model.classifier.to(llm_device)
    if hasattr(model, 'culture_classifier'):
        model.culture_classifier = model.culture_classifier.to(llm_device)

    print(f"   ✅ MoE layers moved to {llm_device}")

    # 7. 设置为评估模式
    model.eval()

    print("\n" + "="*80)
    print("✅ CultureMoE Model Loaded Successfully!")
    print("="*80)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*80)
    print("")

    return model, tokenizer, moe_config


def evaluate_model(model, tokenizer, test_dataset, device, batch_size=8, num_classes=2):
    """
    在测试集上评估模型

    Args:
        model: CultureMoE 模型
        tokenizer: Tokenizer
        test_dataset: 测试数据集
        device: 设备
        batch_size: 批次大小
        num_classes: 分类数量

    Returns:
        dict: 评估结果
    """
    print("\n" + "="*80)
    print("Evaluating Model on Test Set")
    print("="*80)
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("="*80)
    print("")

    # 创建 DataLoader
    data_collator = DualClassificationDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        padding=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2
    )

    # 评估
    model.eval()
    all_preds = []
    all_labels = []
    all_culture_preds = []
    all_culture_labels = []

    print("Running evaluation...")

    # ✅ 获取模型的实际设备（处理 device_map="auto" 的情况）
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # ✅ 移动到模型所在的设备
            input_ids = batch['input_ids'].to(model_device)
            attention_mask = batch['attention_mask'].to(model_device)

            # ✅ 处理标签（可能是张量或列表）
            if isinstance(batch['labels'], torch.Tensor):
                labels = batch['labels']
                culture_labels = batch['culture_labels']
            else:
                labels = torch.tensor(batch['labels'])
                culture_labels = torch.tensor(batch['culture_labels'])

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # ✅ 处理模型输出（可能是字典或张量）
            if isinstance(outputs, dict):
                # 如果是字典，提取 logits 和 culture_logits
                logits = outputs['logits']
                culture_logits = outputs.get('culture_logits', logits)  # 如果没有 culture_logits，使用 logits
            else:
                # 如果是张量，直接使用
                logits = outputs
                culture_logits = outputs  # 使用相同的 logits

            preds = torch.argmax(logits, dim=-1)
            culture_preds = torch.argmax(culture_logits, dim=-1)

            # 收集结果（移到 CPU）
            preds_np = preds.cpu().numpy()
            if preds_np.ndim > 1:
                preds_np = preds_np.flatten()
            all_preds.extend(preds_np.tolist())

            # ✅ 统一处理函数：将任何格式转换为一维列表
            def to_flat_list(data):
                """将张量/数组/列表转换为一维列表"""
                if isinstance(data, torch.Tensor):
                    data = data.cpu().numpy()
                elif isinstance(data, list):
                    # 尝试转换为数组
                    try:
                        data = np.array(data)
                    except (ValueError, TypeError):
                        # 如果失败，手动展平
                        flat = []
                        for item in data:
                            if isinstance(item, (list, tuple)):
                                flat.extend(item)
                            else:
                                flat.append(item)
                        return flat

                # 现在 data 应该是 numpy 数组
                if isinstance(data, np.ndarray):
                    if data.ndim > 1:
                        data = data.flatten()
                    return data.tolist()

                return data

            # 处理所有数据
            all_labels.extend(to_flat_list(labels))
            all_culture_preds.extend(to_flat_list(culture_preds))
            all_culture_labels.extend(to_flat_list(culture_labels))

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_culture_preds = np.array(all_culture_preds)
    all_culture_labels = np.array(all_culture_labels)

    # 计算指标
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)

    # 主任务指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print("\n📊 Main Task (Classification):")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1:        {f1:.4f}")

    # 文化任务指标
    culture_accuracy = accuracy_score(all_culture_labels, all_culture_preds)
    culture_precision, culture_recall, culture_f1, _ = precision_recall_fscore_support(
        all_culture_labels, all_culture_preds, average='macro', zero_division=0
    )

    print("\n📊 Culture Task:")
    print(f"   Accuracy:  {culture_accuracy:.4f}")
    print(f"   Precision: {culture_precision:.4f}")
    print(f"   Recall:    {culture_recall:.4f}")
    print(f"   F1:        {culture_f1:.4f}")

    # 详细分类报告
    print("\n📋 Detailed Classification Report (Main Task):")
    print(classification_report(all_labels, all_preds, zero_division=0))

    print("\n📋 Detailed Classification Report (Culture Task):")
    print(classification_report(all_culture_labels, all_culture_preds, zero_division=0))

    # 返回结果
    results = {
        "main_task": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist()
        },
        "culture_task": {
            "accuracy": float(culture_accuracy),
            "precision": float(culture_precision),
            "recall": float(culture_recall),
            "f1": float(culture_f1),
            "predictions": all_culture_preds.tolist(),
            "labels": all_culture_labels.tolist()
        },
        "num_samples": len(all_labels),
        "num_classes": num_classes
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="加载 CultureMoE 模型并在测试集上评估")
    parser.add_argument("--moe_weights_path", type=str, required=True,
                        help="MoE 权重路径")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="评估结果输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="评估批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("CultureMoE Model Evaluation")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MoE weights: {args.moe_weights_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    print("")

    # 1. 加载模型
    model, tokenizer, moe_config = load_culturemoe_model(
        moe_weights_path=args.moe_weights_path,
        device=args.device
    )

    num_classes = moe_config['num_classes']

    # 2. 加载测试数据
    print("\n" + "="*80)
    print("Loading Test Dataset")
    print("="*80)
    print(f"Test file: {args.test_file}")
    print("="*80)
    print("")

    # 检查测试文件是否存在
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")

    # 加载测试数据（不分割，全部作为测试集）
    datasets = load_and_process_dual_classification_data(
        data_path=args.test_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=0.0  # 不分割，全部作为测试集
    )
    test_dataset = datasets['train']  # val_split=0 时，所有数据在 train 中

    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

    # 3. 评估模型
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=args.device,
        batch_size=args.batch_size,
        num_classes=num_classes
    )

    # 4. 保存评估结果
    print("\n" + "="*80)
    print("Saving Evaluation Results")
    print("="*80)

    # 保存详细结果
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Detailed results saved to: {results_file}")

    # 保存摘要
    summary = {
        "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "moe_weights_path": args.moe_weights_path,
        "test_file": args.test_file,
        "num_samples": results['num_samples'],
        "num_classes": results['num_classes'],
        "main_task": {
            "accuracy": results['main_task']['accuracy'],
            "precision": results['main_task']['precision'],
            "recall": results['main_task']['recall'],
            "f1": results['main_task']['f1']
        },
        "culture_task": {
            "accuracy": results['culture_task']['accuracy'],
            "precision": results['culture_task']['precision'],
            "recall": results['culture_task']['recall'],
            "f1": results['culture_task']['f1']
        }
    }

    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Summary saved to: {summary_file}")

    print("\n" + "="*80)
    print("✅ Evaluation Completed Successfully!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print("")
    print("📊 Summary:")
    print(f"   Main Task Accuracy:    {results['main_task']['accuracy']:.4f}")
    print(f"   Main Task F1:          {results['main_task']['f1']:.4f}")
    print(f"   Culture Task Accuracy: {results['culture_task']['accuracy']:.4f}")
    print(f"   Culture Task F1:       {results['culture_task']['f1']:.4f}")
    print("="*80)
    print("")


if __name__ == "__main__":
    main()

