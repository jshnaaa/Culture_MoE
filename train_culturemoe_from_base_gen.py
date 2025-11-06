#!/usr/bin/env python3
"""
从 Base 模型 + LoRA 权重训练 CultureMoE（生成式版本）

使用方法：
    python train_culturemoe_from_base_gen.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --num_classes 5 \
        --use_culture_loss True
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator
from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


def load_model_from_components(
    base_model_path: str,
    lora_weights_path: str,
    moe_args: ModelArgs,
    device: str = "cuda"
):
    """
    从 Base 模型 + LoRA 权重创建 CultureMoE 模型

    Args:
        base_model_path: Base 模型路径
        lora_weights_path: LoRA 权重路径
        moe_args: MoE 配置
        device: 设备

    Returns:
        model: CultureMoE 模型
        tokenizer: Tokenizer
    """
    print("\n" + "="*80)
    print("Loading Model from Components")
    print("="*80)
    print(f"Base model: {base_model_path}")
    print(f"LoRA weights: {lora_weights_path}")
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

    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        is_trainable=False
    )
    print("   ✅ LoRA weights loaded")

    print("   Merging LoRA weights into base model...")
    merged_model = model_with_lora.merge_and_unload()
    print("   ✅ LoRA weights merged")

    # 4. 冻结 LLaMA 参数
    print("\n4. Freezing LLaMA parameters...")
    for param in merged_model.parameters():
        param.requires_grad = False
    print("   ✅ LLaMA parameters frozen")

    # 5. 创建 CultureMoE 模型
    print("\n5. Creating CultureMoE model...")
    culturemoe_model = LlamaSharedRouterExpertsModel(
        llama_model=merged_model,
        config=merged_model.config,
        args=moe_args
    )
    print("   ✅ CultureMoE model created")

    # 6. 移动到设备
    print(f"\n6. Moving model to {device}...")
    culturemoe_model = culturemoe_model.to(device)
    print("   ✅ Model ready")

    # 打印参数统计
    total_params = sum(p.numel() for p in culturemoe_model.parameters())
    trainable_params = sum(p.numel() for p in culturemoe_model.parameters() if p.requires_grad)
    print(f"\n📊 Model Info:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"   Device: {device}")

    print("\n" + "="*80)
    print("✅ Model Loaded Successfully!")
    print("="*80)
    print("")

    return culturemoe_model, tokenizer


def train_epoch(model, train_loader, optimizer, device, use_culture_loss, culture_loss_lambda):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_culture_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        input_ids_mask = batch['input_ids_mask'].to(device)
        attention_mask_mask = batch['attention_mask_mask'].to(device)
        labels = batch['labels'].to(device)
        culture_labels = batch['culture_labels']

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids_mask=input_ids_mask,
            attention_mask_mask=attention_mask_mask,
            labels=labels,
            culture_labels=culture_labels,
            use_culture_loss=use_culture_loss,
            culture_loss_lambda=culture_loss_lambda
        )

        loss = outputs['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_cls_loss += outputs['classification_loss'].item()
        if use_culture_loss:
            total_culture_loss += outputs['culture_loss'].item()

        # 收集预测和标签
        preds = torch.argmax(outputs['logits'], dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls_loss': f"{outputs['classification_loss'].item():.4f}"
        })

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_culture_loss = total_culture_loss / len(train_loader) if use_culture_loss else 0.0

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'culture_loss': avg_culture_loss,
        'accuracy': accuracy
    }


def evaluate(model, val_loader, device, use_culture_loss, culture_loss_lambda, num_classes):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_culture_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids_mask = batch['input_ids_mask'].to(device)
            attention_mask_mask = batch['attention_mask_mask'].to(device)
            labels = batch['labels'].to(device)
            culture_labels = batch['culture_labels']

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_mask=input_ids_mask,
                attention_mask_mask=attention_mask_mask,
                labels=labels,
                culture_labels=culture_labels,
                use_culture_loss=use_culture_loss,
                culture_loss_lambda=culture_loss_lambda
            )

            # 统计
            total_loss += outputs['loss'].item()
            total_cls_loss += outputs['classification_loss'].item()
            if use_culture_loss:
                total_culture_loss += outputs['culture_loss'].item()

            # 收集预测和标签
            preds = torch.argmax(outputs['logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    avg_cls_loss = total_cls_loss / len(val_loader)
    avg_culture_loss = total_culture_loss / len(val_loader) if use_culture_loss else 0.0

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)

    if num_classes == 2:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', pos_label=1, zero_division=0
        )
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'culture_loss': avg_culture_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description="从 Base + LoRA 训练 CultureMoE")

    # 模型路径
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, required=True)

    # 数据
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # 训练参数
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--use_culture_loss", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--culture_loss_lambda", type=float, default=0.5)
    parser.add_argument("--use_instruction_mask", type=lambda x: x.lower() == 'true', default=True)

    # MoE 参数
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--shared_hidden_dim", type=int, default=2048)
    parser.add_argument("--router_hidden_dim", type=int, default=1024)
    parser.add_argument("--experts_hidden_dim", type=int, default=2048)
    parser.add_argument("--moe_lora_rank", type=int, default=16)
    parser.add_argument("--classification_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=8)

    # 优化器参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)

    # 保存选项
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--model_save_path", type=str, default=None)

    # 设备
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("CultureMoE Training (From Base + LoRA)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.lora_weights_path}")
    print(f"Train file: {args.train_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Num classes: {args.num_classes}")
    print(f"Use culture loss: {args.use_culture_loss}")
    print(f"Use instruction mask: {args.use_instruction_mask}")
    print("="*80)
    print("")

    # 创建 MoE 配置
    moe_args = ModelArgs(
        num_experts=args.num_experts,
        shared_hidden_dim=args.shared_hidden_dim,
        router_hidden_dim=args.router_hidden_dim,
        experts_hidden_dim=args.experts_hidden_dim,
        lora_rank=args.moe_lora_rank,
        num_classes=args.num_classes,
        classification_hidden_dim=args.classification_hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads
    )

    # 加载模型
    model, tokenizer = load_model_from_components(
        args.base_model_path,
        args.lora_weights_path,
        moe_args,
        args.device
    )

    # 加载数据
    print("Loading and processing data...")
    datasets = load_and_process_dual_classification_data(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=args.val_split,
        use_instruction_mask=args.use_instruction_mask
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")

    # 创建 DataLoader
    data_collator = DualClassificationDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.num_workers
    )

    # 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 训练循环
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    print("")

    best_accuracy = 0.0
    best_epoch = 0
    epoch_results = []

    for epoch in range(args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            args.use_culture_loss, args.culture_loss_lambda
        )

        # 评估
        val_metrics = evaluate(
            model, val_loader, args.device,
            args.use_culture_loss, args.culture_loss_lambda,
            args.num_classes
        )

        # 保存结果
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'eval_loss': val_metrics['loss'],
            'eval_accuracy': val_metrics['accuracy'],
            'eval_precision': val_metrics['precision'],
            'eval_recall': val_metrics['recall'],
            'eval_f1': val_metrics['f1']
        }
        epoch_results.append(epoch_result)

        # 打印结果
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"   Eval Loss:  {val_metrics['loss']:.4f}, Eval Acc:  {val_metrics['accuracy']:.4f}")
        print(f"   Eval Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

        # 保存最佳模型
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            best_epoch = epoch + 1

            # 保存 MoE 权重
            best_moe_dir = os.path.join(args.output_dir, "best_moe")
            os.makedirs(best_moe_dir, exist_ok=True)

            # 保存 MoE 配置
            moe_config = {
                'num_experts': args.num_experts,
                'shared_hidden_dim': args.shared_hidden_dim,
                'router_hidden_dim': args.router_hidden_dim,
                'experts_hidden_dim': args.experts_hidden_dim,
                'moe_lora_rank': args.moe_lora_rank,
                'num_classes': args.num_classes,
                'classification_hidden_dim': args.classification_hidden_dim,
                'dropout': args.dropout,
                'num_heads': args.num_heads
            }

            with open(os.path.join(best_moe_dir, "moe_config.json"), 'w') as f:
                json.dump(moe_config, f, indent=2)

            # 保存 MoE 权重
            torch.save(model.state_dict(), os.path.join(best_moe_dir, "moe_state_dict.pt"))

            print(f"   🏆 New best model! Accuracy: {best_accuracy:.4f}")
            print(f"   ✅ Best MoE weights saved to: {best_moe_dir}")

        print(f"   Best so far: Epoch {best_epoch}, Accuracy: {best_accuracy:.4f}")

        # 保存 epoch 结果
        with open(os.path.join(args.output_dir, "epoch_eval_results.json"), 'w') as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)

    # 保存最终配置
    final_config = {
        'base_model': args.base_model_path,
        'lora_weights': args.lora_weights_path,
        'num_classes': args.num_classes,
        'use_culture_loss': args.use_culture_loss,
        'culture_loss_lambda': args.culture_loss_lambda,
        'num_epochs': args.num_epochs,
        'best_epoch': best_epoch,
        'best_accuracy': best_accuracy,
        'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(final_config, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ Training Completed!")
    print("="*80)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)
    print("")


if __name__ == "__main__":
    main()

