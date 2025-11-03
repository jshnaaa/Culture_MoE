#!/usr/bin/env python3
"""
训练 CultureMoE 模型（生成式版本）

注意：生成式版本的 CultureMoE 仍然使用分类头，但训练数据是生成式格式
模型会学习从生成式数据中提取特征并进行分类

使用方法：
    python train_culturemoe_gen.py \
        --merged_llm_path /path/to/merged_llm \
        --data_path /path/to/train_data.json \
        --output_dir /path/to/output \
        --num_experts 6 \
        --learning_rate 1e-4 \
        --num_train_epochs 10
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs
from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data
from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator


def train_culturemoe(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    output_dir,
    save_steps=500
):
    """训练 CultureMoE 模型"""

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")

        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # 计算损失
            loss = nn.CrossEntropyLoss()(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 统计
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct / train_total:.4f}'
            })

            global_step += 1

            # 定期保存
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                save_model(model, checkpoint_dir)
                print(f"\n✅ Checkpoint saved to: {checkpoint_dir}")

        # 计算训练指标
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        print(f"\nTraining - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # 验证
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_dir = os.path.join(output_dir, "best_model")
                save_model(model, best_model_dir)
                print(f"✅ Best model saved (acc: {best_val_acc:.4f})")

    return best_val_acc


def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def save_model(model, output_dir):
    """保存模型"""
    os.makedirs(output_dir, exist_ok=True)

    # 只保存 MoE 部分的权重
    moe_state_dict = {}
    for name, param in model.named_parameters():
        if not name.startswith('llama_model.'):
            moe_state_dict[name] = param.cpu()

    torch.save(moe_state_dict, os.path.join(output_dir, "moe_weights.bin"))


def main():
    parser = argparse.ArgumentParser(description="训练 CultureMoE 模型（生成式版本）")
    parser.add_argument("--merged_llm_path", type=str, required=True,
                        help="合并后的 LLM 路径")
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--num_experts", type=int, default=6,
                        help="专家数量")
    parser.add_argument("--shared_hidden_dim", type=int, default=2048,
                        help="Shared 层隐藏维度")
    parser.add_argument("--router_hidden_dim", type=int, default=512,
                        help="Router 隐藏维度")
    parser.add_argument("--experts_hidden_dim", type=int, default=1024,
                        help="Experts 隐藏维度")
    parser.add_argument("--moe_lora_rank", type=int, default=8,
                        help="MoE LoRA rank")
    parser.add_argument("--classification_hidden_dim", type=int, default=512,
                        help="分类头隐藏维度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="注意力头数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Training CultureMoE Model (Generative Version)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Merged LLM: {args.merged_llm_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    print("")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.merged_llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded\n")

    # 加载合并后的 LLM
    print("Loading merged LLM...")
    llama_model = AutoModelForCausalLM.from_pretrained(
        args.merged_llm_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 冻结 LLM
    for param in llama_model.parameters():
        param.requires_grad = False

    print("✅ Merged LLM loaded and frozen\n")

    # 从数据中推断类别数
    print("Loading data to infer num_classes...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 从 output 字段推断类别数（生成式数据的 output 是字符串）
    unique_labels = set(int(item['output']) for item in data)
    num_classes = len(unique_labels)
    print(f"✅ Inferred num_classes: {num_classes}\n")

    # 创建 CultureMoE 模型
    print("Creating CultureMoE model...")
    moe_args = ModelArgs(
        num_experts=args.num_experts,
        shared_hidden_dim=args.shared_hidden_dim,
        router_hidden_dim=args.router_hidden_dim,
        experts_hidden_dim=args.experts_hidden_dim,
        lora_rank=args.moe_lora_rank,
        num_classes=num_classes,
        classification_hidden_dim=args.classification_hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads
    )

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    )

    # 将 MoE 层移到正确的设备
    llm_device = next(llama_model.parameters()).device
    if hasattr(model, 'shared'):
        model.shared = model.shared.to(llm_device)
    if hasattr(model, 'router'):
        model.router = model.router.to(llm_device)
    if hasattr(model, 'experts_layer'):
        model.experts_layer = model.experts_layer.to(llm_device)
    if hasattr(model, 'classifier'):
        model.classifier = model.classifier.to(llm_device)

    print("✅ CultureMoE model created\n")

    # 加载数据
    print("Loading and processing data...")
    datasets = load_and_process_dual_classification_data(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=512,
        val_split=args.val_split
    )

    train_dataset = datasets['train']
    val_dataset = datasets.get('val', None)

    print(f"✅ Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"✅ Val samples: {len(val_dataset)}\n")

    # 创建 DataLoader
    data_collator = DualClassificationDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        padding=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator
        )

    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 训练
    print("="*80)
    print("Starting training...")
    print("="*80)
    print("")

    best_val_acc = train_culturemoe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_train_epochs,
        device=llm_device,
        output_dir=args.output_dir,
        save_steps=args.save_steps
    )

    # 保存最终模型
    final_model_dir = os.path.join(args.output_dir, "final_model")
    save_model(model, final_model_dir)

    # 保存配置
    config = {
        "model_type": "CultureMoE_Generative",
        "merged_llm_path": args.merged_llm_path,
        "architecture": "Shared + Router + Experts",
        "num_experts": args.num_experts,
        "num_classes": num_classes,
        "shared_hidden_dim": args.shared_hidden_dim,
        "router_hidden_dim": args.router_hidden_dim,
        "experts_hidden_dim": args.experts_hidden_dim,
        "moe_lora_rank": args.moe_lora_rank,
        "classification_hidden_dim": args.classification_hidden_dim,
        "dropout": args.dropout,
        "num_heads": args.num_heads,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "best_accuracy": float(best_val_acc),
        "training_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    config_file = os.path.join(args.output_dir, "moe_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 保存 tokenizer
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "="*80)
    print("✅ Training completed successfully!")
    print("="*80)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("="*80)
    print("")


if __name__ == "__main__":
    main()

