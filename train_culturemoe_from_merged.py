#!/usr/bin/env python3
"""
从合并后的模型训练 CultureMoE
假设已经完成：
1. LoRA Only 训练（train_and_eval_lora_only.py）
2. 权重合并（merge_lora_weights.py）

本脚本只负责：冻结 LLM，训练 MoE 部分
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict

# ✅ 限制只使用一个 GPU（避免 DataParallel 导致的设备不匹配问题）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback
)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data
from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs
from src.llamafactory.train.classification.trainer import ClassificationTrainer
from src.llamafactory.train.classification.metrics import compute_classification_metrics


class EpochEvalCallback(TrainerCallback):
    """每个 epoch 结束后保存评估结果的回调"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.epoch_results = []
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估结束后保存结果"""
        if metrics is not None and state.epoch is not None:
            epoch_result = {
                "epoch": int(state.epoch),
                "step": state.global_step,
                **{k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
            }
            self.epoch_results.append(epoch_result)

            # 保存到文件
            results_file = os.path.join(self.output_dir, "epoch_eval_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.epoch_results, f, indent=2, ensure_ascii=False)

            # 跟踪最佳准确率
            current_accuracy = metrics.get('eval_accuracy', 0)
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.best_epoch = int(state.epoch)

            print(f"\n📊 Epoch {int(state.epoch)} Evaluation Results:")
            print(f"   Accuracy:  {current_accuracy:.4f}")
            print(f"   Precision: {metrics.get('eval_precision', 0):.4f}")
            print(f"   Recall:    {metrics.get('eval_recall', 0):.4f}")
            print(f"   F1:        {metrics.get('eval_f1', 0):.4f}")
            print(f"   Loss:      {metrics.get('eval_loss', 0):.4f}")
            if current_accuracy > self.best_accuracy - 0.0001:  # 当前是最佳
                print(f"   🏆 New best accuracy!")
            print(f"   Saved to: {results_file}\n")


def train_culturemoe(args):
    """
    从合并后的模型训练 CultureMoE
    """
    print("\n" + "="*80)
    print("Training CultureMoE (Frozen LLM)")
    print("="*80)
    print(f"Merged model: {args.merged_model_path}")
    print(f"Dataset: {args.train_file}")
    print(f"Num classes: {args.num_classes}")
    print(f"Num experts: {args.num_experts}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.merged_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载合并后的 LLM（冻结）
    print("\n2. Loading merged LLM (will be frozen)...")
    llama_model = AutoModelForCausalLM.from_pretrained(
        args.merged_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 冻结 LLM
    for param in llama_model.parameters():
        param.requires_grad = False

    print("   ✅ LLM loaded and frozen")
    print(f"   LLM parameters: {sum(p.numel() for p in llama_model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in llama_model.parameters() if p.requires_grad):,}")

    # 3. 创建 MoE 模型
    print("\n3. Creating CultureMoE model...")
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

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    )

    print("   ✅ CultureMoE model created")

    # 打印可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    # 4. 加载数据
    print("\n4. Loading dataset...")
    datasets = load_and_process_dual_classification_data(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=args.val_split
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    print(f"   ✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 5. Data collator
    data_collator = DualClassificationDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        padding=True
    )

    # 6. Training arguments
    print("\n5. Creating training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        # 评估策略
        eval_strategy="epoch",
        save_strategy="no",  # 不保存 checkpoint

        # 日志
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,

        # 优化
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",

        # 其他
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=42,
        dataloader_num_workers=args.num_workers,
    )

    # 7. Callback
    epoch_callback = EpochEvalCallback(args.output_dir)

    # 8. Compute metrics
    def compute_metrics_fn(eval_pred):
        return compute_classification_metrics(eval_pred, num_classes=args.num_classes)

    # 9. Trainer
    print("\n6. Creating trainer...")
    print(f"   Use culture loss: {args.use_culture_loss}")
    print(f"   Culture loss lambda: {args.culture_loss_lambda}")

    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[epoch_callback],
        use_culture_loss=args.use_culture_loss,
        lambda_weight=args.culture_loss_lambda
    )

    # 10. Train
    print("\n7. Starting training...")
    print("="*80)
    trainer.train()

    # 11. 最终评估
    print("\n8. Final evaluation...")
    final_metrics = trainer.evaluate()

    # 保存最终评估结果
    with open(os.path.join(args.output_dir, "final_eval_results.json"), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "="*80)
    print("Training Completed!")
    print("="*80)
    print(f"   Final Accuracy: {final_metrics.get('eval_accuracy', 0):.4f}")
    print(f"   Final F1:       {final_metrics.get('eval_f1', 0):.4f}")
    print(f"   Best Epoch:     {epoch_callback.best_epoch}")
    print(f"   Best Accuracy:  {epoch_callback.best_accuracy:.4f}")
    print("="*80)
    print("")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="从合并后的模型训练 CultureMoE")

    # 基本参数
    parser.add_argument("--merged_model_path", type=str, required=True, help="合并后的模型路径")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_classes", type=int, default=2, help="分类数量")
    parser.add_argument("--use_culture_loss", type=lambda x: x.lower() == 'true', default=True, help="是否使用文化损失")
    parser.add_argument("--culture_loss_lambda", type=float, default=0.5, help="文化损失权重")

    # MoE 参数
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--num_experts", type=int, default=6, help="专家数量")
    parser.add_argument("--shared_hidden_dim", type=int, default=2048, help="Shared层隐藏维度")
    parser.add_argument("--router_hidden_dim", type=int, default=1024, help="Router隐藏维度")
    parser.add_argument("--experts_hidden_dim", type=int, default=2048, help="Experts隐藏维度")
    parser.add_argument("--moe_lora_rank", type=int, default=16, help="MoE LoRA rank")
    parser.add_argument("--classification_hidden_dim", type=int, default=512, help="分类头隐藏维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup比例")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志步数")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")

    args = parser.parse_args()

    # 检查合并后的模型是否存在
    if not os.path.exists(args.merged_model_path):
        print(f"❌ Error: Merged model not found: {args.merged_model_path}")
        print("\nPlease run the following steps first:")
        print("1. Train LoRA Only:")
        print("   sh run_train_lora_only.sh llama 2 true")
        print("2. Merge weights:")
        print("   sh run_merge_lora.sh llama /path/to/lora /path/to/output")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "="*80)
    print("CultureMoE Training (From Merged Model)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Merged model: {args.merged_model_path}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    print("")

    # 训练 MoE
    final_metrics = train_culturemoe(args)

    print("\n✅ Training completed!")
    print(f"   Results saved to: {args.output_dir}")
    print("")


if __name__ == "__main__":
    main()

