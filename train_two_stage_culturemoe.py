#!/usr/bin/env python3
"""
两阶段训练 CultureMoE：
阶段 1：自动调用 LoRA Only 训练（保存最佳模型）+ 自动合并权重
阶段 2：从合并后的模型开始，冻结 LLM，训练 MoE 部分
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn
from peft import PeftModel
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

    def __init__(self, output_dir: str, stage: str):
        self.output_dir = output_dir
        self.stage = stage
        self.epoch_results = []
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估结束后保存结果"""
        if metrics is not None and state.epoch is not None:
            epoch_result = {
                "stage": self.stage,
                "epoch": int(state.epoch),
                "step": state.global_step,
                **{k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
            }
            self.epoch_results.append(epoch_result)

            # 保存到文件
            results_file = os.path.join(self.output_dir, f"{self.stage}_epoch_eval_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.epoch_results, f, indent=2, ensure_ascii=False)

            # 跟踪最佳准确率
            current_accuracy = metrics.get('eval_accuracy', 0)
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.best_epoch = int(state.epoch)

            print(f"\n📊 [{self.stage.upper()}] Epoch {int(state.epoch)} Evaluation Results:")
            print(f"   Accuracy:  {current_accuracy:.4f}")
            print(f"   Precision: {metrics.get('eval_precision', 0):.4f}")
            print(f"   Recall:    {metrics.get('eval_recall', 0):.4f}")
            print(f"   F1:        {metrics.get('eval_f1', 0):.4f}")
            print(f"   Loss:      {metrics.get('eval_loss', 0):.4f}")
            if current_accuracy > self.best_accuracy - 0.0001:  # 当前是最佳
                print(f"   🏆 New best accuracy!")
            print(f"   Saved to: {results_file}\n")


def stage1_train_lora(args):
    """
    阶段 1：自动调用 LoRA Only 训练 + 合并权重
    """
    print("\n" + "="*80)
    print("STAGE 1: LoRA Fine-tuning (Auto Call)")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.train_file}")
    print(f"Num classes: {args.num_classes}")
    print(f"Epochs: {args.stage1_epochs}")
    print("="*80)
    print("")

    # 1. 调用 LoRA Only 训练脚本
    stage1_output = os.path.join(args.output_dir, "stage1_lora")
    os.makedirs(stage1_output, exist_ok=True)

    print("1. Calling LoRA Only training script...")
    print(f"   Output: {stage1_output}")

    # 构建训练命令
    train_cmd = [
        "python", "train_and_eval_lora_only.py",
        "--model_path", args.model_path,
        "--train_file", args.train_file,
        "--output_dir", stage1_output,
        "--num_train_epochs", str(args.stage1_epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--per_device_eval_batch_size", str(args.eval_batch_size),
        "--learning_rate", str(args.learning_rate),
        "--lora_rank", str(args.lora_rank),
        "--max_length", str(args.max_length),
        "--val_split", str(args.val_split),
        "--num_classes", str(args.num_classes),
        "--save_model"  # 保存最佳模型
    ]

    # 运行训练
    result = subprocess.run(train_cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("LoRA Only training failed!")

    print("   ✅ LoRA Only training completed")

    # 2. 读取评估结果
    eval_results_path = os.path.join(stage1_output, "eval_results.json")
    with open(eval_results_path, 'r') as f:
        stage1_metrics = json.load(f)

    print(f"\n2. Stage 1 Results:")
    print(f"   Accuracy: {stage1_metrics.get('eval_accuracy', 0):.4f}")
    print(f"   F1:       {stage1_metrics.get('eval_f1', 0):.4f}")

    # 3. 合并 LoRA 权重
    print("\n3. Merging LoRA weights...")
    merged_model_path = os.path.join(args.output_dir, "merged_model_temp")

    # 调用合并脚本
    merge_cmd = [
        "python", "merge_lora_weights.py",
        "--base_model_path", args.model_path,
        "--lora_model_path", stage1_output,
        "--output_path", merged_model_path,
        "--device", "auto"
    ]

    result = subprocess.run(merge_cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("LoRA merging failed!")

    print("   ✅ LoRA weights merged")
    print(f"   Merged model: {merged_model_path}")

    # 4. 清理 checkpoints（如果不保存模型）
    if not args.save_model:
        print("\n4. Cleaning up checkpoints...")
        import shutil
        checkpoints = [d for d in os.listdir(stage1_output) if d.startswith("checkpoint-")]
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(stage1_output, checkpoint)
            if os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path)
        print(f"   ✅ Removed {len(checkpoints)} checkpoints")

    print("\n" + "="*80)
    print("Stage 1 Completed!")
    print("="*80)
    print(f"   Accuracy: {stage1_metrics.get('eval_accuracy', 0):.4f}")
    print(f"   F1:       {stage1_metrics.get('eval_f1', 0):.4f}")
    print(f"   Merged model: {merged_model_path}")
    print("="*80)
    print("")

    return merged_model_path, stage1_metrics


def stage2_train_moe(args, merged_model_path: str, stage1_metrics: Dict):
    """
    阶段 2：冻结 LLM，训练 MoE 部分
    """
    print("\n" + "="*80)
    print("STAGE 2: Training MoE (Frozen LLM)")
    print("="*80)
    print(f"Merged model: {merged_model_path}")
    print(f"Num experts: {args.num_experts}")
    print(f"Epochs: {args.stage2_epochs}")
    print(f"Output: {args.output_dir}/stage2")
    print("="*80)
    print("")

    stage2_output = os.path.join(args.output_dir, "stage2")
    os.makedirs(stage2_output, exist_ok=True)

    # 1. 加载 tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载合并后的 LLM（冻结）
    print("\n2. Loading merged LLM (will be frozen)...")
    llama_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
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

    # 加载阶段1的分类头（可选）
    classification_head_path = os.path.join(merged_model_path, "classification_head.pt")
    if os.path.exists(classification_head_path):
        print("   Loading Stage 1 classification head...")
        # 注意：这里可能需要调整，因为 MoE 的分类头结构可能不同
        # 如果结构不同，就不加载，让 MoE 从头训练分类头

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
        output_dir=stage2_output,
        num_train_epochs=args.stage2_epochs,
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
        logging_dir=os.path.join(stage2_output, "logs"),
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
    epoch_callback = EpochEvalCallback(args.output_dir, "stage2")

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
    print("\n7. Starting Stage 2 training...")
    print("="*80)
    trainer.train()

    # 11. 最终评估
    print("\n8. Final evaluation...")
    final_metrics = trainer.evaluate()

    # 保存最终评估结果
    with open(os.path.join(args.output_dir, "stage2_final_eval.json"), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "="*80)
    print("Stage 2 Completed!")
    print("="*80)
    print(f"   Final Accuracy: {final_metrics.get('eval_accuracy', 0):.4f}")
    print(f"   Final F1:       {final_metrics.get('eval_f1', 0):.4f}")
    print("="*80)
    print("")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="两阶段训练 CultureMoE")

    # 基本参数
    parser.add_argument("--model_path", type=str, required=True, help="基座模型路径")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_classes", type=int, default=2, help="分类数量")
    parser.add_argument("--use_culture_loss", type=lambda x: x.lower() == 'true', default=True, help="是否使用文化损失")
    parser.add_argument("--culture_loss_lambda", type=float, default=0.05, help="文化损失权重")
    parser.add_argument("--save_model", type=lambda x: x.lower() == 'true', default=False, help="是否保存模型")

    # 阶段1参数（LoRA）
    parser.add_argument("--stage1_epochs", type=int, default=3, help="阶段1训练轮数")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    # 阶段2参数（MoE）
    parser.add_argument("--stage2_epochs", type=int, default=5, help="阶段2训练轮数")
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

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "="*80)
    print("Two-Stage CultureMoE Training")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    print("")

    # 阶段1：LoRA 微调
    merged_model_path, stage1_metrics = stage1_train_lora(args)

    # 阶段2：训练 MoE
    stage2_metrics = stage2_train_moe(args, merged_model_path, stage1_metrics)

    # 生成最终报告
    print("\n" + "="*80)
    print("Final Report")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    print("Stage 1 (LoRA Fine-tuning):")
    print(f"   Accuracy: {stage1_metrics.get('eval_accuracy', 0):.4f}")
    print(f"   F1:       {stage1_metrics.get('eval_f1', 0):.4f}")
    print("")
    print("Stage 2 (MoE Training):")
    print(f"   Accuracy: {stage2_metrics.get('eval_accuracy', 0):.4f}")
    print(f"   F1:       {stage2_metrics.get('eval_f1', 0):.4f}")
    print("")
    print(f"Improvement: {(stage2_metrics.get('eval_accuracy', 0) - stage1_metrics.get('eval_accuracy', 0)):.4f}")
    print("="*80)

    # 保存最终报告
    final_report = {
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "improvement": stage2_metrics.get('eval_accuracy', 0) - stage1_metrics.get('eval_accuracy', 0)
    }

    with open(os.path.join(args.output_dir, "final_report.json"), 'w') as f:
        json.dump(final_report, f, indent=2)

    # 清理临时文件
    print("\nCleaning up temporary files...")
    import shutil
    if os.path.exists(merged_model_path):
        shutil.rmtree(merged_model_path)
    print("   ✅ Temporary files removed")

    print("\n✅ Two-stage training completed!")
    print(f"   Results saved to: {args.output_dir}")
    print("")


if __name__ == "__main__":
    main()

