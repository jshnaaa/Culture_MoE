#!/usr/bin/env python3
"""
使用 LoRA 微调 LLaMA 3.1 模型（无 MoE）
90% 训练，10% 验证，二分类任务
"""

import json
import os
import sys
from dataclasses import dataclass, field

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data
from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class LoRATrainingArguments:
    """LoRA 微调参数"""

    # 模型参数
    model_path: str = field(
        metadata={"help": "LLaMA 模型路径"}
    )

    # 数据参数
    train_file: str = field(
        metadata={"help": "训练数据文件"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    val_split: float = field(
        default=0.1,
        metadata={"help": "验证集比例"}
    )

    # LoRA 参数
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA 目标模块"}
    )

    # 训练参数
    output_dir: str = field(
        default="./output/lora_only",
        metadata={"help": "输出目录"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "训练批次大小"}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "评估批次大小"}
    )
    learning_rate: float = field(
        default=5e-6,  # ✅ 降低学习率：从 2e-5 改为 5e-6
        metadata={"help": "学习率"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志步数"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "保存步数"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "评估步数"}
    )


class BinaryClassificationModel(torch.nn.Module):
    """带分类头的 LLaMA 模型"""

    def __init__(self, llama_model, num_classes=2, use_fp16=True):
        super().__init__()
        self.llama_model = llama_model
        self.config = llama_model.config
        self.use_fp16 = use_fp16

        # 分类头（添加 LayerNorm 以稳定训练）
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.config.hidden_size),  # ✅ 添加 LayerNorm
            torch.nn.Linear(self.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_classes)
        )

        # ✅ 如果使用 fp16，将分类头转换为 float16
        if use_fp16:
            self.classifier = self.classifier.half()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        前向传播

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            labels: [B]
            **kwargs: 其他字段（input_ids_mask, attention_mask_mask, culture_labels）会被忽略
        """
        # LLaMA 前向传播（只使用第一路输入）
        outputs = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 使用最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [B, L, H]

        # 池化：使用平均池化
        pooled = hidden_states.mean(dim=1)  # [B, H]

        # 分类
        logits = self.classifier(pooled)  # [B, num_classes]

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_lora_only(args: LoRATrainingArguments):
    """
    训练 LoRA 微调模型

    数据格式：
    {
        "instruction": "...",
        "instruction_mask": "...",
        "input": "...",
        "output": 0,  # 整数：0, 1, 2
        "label": "0"  # 字符串：文化维度标签
    }
    """

    print("="*60)
    print("Training LLaMA 3.1 with LoRA (No MoE)")
    print("="*60)

    # 1. 加载 tokenizer
    print(f"\n1. Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载数据（使用相同的数据处理器）
    print(f"\n2. Loading data from {args.train_file}...")
    print(f"   Data format: instruction/instruction_mask/input/output/label")
    data = load_and_process_dual_classification_data(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=args.val_split,
        num_proc=4
    )
    train_dataset = data['train']
    val_dataset = data['validation']

    print(f"   ✅ Train dataset size: {len(train_dataset)}")
    print(f"   ✅ Validation dataset size: {len(val_dataset)}")

    # 检查数据格式
    print(f"\n   Dataset columns: {train_dataset.column_names}")
    if 'culture_labels' in train_dataset.column_names:
        print(f"   ✅ Culture labels found in dataset")

    # 3. 加载基础模型
    print(f"\n3. Loading base LLaMA model from {args.model_path}...")

    # ✅ 检查是否支持 BF16（更稳定）
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    llama_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    print(f"   ✅ Base model loaded (using {'bfloat16' if use_bf16 else 'float16'})")

    # 4. 应用 LoRA
    print(f"\n4. Applying LoRA to LLaMA model...")
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    llama_model = get_peft_model(llama_model, lora_config)

    print("   LoRA Configuration:")
    print(f"     Rank: {args.lora_rank}")
    print(f"     Alpha: {args.lora_alpha}")
    print(f"     Dropout: {args.lora_dropout}")
    print(f"     Target modules: {target_modules}")
    print("\n   Trainable Parameters:")
    llama_model.print_trainable_parameters()
    print("   ✅ LoRA applied")

    # 5. 创建分类模型
    print(f"\n5. Creating classification model...")
    model = BinaryClassificationModel(llama_model, num_classes=2, use_fp16=True)
    print("   ✅ Classification model created (using fp16)")

    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        fp16_full_eval=True,  # 评估时也使用 fp16
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        # ✅ 梯度裁剪，防止梯度爆炸
        max_grad_norm=1.0,
        # ✅ 使用更稳定的优化器设置
        optim="adamw_torch",
        warmup_steps=100,  # 添加 warmup
    )

    # 7. Data Collator（使用自定义 collator）
    data_collator = DualClassificationDataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_length
    )

    # 8. 创建 Trainer
    print(f"\n6. Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9. 训练
    print(f"\n7. Starting training...")
    print("="*60)
    train_result = trainer.train()

    # 10. 保存模型
    print(f"\n8. Saving model to {args.output_dir}...")
    trainer.save_model()
    trainer.save_state()

    # 保存 LoRA 权重
    model.llama_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("   ✅ Model saved")

    # 11. 最终评估
    print(f"\n9. Final evaluation on validation set...")
    metrics = trainer.evaluate()

    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    print(f"   Accuracy:   {metrics['eval_accuracy']:.4f}")
    print(f"   Precision:  {metrics['eval_precision']:.4f}")
    print(f"   Recall:     {metrics['eval_recall']:.4f}")
    print(f"   F1:         {metrics['eval_f1']:.4f}")
    print("="*60)

    # 保存指标
    with open(os.path.join(args.output_dir, "eval_results.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Training and evaluation completed!")
    print(f"   Model saved to: {args.output_dir}")

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LoRA 微调 LLaMA 3.1（无 MoE）")
    parser.add_argument("--model_path", type=str, required=True,
                        help="LLaMA 模型路径")
    parser.add_argument("--train_file", type=str, required=True,
                        help="训练数据文件")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available!")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

    # 创建训练参数
    training_args = LoRATrainingArguments(
        model_path=args.model_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_length=args.max_length,
        val_split=args.val_split,
    )

    # 训练
    metrics = train_lora_only(training_args)


if __name__ == "__main__":
    main()

