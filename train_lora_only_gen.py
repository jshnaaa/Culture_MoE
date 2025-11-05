#!/usr/bin/env python3
"""
训练 LoRA Only 模型（生成式版本）

使用方法：
    python train_lora_only_gen.py \
        --model_name_or_path /path/to/base_model \
        --data_path /path/to/train_data.json \
        --output_dir /path/to/output \
        --lora_rank 8 \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_and_process_data(data_path: str, tokenizer, max_length: int = 512):
    """加载并处理生成式数据（只计算答案部分的损失）"""
    print(f"Loading data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    def preprocess_function(examples):
        """预处理函数 - 只计算答案部分的损失"""
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i]
            output = str(examples['output'][i])  # 确保是字符串

            # 构建 prompt（不包含答案）
            prompt = f"{instruction}\n{input_text}\nAnswer:"

            # 构建完整文本（包含答案）
            full_text = f"{prompt} {output}"

            # Tokenize prompt（用于确定忽略的位置）
            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length - 10  # 留空间给答案
            )

            # Tokenize 完整文本
            full_tokens = tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length
            )

            # 创建 labels：prompt 部分用 -100（忽略），答案部分正常
            prompt_len = len(prompt_tokens["input_ids"])
            labels = [-100] * prompt_len + full_tokens["input_ids"][prompt_len:]

            # 确保 labels 和 input_ids 长度一致
            if len(labels) < len(full_tokens["input_ids"]):
                labels.extend(full_tokens["input_ids"][len(labels):])
            elif len(labels) > len(full_tokens["input_ids"]):
                labels = labels[:len(full_tokens["input_ids"])]

            model_inputs["input_ids"].append(full_tokens["input_ids"])
            model_inputs["attention_mask"].append(full_tokens["attention_mask"])
            model_inputs["labels"].append(labels)

        return model_inputs

    # 转换为 Dataset
    dataset = Dataset.from_dict({
        'instruction': [item['instruction'] for item in data],
        'input': [item['input'] for item in data],
        'output': [item['output'] for item in data]
    })

    # 预处理
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing data (answer-only loss)"
    )

    print("✅ Using answer-only loss (prompt tokens will be ignored)")

    return processed_dataset


def main():
    parser = argparse.ArgumentParser(description="训练 LoRA Only 模型（生成式版本）")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="基础模型路径")
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="每个设备的批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Training LoRA Only Model (Generative Version)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {args.model_name_or_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    print("")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded\n")

    # 加载模型
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        # 禁用 tensor parallel（需要 torch >= 2.5）
        attn_implementation="eager"  # 使用标准注意力实现
    )
    print("✅ Base model loaded\n")

    # 配置 LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA configured\n")

    # 加载数据
    print("Loading and processing data...")
    train_dataset = load_and_process_data(
        args.data_path,
        tokenizer,
        args.max_length
    )
    print(f"✅ Processed {len(train_dataset)} training samples\n")

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        # 多卡训练配置
        ddp_find_unused_parameters=False,  # 加速训练
        dataloader_pin_memory=True,        # 加速数据加载
    )

    # 数据整理器 - 使用自定义的 collator
    from transformers import default_data_collator

    def custom_data_collator(features):
        """自定义数据整理器，正确处理 labels（支持多卡训练）"""
        import torch

        # 获取最大长度
        max_length = max(len(f["input_ids"]) for f in features)

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            # Padding
            padding_length = max_length - len(input_ids)

            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length  # padding 的 labels 也用 -100

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # 转换为 tensor（不指定设备，让 Trainer 自动处理）
        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

        return batch

    data_collator = custom_data_collator

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # 训练
    print("="*80)
    print("Starting training...")
    print("="*80)
    print("")

    trainer.train()

    # 保存模型
    print("\n" + "="*80)
    print("Saving model...")
    print("="*80)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Model saved to: {args.output_dir}")

    # 保存训练配置
    config = {
        "model_type": "LoRA_Only_Generative",
        "base_model": args.model_name_or_path,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "training_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    config_file = os.path.join(args.output_dir, "training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ Config saved to: {config_file}")

    print("\n" + "="*80)
    print("✅ Training completed successfully!")
    print("="*80)
    print("")


if __name__ == "__main__":
    main()

