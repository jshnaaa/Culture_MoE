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

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_metrics(eval_preds):
    """
    计算评估指标（准确率、困惑度等）

    对于生成式任务，我们计算：
    - Perplexity (困惑度)
    - Accuracy (token-level 准确率)
    """
    predictions, labels = eval_preds

    # predictions 是 logits，shape: (batch_size, seq_len, vocab_size)
    # labels 是真实标签，shape: (batch_size, seq_len)

    # 将 logits 转换为预测的 token IDs
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # 为了节省内存，分批处理
    batch_size = 1000  # 每次处理 1000 个样本
    total_correct = 0
    total_tokens = 0

    for i in range(0, len(predictions), batch_size):
        batch_preds = predictions[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        # 获取预测的 token IDs
        pred_ids = np.argmax(batch_preds, axis=-1)

        # 只计算非 -100 位置的准确率（即答案部分）
        mask = batch_labels != -100

        # Token-level 准确率
        correct = (pred_ids == batch_labels) & mask
        total_correct += correct.sum()
        total_tokens += mask.sum()

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return {
        "accuracy": float(accuracy),
    }


class EpochEvalCallback(TrainerCallback):
    """每个 epoch 结束后保存评估结果，并保存最佳 LoRA 权重"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.epoch_results = []
        self.best_eval_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Epoch 结束时保存结果和最佳 LoRA 权重"""
        # 获取当前 epoch 的评估指标
        if state.log_history:
            # 找到最近的评估结果
            for log in reversed(state.log_history):
                if 'eval_loss' in log:
                    current_eval_loss = log.get('eval_loss', float('inf'))
                    eval_accuracy = log.get('eval_accuracy', None)

                    epoch_result = {
                        'epoch': int(state.epoch),
                        'eval_loss': current_eval_loss,
                        'eval_accuracy': eval_accuracy,
                        'train_loss': log.get('loss', None),
                    }
                    self.epoch_results.append(epoch_result)

                    # 保存到文件
                    results_file = os.path.join(self.output_dir, "epoch_eval_results.json")
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(self.epoch_results, f, indent=2, ensure_ascii=False)

                    print(f"\n📊 Epoch {int(state.epoch)} Results:")
                    print(f"   Train Loss: {epoch_result['train_loss']:.4f}" if epoch_result['train_loss'] else "   Train Loss: N/A")
                    print(f"   Eval Loss:  {current_eval_loss:.4f}")
                    if eval_accuracy is not None:
                        print(f"   Eval Accuracy: {eval_accuracy:.4f}")

                    # 如果是最佳模型，保存 LoRA 权重
                    if current_eval_loss < self.best_eval_loss:
                        self.best_eval_loss = current_eval_loss
                        self.best_epoch = int(state.epoch)

                        # 只保存 LoRA 权重
                        best_lora_dir = os.path.join(self.output_dir, "best_lora")
                        os.makedirs(best_lora_dir, exist_ok=True)

                        if model is not None:
                            model.save_pretrained(best_lora_dir)
                            print(f"   🏆 New best model! Eval Loss: {current_eval_loss:.4f}")
                            print(f"   ✅ Best LoRA weights saved to: {best_lora_dir}")

                    print(f"   Best so far: Epoch {self.best_epoch}, Eval Loss: {self.best_eval_loss:.4f}\n")
                    break


def load_and_process_data(data_path: str, tokenizer, max_length: int = 512, val_split: float = 0.1):
    """加载并处理生成式数据（只计算答案部分的损失）"""
    print(f"Loading data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # 划分训练集和验证集
    if val_split > 0:
        split_idx = int(len(data) * (1 - val_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        print(f"Split: Train={len(train_data)}, Val={len(val_data)} (val_split={val_split})")
    else:
        train_data = data
        val_data = []
        print(f"No validation split")

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
    train_dataset = Dataset.from_dict({
        'instruction': [item['instruction'] for item in train_data],
        'input': [item['input'] for item in train_data],
        'output': [item['output'] for item in train_data]
    })

    # 预处理训练集
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing train data (answer-only loss)"
    )

    # 处理验证集
    if val_data:
        val_dataset = Dataset.from_dict({
            'instruction': [item['instruction'] for item in val_data],
            'input': [item['input'] for item in val_data],
            'output': [item['output'] for item in val_data]
        })

        val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Processing val data (answer-only loss)"
        )
    else:
        val_dataset = None

    print("✅ Using answer-only loss (prompt tokens will be ignored)")

    return {'train': train_dataset, 'val': val_dataset}


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
        trust_remote_code=True,
        low_cpu_mem_usage=True  # 减少 CPU 内存使用
    )
    # 移动到 GPU
    if torch.cuda.is_available():
        model = model.to("cuda")
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
    datasets = load_and_process_data(
        args.data_path,
        tokenizer,
        args.max_length,
        val_split=0.1  # 9:1 划分
    )
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0} samples\n")

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,  # 评估批次大小
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        # 评估策略
        eval_strategy="epoch",  # 每个 epoch 评估一次
        save_strategy="no",  # 不自动保存 checkpoint（由 Callback 手动保存最佳 LoRA）
        load_best_model_at_end=False,  # 不需要自动加载（Callback 已保存最佳）
        # 评估配置 - 只计算 loss，不计算 accuracy（加快评估速度）
        prediction_loss_only=True,  # 只计算 loss，不返回 logits（大幅加速）
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        # 多卡训练配置
        ddp_find_unused_parameters=False,  # 加速训练
        dataloader_pin_memory=True,        # 加速数据加载
    )

    # 数据整理器 - 使用自定义的 collator

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

    # 创建 Callback
    epoch_callback = EpochEvalCallback(args.output_dir)

    # 创建 Trainer
    # 注意：由于设置了 prediction_loss_only=True，不使用 compute_metrics（加快评估速度）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 添加验证集
        data_collator=data_collator,
        # compute_metrics=compute_metrics,  # 暂时禁用以加快评估速度
        callbacks=[epoch_callback]  # 添加 callback
    )

    # 训练
    print("="*80)
    print("Starting training...")
    print("="*80)
    print("")

    trainer.train()

    # 训练结束后，在验证集上进行完整评估
    print("\n" + "="*80)
    print("Final Evaluation on Validation Set")
    print("="*80)
    print("")

    # 加载最佳模型进行评估
    best_lora_dir = os.path.join(args.output_dir, "best_lora")

    print("Loading best LoRA weights for final evaluation...")
    from peft import PeftModel

    # 重新加载 base 模型
    eval_base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载最佳 LoRA 权重
    eval_model = PeftModel.from_pretrained(
        eval_base_model,
        best_lora_dir,
        is_trainable=False
    )
    eval_model.eval()

    print("✅ Best model loaded for evaluation")

    # 在验证集上评估
    print("\nEvaluating on validation set...")

    from torch.utils.data import DataLoader

    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator
    )

    total_correct = 0
    total_tokens = 0
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # 移动到设备
            input_ids = batch['input_ids'].to(eval_model.device)
            attention_mask = batch['attention_mask'].to(eval_model.device)
            labels = batch['labels'].to(eval_model.device)

            # 前向传播
            outputs = eval_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 累积 loss
            total_loss += outputs.loss.item()
            num_batches += 1

            # 计算 accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # 只计算非 -100 位置的准确率
            mask = labels != -100
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    # 计算最终指标
    final_eval_loss = total_loss / num_batches
    final_eval_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    final_eval_perplexity = np.exp(final_eval_loss)

    print("\n" + "="*80)
    print("Final Evaluation Results")
    print("="*80)
    print(f"Eval Loss:       {final_eval_loss:.4f}")
    print(f"Eval Accuracy:   {final_eval_accuracy:.4f}")
    print(f"Eval Perplexity: {final_eval_perplexity:.4f}")
    print("="*80)

    # 保存评估结果
    eval_results = {
        "eval_loss": final_eval_loss,
        "eval_accuracy": final_eval_accuracy,
        "eval_perplexity": final_eval_perplexity,
        "num_eval_samples": len(val_dataset),
        "num_eval_tokens": total_tokens,
        "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    eval_results_file = os.path.join(args.output_dir, "final_eval_results.json")
    with open(eval_results_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation results saved to: {eval_results_file}")

    # 保存 tokenizer 到 best_lora 目录
    tokenizer.save_pretrained(best_lora_dir)
    print(f"✅ Tokenizer saved to: {best_lora_dir}")

    # 保存训练配置和最佳结果
    config = {
        "model_type": "LoRA_Only_Generative",
        "base_model": args.model_name_or_path,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "best_epoch": epoch_callback.best_epoch,
        "best_eval_loss": epoch_callback.best_eval_loss,
        "final_eval_loss": final_eval_loss,
        "final_eval_accuracy": final_eval_accuracy,
        "final_eval_perplexity": final_eval_perplexity,
        "training_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    config_file = os.path.join(args.output_dir, "training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ Config saved to: {config_file}")
    print(f"\n📊 Training Summary:")
    print(f"   Best Epoch: {epoch_callback.best_epoch}")
    print(f"   Best Eval Loss: {epoch_callback.best_eval_loss:.4f}")
    print(f"   Final Eval Accuracy: {final_eval_accuracy:.4f}")
    print(f"   Best LoRA weights: {best_lora_dir}")

    print("\n" + "="*80)
    print("✅ Training completed successfully!")
    print("="*80)
    print(f"\n💡 Next steps:")
    print(f"   1. Merge LoRA with base model:")
    print(f"      python merge_lora.py \\")
    print(f"        --base_model {args.model_name_or_path} \\")
    print(f"        --lora_weights {best_lora_dir} \\")
    print(f"        --output_dir /path/to/merged_model")
    print(f"\n   2. Or use LoRA directly for inference")
    print("")


if __name__ == "__main__":
    main()

