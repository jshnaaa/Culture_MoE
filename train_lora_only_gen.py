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


def compute_metrics(eval_preds, tokenizer=None):
    """
    计算评估指标（样本级别准确率）

    对于生成式任务，我们计算：
    - Sample-level Accuracy: 整个答案完全正确才算对
    - Token-level Accuracy: 答案部分每个 token 的准确率

    注意：predictions 已经通过 preprocess_logits_for_metrics 转换为 token IDs
    """
    import numpy as np
    import os

    predictions, labels = eval_preds

    # predictions 已经是 token IDs，shape: (batch_size, seq_len)
    # labels 是真实标签，shape: (batch_size, seq_len)

    # 转换为 numpy 数组
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()

    # 只计算非 -100 位置（即答案部分）
    mask = labels != -100

    # 只在主进程中打印调试信息
    is_main_process = int(os.environ.get('RANK', 0)) == 0

    if is_main_process:
        print(f"\n[DEBUG] compute_metrics called:")
        print(f"  predictions shape: {predictions.shape}")
        print(f"  labels shape: {labels.shape}")
        print(f"  mask sum: {mask.sum()}")

        # 打印第一个样本的答案部分
        if len(predictions) > 0 and tokenizer is not None:
            first_mask = mask[0]
            first_pred = predictions[0][first_mask]
            first_label = labels[0][first_mask]

            # 解码为文本
            try:
                pred_text = tokenizer.decode(first_pred, skip_special_tokens=True)
                label_text = tokenizer.decode(first_label, skip_special_tokens=True)
                print(f"  First sample answer:")
                print(f"    Predicted tokens: {first_pred[:10]}")
                print(f"    Predicted text:   '{pred_text}'")
                print(f"    Label tokens:     {first_label[:10]}")
                print(f"    Label text:       '{label_text}'")
                print(f"    Match:            {np.array_equal(first_pred, first_label)}")
            except:
                print(f"  First sample answer tokens:")
                print(f"    Predicted: {first_pred[:10]}")
                print(f"    Label:     {first_label[:10]}")
                print(f"    Match:     {np.array_equal(first_pred, first_label)}")

    # 1. Token-level 准确率（每个 token 的准确率）
    token_correct = (predictions == labels) & mask
    total_correct_tokens = token_correct.sum()
    total_tokens = mask.sum()
    token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0

    if is_main_process:
        print(f"  Token accuracy: {token_accuracy:.4f} ({total_correct_tokens}/{total_tokens})")

    # 2. Sample-level 准确率（整个答案完全正确才算对）
    num_samples = predictions.shape[0]
    sample_correct = 0

    for i in range(num_samples):
        # 获取当前样本的答案部分
        sample_mask = mask[i]
        sample_pred = predictions[i][sample_mask]
        sample_label = labels[i][sample_mask]

        # 检查答案是否完全匹配
        if len(sample_pred) == len(sample_label) and np.array_equal(sample_pred, sample_label):
            sample_correct += 1

    sample_accuracy = sample_correct / num_samples if num_samples > 0 else 0.0

    if is_main_process:
        print(f"  Sample accuracy: {sample_accuracy:.4f} ({sample_correct}/{num_samples})\n")

    return {
        "accuracy": float(sample_accuracy),  # 主要指标：样本级别准确率
        "token_accuracy": float(token_accuracy),  # 辅助指标：token 级别准确率
    }


class EpochEvalCallback(TrainerCallback):
    """每个 epoch 结束后保存评估结果，并保存最佳 LoRA 权重"""

    def __init__(self, output_dir, tokenizer=None):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
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
                            # 同时保存 tokenizer
                            if self.tokenizer is not None:
                                self.tokenizer.save_pretrained(best_lora_dir)
                            print(f"   🏆 New best model! Eval Loss: {current_eval_loss:.4f}")
                            print(f"   ✅ Best LoRA weights saved to: {best_lora_dir}")

                    print(f"   Best so far: Epoch {self.best_epoch}, Eval Loss: {self.best_eval_loss:.4f}\n")
                    break


def load_and_process_data(data_path: str, tokenizer, max_length: int = 512, val_split: float = 0.1):
    """加载并处理生成式数据（只计算答案部分的损失）

    支持单个文件或多个文件（用逗号分隔）
    """
    # 支持多个数据文件（用逗号分隔）
    data_files = [f.strip() for f in data_path.split(',')]

    print(f"Loading data from {len(data_files)} file(s):")
    for f in data_files:
        print(f"  - {f}")

    # 加载所有数据文件
    all_data = []
    for file_path in data_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            all_data.extend(file_data)
            print(f"  Loaded {len(file_data)} samples from {os.path.basename(file_path)}")

    data = all_data
    print(f"Total: {len(data)} samples")

    # 分析数据集标签分布（用于信息展示）
    output_type = "number"  # 默认为数字类型
    try:
        # 检查第一个样本的输出类型
        first_output = str(data[0]['output']).lower().strip()
        if first_output in ['yes', 'no', 'neutral']:
            # 文本类型
            output_type = "text"
            unique_labels = set(str(item['output']).lower().strip() for item in data)
            num_classes = 3
            print(f"📊 Dataset Statistics:")
            print(f"   Output type: text (yes/no/neutral)")
            print(f"   Inferred num_classes: {num_classes}")
            print(f"   Unique labels: {sorted(unique_labels)}")
        elif first_output in ['true', 'false']:
            # 布尔类型
            output_type = "bool"
            unique_labels = set(str(item['output']).upper().strip() for item in data)
            num_classes = 2
            print(f"📊 Dataset Statistics:")
            print(f"   Output type: bool (TRUE/FALSE)")
            print(f"   Inferred num_classes: {num_classes}")
            print(f"   Unique labels: {sorted(unique_labels)}")
        else:
            # 数字类型
            output_type = "number"
            unique_labels = set()
            for item in data:
                try:
                    label = int(item['output'])
                    unique_labels.add(label)
                except (ValueError, KeyError):
                    pass

            if unique_labels:
                min_label = min(unique_labels)
                max_label = max(unique_labels)

                # 判断标签范围
                if min_label == 0:
                    num_classes = max_label + 1
                    label_type = "0-indexed"
                else:
                    num_classes = max_label
                    label_type = "1-indexed"

                print(f"📊 Dataset Statistics:")
                print(f"   Output type: number")
                print(f"   Label range: {min_label} to {max_label} ({label_type})")
                print(f"   Inferred num_classes: {num_classes}")
                print(f"   Unique labels: {sorted(unique_labels)}")
    except Exception as e:
        print(f"⚠️  Could not analyze label distribution: {e}")

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
            # 注意：如果 input_text 已经包含了提示语（如 "Your answer is:"），则不需要额外添加
            if output_type == "text":
                # 文本类型：检查 input 是否已包含提示语
                if "your answer is:" in input_text.lower() or "answer one of" in input_text.lower():
                    # input 已包含提示语，直接使用
                    prompt = f"{instruction}\n{input_text}"
                    # 答案直接跟在后面，不加空格
                    full_text = f"{prompt}{output}"
                else:
                    # input 不包含提示语，添加标准提示
                    prompt = f"{instruction}\n{input_text}\nAnswer one of 'yes', 'no', or 'neutral'. Your answer is:"
                    # 答案前加空格
                    full_text = f"{prompt} {output}"
            elif output_type == "bool":
                # 布尔类型：检查 input 是否已包含提示语
                if "your answer is:" in input_text.lower() or ("true" in input_text.lower() and "false" in input_text.lower()):
                    # input 已包含提示语，直接使用
                    prompt = f"{instruction}\n{input_text}"
                    # 答案直接跟在后面，不加空格
                    full_text = f"{prompt}{output}"
                else:
                    # input 不包含提示语，添加标准提示
                    prompt = f"{instruction}\n{input_text}\nAnswer one of 'TRUE' or 'FALSE'. Your answer is:"
                    # 答案前加空格
                    full_text = f"{prompt} {output}"
            else:
                # 数字类型
                prompt = f"{instruction}\n{input_text}\nAnswer:"
                # 答案前加空格
                full_text = f"{prompt} {output}"

            # Tokenize prompt 和答案（分别）
            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,  # 不截断
                max_length=None
            )

            answer_tokens = tokenizer(
                output,
                add_special_tokens=False,  # 答案不需要特殊 token
                truncation=False,
                max_length=None
            )

            # 合并 prompt 和答案的 token IDs
            full_input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
            full_attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"]

            # 截断到最大长度
            if len(full_input_ids) > max_length:
                full_input_ids = full_input_ids[:max_length]
                full_attention_mask = full_attention_mask[:max_length]

            # 创建 labels：prompt 部分用 -100（忽略），答案部分正常
            prompt_len = len(prompt_tokens["input_ids"])
            labels = [-100] * prompt_len + answer_tokens["input_ids"]

            # 截断 labels
            if len(labels) > max_length:
                labels = labels[:max_length]

            # 确保长度一致
            assert len(full_input_ids) == len(full_attention_mask) == len(labels), \
                f"Length mismatch: input_ids={len(full_input_ids)}, attention_mask={len(full_attention_mask)}, labels={len(labels)}"

            model_inputs["input_ids"].append(full_input_ids)
            model_inputs["attention_mask"].append(full_attention_mask)
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
        prediction_loss_only=False,  # 返回 predictions 以计算 accuracy
        # 评估配置 - 需要返回 predictions 以计算 accuracy
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
    epoch_callback = EpochEvalCallback(args.output_dir, tokenizer=tokenizer)

    # 预处理 logits 以节省显存
    def preprocess_logits_for_metrics(logits, labels):
        """
        只保留预测的 token IDs，而不是完整的 logits
        这样可以大幅减少显存占用
        """
        import os

        if isinstance(logits, tuple):
            logits = logits[0]

        # 只在主进程中打印调试信息
        is_main_process = int(os.environ.get('RANK', 0)) == 0

        if is_main_process:
            print(f"\n[DEBUG] preprocess_logits_for_metrics:")
            print(f"  logits shape: {logits.shape}")
            print(f"  labels shape: {labels.shape}")

        # 只保存预测的 token IDs
        pred_ids = logits.argmax(dim=-1)

        if is_main_process:
            print(f"  pred_ids shape: {pred_ids.shape}")
            print(f"  First sample pred_ids: {pred_ids[0, :10].tolist()}")
            print(f"  First sample labels: {labels[0, :10].tolist()}")

        return pred_ids

    # 创建一个闭包来传递 tokenizer 给 compute_metrics
    def compute_metrics_with_tokenizer(eval_preds):
        return compute_metrics(eval_preds, tokenizer=tokenizer)

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 添加验证集
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_tokenizer,  # 计算准确率（带 tokenizer）
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # 预处理 logits
        callbacks=[epoch_callback]  # 添加 callback
    )

    # 训练
    print("="*80)
    print("Starting training...")
    print("="*80)
    print("")

    trainer.train()

    # 只在主进程中进行最终评估和保存
    import torch.distributed as dist
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    if is_main_process:
        # 确保保存最终的 LoRA 权重（如果 Callback 没有保存）
        best_lora_dir = os.path.join(args.output_dir, "best_lora")
        if not os.path.exists(os.path.join(best_lora_dir, "adapter_config.json")):
            print("\n⚠️  Best LoRA weights not found, saving current model...")
            os.makedirs(best_lora_dir, exist_ok=True)
            model.save_pretrained(best_lora_dir)
            tokenizer.save_pretrained(best_lora_dir)
            print(f"✅ LoRA weights saved to: {best_lora_dir}")

        # 训练完成
        print("\n" + "="*80)
        print("✅ Training Completed!")
        print("="*80)
        print(f"Best LoRA weights saved to: {best_lora_dir}")
        print(f"\n💡 To evaluate the model, run:")
        print(f"   sh run_eval_lora_only_from_components.sh llama")
        print("="*80)
        print("")


if __name__ == "__main__":
    main()
