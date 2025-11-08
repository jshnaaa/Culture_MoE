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


def generate_and_evaluate(model, tokenizer, eval_dataset, output_dir, output_type="number", max_new_tokens=10):
    """
    生成答案并评估（使用真正的生成，而不是 teacher forcing）

    这个函数会：
    1. 对验证集中的每个样本生成答案
    2. 保存到 generated_answers.json
    3. 调用 post-eval 脚本计算指标
    """
    import os
    import subprocess
    from tqdm import tqdm
    import re

    print("\n" + "="*80)
    print("Generating answers for evaluation...")
    print("="*80)

    model.eval()
    generated_answers = []

    for i in tqdm(range(len(eval_dataset)), desc="Generating"):
        sample = eval_dataset[i]
        input_ids = sample['input_ids']
        labels = sample['labels']

        # 找到 prompt 的结束位置（第一个非 -100 的 label 之前）
        label_mask = [l != -100 for l in labels]
        if any(label_mask):
            prompt_end = label_mask.index(True)
            prompt_ids = input_ids[:prompt_end]
        else:
            prompt_ids = input_ids

        # 生成答案
        with torch.no_grad():
            prompt_tensor = torch.tensor([prompt_ids]).to(model.device)
            outputs = model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

            generated_ids = outputs[0][len(prompt_ids):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # 对于数字类型，只提取第一个数字
            if output_type == "number":
                numbers = re.findall(r'\d+', generated_text)
                if numbers:
                    generated_text = numbers[0]

        # 提取真实答案
        answer_ids = [l for l in labels if l != -100]
        true_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        generated_answers.append({
            "predicted": generated_text,
            "true": true_text
        })

    # 保存生成的答案
    answers_file = os.path.join(output_dir, "generated_answers.json")
    with open(answers_file, 'w', encoding='utf-8') as f:
        json.dump(generated_answers, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated answers saved to: {answers_file}")

    # 调用 post-eval 脚本计算指标
    eval_script = os.path.join(os.path.dirname(__file__), "eval_from_generated_answers.py")
    metrics_file = os.path.join(output_dir, "eval_metrics.json")

    if os.path.exists(eval_script):
        print("\nRunning post-evaluation...")
        try:
            result = subprocess.run(
                ["python", eval_script, "--input", answers_file, "--output", metrics_file],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=True
            )
            print(result.stdout)

            if os.path.exists(metrics_file):
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                return metrics
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Post-evaluation failed: {e}")
            if e.stderr:
                print(e.stderr)
        except Exception as e:
            print(f"⚠️  Post-evaluation error: {e}")
    else:
        print(f"⚠️  Evaluation script not found: {eval_script}")

    # 如果 post-eval 失败，返回简单的准确率
    correct = sum(1 for item in generated_answers if item["predicted"].lower() == item["true"].lower())
    accuracy = correct / len(generated_answers) if len(generated_answers) > 0 else 0.0
    return {"accuracy": accuracy}


class EpochEvalCallback(TrainerCallback):
    """每个 epoch 结束后生成答案、评估并保存最佳 LoRA 权重"""

    def __init__(self, output_dir, tokenizer=None, eval_dataset=None, output_type="number"):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_type = output_type
        self.epoch_results = []
        self.best_eval_accuracy = -1.0
        self.best_epoch = 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Epoch 结束时生成答案、评估并保存最佳 LoRA 权重"""
        import torch.distributed as dist

        # 只在主进程中进行评估
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        if not is_main_process:
            return

        if model is None or self.eval_dataset is None:
            return

        # 获取训练 loss
        train_loss = None
        for log in reversed(state.log_history):
            if 'loss' in log:
                train_loss = log.get('loss', None)
                break

        # 生成答案并评估
        print(f"\n{'='*80}")
        print(f"📊 Epoch {int(state.epoch)} Evaluation")
        print(f"{'='*80}")

        # 调用生成和评估函数
        metrics = generate_and_evaluate(
            model=model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir,
            output_type=self.output_type,
            max_new_tokens=10
        )

        eval_accuracy = metrics.get('accuracy', 0.0)

        # 保存结果
        epoch_result = {
            'epoch': int(state.epoch),
            'eval_accuracy': eval_accuracy,
            'train_loss': train_loss,
        }

        # 添加其他指标
        for key in ['precision', 'recall', 'f1']:
            if key in metrics:
                epoch_result[f'eval_{key}'] = metrics[key]

        self.epoch_results.append(epoch_result)

        # 保存到文件
        results_file = os.path.join(self.output_dir, "epoch_eval_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.epoch_results, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Epoch {int(state.epoch)} Results:")
        if train_loss is not None:
            print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Eval Accuracy: {eval_accuracy:.4f}")

        # 打印其他指标
        for key in ['precision', 'recall', 'f1']:
            if key in metrics:
                print(f"   Eval {key.capitalize()}: {metrics[key]:.4f}")

        # 如果是最佳模型，保存 LoRA 权重
        if eval_accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = eval_accuracy
            self.best_epoch = int(state.epoch)

            best_lora_dir = os.path.join(self.output_dir, "best_lora")
            os.makedirs(best_lora_dir, exist_ok=True)

            if model is not None:
                model.save_pretrained(best_lora_dir)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(best_lora_dir)
                print(f"   🏆 New best model! Eval Accuracy: {eval_accuracy:.4f}")
                print(f"   ✅ Best LoRA weights saved to: {best_lora_dir}")

        print(f"   Best so far: Epoch {self.best_epoch}, Eval Accuracy: {self.best_eval_accuracy:.4f}")
        print(f"{'='*80}\n")


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

    return {'train': train_dataset, 'val': val_dataset, 'output_type': output_type}


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

    # 检测模型类型
    model_type = model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
    print(f"Detected model type: {model_type}\n")

    # 配置 LoRA（根据模型类型）
    print("Configuring LoRA...")
    if model_type == 'qwen':
        print("  Using Qwen-specific LoRA configuration")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,  # ✅ Qwen 需要更高的 dropout
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            init_lora_weights="gaussian"  # ✅ 使用高斯初始化
        )
    else:
        print("  Using LLaMA-specific LoRA configuration")
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
    output_type = datasets.get('output_type', 'number')
    print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0} samples\n")

    # 配置训练参数（根据模型类型调整）
    if model_type == 'qwen':
        # ✅ Qwen 需要保守但不过度的配置
        learning_rate = args.learning_rate * 0.5  # 降低 50%（不是 90%）
        max_grad_norm = 0.5  # 适中的梯度裁剪（不是 0.1）
        warmup_ratio = 0.1  # ✅ 10% 预热（不是 50%）
        use_fp32 = True  # 使用 fp32
        print("  Using Qwen-specific training configuration")
        print(f"    Learning rate: {learning_rate} (50% of {args.learning_rate})")
        print(f"    Max grad norm: {max_grad_norm}")
        print(f"    Warmup ratio: {warmup_ratio * 100:.0f}%")
        print(f"    Using fp32 (not fp16)")
    else:
        # LLaMA 配置
        learning_rate = args.learning_rate
        max_grad_norm = 1.0
        warmup_ratio = 0.1
        use_fp32 = False
        print("  Using LLaMA-specific training configuration")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="no",
        fp16=not use_fp32,  # ✅ Qwen 使用 fp32（fp16=False）
        fp16_full_eval=False,  # 评估时不使用 fp16
        fp16_opt_level="O1",   # 使用 O1 混合精度（更稳定）
        max_grad_norm=max_grad_norm,     # ✅ 根据模型类型调整
        warmup_ratio=warmup_ratio,      # ✅ 使用 warmup_ratio（不使用 warmup_steps）
        weight_decay=0.01,     # 权重衰减
        adam_epsilon=1e-8,     # Adam 优化器的 epsilon
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # 如果内存不够可以开启
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
    epoch_callback = EpochEvalCallback(
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        output_type=output_type
    )

    # 创建 Trainer（不使用 compute_metrics 和 preprocess_logits_for_metrics）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[epoch_callback]
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
