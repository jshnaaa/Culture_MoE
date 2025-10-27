#!/usr/bin/env python3
"""
使用 LoRA 微调 LLaMA 3.1 模型（灵活标签版本）
支持不同类别数和标签名称的数据集（如 WVS）
"""

import json
import os
import re
import sys
from collections import defaultdict
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
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def extract_label_info_from_text(text: str):
    """
    从文本中提取标签信息

    例如：
    "1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree"
    返回：["Strongly agree", "agree", "Disagree", "Strongly disagree"]
    """
    # 先移除干扰文本
    text = re.sub(r'You can only choose one option[.\s]*', '', text)
    text = re.sub(r'###[.\s]*', '', text)

    # 匹配模式：数字. 文本（直到下一个数字或结束）
    pattern = r'(\d+)\.\s*([^0-9]+?)(?=\s*\d+\.|$)'
    matches = re.findall(pattern, text)

    if matches:
        # 清理标签文本
        labels = []
        for num, label_text in matches:
            # 移除末尾的标点和空格
            cleaned = label_text.strip().rstrip('.,;:\n')
            if cleaned:  # 确保不是空字符串
                labels.append(cleaned)
        return labels if labels else None

    return None


def load_flexible_classification_data(data_path: str, tokenizer, max_length: int = 512, val_split: float = 0.1):
    """
    加载灵活标签的分类数据（支持不同类别数和标签名称）

    Args:
        data_path: 数据文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度
        val_split: 验证集比例

    Returns:
        {'train': train_dataset, 'val': val_dataset, 'num_classes': num_classes}
    """
    print(f"Loading data from {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # 处理数据
    processed_data = []
    label_info_stats = defaultdict(int)
    all_num_classes = []

    for item in data:
        instruction = item['instruction']
        input_text = item.get('input', '')
        output = item['output']

        # ✅ 优先从 input 字段提取标签，如果没有则从 instruction 提取
        label_options = None
        if input_text:
            label_options = extract_label_info_from_text(input_text)

        if label_options is None and instruction:
            label_options = extract_label_info_from_text(instruction)

        if label_options is None:
            print(f"Warning: Cannot extract labels from instruction or input")
            print(f"  Instruction: {instruction[:100]}...")
            print(f"  Input: {input_text[:100]}...")
            continue

        num_classes = len(label_options)
        label_info_stats[num_classes] += 1
        all_num_classes.append(num_classes)

        # 组合文本
        if input_text:
            full_text = f"{instruction}\n{input_text}"
        else:
            full_text = instruction

        # Tokenize
        encoded = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        processed_data.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': output,
            'num_classes': num_classes,
            'label_options': label_options
        })

    print(f"\nLabel distribution:")
    for num_classes, count in sorted(label_info_stats.items()):
        print(f"  {num_classes}-class: {count} samples ({count/len(processed_data)*100:.1f}%)")

    # 确定主要的类别数（用于模型）
    from collections import Counter
    num_classes_counter = Counter(all_num_classes)
    main_num_classes = num_classes_counter.most_common(1)[0][0]
    print(f"\nUsing {main_num_classes} classes for model (most common)")

    # 划分训练集和验证集
    if val_split > 0:
        split_idx = int(len(processed_data) * (1 - val_split))
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")
        return {'train': train_data, 'val': val_data, 'num_classes': main_num_classes}
    else:
        print(f"Train: {len(processed_data)}")
        return {'train': processed_data, 'val': [], 'num_classes': main_num_classes}


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
        default="./output/lora_only_flexible",
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
        default=5e-6,
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


class FlexibleClassificationModel(torch.nn.Module):
    """带分类头的 LLaMA 模型（支持多分类）"""

    def __init__(self, llama_model, num_classes=4):
        super().__init__()
        self.llama_model = llama_model
        self.config = llama_model.config
        self.num_classes = num_classes

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.config.hidden_size),
            torch.nn.Linear(self.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        前向传播

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            labels: [B]
            **kwargs: 其他字段会被忽略
        """
        # LLaMA 前向传播
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

    # 使用 macro 平均（适用于多分类）
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_lora_only_flexible(args: LoRATrainingArguments):
    """
    训练 LoRA 微调模型（灵活标签版本）

    数据格式：
    {
        "instruction": "Question: ...",
        "input": "1. Option1 2. Option2 3. Option3 4. Option4",
        "output": 0  # 0-based index
    }
    """

    # ✅ 检测分布式训练环境
    import torch.distributed as dist
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
        print(f"[Rank {local_rank}/{world_size}] Distributed training initialized")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Single GPU training")

    if not is_distributed or local_rank == 0:
        print("="*60)
        print("Training LLaMA 3.1 with LoRA (Flexible Labels)")
        print("="*60)

    # 1. 加载 tokenizer
    print(f"\n1. Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载数据（灵活标签）
    print(f"\n2. Loading data from {args.train_file}...")
    print(f"   Data format: instruction/input/output (flexible labels)")
    data = load_flexible_classification_data(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=args.val_split
    )
    train_dataset = data['train']
    val_dataset = data['val']
    num_classes = data['num_classes']

    print(f"   ✅ Train dataset size: {len(train_dataset)}")
    print(f"   ✅ Validation dataset size: {len(val_dataset)}")
    print(f"   ✅ Number of classes: {num_classes}")

    # 3. 加载基础模型
    print(f"\n3. Loading base LLaMA model from {args.model_path}...")
    llama_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    print("   ✅ Base model loaded (using float32)")

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
    model = FlexibleClassificationModel(llama_model, num_classes=num_classes)

    # ✅ 将模型移到正确的设备
    if not is_distributed:
        model = model.to(device)

    print(f"   ✅ Classification model created ({num_classes} classes, using float32)")

    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
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
        max_grad_norm=1.0,
        optim="adamw_torch",
        warmup_steps=100,
        # ✅ 禁用 DataParallel，避免多 GPU 问题
        dataloader_drop_last=False,
        ddp_find_unused_parameters=False if is_distributed else None,
    )

    # 7. Data Collator
    def simple_data_collator(features):
        """简单的数据整理器，只处理 input_ids, attention_mask, labels"""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        # 提取字段
        input_ids = [torch.tensor(f['input_ids']) for f in features]
        attention_mask = [torch.tensor(f['attention_mask']) for f in features]
        labels = [f['labels'] for f in features]

        # Padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # 确保 labels 是正确的形状 [batch_size]
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    data_collator = simple_data_collator

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

    # 10. 保存模型（可选）
    if args.save_model:
        print(f"\n8. Saving model to {args.output_dir}...")
        trainer.save_model()
        trainer.save_state()

        # 保存 LoRA 权重
        model.llama_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print("   ✅ Model saved")
    else:
        print(f"\n8. Skipping model saving (--save_model not set)")

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

    # 保存指标（总是保存）
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Training and evaluation completed!")
    if args.save_model:
        print(f"   Model saved to: {args.output_dir}")
    print(f"   Eval results saved to: {os.path.join(args.output_dir, 'eval_results.json')}")

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LoRA 微调 LLaMA 3.1（灵活标签）")
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
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="评估批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="学习率")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--save_model", action="store_true",
                        help="是否保存模型（默认不保存，只保存评估结果）")

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
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_length=args.max_length,
        val_split=args.val_split,
    )

    # 训练
    metrics = train_lora_only_flexible(training_args)


if __name__ == "__main__":
    main()

