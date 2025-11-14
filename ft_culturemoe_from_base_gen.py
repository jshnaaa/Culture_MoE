#!/usr/bin/env python3
"""
使用标准语言建模损失 + 文化专注性损失微调 CultureMoE 模型（新数据格式）

使用方法：
    python ft_culturemoe_from_base_gen_new_format.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --use_culture_loss True \
        --culture_loss_lambda 0.5

数据格式（新格式）：
    {
        "instruction": "### Question: ... ### Answer: ",
        "instruction_mask": "### Question: ... [MASK] ### Answer: ",
        "input": "",
        "output": "2",
        "label": "0"
    }

关键特性：
    1. MOE 专家使用 instruction 字段
    2. Shared 专家使用 instruction_mask 字段
    3. 使用标准语言建模损失 + 文化专注性损失
    4. 支持 Post Eval（生成答案并评估准确率）
    5. 每个 epoch 生成答案并评估
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 CultureMoE 模型
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


class CultureMoENewFormatDataset(Dataset):
    """
    CultureMoE 新格式数据集

    数据格式：
    {
        "instruction": "### Question: ... ### Answer: ",
        "instruction_mask": "### Question: ... [MASK] ### Answer: ",
        "input": "",
        "output": "2",
        "label": "0"
    }
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        Args:
            data_path: 数据文件路径
            tokenizer: Tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 新格式：instruction + input + output
        instruction = item.get('instruction', '')
        instruction_mask = item.get('instruction_mask', instruction)
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        label = item.get('label', '')

        # 构建完整的输入和输出
        # MOE 专家用 instruction
        if input_text:
            full_input = f"{instruction}{input_text}"
        else:
            full_input = instruction

        # Shared 专家用 instruction_mask
        if input_text:
            full_input_mask = f"{instruction_mask}{input_text}"
        else:
            full_input_mask = instruction_mask

        # 完整的文本（用于语言建模）
        full_text = f"{full_input}{output_text}"
        full_text_mask = f"{full_input_mask}{output_text}"

        # Tokenize for MOE experts (instruction)
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize for Shared experts (instruction_mask)
        encoded_mask = self.tokenizer(
            full_text_mask,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        input_ids_mask = encoded_mask['input_ids'].squeeze(0)
        attention_mask_mask = encoded_mask['attention_mask'].squeeze(0)

        # ✅ 只在答案部分计算 loss
        labels = input_ids.clone()

        # 找到答案开始的位置（prompt 的长度）
        # Tokenize prompt (instruction + input) without output
        prompt_encoded = self.tokenizer(
            full_input,
            add_special_tokens=True,  # 保持与完整文本相同的 special tokens
            truncation=True,
            max_length=self.max_length
        )
        prompt_length = len(prompt_encoded['input_ids'])

        # 将问题部分的 labels 设置为 -100（CrossEntropyLoss 会忽略）
        # 只在答案部分（从 prompt_length 开始）计算损失
        labels[:prompt_length] = -100

        # 将 padding 部分也设置为 -100
        labels[attention_mask == 0] = -100

        # 将 label 转换为整数（用于文化损失）
        try:
            label_int = int(label)
        except:
            label_int = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'input_ids_mask': input_ids_mask,
            'attention_mask_mask': attention_mask_mask,
            'labels': labels,  # ✅ 只在答案部分有有效标签
            'label': label_int,
            'instruction': instruction,
            'input': input_text,
            'output': output_text
        }


def compute_class_weights(dataset, num_classes=10):
    """
    计算类别权重以处理类别不平衡

    Args:
        dataset: 数据集
        num_classes: 类别数量

    Returns:
        torch.Tensor: 类别权重
    """
    from collections import Counter

    # 统计每个类别的样本数
    labels = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        # 从 output 中提取标签
        output_text = item.get('output', '')
        try:
            label = int(output_text.strip())
            labels.append(label)
        except:
            pass

    # 计算类别分布
    label_counts = Counter(labels)
    print(f"\n📊 Class distribution in dataset:")
    for label in sorted(label_counts.keys()):
        print(f"   Class {label}: {label_counts[label]} samples")

    # 计算类别权重：1 / count
    class_counts = torch.zeros(num_classes)
    for label, count in label_counts.items():
        if 0 <= label < num_classes:
            class_counts[label] = count

    # 避免除以零
    class_counts = torch.clamp(class_counts, min=1.0)

    # 计算权重：1 / count，然后归一化
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes

    print(f"\n📊 Computed class weights:")
    for i in range(num_classes):
        if class_counts[i] > 1:
            print(f"   Class {i}: weight={class_weights[i]:.4f} (count={int(class_counts[i])})")

    return class_weights


def load_and_process_data(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    val_split: float = 0.1
):
    """
    加载并处理数据，按 9:1 比例划分训练集和验证集

    Args:
        data_path: 数据文件路径
        tokenizer: Tokenizer
        max_length: 最大序列长度
        val_split: 验证集比例

    Returns:
        dict: 包含 'train' 和 'validation' 的字典
    """
    dataset = CultureMoENewFormatDataset(data_path, tokenizer, max_length)

    # 按 9:1 比例划分
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    return {
        'train': train_dataset,
        'validation': val_dataset
    }


def extract_answer_from_text(text: str) -> str:
    """
    从生成的文本中提取答案

    使用正则表达式查找数字

    Args:
        text: 生成的文本

    Returns:
        提取的答案（数字字符串）
    """
    # 查找数字（1-10 或 1-4）
    match = re.search(r'\d+', text)
    if match:
        return match.group(0)

    return ""


def generate_answer(model, tokenizer, instruction: str, input_text: str, device: str = 'cuda', max_new_tokens: int = 10) -> str:
    """
    使用模型生成答案

    Args:
        model: 模型
        tokenizer: tokenizer
        instruction: 指令
        input_text: 输入文本
        device: 设备
        max_new_tokens: 最大生成 token 数

    Returns:
        生成的文本
    """
    # ✅ 处理 DataParallel 包装的模型
    if isinstance(model, torch.nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model

    # 构建输入
    if input_text:
        full_input = f"{instruction}{input_text}"
    else:
        full_input = instruction

    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = actual_model.generate(
            **inputs,
            max_new_tokens=1,  # ✅ 只生成 1 个 token（数字）
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None
        )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ✅ 额外的后处理：只保留第一个数字
    # 如果生成的文本包含非数字字符，只保留开头的数字部分
    import re
    match = re.match(r'^(\d+)', generated_text)
    if match:
        generated_text = match.group(1)
    else:
        # 如果没有匹配到数字，尝试从整个文本中提取第一个数字
        match = re.search(r'\d+', generated_text)
        if match:
            generated_text = match.group(0)

    return generated_text


def train_epoch(model, train_loader, optimizer, device, scheduler=None, use_culture_loss=False, culture_loss_lambda=0.5, culture_loss_alpha=2.0, culture_loss_beta=1.0, router_temperature=2.0, load_balance_weight=0.01, entropy_weight=0.1, num_accumulation_steps=1, class_weights=None):
    """
    训练一个 epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        scheduler: 学习率调度器（可选）
        use_culture_loss: 是否使用文化损失
        culture_loss_lambda: 文化损失权重 (lambda)
        culture_loss_alpha: specialization 损失权重 (alpha)
        culture_loss_beta: diversity 损失权重 (beta)
        num_accumulation_steps: 梯度累积步数
        class_weights: 类别权重（用于处理类别不平衡）

    Returns:
        dict: 包含训练指标的字典
    """
    model.train()
    total_loss = 0
    total_gen_loss = 0
    total_culture_loss = 0
    total_spec_loss = 0
    total_div_loss = 0
    total_lambda_value = 0  # ✅ 累积 lambda 值
    num_batches = 0
    nan_count = 0

    # ✅ 用于统计 Router 权重分布
    all_expert_weights = []

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        input_ids_mask = batch['input_ids_mask'].to(device)
        attention_mask_mask = batch['attention_mask_mask'].to(device)
        labels = batch['labels'].to(device)
        culture_labels = batch['label'].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids_mask=input_ids_mask,
            attention_mask_mask=attention_mask_mask,
            labels=labels,
            culture_labels=culture_labels if use_culture_loss else None,
            use_culture_loss=use_culture_loss,
            culture_loss_lambda=culture_loss_lambda,
            culture_loss_alpha=culture_loss_alpha,
            culture_loss_beta=culture_loss_beta
        )

        loss = outputs['loss']
        gen_loss = outputs.get('generation_loss', loss)
        culture_loss = outputs.get('culture_loss', torch.tensor(0.0, device=device))
        spec_loss = outputs.get('specialization_loss', torch.tensor(0.0, device=device))
        div_loss = outputs.get('diversity_loss', torch.tensor(0.0, device=device))
        lambda_value = outputs.get('culture_loss_lambda', 0.0)  # ✅ 获取当前 lambda 值

        # ✅ 收集 expert_weights 用于诊断
        if 'expert_weights' in outputs:
            all_expert_weights.append(outputs['expert_weights'].detach().cpu())

        # ✅ 检查 NaN loss 并诊断
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            print(f"\n❌ NaN or Inf loss detected at batch {batch_idx}")
            print(f"   Loss: {loss.item()}")
            print(f"   Gen Loss: {gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss}")
            print(f"   Culture Loss: {culture_loss.item() if isinstance(culture_loss, torch.Tensor) else culture_loss}")
            print(f"   Spec Loss: {spec_loss.item() if isinstance(spec_loss, torch.Tensor) else spec_loss}")
            print(f"   Div Loss: {div_loss.item() if isinstance(div_loss, torch.Tensor) else div_loss}")

            # 诊断：检查 logits 的范围
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                print(f"   Logits max: {logits.abs().max().item():.4f}")
                print(f"   Logits has NaN: {torch.isnan(logits).any().item()}")
                print(f"   Logits has Inf: {torch.isinf(logits).any().item()}")

            # 跳过这个批次
            continue

        # 梯度累积
        loss = loss / num_accumulation_steps
        loss.backward()

        total_loss += loss.item() * num_accumulation_steps
        total_gen_loss += gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss
        total_culture_loss += culture_loss.item() if isinstance(culture_loss, torch.Tensor) else culture_loss
        total_spec_loss += spec_loss.item() if isinstance(spec_loss, torch.Tensor) else spec_loss
        total_div_loss += div_loss.item() if isinstance(div_loss, torch.Tensor) else div_loss
        total_lambda_value += lambda_value  # ✅ 累积 lambda 值
        num_batches += 1

        # 梯度更新
        if (batch_idx + 1) % num_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            # ✅ 更新学习率（如果提供了 scheduler）
            if scheduler is not None:
                scheduler.step()

        pbar.set_postfix({
            'loss': f"{loss.item() * num_accumulation_steps:.4f}",
            'gen_loss': f"{gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss:.4f}",
            'culture_loss': f"{culture_loss.item() if isinstance(culture_loss, torch.Tensor) else culture_loss:.4f}"
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_gen_loss = total_gen_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0
    avg_spec_loss = total_spec_loss / num_batches if num_batches > 0 else 0
    avg_div_loss = total_div_loss / num_batches if num_batches > 0 else 0
    avg_lambda = total_lambda_value / num_batches if num_batches > 0 else 0  # ✅ 计算平均 lambda 值

    if nan_count > 0:
        print(f"\n⚠️  WARNING: {nan_count} batches had NaN/Inf loss (skipped)")

    # ✅ 打印 Router 权重分布诊断
    if len(all_expert_weights) > 0:
        all_expert_weights = torch.cat(all_expert_weights, dim=0)  # [total_samples, num_experts]
        avg_expert_weights = all_expert_weights.mean(dim=0)  # [num_experts]

        print(f"\n📊 Router Average Weights (across all batches):")
        for i, weight in enumerate(avg_expert_weights):
            print(f"   Expert {i}: {weight.item():.4f}")

        # ✅ 检查 Router 是否塌陷
        max_weight = avg_expert_weights.max().item()
        if max_weight > 0.7:
            print(f"\n⚠️  WARNING: Router may have collapsed! Expert {avg_expert_weights.argmax().item()} has weight {max_weight:.4f}")
            print(f"   This suggests the router is always selecting the same expert.")
        else:
            print(f"\n✅ Router weights are balanced (max weight: {max_weight:.4f})")

    return {
        'loss': avg_loss,
        'gen_loss': avg_gen_loss,
        'culture_loss': avg_culture_loss,
        'spec_loss': avg_spec_loss,
        'div_loss': avg_div_loss,
        'culture_loss_lambda': avg_lambda,  # ✅ 返回平均 lambda 值
        'load_balance_loss': 0.0,  # ✅ 占位符，实际值在 forward 中计算
        'entropy_loss': 0.0,  # ✅ 占位符
        'neg_entropy_loss': 0.0,  # ✅ 占位符，负熵损失
        'num_batches': num_batches,
        'nan_count': nan_count
    }


def evaluate(model, val_loader, device, use_culture_loss=False, culture_loss_lambda=0.5):
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        use_culture_loss: 是否使用文化损失
        culture_loss_lambda: 文化损失权重

    Returns:
        dict: 包含评估指标的字典
    """
    model.eval()
    total_loss = 0
    total_gen_loss = 0
    total_culture_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Evaluating")

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids_mask = batch['input_ids_mask'].to(device)
            attention_mask_mask = batch['attention_mask_mask'].to(device)
            labels = batch['labels'].to(device)
            culture_labels = batch['label'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_mask=input_ids_mask,
                attention_mask_mask=attention_mask_mask,
                labels=labels,
                culture_labels=culture_labels if use_culture_loss else None,
                use_culture_loss=use_culture_loss,
                culture_loss_lambda=culture_loss_lambda
            )

            loss = outputs['loss']
            gen_loss = outputs.get('generation_loss', loss)
            culture_loss = outputs.get('culture_loss', torch.tensor(0.0, device=device))

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            total_gen_loss += gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss
            total_culture_loss += culture_loss.item() if isinstance(culture_loss, torch.Tensor) else culture_loss
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'gen_loss': f"{gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss:.4f}",
                'culture_loss': f"{culture_loss.item() if isinstance(culture_loss, torch.Tensor) else culture_loss:.4f}"
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_gen_loss = total_gen_loss / num_batches if num_batches > 0 else 0
    avg_culture_loss = total_culture_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'gen_loss': avg_gen_loss,
        'culture_loss': avg_culture_loss,
        'num_batches': num_batches
    }


def generate_and_evaluate_answers(model, val_dataset, tokenizer, device, output_dir, epoch=None):
    """
    在验证集上生成答案并评估准确率（Post Eval）

    Args:
        model: 模型
        val_dataset: 验证数据集
        tokenizer: tokenizer
        device: 设备
        output_dir: 输出目录
        epoch: 当前 epoch 数（用于保存文件名）

    Returns:
        dict: 包含准确率等指标的字典
    """
    model.eval()

    correct = 0
    total = 0
    generated_data = []

    print("\nGenerating answers on validation set (Post Eval)...")

    for idx in tqdm(range(len(val_dataset)), desc="Generating"):
        sample = val_dataset[idx]
        instruction = sample['instruction']
        input_text = sample['input']
        true_output = sample['output']
        label = sample['label']

        # 生成答案
        generated_text = generate_answer(model, tokenizer, instruction, input_text, device)

        # 提取答案
        predicted_answer = extract_answer_from_text(generated_text)

        # 比对答案
        if predicted_answer == true_output:
            correct += 1
        total += 1

        # 保存生成的数据
        generated_data.append({
            'instruction': instruction,
            'input': input_text,
            'true_output': true_output,
            'label': label,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': predicted_answer == true_output
        })

    accuracy = correct / total if total > 0 else 0

    # 保存生成的答案（最新的）
    with open(os.path.join(output_dir, 'generated_answers.json'), 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # ✅ 保存每个 epoch 的生成答案（不覆盖）
    if epoch is not None:
        epoch_answers_file = os.path.join(output_dir, f'generated_answers_epoch_{epoch}.json')
        with open(epoch_answers_file, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # 打印前五条生成的答案
    print("\n📋 前五条生成的答案:")
    print("-" * 100)
    for idx in range(min(5, len(generated_data))):
        item = generated_data[idx]
        print(f"\n样本 {idx + 1}:")
        print(f"  Instruction: {item['instruction'][:80]}...")
        print(f"  Input: {item['input']}")
        print(f"  True Output: {item['true_output']}")
        print(f"  Generated Text: {item['generated_text']}")
        print(f"  Predicted Answer: {item['predicted_answer']}")
        print(f"  Correct: {'✅' if item['correct'] else '❌'}")
    print("\n" + "-" * 100)

    # ✅ 打印准确率统计
    print(f"\n📊 Accuracy Statistics:")
    print(f"   Correct: {correct}/{total}")
    print(f"   Accuracy: {accuracy:.4f}")

    # ✅ 统计生成的答案分布
    predicted_answers = [item['predicted_answer'] for item in generated_data]
    from collections import Counter
    answer_dist = Counter(predicted_answers)
    print(f"\n   Generated Answer Distribution:")
    for ans, count in sorted(answer_dist.items()):
        print(f"      Answer '{ans}': {count} times")

    # ✅ 统计真实答案分布
    true_answers = [item['true_output'] for item in generated_data]
    true_dist = Counter(true_answers)
    print(f"\n   True Answer Distribution:")
    for ans, count in sorted(true_dist.items()):
        print(f"      Answer '{ans}': {count} times")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CultureMoE model with new data format")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--lora_weights_path", type=str, required=True,
                        help="Path to LoRA weights")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (new format JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    # 训练参数
    parser.add_argument("--use_culture_loss", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to use culture loss")
    parser.add_argument("--culture_loss_lambda", type=float, default=0.5,
                        help="Culture loss weight (lambda)")
    parser.add_argument("--culture_loss_alpha", type=float, default=2.0,
                        help="Specialization loss weight (alpha)")
    parser.add_argument("--culture_loss_beta", type=float, default=1.0,
                    help="Diversity loss weight (beta)")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=3,
                        help="Evaluate every N epochs")

    # MOE 参数
    parser.add_argument("--num_experts", type=int, default=6,
                        help="Number of experts")
    parser.add_argument("--use_shared_experts", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to use shared experts (True) or only MoE experts (False)")
    parser.add_argument("--moe_fusion", type=float, default=0.4,
                        help="MoE fusion coefficient (default 0.4)")
    parser.add_argument("--shared_hidden_dim", type=int, default=4096,
                        help="Shared layer hidden dimension")
    parser.add_argument("--router_hidden_dim", type=int, default=2048,
                    help="Router hidden dimension")
    parser.add_argument("--experts_hidden_dim", type=int, default=4096,
                        help="Experts hidden dimension")
    parser.add_argument("--moe_lora_rank", type=int, default=32,
                        help="MOE LoRA rank")
    parser.add_argument("--classification_hidden_dim", type=int, default=512,
                        help="Classification head hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate")
    parser.add_argument("--num_heads", type=int, default=8,
                    help="Number of attention heads")

    # 防塌陷参数
    parser.add_argument("--router_temperature", type=float, default=2.0,
                        help="Router temperature parameter (default 2.0, prevents collapse)")
    parser.add_argument("--load_balance_weight", type=float, default=0.01,
                        help="Load balance loss weight (default 0.01, prevents collapse)")
    parser.add_argument("--entropy_weight", type=float, default=0.1,
                        help="Entropy regularization weight (default 0.1, prevents collapse)")

    parser.add_argument("--device", type=str, default='cuda',
                    help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Fine-tuning CultureMoE Model with New Data Format")
    print("="*80)
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.lora_weights_path}")
    print(f"Training data: {args.train_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Use culture loss: {args.use_culture_loss}")
    print(f"Culture loss weight: {args.culture_loss_lambda}")
    print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded")

    # 加载数据
    print("\nLoading and processing data...")
    datasets = load_and_process_data(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        val_split=args.val_split
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    print("✅ Data loaded")

    # ✅ 计算类别权重以处理类别不平衡
    print("\nComputing class weights...")
    # 从原始数据集计算权重
    full_dataset = CultureMoENewFormatDataset(args.train_file, tokenizer, args.max_length)
    class_weights = compute_class_weights(full_dataset, num_classes=10)
    class_weights = class_weights.to(args.device)
    print("✅ Class weights computed")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 加载 Base 模型
    print("\nLoading base model...")
    # ✅ 使用 bfloat16 而不是 float32（节省内存，避免 OOM）
    # bfloat16 数值范围与 float32 相同，但内存占用与 float16 相同
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Base model loaded (bfloat16)")
    except Exception as e:
        print(f"⚠️  bfloat16 not supported, falling back to float16")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16,  # 降级到 float16
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Base model loaded (float16)")

    # 加载 LoRA 权重
    print("\nLoading LoRA weights...")
    # ✅ 使用与 base_model 相同的精度
    model_dtype = base_model.dtype
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_weights_path,
        is_trainable=False,  # LoRA 权重不训练
        torch_dtype=model_dtype
    )
    print(f"✅ LoRA weights loaded ({model_dtype})")

    # 合并 LoRA 权重
    print("\nMerging LoRA weights...")
    model = model.merge_and_unload()
    print("✅ LoRA weights merged")

    # 创建 CultureMoE 模型
    print("\nCreating CultureMoE model...")

    # 创建 ModelArgs
    moe_args = ModelArgs(
        num_experts=args.num_experts,
        shared_hidden_dim=args.shared_hidden_dim,
        router_hidden_dim=args.router_hidden_dim,
        experts_hidden_dim=args.experts_hidden_dim,
        lora_rank=args.moe_lora_rank,
        classification_hidden_dim=args.classification_hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads
    )

    # 创建 CultureMoE 模型
    model = LlamaSharedRouterExpertsModel(
        llama_model=model,
        config=model.config,
        args=moe_args,
        culture_loss_lambda=args.culture_loss_lambda,  # ✅ 传递 lambda 参数
        moe_fusion=args.moe_fusion  # ✅ 传递 moe_fusion 参数
    )

    # ✅ 打印权重学习模式
    if model.culture_loss_lambda_learnable:
        print(f"✅ Culture loss weight is learnable (initial value: {model.culture_loss_lambda.item():.4f})")
    else:
        print(f"✅ Culture loss weight is fixed (value: {model.culture_loss_lambda.item():.4f})")
    print(f"✅ MoE fusion coefficient: {model.moe_fusion_alpha.item():.4f} (learnable)")

    # ✅ 强制 MoE 部分使用 float32（防止 NaN）
    print("Setting MoE layers to float32...")
    # ✅ 获取 base_model 的设备
    device = next(base_model.parameters()).device
    model.shared = model.shared.to(device=device, dtype=torch.float32)
    model.router = model.router.to(device=device, dtype=torch.float32)
    model.experts_layer = model.experts_layer.to(device=device, dtype=torch.float32)

    print(f"✅ CultureMoE model created (MoE layers in float32 on {device})")

    # 诊断：打印所有参数名称（前 20 个）
    print("\n📋 Model parameter names (first 20):")
    param_names = list(dict(model.named_parameters()).keys())
    for i, name in enumerate(param_names[:20]):
        print(f"   {i+1}. {name}")
    if len(param_names) > 20:
        print(f"   ... and {len(param_names) - 20} more parameters")

    # 冻结 Base 模型的所有参数（只训练 MOE 层）
    print("\nFreezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False
    print("✅ Base model parameters frozen")

    # 解冻 MOE 层的参数（如果存在）
    print("\nUnfreezing MOE layer parameters...")
    moe_param_count = 0
    moe_param_names = []

    # 尝试多种参数名称模式
    # 注意：不包括 'gate'，因为 gate_proj 是标准的 MLP 门控，不是 MOE 层
    moe_patterns = [
        'moe', 'expert', 'router', 'mixture',
        'shared', 'lora', 'adapter'
    ]

    for name, param in model.named_parameters():
        # 检查是否匹配任何 MOE 相关的模式
        if any(pattern in name.lower() for pattern in moe_patterns):
            param.requires_grad = True
            moe_param_count += 1
            moe_param_names.append(name)

    print(f"✅ MOE layer parameters unfrozen ({moe_param_count} parameters)")

    # 如果没有找到 MOE 参数，打印警告并列出所有参数名称
    if moe_param_count == 0:
        print("\n⚠️  WARNING: No MOE parameters found!")
        print("   Possible reasons:")
        print("   1. Model doesn't have MOE layers")
        print("   2. MOE layer names don't match the patterns")
        print("\n   All parameter names in the model:")
        for i, name in enumerate(param_names):
            print(f"   {i+1}. {name}")

        # 如果没有 MOE 参数，训练所有参数
        print("\n   Training all parameters instead...")
        for param in model.parameters():
            param.requires_grad = True
        moe_param_count = sum(1 for p in model.parameters() if p.requires_grad)
    else:
        print("\n   MOE parameters found:")
        for name in moe_param_names[:10]:
            print(f"   - {name}")
        if len(moe_param_names) > 10:
            print(f"   ... and {len(moe_param_names) - 10} more")

    # 设置模型为训练模式
    model.train()

    # 获取可训练的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n📊 Trainable parameters: {len(trainable_params)}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    # 优化器（只优化可训练的参数）
    if len(trainable_params) == 0:
        print("\n❌ ERROR: No trainable parameters found!")
        print("   Please check your model structure.")
        sys.exit(1)

    # ✅ 使用合理的学习率
    # 注意：学习率太小（1e-8）会导致损失不下降
    learning_rate = args.learning_rate  # 1e-6 * 0.1 = 1e-7（合理的学习率）

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,  # 增加数值稳定性
        betas=(0.9, 0.999)  # 标准 Adam 参数
    )

    # ✅ 添加学习率调度（余弦退火）
    total_steps = args.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate * 0.1
    )

    print(f"\n📊 Optimizer configuration:")
    print(f"   Learning rate: {learning_rate:.2e}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Gradient clipping: 1.0")
    print(f"   Learning rate scheduler: CosineAnnealingLR")
    print(f"   Total training steps: {total_steps}")

    # 训练循环
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    best_eval_accuracy = 0.0  # ✅ 改为根据 accuracy 保存最佳模型
    best_moe_dir = os.path.join(args.output_dir, 'best_moe')
    best_lambda = 0.0  # ✅ 记录最佳模型的 lambda 值
    best_epoch = 0  # ✅ 记录最佳 epoch

    epoch_results = []

    for epoch in range(args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            scheduler=scheduler,
            use_culture_loss=args.use_culture_loss,
            culture_loss_lambda=args.culture_loss_lambda,
            culture_loss_alpha=args.culture_loss_alpha,
            culture_loss_beta=args.culture_loss_beta,
            router_temperature=args.router_temperature,
            load_balance_weight=args.load_balance_weight,
            entropy_weight=args.entropy_weight,
            num_accumulation_steps=args.gradient_accumulation_steps,
            class_weights=class_weights  # ✅ 传递类别权重
        )

        # ✅ 获取当前 lambda 值
        current_lambda = train_metrics.get('culture_loss_lambda', 0.0)

        print(f"\n📊 Epoch {epoch + 1} Training Results:")
        print(f"   Train Loss: {train_metrics['loss']:.4f}")
        print(f"   Train Gen Loss: {train_metrics['gen_loss']:.4f}")
        print(f"   Train Culture Loss: {train_metrics['culture_loss']:.4f}")
        print(f"   Train Spec Loss: {train_metrics['spec_loss']:.4f}")
        print(f"   Train Div Loss: {train_metrics['div_loss']:.4f}")
        print(f"   Culture Loss Lambda: {current_lambda:.6f}")  # ✅ 打印 lambda 值

        # ✅ 打印防塌陷损失
        if 'load_balance_loss' in train_metrics:
            print(f"   Load Balance Loss: {train_metrics['load_balance_loss']:.6f}")
        if 'entropy_loss' in train_metrics:
            print(f"   Entropy Loss: {train_metrics['entropy_loss']:.6f}")
        if 'neg_entropy_loss' in train_metrics:
            print(f"   Neg Entropy Loss: {train_metrics['neg_entropy_loss']:.6f}")  # ✅ 打印负熵损失

        # ✅ 每 eval_interval 个 epoch 进行一次评估
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.num_epochs:
            print(f"\n{'='*80}")
            print(f"Evaluating at Epoch {epoch + 1}...")
            print(f"{'='*80}")

            # 验证
            val_metrics = evaluate(
                model, val_loader, args.device,
                use_culture_loss=args.use_culture_loss,
                culture_loss_lambda=args.culture_loss_lambda
            )

            # 生成答案并评估准确率（Post Eval）
            gen_metrics = generate_and_evaluate_answers(
                model, val_dataset, tokenizer, args.device, args.output_dir, epoch=epoch+1
            )

            print(f"\n📊 Epoch {epoch + 1} Evaluation Results:")
            print(f"   Eval Loss: {val_metrics['loss']:.4f}")
            print(f"   Eval Gen Loss: {val_metrics['gen_loss']:.4f}")
            print(f"   Eval Culture Loss: {val_metrics['culture_loss']:.4f}")
            print(f"   Eval Accuracy: {gen_metrics['accuracy']:.4f} ({gen_metrics['correct']}/{gen_metrics['total']})")

            # ✅ 根据 accuracy 保存最好的模型
            if gen_metrics['accuracy'] > best_eval_accuracy:
                best_eval_accuracy = gen_metrics['accuracy']
                best_epoch = epoch + 1  # ✅ 记录最佳 epoch
                best_lambda = current_lambda  # ✅ 记录最佳模型的 lambda 值

                # 删除旧的最好模型
                if os.path.exists(best_moe_dir):
                    import shutil
                    shutil.rmtree(best_moe_dir)

                # 保存新的最好模型
                os.makedirs(best_moe_dir, exist_ok=True)

                try:
                    # ✅ 只保存 MoE 部分的参数
                    moe_state_dict = {}
                    for name, param in model.named_parameters():
                        # 只保存 MoE 相关的参数
                        if any(keyword in name.lower() for keyword in ['expert', 'router', 'shared', 'moe']):
                            moe_state_dict[name] = param.data.cpu()

                    torch.save(moe_state_dict, os.path.join(best_moe_dir, "moe_weights.bin"))
                    print(f"   ✅ Best MoE model saved (accuracy: {best_eval_accuracy:.4f})")
                    print(f"   Saved {len(moe_state_dict)} MoE parameters")
                except Exception as e:
                    print(f"   ⚠️  Warning: Failed to save MoE model: {str(e)}")
                    print(f"   Continuing training without saving...")

            # 记录结果（包含评估指标）
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_gen_loss': train_metrics['gen_loss'],
                'train_culture_loss': train_metrics['culture_loss'],
                'culture_loss_lambda': current_lambda,  # ✅ 记录 lambda 值
                'eval_loss': val_metrics['loss'],
                'eval_gen_loss': val_metrics['gen_loss'],
                'eval_culture_loss': val_metrics['culture_loss'],
                'eval_accuracy': gen_metrics['accuracy'],
                'correct': gen_metrics['correct'],
                'total': gen_metrics['total'],
                'is_best': gen_metrics['accuracy'] == best_eval_accuracy
            })
        else:
            # 不评估的 epoch，只记录训练指标
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_gen_loss': train_metrics['gen_loss'],
                'train_culture_loss': train_metrics['culture_loss'],
                'eval_loss': None,
                'eval_gen_loss': None,
                'eval_culture_loss': None,
                'eval_accuracy': None,
                'correct': None,
                'total': None,
                'is_best': False
            })

    # 保存训练结果
    with open(os.path.join(args.output_dir, 'epoch_eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(epoch_results, f, indent=2, ensure_ascii=False)

    # 保存配置
    config = {
    'base_model': args.base_model_path,
    'lora_weights': args.lora_weights_path,
    'num_epochs': args.num_epochs,
    'batch_size': args.batch_size,
    'learning_rate': learning_rate,  # ✅ 保存实际使用的学习率
    'use_culture_loss': args.use_culture_loss,
    'culture_loss_lambda_initial': args.culture_loss_lambda,  # ✅ 初始 lambda 值
    'culture_loss_lambda_learnable': model.culture_loss_lambda_learnable,  # ✅ 是否可学习
    'num_experts': args.num_experts,
    'eval_interval': args.eval_interval,
    'best_epoch': best_epoch,  # ✅ 最佳 epoch
    'best_eval_accuracy': best_eval_accuracy,  # ✅ 最佳准确率
    'best_culture_loss_lambda': best_lambda,  # ✅ 最佳模型的 lambda 值
    'moe_fusion': args.moe_fusion,  # ✅ MoE 融合系数
    'data_format': 'new_format (instruction + instruction_mask + input + output + label)'
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ Training completed!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"\nFiles generated:")
    print(f"  - best_moe/ (Best MOE weights)")
    print(f"  - epoch_eval_results.json (Epoch-by-epoch results)")
    print(f"  - generated_answers.json (Generated answers on validation set)")
    print(f"  - config.json (Training configuration)")
    print("="*80)


if __name__ == "__main__":
    main()

