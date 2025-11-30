#!/usr/bin/env python3
"""
使用标准语言建模损失微调 LoRA Only 模型（新数据格式）

使用方法：
    python ft_lora_only_gen.py \
        --base_model_path /path/to/base_model \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --num_epochs 6

数据格式（新格式）：
    {
        "instruction": "Give me the answer from 1 to 4: Do you agree with ...",
        "instruction_mask": "Give me the answer from 1 to 4: Do you agree with ... [MASK]",
        "input": "This question is for a country or language that is Arabic.",
        "output": "2",
        "label": "0"
    }

关键特性：
    1. 使用 instruction + input 作为输入
    2. 使用 output 作为目标输出
    3. 使用标准语言建模损失
    4. 支持 post eval（生成答案并评估准确率）
    5. instruction_mask 和 label 字段被保存但不用于训练
"""

import argparse
import json
import os
import re
import sys

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CultureLLMNewFormatDataset(Dataset):
    """
    CultureLLM 新格式数据集

    数据格式：
    {
        "instruction": "Give me the answer from 1 to 4: ...",
        "instruction_mask": "Give me the answer from 1 to 4: ... [MASK]",
        "input": "This question is for a country or language that is Arabic.",
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
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        label = item.get('label', '')

        # 构建完整的输入和输出
        # 格式：instruction + input → output
        if input_text:
            full_input = f"{instruction}\n{input_text}"
        else:
            full_input = instruction

        # 完整的文本（用于语言建模）
        # 这样模型会学习：给定 instruction + input，生成 output
        full_text = f"{full_input}\n{output_text}"

        # 🔧 关键修复：正确的标签掩码和tokenizer一致性
        # 1. 先tokenize完整文本（统一使用add_special_tokens=True）
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            add_special_tokens=True  # 明确指定
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # 2. 正确计算input_length - 确保tokenizer参数一致！
        input_with_newline = f"{full_input}\n"
        encoded_input = self.tokenizer(
            input_with_newline,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            add_special_tokens=True  # 与完整文本保持一致
        )

        # ✅ 关键修复：正确计算input_length，避免padding污染
        # 方法：直接tokenize输入部分，不使用padding，然后计算实际长度
        encoded_input_no_pad = self.tokenizer(
            input_with_newline,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
            padding=False  # 不使用padding！
        )

        # 获取真实的输入长度（不包含padding）
        input_length = len(encoded_input_no_pad['input_ids'][0])

        # 🔧 Llama特殊token处理：正确识别padding token
        # 获取正确的pad_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            # 如果没有设置pad_token，使用默认的eos_token
            pad_token_id = self.tokenizer.eos_token_id

        # 🔧 验证：确保input_length不会超出完整序列的非padding部分
        total_non_pad = (input_ids != pad_token_id).sum().item()
        if input_length >= total_non_pad:
            # 如果输入部分已经占满了非padding部分，至少保留1个token给输出
            input_length = max(0, total_non_pad - 1)

        # 🔧 额外验证：检查是否正确识别了<|eot_id|>
        eot_token_id = 128009  # Llama的<|eot_id|>
        eot_positions = (input_ids == eot_token_id).nonzero(as_tuple=True)[0]
        if len(eot_positions) > 0:
            first_eot_pos = eot_positions[0].item()
            # 如果输入部分超过了第一个<|eot_id|>位置，需要调整
            if input_length > first_eot_pos:
                input_length = first_eot_pos

        # 3. 创建正确的标签
        labels = input_ids.clone()

        # 4. 掩码输入部分（设为-100，不计算损失）
        labels[:input_length] = -100

        # 5. 计算有效的训练token数量
        valid_labels = (labels != -100).sum().item()
        total_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()

        # 🔍 详细的labels调试信息（前2个样本）
        if idx < 2:
            print(f"\n📋 样本 {idx} - 有效标签数: {valid_labels}")

            # 检查tokenizer配置
            eot_token_id = 128009  # <|eot_id|>
            actual_pad_token_id = self.tokenizer.pad_token_id

            print(f"  🔧 Tokenizer状态:")
            print(f"    pad_token: {repr(self.tokenizer.pad_token)}")
            print(f"    pad_token_id: {actual_pad_token_id}")
            print(f"    eos_token_id: {self.tokenizer.eos_token_id}")

            # 分析labels中的token分布
            eot_in_labels = (labels == eot_token_id).sum().item()
            if actual_pad_token_id is not None:
                pad_in_labels = (labels == actual_pad_token_id).sum().item()
            else:
                pad_in_labels = 0

            # 统计所有非-100的token
            non_mask_indices = (labels != -100).nonzero(as_tuple=True)[0]

            print(f"  📊 Labels分析:")
            print(f"    总序列长度: {len(labels)}")
            print(f"    有效标签数: {valid_labels}")
            print(f"    <|eot_id|>(128009)数量: {eot_in_labels}")
            print(f"    padding token数量: {pad_in_labels}")

            if valid_labels > 10:
                print(f"  ⚠️ 标签数过多({valid_labels})，可能仍有padding问题")

                # 显示labels的分布情况
                unique_tokens = {}
                for pos in non_mask_indices[:50]:  # 检查前50个有效标签
                    token_id = labels[pos.item()].item()
                    unique_tokens[token_id] = unique_tokens.get(token_id, 0) + 1

                print(f"  🔍 前50个有效标签的token分布:")
                for token_id, count in sorted(unique_tokens.items(), key=lambda x: x[1], reverse=True)[:10]:
                    try:
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        print(f"    token_id={token_id}('{token_text}'): {count}次")
                    except:
                        print(f"    token_id={token_id}(解码失败): {count}次")

                # 显示labels的具体位置和值
                print(f"  📋 前20个有效标签详情:")
                for i, pos in enumerate(non_mask_indices[:20]):
                    pos_idx = pos.item()
                    token_id = labels[pos_idx].item()
                    try:
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        print(f"    位置{pos_idx}: {token_id}='{token_text}'")
                    except:
                        print(f"    位置{pos_idx}: {token_id}=(解码失败)")

            elif eot_in_labels > 1:
                print(f"  ⚠️ 训练标签包含{eot_in_labels}个<|eot_id|>")
                # 显示所有有效标签
                print(f"  📋 所有有效标签:")
                for i, pos in enumerate(non_mask_indices):
                    pos_idx = pos.item()
                    token_id = labels[pos_idx].item()
                    try:
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        print(f"    位置{pos_idx}: {token_id}='{token_text}'")
                    except:
                        print(f"    位置{pos_idx}: {token_id}=(解码失败)")

            elif valid_labels <= 5:
                print(f"  ✅ 标签数正常({valid_labels})")
                # 显示所有有效标签
                print(f"  📋 所有有效标签:")
                for i, pos in enumerate(non_mask_indices):
                    pos_idx = pos.item()
                    token_id = labels[pos_idx].item()
                    try:
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        print(f"    位置{pos_idx}: {token_id}='{token_text}'")
                    except:
                        print(f"    位置{pos_idx}: {token_id}=(解码失败)")

            # 检查input_length计算是否正确
            print(f"  🔧 Input length计算:")
            print(f"    计算的input_length: {input_length}")
            print(f"    序列总长度: {len(input_ids)}")
            print(f"    非padding长度: {(input_ids != (actual_pad_token_id or 0)).sum().item()}")

            # 检查是否pad_token_id配置错误导致问题
            if actual_pad_token_id == eot_token_id:
                print(f"  🚨 发现问题: pad_token_id == <|eot_id|> ({actual_pad_token_id})")
                print(f"    这会导致padding区域填充<|eot_id|>，造成大量有效标签!")
            elif actual_pad_token_id is None:
                print(f"  🚨 发现问题: pad_token_id is None!")
                print(f"    tokenizer配置可能没有正确应用")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'instruction': instruction,
            'input': input_text,
            'output': output_text,
            'label': label,
            'valid_labels': valid_labels,
            'total_tokens': total_tokens,
            'input_length': input_length  # 添加调试信息
        }


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
    dataset = CultureLLMNewFormatDataset(data_path, tokenizer, max_length)

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

    关键改进：
    - 只输入 instruction + input，不输入 output
    - 让模型生成 output

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
    # 构建输入 - 与训练时格式完全保持一致
    # 训练时的格式：full_input = f"{instruction}\n{input_text}"，然后添加\n{output}
    # 所以生成时应该给模型：full_input + \n，让它生成output
    if input_text:
        full_input = f"{instruction}\n{input_text}\n"  # 与训练时的full_text开头一致
    else:
        full_input = f"{instruction}\n"  # 与训练时的full_text开头一致

    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # 检查模型类型，确定使用哪种generate方法
        model_class_name = model.__class__.__name__
        # print(f"🔍 Model class: {model_class_name}")  # 注释掉详细调试

        if 'JointLoRAMoE' in model_class_name or hasattr(model, 'moe_layer'):
            # 使用联合模型的自定义generate方法，确保通过MoE层
            # print(f"🔍 Using custom joint model generate method")  # 注释掉详细调试
            # print(f"🔍 Input shape: {inputs['input_ids'].shape}")  # 注释掉详细调试
            # print(f"🔍 Input tokens: {inputs['input_ids'][0].tolist()}")  # 注释掉详细调试

            # 检查模型是否真的是联合模型
            # if hasattr(model, 'base_model') and hasattr(model, 'moe_layer'):
            #     print(f"🔍 Confirmed: Model has both base_model and moe_layer")
            # else:
            #     print(f"⚠️ Warning: Model structure unexpected")

            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,  # 启用采样避免重复
                temperature=0.8,  # 适中的温度
                repetition_penalty=1.1  # 添加重复惩罚
            )
        else:
            # 回退到标准generate方法
            # print(f"🔍 Using standard model generate method")  # 注释掉详细调试
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,  # 贪婪解码
                num_beams=1,      # 禁用 beam search
                repetition_penalty=1.0
            )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 简化的生成信息（只显示结果）
    if len(generated_text.strip()) == 0:
        print(f"⚠️ Empty generation")
    # print(f"🔍 Generated token IDs: {generated_ids.tolist()}")  # 注释掉详细调试
    # print(f"🔍 Generated text: {repr(generated_text)}")  # 注释掉详细调试
    # print(f"🔍 Generated text length: {len(generated_text)}")  # 注释掉详细调试

    # 检查生成的token是否在合理范围内
    # vocab_size = tokenizer.vocab_size
    # valid_tokens = [tid for tid in generated_ids.tolist() if 0 <= tid < vocab_size]
    # print(f"🔍 Valid tokens: {len(valid_tokens)}/{len(generated_ids)}")  # 注释掉详细调试

    # if len(generated_ids) > 0:
        # 检查前几个生成的token
        # for i, token_id in enumerate(generated_ids[:5].tolist()):
        #     try:
        #         token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        #         print(f"🔍 Token {i}: {token_id} -> {repr(token_text)}")
        #     except Exception as e:
        #         print(f"🔍 Token {i}: {token_id} -> ERROR: {e}")  # 注释掉详细调试

    return generated_text


def train_epoch(model, train_loader, optimizer, device, num_accumulation_steps=1):
    """
    训练一个 epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        num_accumulation_steps: 梯度累积步数

    Returns:
        dict: 包含训练指标的字典
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", disable=False, mininterval=1.0)

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # ✅ 检查 NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ NaN or Inf loss detected at batch {batch_idx}")
            continue

        # 梯度累积
        loss = loss / num_accumulation_steps
        loss.backward()

        total_loss += loss.item() * num_accumulation_steps
        num_batches += 1

        # 梯度更新
        if (batch_idx + 1) % num_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix({'loss': f"{loss.item() * num_accumulation_steps:.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'num_batches': num_batches
    }


def evaluate(model, val_loader, device):
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备

    Returns:
        dict: 包含评估指标的字典
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    # 获取实际的模型（如果被 DDP 包装）
    actual_model = model.module if isinstance(model, DDP) else model

    pbar = tqdm(val_loader, desc="Evaluating", disable=False, mininterval=1.0)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'num_batches': num_batches
    }


def generate_and_evaluate_answers(model, val_dataset, tokenizer, device, output_dir, epoch=None):
    """
    在验证集上生成答案并评估准确率（Post Eval）

    关键改进：
    - 只输入 instruction + input，不输入 output
    - 让模型生成答案
    - 通过正则表达式提取数字答案
    - 与真实答案比对

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

    for idx in tqdm(range(len(val_dataset)), desc="Generating", disable=False, mininterval=1.0):
        # 获取原始数据集（处理 Subset 对象）
        if hasattr(val_dataset, 'dataset'):
            # val_dataset 是 Subset 对象
            original_idx = val_dataset.indices[idx]
            sample = val_dataset.dataset[original_idx]
        else:
            # val_dataset 是普通 Dataset 对象
            sample = val_dataset[idx]

        instruction = sample['instruction']
        input_text = sample['input']
        true_output = sample['output']
        label = sample['label']

        # 生成答案（只输入 instruction + input）
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

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LoRA Only model with new data format")

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (new format JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=6,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval (every N epochs)")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Fine-tuning LoRA Only Model with New Data Format")
    print("="*80)
    print(f"Base model: {args.base_model_path}")
    print(f"Training data: {args.train_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    # 🔧 修复Llama 3.1 tokenizer配置问题
    if tokenizer.pad_token is None:
        # 检查是否是Llama 3.1 (eos_token_id是128009)
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128009:
            # Llama 3.1: 查找真正的<unk> token
            if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
                # 使用真正的<unk> token
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
                print(f"🔧 Llama 3.1: 使用真正的<unk> token: '{tokenizer.unk_token}' (id={tokenizer.pad_token_id})")
            else:
                # 如果没有<unk>，使用eos_token作为padding（避免添加新token）
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"🔧 Llama 3.1: 使用eos_token作为padding: '{tokenizer.eos_token}' (id={tokenizer.pad_token_id})")
        else:
            # 其他模型：使用标准配置
            tokenizer.pad_token = tokenizer.eos_token
            print(f"🔧 标准配置: pad_token = eos_token")

    tokenizer.padding_side = "right"

    # 验证tokenizer配置
    print(f"✅ Tokenizer配置验证:")
    print(f"  pad_token: {repr(tokenizer.pad_token)}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  eos_token: {repr(tokenizer.eos_token)}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  unk_token: {repr(tokenizer.unk_token)}")
    if hasattr(tokenizer, 'unk_token_id'):
        print(f"  unk_token_id: {tokenizer.unk_token_id}")

    # 🚨 强制验证和修复
    if tokenizer.pad_token_id == 128009:
        print(f"🚨 严重错误: pad_token_id仍然是128009 (<|eot_id|>)!")
        print(f"   强制修复tokenizer配置...")

        # 使用eos_token作为padding（避免添加新token）
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"   修复后pad_token_id: {tokenizer.pad_token_id}")
        print(f"   修复后pad_token: {repr(tokenizer.pad_token)}")

    elif tokenizer.pad_token_id is None:
        print(f"🚨 错误: pad_token_id is None!")
        print(f"   强制设置专用padding token...")

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"   设置后pad_token_id: {tokenizer.pad_token_id}")
        print(f"   设置后pad_token: {repr(tokenizer.pad_token)}")

    elif tokenizer.pad_token_id == 0:
        # 检查token_id=0对应的实际字符
        token_0_text = tokenizer.decode([0], skip_special_tokens=True)
        print(f"⚠️ pad_token_id = 0，对应字符: '{token_0_text}'")

        if token_0_text.strip() not in ['<unk>', '']:  # 如果不是真正的<unk>
            print(f"🚨 问题: token_id=0 不是真正的<unk>，而是'{token_0_text}'!")
            print(f"   这会导致padding区域填充'{token_0_text}'字符")
            print(f"   强制创建专用padding token...")

            # 使用一个现有的低频token作为padding，避免添加新token
            # 找一个不常用的标点符号token
            candidate_tokens = ['~', '`', '|', '^']
            chosen_pad_token = None

            for token in candidate_tokens:
                try:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if token_id != tokenizer.unk_token_id:  # 确保不是unk
                        chosen_pad_token = token
                        tokenizer.pad_token = token
                        tokenizer.pad_token_id = token_id
                        break
                except:
                    continue

            if chosen_pad_token is None:
                # 如果找不到合适的token，使用eos_token
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                chosen_pad_token = tokenizer.eos_token

            print(f"   修复后pad_token_id: {tokenizer.pad_token_id}")
            print(f"   修复后pad_token: {repr(chosen_pad_token)}")
        else:
            print(f"✅ 正确: pad_token_id = 0 是真正的<unk>")
    else:
        print(f"✅ 正确: pad_token_id ({tokenizer.pad_token_id}) != 128009")

    # 最终验证
    print(f"\n🔧 最终tokenizer状态:")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  是否等于<|eot_id|>: {tokenizer.pad_token_id == 128009}")
    print(f"  是否为None: {tokenizer.pad_token_id is None}")

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

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 加载模型
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Base model loaded")

    # 配置 LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA configured")

    # 设置模型为评估模式（禁用 dropout）
    model.eval()

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 训练循环
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    best_eval_accuracy = 0.0  # ✅ 改为根据 accuracy 保存最佳模型
    best_model_dir = os.path.join(args.output_dir, 'best_lora')

    epoch_results = []

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            num_accumulation_steps=args.gradient_accumulation_steps
        )

        print(f"  Train Loss: {train_metrics['loss']:.4f}")

        # 每 eval_interval 个 epoch 进行一次验证
        if (epoch + 1) % args.eval_interval == 0:
            # 验证
            val_metrics = evaluate(model, val_loader, args.device)

            # 生成答案并评估准确率（Post Eval）
            gen_metrics = generate_and_evaluate_answers(
                model, val_dataset, tokenizer, args.device, args.output_dir, epoch=epoch+1
            )

            print(f"  Eval Loss: {val_metrics['loss']:.4f}")
            print(f"  Eval Accuracy: {gen_metrics['accuracy']:.4f} ({gen_metrics['correct']}/{gen_metrics['total']})")

            # ✅ 根据 accuracy 保存最好的模型
            if gen_metrics['accuracy'] > best_eval_accuracy:
                best_eval_accuracy = gen_metrics['accuracy']

                # 删除旧的最好模型
                if os.path.exists(best_model_dir):
                    import shutil
                    shutil.rmtree(best_model_dir)

                # 保存新的最好模型
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                print(f"  ✅ Best model saved (accuracy: {best_eval_accuracy:.4f})")

            # 记录结果
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'eval_loss': val_metrics['loss'],
                'eval_accuracy': gen_metrics['accuracy'],
                'correct': gen_metrics['correct'],
                'total': gen_metrics['total'],
                'is_best': gen_metrics['accuracy'] == best_eval_accuracy  # ✅ 标记是否为最佳
            })
        else:
            # 不评估的 epoch，只记录训练损失
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'eval_loss': None,
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
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'eval_interval': args.eval_interval,
        'best_eval_accuracy': best_eval_accuracy,  # ✅ 改为保存最佳准确率
        'data_format': 'new_format (instruction + input + output)'
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ Training completed!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"\nFiles generated:")
    print(f"  - best_lora/ (Best LoRA weights)")
    print(f"  - epoch_eval_results.json (Epoch-by-epoch results)")
    print(f"  - generated_answers.json (Generated answers on validation set)")
    print(f"  - config.json (Training configuration)")
    print("="*80)


if __name__ == "__main__":
    main()

