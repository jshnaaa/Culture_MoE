#!/usr/bin/env python3
"""
从 Base 模型 + LoRA 权重训练 CultureMoE（生成式版本 - 支持双卡训练）

使用方法：
    # 单卡训练
    python train_culturemoe_from_base_gen.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --use_culture_loss True

    # 双卡训练
    torchrun --nproc_per_node=2 train_culturemoe_from_base_gen.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --use_culture_loss True
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ✅ 使用生成式数据处理，而不是分类数据处理
# from src.llamafactory.data.dual_classification_collator import DualClassificationDataCollator
# from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


def load_and_process_generative_data(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    val_split: float = 0.1,
    use_instruction_mask: bool = True
):
    """
    加载并处理生成式数据（双路输入）

    Args:
        data_path: 数据文件路径
        tokenizer: Tokenizer
        max_length: 最大序列长度
        val_split: 验证集比例
        use_instruction_mask: 是否使用 instruction_mask

    Returns:
        dict: 包含 'train', 'validation', 'validation_raw' 的字典
    """
    import json
    from datasets import Dataset

    print(f"Loading data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # 处理数据
    processed_data = []
    for item in data:
        instruction = item.get('instruction', '')
        instruction_mask = item.get('instruction_mask', instruction) if use_instruction_mask else instruction
        input_text = item.get('input', '')
        output = str(item['output'])
        culture_label = item.get('label', '')

        # 第一路：instruction + input
        if input_text and input_text.strip():
            full_text = f"{instruction}\n{input_text}\nAnswer:"
        else:
            full_text = f"{instruction}\nAnswer:"

        # 第二路：instruction_mask + input
        if input_text and input_text.strip():
            mask_text = f"{instruction_mask}\n{input_text}\nAnswer:"
        else:
            mask_text = f"{instruction_mask}\nAnswer:"

        # Tokenize 第一路（instruction + input）
        encoded_full = tokenizer(
            full_text,
            max_length=max_length - 10,  # 留出空间给答案
            truncation=True,
            padding=False,
            return_tensors=None
        )

        # Tokenize 第二路（instruction_mask + input）
        encoded_mask = tokenizer(
            mask_text,
            max_length=max_length - 10,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        # Tokenize 答案
        answer_tokens = tokenizer(
            f" {output}",
            add_special_tokens=False,
            return_tensors=None
        )

        # 合并：prompt + answer
        input_ids_full = encoded_full['input_ids'] + answer_tokens['input_ids']
        attention_mask_full = encoded_full['attention_mask'] + answer_tokens['attention_mask']

        # Labels：prompt 部分用 -100，answer 部分用真实 token
        labels = [-100] * len(encoded_full['input_ids']) + answer_tokens['input_ids']

        # 处理 culture_label
        if isinstance(culture_label, str) and culture_label.strip():
            culture_labels = [int(x.strip()) for x in culture_label.split(',') if x.strip().isdigit()]
        elif isinstance(culture_label, int):
            culture_labels = [culture_label]
        else:
            culture_labels = []

        processed_data.append({
            'input_ids': input_ids_full,
            'attention_mask': attention_mask_full,
            'input_ids_mask': encoded_mask['input_ids'] + answer_tokens['input_ids'],
            'attention_mask_mask': encoded_mask['attention_mask'] + answer_tokens['attention_mask'],
            'labels': labels,
            'culture_labels': culture_labels,
            'original_output': output
        })

    # ✅ 随机划分训练集和验证集（保证标签均衡）
    import random
    random.seed(42)  # 固定随机种子，保证可复现

    # ✅ 同时打乱 processed_data 和原始 data（保持对应关系）
    indices = list(range(len(data)))
    random.shuffle(indices)  # 打乱索引

    # 使用相同的索引划分
    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # 根据索引获取数据
    train_data = [processed_data[i] for i in train_indices]
    val_data = [processed_data[i] for i in val_indices]
    val_data_raw = [data[i] for i in val_indices]  # ✅ 使用相同的索引

    print(f"Train: {len(train_data)}, Val: {len(val_data)} (randomly shuffled)")

    # 创建 Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # 保存原始验证集（用于生成式评估）
    val_dataset_raw = val_data_raw  # ✅ 使用相同索引的原始数据

    return {
        'train': train_dataset,
        'validation': val_dataset,
        'validation_raw': val_dataset_raw
    }


def generative_data_collator(features, tokenizer):
    """
    生成式数据整理器（双路输入）

    Args:
        features: 样本列表
        tokenizer: Tokenizer

    Returns:
        批次数据字典
    """
    # 获取最大长度
    max_len_full = max(len(f['input_ids']) for f in features)
    max_len_mask = max(len(f['input_ids_mask']) for f in features)

    batch = {
        'input_ids': [],
        'attention_mask': [],
        'input_ids_mask': [],
        'attention_mask_mask': [],
        'labels': [],
        'culture_labels': []
    }

    for feature in features:
        # 第一路 padding
        padding_len_full = max_len_full - len(feature['input_ids'])
        batch['input_ids'].append(feature['input_ids'] + [tokenizer.pad_token_id] * padding_len_full)
        batch['attention_mask'].append(feature['attention_mask'] + [0] * padding_len_full)
        batch['labels'].append(feature['labels'] + [-100] * padding_len_full)

        # 第二路 padding
        padding_len_mask = max_len_mask - len(feature['input_ids_mask'])
        batch['input_ids_mask'].append(feature['input_ids_mask'] + [tokenizer.pad_token_id] * padding_len_mask)
        batch['attention_mask_mask'].append(feature['attention_mask_mask'] + [0] * padding_len_mask)

        # Culture labels
        batch['culture_labels'].append(feature['culture_labels'])

    # 转换为 tensor
    return {
        'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
        'input_ids_mask': torch.tensor(batch['input_ids_mask'], dtype=torch.long),
        'attention_mask_mask': torch.tensor(batch['attention_mask_mask'], dtype=torch.long),
        'labels': torch.tensor(batch['labels'], dtype=torch.long),
        'culture_labels': batch['culture_labels']
    }


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model_from_components(
    base_model_path: str,
    lora_weights_path: str,
    moe_args: ModelArgs,
    device: str = "cuda",
    is_distributed: bool = False,
    local_rank: int = 0
):
    """
    从 Base 模型 + LoRA 权重创建 CultureMoE 模型

    Args:
        base_model_path: Base 模型路径
        lora_weights_path: LoRA 权重路径
        moe_args: MoE 配置
        device: 设备
        is_distributed: 是否使用分布式训练
        local_rank: 本地 rank

    Returns:
        model: CultureMoE 模型
        tokenizer: Tokenizer
    """
    # 只在主进程打印
    if not is_distributed or local_rank == 0:
        print("\n" + "="*80)
        print("Loading Model from Components")
        print("="*80)
        print(f"Base model: {base_model_path}")
        print(f"LoRA weights: {lora_weights_path}")
        print(f"Distributed: {is_distributed}")
        if is_distributed:
            print(f"Local rank: {local_rank}")
        print("="*80)
        print("")

    # 1. 加载 Tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    # 2. 加载 Base 模型
    print("\n2. Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # 强制使用单GPU，避免DTensor问题
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("   ✅ Base model loaded")

    # 3. 加载 LoRA 权重并合并
    print("\n3. Loading and merging LoRA weights...")
    from peft import PeftModel

    # 检查 LoRA 权重目录
    if os.path.exists(lora_weights_path):
        lora_files = os.listdir(lora_weights_path)
        print(f"   LoRA directory contents: {lora_files}")
    else:
        raise FileNotFoundError(f"LoRA weights directory not found: {lora_weights_path}")

    # 检查 adapter_config.json 是否存在
    adapter_config_path = os.path.join(lora_weights_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {lora_weights_path}")

    print(f"   Found adapter_config.json: {adapter_config_path}")

    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        is_trainable=False,
        torch_dtype=torch.float16
    )
    print("   ✅ LoRA weights loaded")

    print("   Merging LoRA weights into base model...")
    merged_model = model_with_lora.merge_and_unload()
    print("   ✅ LoRA weights merged")

    # 清理内存
    del model_with_lora
    torch.cuda.empty_cache()

    # 4. 冻结 LLaMA 参数
    print("\n4. Freezing LLaMA parameters...")
    for param in merged_model.parameters():
        param.requires_grad = False
    print("   ✅ LLaMA parameters frozen")

    # 5. 创建 CultureMoE 模型
    print("\n5. Creating CultureMoE model...")
    culturemoe_model = LlamaSharedRouterExpertsModel(
        llama_model=merged_model,
        config=merged_model.config,
        args=moe_args
    )
    print("   ✅ CultureMoE model created")

    # 6. 移动到设备
    print(f"\n6. Moving model to {device}...")
    culturemoe_model = culturemoe_model.to(device)
    print("   ✅ Model ready")

    # 打印参数统计
    total_params = sum(p.numel() for p in culturemoe_model.parameters())
    trainable_params = sum(p.numel() for p in culturemoe_model.parameters() if p.requires_grad)
    print(f"\n📊 Model Info:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"   Device: {device}")

    print("\n" + "="*80)
    print("✅ Model Loaded Successfully!")
    print("="*80)
    print("")

    return culturemoe_model, tokenizer


def train_epoch(model, train_loader, optimizer, device, use_culture_loss, culture_loss_lambda,
                lambda_entropy=0.05, lambda_load=0.01):
    """
    训练一个 epoch

    Args:
        lambda_entropy: 熵正则化系数（推荐 0.01-0.1）
        lambda_load: 负载均衡系数（推荐 0.01）
    """
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_culture_loss = 0
    total_entropy_loss = 0
    total_load_loss = 0
    all_preds = []
    all_labels = []
    nan_count = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        input_ids_mask = batch['input_ids_mask'].to(device)
        attention_mask_mask = batch['attention_mask_mask'].to(device)
        labels = batch['labels'].to(device)
        culture_labels = batch['culture_labels']

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids_mask=input_ids_mask,
            attention_mask_mask=attention_mask_mask,
            labels=labels,
            culture_labels=culture_labels,
            use_culture_loss=use_culture_loss,
            culture_loss_lambda=culture_loss_lambda
        )

        loss = outputs['loss']

        # ✅ 计算熵正则化（Router 熵）
        entropy_loss = torch.tensor(0.0, device=device)
        if lambda_entropy > 0:
            # 从模型获取专家权重
            actual_model = model.module if isinstance(model, DDP) else model
            expert_weights = actual_model.get_expert_weights()  # [batch, num_experts]

            if expert_weights is not None:
                # 计算熵：H = -sum(p * log(p))
                entropy = -(expert_weights * torch.log(expert_weights + 1e-12)).sum(dim=-1).mean()
                # 负熵项：惩罚过低熵（鼓励保持一定的熵）
                entropy_loss = -entropy
                loss = loss + lambda_entropy * entropy_loss

        # ✅ 计算负载均衡损失
        load_loss = torch.tensor(0.0, device=device)
        if lambda_load > 0:
            # 从模型获取专家权重
            actual_model = model.module if isinstance(model, DDP) else model
            expert_weights = actual_model.get_expert_weights()  # [batch, num_experts]

            if expert_weights is not None:
                # 计算每个专家的重要性（接收的总权重）
                importance = expert_weights.sum(dim=0)  # [num_experts]
                importance_norm = importance / (importance.sum() + 1e-12)
                # 负载均衡：惩罚分布尖峰（鼓励均匀分布）
                load_loss = (importance_norm ** 2).sum()
                loss = loss + lambda_load * load_loss

        # ✅ 检查 NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            print(f"\n⚠️  NaN/Inf loss detected! Skipping batch...")
            print(f"   Loss: {loss.item()}")
            print(f"   Generation loss: {outputs['generation_loss'].item()}")
            if use_culture_loss:
                print(f"   Culture loss: {outputs['culture_loss'].item()}")
            optimizer.zero_grad()  # 清空梯度
            continue

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # ✅ 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ✅ 更新 MoE 预热步数
        actual_model = model.module if isinstance(model, DDP) else model
        actual_model.step_moe_warmup()

        # 统计
        total_loss += loss.item()
        total_cls_loss += outputs['generation_loss'].item()  # ✅ 修改为 generation_loss
        if use_culture_loss:
            total_culture_loss += outputs['culture_loss'].item()
        total_entropy_loss += entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss
        total_load_loss += load_loss.item() if isinstance(load_loss, torch.Tensor) else load_loss

        # ✅ 生成式模型不需要在训练时收集预测（会很慢）
        # 只在评估时使用 generate() 方法

        # 更新进度条
        warmup_weight = actual_model.get_moe_warmup_weight()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'gen_loss': f"{outputs['generation_loss'].item():.4f}",  # ✅ 修改为 gen_loss
            'entropy': f"{entropy_loss.item():.4f}" if isinstance(entropy_loss, torch.Tensor) else "0.0000",
            'load': f"{load_loss.item():.4f}" if isinstance(load_loss, torch.Tensor) else "0.0000",
            'moe_warmup': f"{warmup_weight:.2f}"  # ✅ 显示预热权重
        })

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_gen_loss = total_cls_loss / len(train_loader)  # ✅ 修改为 gen_loss
    avg_culture_loss = total_culture_loss / len(train_loader) if use_culture_loss else 0.0
    avg_entropy_loss = total_entropy_loss / len(train_loader)
    avg_load_loss = total_load_loss / len(train_loader)

    # ✅ 生成式模型在训练时不计算准确率（太慢）
    # 准确率只在评估时通过 generate() 计算

    return {
        'loss': avg_loss,
        'gen_loss': avg_gen_loss,  # ✅ 修改为 gen_loss
        'culture_loss': avg_culture_loss,
        'entropy_loss': avg_entropy_loss,  # ✅ 新增
        'load_loss': avg_load_loss,  # ✅ 新增
        'accuracy': 0.0  # ✅ 训练时不计算准确率
    }


def evaluate(model, val_loader, device, use_culture_loss, culture_loss_lambda, num_classes):
    """评估模型（只计算 loss，不计算准确率）"""
    model.eval()
    total_loss = 0
    total_gen_loss = 0  # ✅ 修改为 gen_loss
    total_culture_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating (loss only)"):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids_mask = batch['input_ids_mask'].to(device)
            attention_mask_mask = batch['attention_mask_mask'].to(device)
            labels = batch['labels'].to(device)
            culture_labels = batch['culture_labels']

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_mask=input_ids_mask,
                attention_mask_mask=attention_mask_mask,
                labels=labels,
                culture_labels=culture_labels,
                use_culture_loss=use_culture_loss,
                culture_loss_lambda=culture_loss_lambda
            )

            # 统计
            total_loss += outputs['loss'].item()
            total_gen_loss += outputs['generation_loss'].item()  # ✅ 修改为 generation_loss
            if use_culture_loss:
                total_culture_loss += outputs['culture_loss'].item()

    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    avg_gen_loss = total_gen_loss / len(val_loader)  # ✅ 修改为 gen_loss
    avg_culture_loss = total_culture_loss / len(val_loader) if use_culture_loss else 0.0

    # ✅ 生成式模型的准确率通过 generate_and_evaluate() 计算
    return {
        'loss': avg_loss,
        'gen_loss': avg_gen_loss,  # ✅ 修改为 gen_loss
        'culture_loss': avg_culture_loss
    }


def generate_and_evaluate(model, tokenizer, val_dataset, output_dir, num_classes, max_new_tokens=10):
    """
    生成答案并评估（使用真正的生成，而不是 teacher forcing）

    这个函数会：
    1. 对验证集中的每个样本生成答案
    2. 保存到 generated_answers.json
    3. 调用 post-eval 脚本计算指标
    """
    import subprocess
    import re

    print("\n" + "="*80)
    print("Generating answers for evaluation...")
    print("="*80)

    model.eval()
    generated_answers = []
    device = next(model.parameters()).device

    for i in tqdm(range(len(val_dataset)), desc="Generating"):
        sample = val_dataset[i]
        instruction = sample['instruction']
        input_text = sample['input']
        # ✅ 修复：原始数据字段名是 'output'，不是 'label'
        true_label = sample.get('output', sample.get('label', ''))  # 兼容两种字段名

        # 构建 prompt
        if num_classes <= 10:
            full_input = f"{instruction}\n{input_text}\n\nPlease answer with ONLY ONE NUMBER (1 to {num_classes}).\nYour answer:"
        else:
            full_input = f"{instruction}\n{input_text}\n\nYour answer:"

        # Tokenize
        inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ✅ 使用 model.forward() 而不是 model.llama_model.generate()
        # 这样可以确保经过 MoE 层
        with torch.no_grad():
            # 构建 input_ids_mask（如果需要）
            input_ids_mask = inputs['input_ids'].clone()
            attention_mask_mask = inputs['attention_mask'].clone()

            # ✅ 生成多个 token（最多 max_new_tokens 个）
            current_ids = inputs['input_ids'].clone()
            current_mask = inputs['attention_mask'].clone()
            generated_tokens = []

            for _ in range(max_new_tokens):
                # 调用 model.forward() 获取 logits
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    input_ids_mask=input_ids_mask,
                    attention_mask_mask=attention_mask_mask,
                    labels=None,
                    use_culture_loss=False
                )

                # 获取 logits [B, L, vocab_size]
                logits = outputs['logits']

                # 获取最后一个 token 的 logits
                last_token_logits = logits[:, -1, :]  # [B, vocab_size]

                # 贪婪解码：选择概率最高的 token
                predicted_token_id = torch.argmax(last_token_logits, dim=-1)  # [B]

                # 添加到生成的 tokens
                generated_tokens.append(predicted_token_id.item())

                # 检查是否生成了 EOS token
                if predicted_token_id.item() == tokenizer.eos_token_id:
                    break

                # 拼接到 input_ids
                current_ids = torch.cat([current_ids, predicted_token_id.unsqueeze(-1)], dim=1)
                current_mask = torch.cat([current_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)

                # 更新 input_ids_mask（保持相同长度）
                input_ids_mask = torch.cat([input_ids_mask, predicted_token_id.unsqueeze(-1)], dim=1)
                attention_mask_mask = torch.cat([attention_mask_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)

            # 解码生成的 tokens
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # 提取数字
        numbers = re.findall(r'\d+', generated_text)
        if numbers:
            predicted_label = numbers[0]
        else:
            predicted_label = generated_text

        # ✅ 调试：打印前几个样本的生成结果
        if i < 5:
            print(f"\n🔍 Sample {i}:")
            print(f"  Input: {full_input[:100]}...")
            print(f"  Generated text: '{generated_text}'")
            print(f"  Predicted: '{predicted_label}'")
            print(f"  True: '{true_label}'")

        generated_answers.append({
            "predicted": predicted_label,
            "true": str(true_label)
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
    correct = sum(1 for item in generated_answers if item["predicted"] == item["true"])
    accuracy = correct / len(generated_answers) if len(generated_answers) > 0 else 0.0
    return {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="从 Base + LoRA 训练 CultureMoE")

    # 模型路径
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, required=True)

    # 数据
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # 训练参数
    parser.add_argument("--use_culture_loss", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--culture_loss_lambda", type=float, default=0.5)
    parser.add_argument("--use_instruction_mask", type=lambda x: x.lower() == 'true', default=True)

    # ✅ MoE 正则化参数
    parser.add_argument("--lambda_entropy", type=float, default=0.05,
                        help="Entropy regularization coefficient (0.01-0.1)")
    parser.add_argument("--lambda_load", type=float, default=0.01,
                        help="Load balancing coefficient (0.01)")

    # MoE 参数
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--shared_hidden_dim", type=int, default=2048)
    parser.add_argument("--router_hidden_dim", type=int, default=1024)
    parser.add_argument("--experts_hidden_dim", type=int, default=2048)
    parser.add_argument("--moe_lora_rank", type=int, default=16)
    parser.add_argument("--classification_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=8)

    # 优化器参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)

    # 设备
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # ✅ 初始化分布式训练
    is_distributed, rank, world_size, local_rank = setup_distributed()

    # 设置设备
    if is_distributed:
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    # 只在主进程创建输出目录和打印信息
    if not is_distributed or rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("CultureMoE Training (From Base + LoRA)")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Distributed: {is_distributed}")
        if is_distributed:
            print(f"World size: {world_size}")
            print(f"Rank: {rank}")
            print(f"Local rank: {local_rank}")
        print(f"Base model: {args.base_model_path}")
        print(f"LoRA weights: {args.lora_weights_path}")
        print(f"Train file: {args.train_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Use culture loss: {args.use_culture_loss}")
        print(f"Use instruction mask: {args.use_instruction_mask}")
        print("="*80)
        print("")

    # 创建 MoE 配置
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

    # 加载模型
    model, tokenizer = load_model_from_components(
        args.base_model_path,
        args.lora_weights_path,
        moe_args,
        device,
        is_distributed=is_distributed,
        local_rank=local_rank
    )

    # ✅ 如果是分布式训练，包装模型
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # ✅ 设置 MoE 预热
    # 计算总训练步数
    print("Loading and processing data...")
    datasets = load_and_process_generative_data(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=args.val_split,
        use_instruction_mask=args.use_instruction_mask
    )
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    val_dataset_raw = datasets.get('validation_raw', None)  # ✅ 获取原始验证集
    print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")

    # 创建 DataLoader（使用生成式数据整理器）
    from functools import partial
    data_collator = partial(generative_data_collator, tokenizer=tokenizer)

    # ✅ 如果是分布式训练，使用 DistributedSampler
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,  # 使用 sampler 而不是 shuffle
            collate_fn=data_collator,
            num_workers=args.num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            sampler=val_sampler,  # 使用 sampler 而不是 shuffle
            collate_fn=data_collator,
            num_workers=args.num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=args.num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=args.num_workers
        )

    # ✅ 解冻 LoRA 权重（微调）
    print("Unfreezing LoRA weights for fine-tuning...")
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    # ✅ 分离参数组（使用分层学习率）
    lora_params = []
    moe_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora' in name.lower():
                lora_params.append(param)
            elif 'moe' in name.lower() or 'router' in name.lower() or 'shared' in name.lower() or 'experts' in name.lower():
                moe_params.append(param)
            else:
                other_params.append(param)

    if not is_distributed or rank == 0:
        print(f"✅ Parameter groups:")
        print(f"   LoRA params: {len(lora_params)} ({sum(p.numel() for p in lora_params):,} parameters)")
        print(f"   MoE params: {len(moe_params)} ({sum(p.numel() for p in moe_params):,} parameters)")
        print(f"   Other params: {len(other_params)} ({sum(p.numel() for p in other_params):,} parameters)")
        print("")

    # ✅ 创建优化器（使用分层学习率）
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': 1e-6},           # LoRA：低学习率（微调）
        {'params': moe_params, 'lr': args.learning_rate},  # MoE：原始学习率（快速学习）
        {'params': other_params, 'lr': 1e-6}           # 其他：低学习率
    ], weight_decay=args.weight_decay)

    if not is_distributed or rank == 0:
        print(f"✅ Optimizer created with layered learning rates:")
        print(f"   LoRA learning rate: 1e-6 (fine-tuning)")
        print(f"   MoE learning rate: {args.learning_rate} (fast learning)")
        print(f"   Other learning rate: 1e-6 (fine-tuning)")
        print("")

    # ✅ 计算总训练步数并设置 MoE 预热
    total_steps = len(train_loader) * args.num_epochs

    # 获取实际的模型对象（处理 DDP 包装）
    actual_model = model.module if is_distributed else model
    actual_model.set_moe_warmup(total_steps=total_steps, enabled=True)

    if not is_distributed or rank == 0:
        print(f"✅ MoE Warmup enabled: {total_steps} total steps")
        print(f"   Warmup phase: first {int(total_steps * 0.2)} steps (20%)")
        print("")

    # 训练循环
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    print("")

    best_accuracy = 0.0
    best_epoch = 0
    epoch_results = []

    for epoch in range(args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            args.use_culture_loss, args.culture_loss_lambda,
            lambda_entropy=args.lambda_entropy,  # ✅ 新增
            lambda_load=args.lambda_load  # ✅ 新增
        )

        # 评估（使用 forward pass 计算 loss）
        val_metrics = evaluate(
            model, val_loader, args.device,
            args.use_culture_loss, args.culture_loss_lambda,
            moe_args.num_classes
        )

        # 生成式评估（使用真正的生成）
        print(f"\n{'='*80}")
        print(f"📊 Epoch {epoch + 1} Generative Evaluation")
        print(f"{'='*80}")

        gen_metrics = generate_and_evaluate(
            model=model,
            tokenizer=tokenizer,
            val_dataset=val_dataset_raw,  # ✅ 使用原始验证集
            output_dir=args.output_dir,
            num_classes=moe_args.num_classes,
            max_new_tokens=10
        )

        gen_accuracy = gen_metrics.get('accuracy', 0.0)

        # 保存结果
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_gen_loss': train_metrics['gen_loss'],  # ✅ 添加 gen_loss
            'train_entropy_loss': train_metrics['entropy_loss'],  # ✅ 新增
            'train_load_loss': train_metrics['load_loss'],  # ✅ 新增
            'train_accuracy': train_metrics['accuracy'],  # ✅ 训练时为 0.0
            'eval_loss': val_metrics['loss'],
            'eval_gen_loss': val_metrics['gen_loss'],  # ✅ 添加 gen_loss
            'eval_accuracy': gen_accuracy,  # 使用生成式评估的准确率
            'eval_precision': gen_metrics.get('precision', 0.0),  # ✅ 修复默认值
            'eval_recall': gen_metrics.get('recall', 0.0),  # ✅ 修复默认值
            'eval_f1': gen_metrics.get('f1', 0.0)  # ✅ 修复默认值
        }
        epoch_results.append(epoch_result)

        # 打印结果
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {train_metrics['loss']:.4f}, Train Gen Loss: {train_metrics['gen_loss']:.4f}")  # ✅ 修改
        print(f"   Eval Loss:  {val_metrics['loss']:.4f}, Eval Gen Loss: {val_metrics['gen_loss']:.4f}")  # ✅ 修改
        # ✅ 显示正则化损失
        print(f"   Train Entropy Loss: {train_metrics['entropy_loss']:.4f}, Load Loss: {train_metrics['load_loss']:.4f}")
        print(f"   Eval Accuracy (Generative): {gen_accuracy:.4f}")
        if 'precision' in gen_metrics:
            print(f"   Eval Precision: {gen_metrics['precision']:.4f}, Recall: {gen_metrics['recall']:.4f}, F1: {gen_metrics['f1']:.4f}")

        # 保存最佳模型（基于生成式评估的准确率）
        if gen_accuracy > best_accuracy:
            best_accuracy = gen_accuracy
            best_epoch = epoch + 1

            # 保存 MoE 权重（只保存 MoE 部分）
            best_moe_dir = os.path.join(args.output_dir, "best_moe")
            os.makedirs(best_moe_dir, exist_ok=True)

            # 保存 MoE 配置
            moe_config = {
                'num_experts': args.num_experts,
                'shared_hidden_dim': args.shared_hidden_dim,
                'router_hidden_dim': args.router_hidden_dim,
                'experts_hidden_dim': args.experts_hidden_dim,
                'moe_lora_rank': args.moe_lora_rank,
                'num_classes': moe_args.num_classes,
                'classification_hidden_dim': args.classification_hidden_dim,
                'dropout': args.dropout,
                'num_heads': args.num_heads
            }

            with open(os.path.join(best_moe_dir, "moe_config.json"), 'w') as f:
                json.dump(moe_config, f, indent=2)

            # 只保存 MoE 部分的权重（不包括 llama_model）
            moe_state_dict = {}
            for name, param in model.named_parameters():
                if not name.startswith('llama_model.'):
                    moe_state_dict[name] = param.cpu()

            torch.save(moe_state_dict, os.path.join(best_moe_dir, "moe_state_dict.pt"))

            print(f"   🏆 New best model! Accuracy: {best_accuracy:.4f}")
            print(f"   ✅ Best MoE weights saved to: {best_moe_dir} (MoE only, {len(moe_state_dict)} parameters)")

        print(f"   Best so far: Epoch {best_epoch}, Accuracy: {best_accuracy:.4f}")

        # 保存 epoch 结果
        with open(os.path.join(args.output_dir, "epoch_eval_results.json"), 'w') as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)

    # 训练结束后，加载最佳模型进行最终评估
    print("\n" + "="*80)
    print("Final Evaluation on Validation Set")
    print("="*80)
    print("")

    print("Loading best MoE weights for final evaluation...")
    best_moe_dir = os.path.join(args.output_dir, "best_moe")

    # 加载最佳 MoE 权重（只包含 MoE 部分，不包括 llama_model）
    best_state_dict = torch.load(os.path.join(best_moe_dir, "moe_state_dict.pt"))

    # ✅ 使用 strict=False，因为 best_state_dict 只包含 MoE 部分
    # llama_model 部分保持不变（仍然是 Base + LoRA merged）
    model.load_state_dict(best_state_dict, strict=False)
    model.eval()
    print("✅ Best MoE weights loaded (llama_model with LoRA remains unchanged)")

    # 在验证集上进行最终评估
    print("\nEvaluating on validation set...")
    final_val_metrics = evaluate(
        model=model,
        val_loader=val_loader,
        device=args.device,
        num_classes=moe_args.num_classes,
        use_culture_loss=args.use_culture_loss,
        culture_loss_lambda=args.culture_loss_lambda
    )

    print("\n" + "="*80)
    print("Final Evaluation Results")
    print("="*80)
    print(f"Loss:       {final_val_metrics['loss']:.4f}")
    print(f"Accuracy:   {final_val_metrics['accuracy']:.4f}")
    print(f"Precision:  {final_val_metrics['precision']:.4f}")
    print(f"Recall:     {final_val_metrics['recall']:.4f}")
    print(f"F1:         {final_val_metrics['f1']:.4f}")
    print("="*80)

    # 保存最终评估结果
    final_eval_results = {
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "final_eval_loss": final_val_metrics['loss'],
        "final_eval_accuracy": final_val_metrics['accuracy'],
        "final_eval_precision": final_val_metrics['precision'],
        "final_eval_recall": final_val_metrics['recall'],
        "final_eval_f1": final_val_metrics['f1'],
        "num_val_samples": len(val_dataset),
        "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(args.output_dir, "final_eval_results.json"), 'w') as f:
        json.dump(final_eval_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Final evaluation results saved to: {os.path.join(args.output_dir, 'final_eval_results.json')}")

    # 保存最终配置
    final_config = {
        'base_model': args.base_model_path,
        'lora_weights': args.lora_weights_path,
        'num_classes': moe_args.num_classes,
        'use_culture_loss': args.use_culture_loss,
        'culture_loss_lambda': args.culture_loss_lambda,
        'num_epochs': args.num_epochs,
        'best_epoch': best_epoch,
        'best_accuracy': best_accuracy,
        'final_eval_accuracy': final_val_metrics['accuracy'],
        'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(final_config, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ Training Completed!")
    print("="*80)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Final Eval Accuracy: {final_val_metrics['accuracy']:.4f}")
    print(f"\n📁 Output Files:")
    print(f"   Best MoE weights: {best_moe_dir}")
    print(f"   Epoch eval results: {os.path.join(args.output_dir, 'epoch_eval_results.json')}")
    print(f"   Final eval results: {os.path.join(args.output_dir, 'final_eval_results.json')}")
    print(f"   Training config: {os.path.join(args.output_dir, 'config.json')}")
    print("="*80)
    print("")


if __name__ == "__main__":
    main()

