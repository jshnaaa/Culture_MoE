#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单 MoE 模型训练脚本
在 LoRA 微调后的完整模型基础上添加简单的 MoE 结构

使用方法：
    python ft_simple_moe_gen.py \
        --base_model_path /path/to/base_model \
        --lora_weights_path /path/to/lora_weights \
        --train_file /path/to/train_data.json \
        --output_dir /path/to/output \
        --num_experts 12 \
        --top_k 2
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.simple_moe import SimpleMoEModel


class SimpleMoEDataset(Dataset):
    """简单的数据集类"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建输入文本
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')

        # 构建完整文本
        if input_text:
            full_text = f"{instruction}\n{input_text}\n{output}"
        else:
            full_text = f"{instruction}\n{output}"

        # 分词
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),  # 语言建模任务，labels = input_ids
            'text': full_text
        }


def collate_fn(batch):
    """批处理函数"""
    max_len = max([item['input_ids'].size(0) for item in batch])

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        input_id = item['input_ids']
        attn_mask = item['attention_mask']
        label = item['labels']

        # 右侧填充
        pad_len = max_len - input_id.size(0)
        if pad_len > 0:
            input_id = torch.cat([input_id, torch.full((pad_len,), 0)])  # pad_token_id = 0
            attn_mask = torch.cat([attn_mask, torch.zeros(pad_len)])
            label = torch.cat([label, torch.full((pad_len,), -100)])  # ignore_index = -100

        input_ids.append(input_id)
        attention_mask.append(attn_mask)
        labels.append(label)

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }


class SimpleMoETrainer:
    """简单 MoE 训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 设置随机种子
        set_seed(42)

        # 设置日志
        self.setup_logging()

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 保存配置
        self.save_config()

    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.args.output_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info("=" * 80)
        logging.info("Simple MoE Training Started")
        logging.info("=" * 80)
        logging.info(f"Arguments: {vars(self.args)}")

    def save_config(self):
        """保存训练配置"""
        config = {
            'training_time': datetime.now().isoformat(),
            'base_model_path': self.args.base_model_path,
            'lora_weights_path': self.args.lora_weights_path,
            'train_file': self.args.train_file,
            'num_experts': self.args.num_experts,
            'top_k': self.args.top_k,
            'expert_hidden_dim': self.args.expert_hidden_dim,
            'router_hidden_dim': self.args.router_hidden_dim,
            'learning_rate': self.args.learning_rate,
            'num_epochs': self.args.num_epochs,
            'batch_size': self.args.batch_size,
            'max_length': self.args.max_length,
            'device': str(self.device)
        }

        config_file = os.path.join(self.args.output_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logging.info(f"Configuration saved to {config_file}")

    def load_model_from_components(self):
        """从 Base 模型和 LoRA 权重加载完整模型"""
        logging.info("=" * 80)
        logging.info("Loading Model from Components")
        logging.info("=" * 80)
        logging.info(f"Base model: {self.args.base_model_path}")
        logging.info(f"LoRA weights: {self.args.lora_weights_path}")

        # 1. 加载 Tokenizer
        logging.info("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.args.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logging.info("   ✅ Tokenizer loaded")

        # 2. 加载 Base 模型
        logging.info("2. Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None  # 手动管理设备
        )
        logging.info("   ✅ Base model loaded")

        # 3. 加载 LoRA 权重并合并
        logging.info("3. Loading and merging LoRA weights...")
        model_with_lora = PeftModel.from_pretrained(
            base_model,
            self.args.lora_weights_path,
            is_trainable=False
        )
        logging.info("   ✅ LoRA weights loaded")

        # 合并 LoRA 权重到 base 模型
        logging.info("   Merging LoRA weights into base model...")
        merged_model = model_with_lora.merge_and_unload()
        logging.info("   ✅ LoRA weights merged")

        # 4. 创建简单 MoE 模型
        logging.info("4. Creating Simple MoE model...")
        simple_moe_model = SimpleMoEModel(
            llama_model=merged_model,
            config=merged_model.config,
            num_experts=self.args.num_experts,
            top_k=self.args.top_k,
            expert_hidden_dim=self.args.expert_hidden_dim,
            router_hidden_dim=self.args.router_hidden_dim,
            dropout=self.args.dropout
        )
        logging.info("   ✅ Simple MoE model created")

        # 5. 冻结 LLaMA 模型，只训练 MoE 组件
        logging.info("5. Freezing LLaMA model parameters...")
        for name, param in simple_moe_model.named_parameters():
            if 'moe_layer' in name or 'moe_fusion_weight' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in simple_moe_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in simple_moe_model.parameters())

        logging.info(f"   Total parameters: {total_params:,}")
        logging.info(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

        # 6. 移动到设备
        logging.info(f"6. Moving model to {self.device}...")
        simple_moe_model = simple_moe_model.to(self.device)
        simple_moe_model.train()
        logging.info("   ✅ Model ready for training")

        logging.info("=" * 80)
        logging.info("✅ Model Loading Completed Successfully!")
        logging.info("=" * 80)

        return simple_moe_model, tokenizer

    def create_dataloader(self, tokenizer):
        """创建数据加载器"""
        logging.info(f"Creating dataloader from: {self.args.train_file}")

        dataset = SimpleMoEDataset(
            data_path=self.args.train_file,
            tokenizer=tokenizer,
            max_length=self.args.max_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers
        )

        logging.info(f"Dataloader created with {len(dataset)} samples, {len(dataloader)} batches")
        return dataloader

    def train(self, model, dataloader, tokenizer):
        """训练模型"""
        logging.info("Starting training...")

        # 优化器
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        # 训练循环
        model.train()
        global_step = 0
        total_loss = 0
        epoch_results = []

        for epoch in range(self.args.num_epochs):
            epoch_loss = 0
            epoch_steps = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

            for batch in progress_bar:
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )

                loss = outputs['loss']

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录损失
                epoch_loss += loss.item()
                total_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/epoch_steps:.4f}'
                })

                # 定期清理内存
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()

            # 计算 epoch 平均损失
            avg_epoch_loss = epoch_loss / epoch_steps

            # 获取专家使用统计
            expert_stats = model.get_expert_usage_stats(outputs['expert_weights'])

            # 记录 epoch 结果
            epoch_result = {
                'epoch': epoch + 1,
                'avg_loss': avg_epoch_loss,
                'expert_stats': expert_stats,
                'timestamp': datetime.now().isoformat()
            }
            epoch_results.append(epoch_result)

            logging.info(f"Epoch {epoch+1}/{self.args.num_epochs} completed:")
            logging.info(f"  Average loss: {avg_epoch_loss:.4f}")
            logging.info(f"  Expert usage std: {expert_stats['usage_std']:.4f}")
            logging.info(f"  Load balance loss: {expert_stats['load_balance_loss']:.4f}")

            # 保存检查点
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(model, tokenizer, epoch + 1, avg_epoch_loss)

        # 保存最终模型
        self.save_final_model(model, tokenizer)

        # 保存训练结果
        results_file = os.path.join(self.args.output_dir, 'training_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Training completed. Results saved to {results_file}")
        return epoch_results

    def save_checkpoint(self, model, tokenizer, epoch, loss):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 只保存 MoE 相关的参数
        moe_state_dict = {}
        for name, param in model.named_parameters():
            if 'moe_layer' in name or 'moe_fusion_weight' in name:
                moe_state_dict[name] = param.cpu().clone()

        torch.save(moe_state_dict, os.path.join(checkpoint_dir, 'moe_weights.pth'))

        # 保存检查点信息
        checkpoint_info = {
            'epoch': epoch,
            'loss': loss,
            'num_experts': self.args.num_experts,
            'top_k': self.args.top_k,
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(checkpoint_dir, 'checkpoint_info.json'), 'w') as f:
            json.dump(checkpoint_info, f, indent=2)

        logging.info(f"Checkpoint saved to {checkpoint_dir}")

    def save_final_model(self, model, tokenizer):
        """保存最终模型"""
        final_dir = os.path.join(self.args.output_dir, 'final_model')
        os.makedirs(final_dir, exist_ok=True)

        # 保存 MoE 权重
        moe_state_dict = {}
        for name, param in model.named_parameters():
            if 'moe_layer' in name or 'moe_fusion_weight' in name:
                moe_state_dict[name] = param.cpu().clone()

        torch.save(moe_state_dict, os.path.join(final_dir, 'moe_weights.pth'))

        # 保存模型配置
        model_config = {
            'num_experts': self.args.num_experts,
            'top_k': self.args.top_k,
            'expert_hidden_dim': self.args.expert_hidden_dim,
            'router_hidden_dim': self.args.router_hidden_dim,
            'dropout': self.args.dropout,
            'base_model_path': self.args.base_model_path,
            'lora_weights_path': self.args.lora_weights_path,
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(final_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)

        logging.info(f"Final model saved to {final_dir}")

    def run_training(self):
        """运行完整的训练流程"""
        start_time = time.time()

        try:
            # 1. 加载模型
            model, tokenizer = self.load_model_from_components()

            # 2. 创建数据加载器
            dataloader = self.create_dataloader(tokenizer)

            # 3. 训练模型
            results = self.train(model, dataloader, tokenizer)

            end_time = time.time()
            training_time = end_time - start_time

            logging.info("=" * 80)
            logging.info("✅ Simple MoE Training Completed Successfully!")
            logging.info("=" * 80)
            logging.info(f"Total training time: {training_time:.2f} seconds")
            logging.info(f"Final average loss: {results[-1]['avg_loss']:.4f}")
            logging.info("=" * 80)

            return True

        except Exception as e:
            logging.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Simple MoE Training")

    # 必需参数
    parser.add_argument('--base_model_path', type=str, required=True, help='基础模型路径')
    parser.add_argument('--lora_weights_path', type=str, required=True, help='LoRA权重路径')
    parser.add_argument('--train_file', type=str, required=True, help='训练数据文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')

    # MoE 参数
    parser.add_argument('--num_experts', type=int, default=12, help='专家数量')
    parser.add_argument('--top_k', type=int, default=2, help='Top-K 路由')
    parser.add_argument('--expert_hidden_dim', type=int, default=None, help='专家隐藏维度')
    parser.add_argument('--router_hidden_dim', type=int, default=512, help='路由器隐藏维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')

    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载器工作进程数')
    parser.add_argument('--save_interval', type=int, default=2, help='保存间隔（轮数）')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    args = parser.parse_args()

    # 创建训练器并运行
    trainer = SimpleMoETrainer(args)
    success = trainer.run_training()

    if success:
        print("\n🎉 Training completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()