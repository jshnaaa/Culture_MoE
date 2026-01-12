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
from functools import partial

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MergedDataset(Dataset):
    """
    合并多个数据集的数据集类

    支持将多个JSON文件的数据合并，并标记来源数据集
    """

    def __init__(self, data_paths: list, dataset_names: list, tokenizer, max_length: int = 512, use_mask_inference: bool = True):
        """
        Args:
            data_paths: 数据文件路径列表
            dataset_names: 数据集名称列表（与data_paths对应）
            tokenizer: Tokenizer
            max_length: 最大序列长度
            use_mask_inference: 推理时是否启用MASK机制（双路输入处理）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_mask_inference = use_mask_inference
        self.dataset_names = dataset_names

        self.data = []
        self.dataset_indices = []  # 记录每个样本来自哪个数据集

        print(f"Loading merged dataset from {len(data_paths)} sources:")
        for i, (data_path, dataset_name) in enumerate(zip(data_paths, dataset_names)):
            print(f"  Loading {dataset_name} from: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)

            # 为每个样本添加数据集来源标识
            for item in dataset_data:
                item['source_dataset'] = dataset_name
                item['source_index'] = i

            self.data.extend(dataset_data)
            self.dataset_indices.extend([i] * len(dataset_data))
            print(f"    Loaded {len(dataset_data)} samples from {dataset_name}")

        print(f"Total merged dataset size: {len(self.data)} samples")

        # 统计各数据集的样本数量
        dataset_counts = {}
        for i, name in enumerate(dataset_names):
            count = self.dataset_indices.count(i)
            dataset_counts[name] = count
            print(f"  {name}: {count} samples ({count/len(self.data)*100:.1f}%)")

        # 🔧 调试：检查原始数据中是否包含country字段
        if self.data and len(self.data) > 0:
            first_item = self.data[0]
            has_country = 'country' in first_item
            print(f"🔍 合并数据country字段检查: {has_country}")
            if has_country:
                print(f"    第一个样本的country: {first_item.get('country', 'None')}")
                # 统计不同country的数量
                countries = set()
                for item in self.data[:100]:  # 检查前100个样本
                    if 'country' in item and item['country']:
                        countries.add(item['country'])
                print(f"    前100个样本中的country类型: {sorted(list(countries))}")
            else:
                print(f"    数据字段: {list(first_item.keys())}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 获取基本信息
        instruction = item['instruction']
        instruction_mask = item.get('instruction_mask', instruction)  # 如果没有mask版本，使用原版
        input_text = item['input']
        output_text = item['output']
        label = item['label']

        # 🔧 新增：获取country字段（用于blend数据集分组统计）
        country = item.get('country', None)

        # 🔧 新增：获取数据集来源信息
        source_dataset = item.get('source_dataset', 'unknown')
        source_index = item.get('source_index', -1)

        # 构建完整输入 - 与联合训练保持一致的格式
        full_input = f"{instruction}\n{input_text}"
        full_input_mask = f"{instruction_mask}\n{input_text}"

        # 编码输入（原始版本）
        encoded_input = self.tokenizer(
            full_input,
            truncation=True,
            max_length=self.max_length - 10,  # 为输出预留空间
            padding=False,
            return_tensors=None
        )

        # 编码输入（MASK版本）
        encoded_input_mask = self.tokenizer(
            full_input_mask,
            truncation=True,
            max_length=self.max_length - 10,  # 为输出预留空间
            padding=False,
            return_tensors=None
        )

        # 编码输出
        encoded_output = self.tokenizer(
            output_text,
            truncation=True,
            max_length=10,  # 输出通常很短
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )

        # 构建完整序列：input + " " + output
        input_ids = encoded_input['input_ids'] + [self.tokenizer.convert_tokens_to_ids(" ")] + encoded_output['input_ids']
        input_ids_mask = encoded_input_mask['input_ids'] + [self.tokenizer.convert_tokens_to_ids(" ")] + encoded_output['input_ids']

        # 创建attention mask
        attention_mask = [1] * len(input_ids)
        attention_mask_mask = [1] * len(input_ids_mask)

        # 创建labels：input部分为-100，output部分为token ids
        labels = [-100] * len(encoded_input['input_ids']) + [-100] + encoded_output['input_ids']  # 空格也设为-100

        # 转换为tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        input_ids_mask = torch.tensor(input_ids_mask, dtype=torch.long)
        attention_mask_mask = torch.tensor(attention_mask_mask, dtype=torch.long)

        # 计算有效标签数量（用于损失计算）
        valid_labels = (labels != -100).sum().item()

        # 计算统计信息
        total_tokens = len(input_ids)
        input_length = len(encoded_input['input_ids'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'input_ids_mask': input_ids_mask,
            'attention_mask_mask': attention_mask_mask,
            'instruction': instruction,
            'instruction_mask': instruction_mask,
            'input': input_text,
            'output': output_text,
            'label': label,
            'country': country,  # 🔧 新增：country字段
            'source_dataset': source_dataset,  # 🔧 新增：数据集来源
            'source_index': source_index,  # 🔧 新增：数据集索引
            'valid_labels': valid_labels,
            'total_tokens': total_tokens,
            'input_length': input_length
        }


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

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, use_mask_inference: bool = True):
        """
        Args:
            data_path: 数据文件路径
            tokenizer: Tokenizer
            max_length: 最大序列长度
            use_mask_inference: 推理时是否启用MASK机制（双路输入处理）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_mask_inference = use_mask_inference

        print(f"Loading data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples")

        # 🔧 调试：检查原始数据中是否包含country字段
        if self.data and len(self.data) > 0:
            first_item = self.data[0]
            has_country = 'country' in first_item
            print(f"🔍 原始数据country字段检查: {has_country}")
            if has_country:
                print(f"    第一个样本的country: {first_item.get('country', 'None')}")
                # 统计不同country的数量
                countries = set()
                for item in self.data[:100]:  # 检查前100个样本
                    if 'country' in item and item['country']:
                        countries.add(item['country'])
                print(f"    前100个样本中的country类型: {sorted(list(countries))}")
            else:
                print(f"    数据字段: {list(first_item.keys())}")

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]

        # 新格式：instruction + input + output
        instruction = item.get('instruction', '')
        instruction_mask = item.get('instruction_mask', instruction)  # 如果没有instruction_mask，使用原instruction
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        label = item.get('label', '')

        # 🔧 推理时MASK机制控制：根据use_mask_inference决定是否启用双路输入处理
        if not self.use_mask_inference:
            # 禁用MASK机制：instruction_mask使用原始instruction，实现单路输入处理
            instruction_mask = instruction

        # 构建完整的输入和输出
        # 格式：instruction + input → output
        if input_text:
            full_input = f"{instruction}\n{input_text}"
            full_input_mask = f"{instruction_mask}\n{input_text}"  # MASK版本的输入
        else:
            full_input = instruction
            full_input_mask = instruction_mask

        # 完整的文本（用于语言建模）
        # 🔧 修复：添加明确分隔符，确保input和output部分可以准确区分
        full_text = f"{full_input.rstrip()}\n{output_text}"
        full_text_mask = f"{full_input_mask.rstrip()}\n{output_text}"  # MASK版本的完整文本

        # 🔍 关键调试：检查构建后的full_text（注释掉详细调试）
        # if idx < 5:
        #     print(f"  构建的full_text末尾: {repr(full_text[-50:])}")
        #     print(f"  full_input末尾: {repr(full_input[-30:])}")
        #     print(f"  期望的输出部分: {repr(output_text)}")

        # 🔧 关键修复：动态padding - 不在这里padding，在collate_fn中处理
        # 1. 先tokenize完整文本（不padding，在batch级别动态处理）
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # 改为False，使用动态padding
            return_tensors='pt',
            add_special_tokens=True  # 明确指定
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # 同时tokenize mask版本的文本
        encoded_mask = self.tokenizer(
            full_text_mask,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt',
            add_special_tokens=True
        )

        input_ids_mask = encoded_mask['input_ids'].squeeze(0)
        attention_mask_mask = encoded_mask['attention_mask'].squeeze(0)

        # 🔍 关键调试：检查tokenization结果（注释掉详细调试）
        # if idx < 5:
        #     print(f"  🔍 Tokenization结果:")
        #     print(f"    input_ids长度: {len(input_ids)}")

        #     # 解码完整序列看看实际内容
        #     full_decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        #     print(f"    完整解码内容: {repr(full_decoded)}")

        #     # 检查最后10个token（应该包含output部分）
        #     last_tokens = input_ids[-15:].tolist()
        #     print(f"    最后15个token_ids: {last_tokens}")
        #     for i, token_id in enumerate(last_tokens):
        #         if token_id != self.tokenizer.pad_token_id:  # 跳过padding token
        #             try:
        #                 token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        #                 print(f"      token_{len(input_ids)-15+i}: {token_id}='{token_text}'")
        #             except:
        #                 print(f"      token_{len(input_ids)-15+i}: {token_id}=(解码失败)")

        # 2. 🔧 关键修复：精确计算input_length，只让模型学习答案部分
        # 我们需要找到"### Answer:"之后空格的位置，让模型从那里开始学习

        # 🔧 修复：精确计算input_length - 确保只学习output部分
        # 实际数据格式：full_input + output_text (不添加空格)
        # 我们需要计算full_input的token数量

        # 🔧 修复：基于明确分隔符的精确input_length计算
        # 现在有了明确的'\n'分隔符，可以准确计算input部分长度
        full_input_with_sep = f"{full_input.rstrip()}\n"
        input_tokens = self.tokenizer(full_input_with_sep, add_special_tokens=True, truncation=False)['input_ids']
        input_length = len(input_tokens)

        # 安全检查：确保input_length不超过总长度
        total_length = len(input_ids)
        if input_length >= total_length:
            input_length = max(10, total_length - 5)  # 至少给output留5个token空间，确保有合理的输入长度

        # 🔍 调试：检查input_length计算
        # 注释掉详细调试信息
        # if idx < 5:
        #     print(f"  🔧 Input length计算:")
        #     print(f"    full_input_with_sep: {repr(full_input_with_sep[-50:])}")
        #     print(f"    计算出的input_length: {input_length}")
        #     print(f"    应该掩码的部分: 0 到 {input_length-1}")
        #     print(f"    应该学习的部分: {input_length} 开始")

        # 🔧 Llama特殊token处理：正确识别padding token
        # 获取正确的pad_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            # 如果没有设置pad_token，使用默认的eos_token
            pad_token_id = self.tokenizer.eos_token_id

        # 🔧 新的精确计算方法基于明确分隔符，不需要额外调整
        # 验证：确保input_length合理
        total_non_pad = (input_ids != pad_token_id).sum().item()

        # 注释掉验证调试信息，避免训练时输出过多日志
        # if idx < 5:
        #     print(f"  🔧 验证: 总非padding长度={total_non_pad}, input_length={input_length}")
        #     if input_length >= total_non_pad:
        #         print(f"  ⚠️ 警告: input_length >= 总长度，这会导致没有训练目标")

        # 🔧 删除eot_id调整逻辑，因为我们的精确计算已经处理了这个问题
        # 新的方法直接定位到"### Answer: "后面，不需要额外调整

        # 3. 创建正确的标签
        labels = input_ids.clone()

        # 4. 掩码输入部分（设为-100，不计算损失）
        labels[:input_length] = -100

        # 5. 🔧 关键修复：掩码所有padding token，不管它们在哪个位置
        # 确保padding token不被当作训练目标
        padding_mask = (input_ids == pad_token_id)
        labels[padding_mask] = -100

        # 6. 计算有效的训练token数量
        valid_labels = (labels != -100).sum().item()
        total_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()

        # 7. 🔧 最终验证：确保padding token被正确掩码
        remaining_pad_tokens = (labels == pad_token_id).sum().item()
        if remaining_pad_tokens > 0:
            # 只在前几个样本或严重问题时打印警告
            if idx < 3 or remaining_pad_tokens > 5:
                print(f"⚠️ 警告: 样本{idx}中仍有{remaining_pad_tokens}个padding token未被掩码!")
            # 强制掩码剩余的padding token
            labels[labels == pad_token_id] = -100
            valid_labels = (labels != -100).sum().item()
            if idx < 3 or remaining_pad_tokens > 5:
                print(f"   强制掩码后有效标签数: {valid_labels}")

        # 🔍 详细的labels调试信息（注释掉详细调试）
        # if idx < 5:
        #     print(f"\n📋 样本 {idx} - 有效标签数: {valid_labels}")
        #     print(f"  🔧 掩码分析:")
        #     print(f"    原始文本: '{full_text[:100]}...'")
        #     print(f"    输入部分: '{full_input[:100]}...'")
        #     print(f"    输出部分: '{output_text}'")
        #     print(f"    精确计算的input_length: {input_length}")
        #     print(f"    总序列长度: {len(input_ids)}")
        #     print(f"    实际非padding长度: {(input_ids != pad_token_id).sum().item()}")

        #     # 检查掩码的具体位置
        #     mask_positions = (labels == -100).nonzero(as_tuple=True)[0]
        #     valid_positions = (labels != -100).nonzero(as_tuple=True)[0]
        #     print(f"    掩码位置数: {len(mask_positions)}")
        #     print(f"    有效位置数: {len(valid_positions)}")

        #     if len(valid_positions) > 0:
        #         print(f"    有效位置范围: {valid_positions[0].item()} - {valid_positions[-1].item()}")
        #         # 显示前几个有效token
        #         for i, pos in enumerate(valid_positions[:3]):
        #             token_id = input_ids[pos].item()
        #             try:
        #                 token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        #                 print(f"      位置{pos.item()}: {token_id}='{token_text}'")
        #             except:
        #                 print(f"      位置{pos.item()}: {token_id}=(解码失败)")

        #     # 检查是否input_length计算有问题
        #     if input_length >= len(input_ids) - 5:  # 如果输入长度几乎占满整个序列
        #         print(f"  🚨 问题: input_length({input_length})过大，几乎占满整个序列!")
        #         print(f"    这会导致几乎没有训练目标")

        # 🔧 只在出现真正异常情况时打印调试信息
        # 检查是否存在潜在问题：有效标签数过少或过多
        if valid_labels == 0:
            print(f"⚠️ 严重警告: 样本 {idx} 没有有效训练标签!")
            print(f"  原始output: '{output_text}'")
            print(f"  output_tokens: {output_tokens}")
            print(f"  input_length: {input_length}, 总长度: {len(input_ids)}")
        elif valid_labels > 50:  # 🔧 提高阈值，只在真正异常时警告
            print(f"⚠️ 严重警告: 样本 {idx} 有效标签数异常多 ({valid_labels})")
            print(f"  可能存在input_length计算错误")
            print(f"  原始output: '{output_text}'")
            print(f"  input_length: {input_length}, 总长度: {len(input_ids)}")
            # 显示实际的有效标签内容帮助调试
            valid_positions = (labels != -100).nonzero(as_tuple=True)[0]
            print(f"  前10个有效标签token:")
            for i, pos in enumerate(valid_positions[:10]):
                pos_idx = pos.item()
                token_id = labels[pos_idx].item()
                try:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    print(f"    位置{pos_idx}: {token_id}='{token_text}'")
                except:
                    print(f"    位置{pos_idx}: {token_id}=(解码失败)")
        # 🔧 移除误导性的input_length检查：对于选择题任务，答案通常只有1个数字token，
        # 所以input_length接近总长度是完全正常的，不需要警告

        # 注释掉其他调试信息
        # if idx < 5:
        #     print(f"    pad_token: {repr(self.tokenizer.pad_token)}")
        #     print(f"    pad_token_id: {pad_token_id}")
        #     print(f"    eos_token_id: {self.tokenizer.eos_token_id}")

        #     # 分析labels中的token分布
        #     eot_in_labels = (labels == eot_token_id).sum().item()
        #     pad_in_labels = (labels == pad_token_id).sum().item()

        #     # 统计所有非-100的token
        #     non_mask_indices = (labels != -100).nonzero(as_tuple=True)[0]

        #     print(f"  📊 Labels分析:")
        #     print(f"    总序列长度: {len(labels)}")
        #     print(f"    有效标签数: {valid_labels}")
        #     print(f"    <|eot_id|>(128009)数量: {eot_in_labels}")
        #     print(f"    padding token({pad_token_id})数量: {pad_in_labels}")

        #     if valid_labels > 10:
        #         print(f"  ⚠️ 标签数过多({valid_labels})，可能仍有padding问题")
        #         print(f"  🚨 CRITICAL: tokenizer修复失效！")
        #         print(f"  当前pad_token_id: {self.tokenizer.pad_token_id}")
        #         print(f"  当前pad_token: {repr(self.tokenizer.pad_token)}")

        #         # 强制检查实际使用的pad_token_id
        #         if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
        #             actual_pad_text = self.tokenizer.decode([self.tokenizer.pad_token_id], skip_special_tokens=True)
        #             print(f"  实际pad_token解码: '{actual_pad_text}'")

        #         # 检查input_ids中实际的token分布
        #         unique_tokens, counts = torch.unique(input_ids, return_counts=True)
        #         print(f"  🔍 input_ids中的token分布 (前10个):")
        #         for i in range(min(10, len(unique_tokens))):
        #             token_id = unique_tokens[i].item()
        #             count = counts[i].item()
        #             try:
        #                 token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        #                 print(f"    token_id={token_id}('{token_text}'): {count}次")
        #             except:
        #                 print(f"    token_id={token_id}(解码失败): {count}次")

        #         # 显示labels的分布情况
        #         unique_tokens = {}
        #         for pos in non_mask_indices[:50]:  # 检查前50个有效标签
        #             token_id = labels[pos.item()].item()
        #             unique_tokens[token_id] = unique_tokens.get(token_id, 0) + 1

        #         print(f"  🔍 前50个有效标签的token分布:")
        #         for token_id, count in sorted(unique_tokens.items(), key=lambda x: x[1], reverse=True)[:10]:
        #             try:
        #                 token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        #                 print(f"    token_id={token_id}('{token_text}'): {count}次")
        #             except:
        #                 print(f"    token_id={token_id}(解码失败): {count}次")

        #         # 显示labels的具体位置和值
        #         print(f"  📋 前20个有效标签详情:")
        #         for i, pos in enumerate(non_mask_indices[:20]):
        #             pos_idx = pos.item()
        #             token_id = labels[pos_idx].item()
        #             try:
        #                 token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        #                 print(f"    位置{pos_idx}: {token_id}='{token_text}'")
        #             except:
        #                 print(f"    位置{pos_idx}: {token_id}=(解码失败)")

        #     elif eot_in_labels > 1:
        #         print(f"  ⚠️ 训练标签包含{eot_in_labels}个<|eot_id|>")
        #         # 显示所有有效标签
        #         print(f"  📋 所有有效标签:")
        #         for i, pos in enumerate(non_mask_indices):
        #             pos_idx = pos.item()
        #             token_id = labels[pos_idx].item()
        #             try:
        #                 token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        #                 print(f"    位置{pos_idx}: {token_id}='{token_text}'")
        #             except:
        #                 print(f"    位置{pos_idx}: {token_id}=(解码失败)")

        #     elif valid_labels <= 5:
        #         print(f"  ✅ 标签数正常({valid_labels})")
        #         # 显示所有有效标签
        #         print(f"  📋 所有有效标签:")
        #         for i, pos in enumerate(non_mask_indices):
        #             pos_idx = pos.item()
        #             token_id = labels[pos_idx].item()
        #             try:
        #                 token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        #                 print(f"    位置{pos_idx}: {token_id}='{token_text}'")
        #             except:
        #                 print(f"    位置{pos_idx}: {token_id}=(解码失败)")

        #     # 🔧 简化的问题检查
        #     if pad_token_id == eot_token_id:
        #         print(f"  🚨 发现问题: pad_token_id == <|eot_id|> ({pad_token_id})")
        #         print(f"    这会导致padding区域填充<|eot_id|>，造成大量有效标签!")
        #     elif pad_token_id is None:
        #         print(f"  🚨 发现问题: pad_token_id is None!")
        #         print(f"    tokenizer配置可能没有正确应用")
        #     elif pad_in_labels > 0:
        #         print(f"  🚨 发现问题: {pad_in_labels}个padding token({pad_token_id})仍在训练标签中!")
        #         print(f"    padding token应该被掩码为-100，不应该出现在有效标签中")
        #     elif valid_labels > 10:
        #         print(f"  🚨 发现问题: 有效标签数过多({valid_labels})，可能有其他token被错误包含")

        # 🔧 新增：获取country字段（用于blend数据集分组统计）
        country = item.get('country', None)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'input_ids_mask': input_ids_mask,
            'attention_mask_mask': attention_mask_mask,
            'instruction': instruction,
            'instruction_mask': instruction_mask,
            'input': input_text,
            'output': output_text,
            'label': label,
            'country': country,  # 🔧 新增：country字段
            'valid_labels': valid_labels,
            'total_tokens': total_tokens,
            'input_length': input_length  # 添加调试信息
        }


def dynamic_padding_collate_fn(batch, tokenizer, max_seq_length=384):
    """
    动态padding collate函数：根据batch内最长样本动态设置padding长度

    Args:
        batch: 数据集返回的样本列表
        tokenizer: tokenizer对象
        max_seq_length: 最大序列长度限制

    Returns:
        批次数据字典
    """
    # 找到batch内最长的序列长度，但限制在指定长度以内
    max_length = min(max(max(len(item['input_ids']), len(item.get('input_ids_mask', []))) for item in batch), max_seq_length)

    # 🔧 额外安全检查：如果仍然过长，强制截断到更安全的长度
    if max_length > max_seq_length:
        max_length = max_seq_length
        # 只在第一次截断时打印警告，避免重复输出
        if not hasattr(dynamic_padding_collate_fn, '_truncation_warned'):
            print(f"⚠️ 强制截断序列长度到 {max_seq_length}")
            dynamic_padding_collate_fn._truncation_warned = True

    # 为每个样本进行padding
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_input_ids_mask = []
    batch_attention_mask_mask = []
    batch_instructions = []
    batch_instruction_masks = []
    batch_inputs = []
    batch_outputs = []
    batch_labels_culture = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        labels = item['labels']

        # 处理mask版本的数据
        input_ids_mask = item.get('input_ids_mask', input_ids)  # 如果没有mask版本，使用原版本
        attention_mask_mask = item.get('attention_mask_mask', attention_mask)

        # 🔧 如果样本超过max_seq_length，进行截断
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            labels = labels[:max_seq_length]

        if len(input_ids_mask) > max_seq_length:
            input_ids_mask = input_ids_mask[:max_seq_length]
            attention_mask_mask = attention_mask_mask[:max_seq_length]

        # 计算需要padding的长度
        pad_length = max_length - len(input_ids)
        pad_length_mask = max_length - len(input_ids_mask)

        if pad_length > 0:
            # 右侧padding
            padded_input_ids = F.pad(input_ids, (0, pad_length), value=tokenizer.pad_token_id)
            padded_attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            padded_labels = F.pad(labels, (0, pad_length), value=-100)
        else:
            padded_input_ids = input_ids
            padded_attention_mask = attention_mask
            padded_labels = labels

        if pad_length_mask > 0:
            # 右侧padding for mask version
            padded_input_ids_mask = F.pad(input_ids_mask, (0, pad_length_mask), value=tokenizer.pad_token_id)
            padded_attention_mask_mask = F.pad(attention_mask_mask, (0, pad_length_mask), value=0)
        else:
            padded_input_ids_mask = input_ids_mask
            padded_attention_mask_mask = attention_mask_mask

        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)
        batch_labels.append(padded_labels)
        batch_input_ids_mask.append(padded_input_ids_mask)
        batch_attention_mask_mask.append(padded_attention_mask_mask)
        batch_instructions.append(item['instruction'])
        batch_instruction_masks.append(item.get('instruction_mask', item['instruction']))
        batch_inputs.append(item['input'])
        batch_outputs.append(item['output'])
        batch_labels_culture.append(item['label'])

    # 堆叠成batch张量
    # 🔧 新增：处理country字段
    batch_countries = [item.get('country', None) for item in batch]

    return {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'labels': torch.stack(batch_labels),
        'input_ids_mask': torch.stack(batch_input_ids_mask),
        'attention_mask_mask': torch.stack(batch_attention_mask_mask),
        'instruction': batch_instructions,
        'instruction_mask': batch_instruction_masks,
        'input': batch_inputs,
        'output': batch_outputs,
        'label': batch_labels_culture,
        'country': batch_countries  # 🔧 新增：country字段
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


def load_and_process_merged_data_8_1_1(
    data_paths: list,
    dataset_names: list,
    tokenizer,
    max_length: int = 512,
    output_dir: str = None,
    seed: int = 42
):
    """
    加载并处理合并数据集，分别对每个数据集按 8:1:1 比例划分，然后合并

    Args:
        data_paths: 数据文件路径列表
        dataset_names: 数据集名称列表
        tokenizer: Tokenizer
        max_length: 最大序列长度
        output_dir: 输出目录，用于保存划分信息
        seed: 随机种子

    Returns:
        dict: 包含 'train', 'validation', 'test' 的字典，每个都包含合并后的数据集
    """
    import pickle
    from sklearn.model_selection import train_test_split

    # 设置随机种子
    torch.manual_seed(seed)

    print(f"Loading and splitting {len(data_paths)} datasets separately...")

    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []
    split_info_all = {}

    for data_path, dataset_name in zip(data_paths, dataset_names):
        print(f"\n🔧 Processing {dataset_name} from {data_path}")

        # 为每个数据集单独创建数据集对象
        dataset = CultureLLMNewFormatDataset(data_path, tokenizer, max_length)

        # 获取所有数据的索引和文化标签，用于分层采样
        indices = list(range(len(dataset)))
        culture_labels = []

        # 提取文化标签用于分层采样
        for i in range(len(dataset)):
            sample = dataset.data[i]
            culture_labels.append(sample.get('label', '0'))

        # 第一次划分：80% 训练，20% 临时（验证+测试）
        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            indices, culture_labels,
            test_size=0.2,
            random_state=seed,
            stratify=culture_labels
        )

        # 第二次划分：从20%中分出10%验证，10%测试
        val_indices, test_indices, _, _ = train_test_split(
            temp_indices, temp_labels,
            test_size=0.5,  # 0.5 * 0.2 = 0.1 (10%)
            random_state=seed,
            stratify=temp_labels
        )

        # 创建子数据集
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        print(f"  {dataset_name} - Train: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
        print(f"  {dataset_name} - Val: {len(val_dataset)} ({len(val_dataset)/len(dataset)*100:.1f}%)")
        print(f"  {dataset_name} - Test: {len(test_dataset)} ({len(test_dataset)/len(dataset)*100:.1f}%)")

        # 保存每个数据集的划分信息
        split_info = {
            'dataset_name': dataset_name,
            'data_path': data_path,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'total_size': len(dataset),
            'seed': seed,
            'split_ratio': '8:1:1'
        }
        split_info_all[dataset_name] = split_info

        # 保存单个数据集的划分信息到pkl文件
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            split_file = os.path.join(output_dir, f'data_split_8_1_1_{dataset_name}.pkl')
            with open(split_file, 'wb') as f:
                pickle.dump(split_info, f)
            print(f"  {dataset_name} split info saved to: {split_file}")

        # 收集到总列表
        all_train_datasets.append(train_dataset)
        all_val_datasets.append(val_dataset)
        all_test_datasets.append(test_dataset)

    # 合并所有数据集
    print(f"\n🔧 Merging all datasets...")
    merged_train = torch.utils.data.ConcatDataset(all_train_datasets)
    merged_val = torch.utils.data.ConcatDataset(all_val_datasets)
    merged_test = torch.utils.data.ConcatDataset(all_test_datasets)

    print(f"Merged Train set size: {len(merged_train)}")
    print(f"Merged Validation set size: {len(merged_val)}")
    print(f"Merged Test set size: {len(merged_test)}")

    # 保存合并后的划分信息
    if output_dir:
        merged_split_info = {
            'datasets': dataset_names,
            'data_paths': data_paths,
            'individual_splits': split_info_all,
            'merged_train_size': len(merged_train),
            'merged_val_size': len(merged_val),
            'merged_test_size': len(merged_test),
            'total_merged_size': len(merged_train) + len(merged_val) + len(merged_test),
            'seed': seed,
            'split_ratio': '8:1:1'
        }

        merged_split_file = os.path.join(output_dir, 'data_split_8_1_1_merged.pkl')
        with open(merged_split_file, 'wb') as f:
            pickle.dump(merged_split_info, f)
        print(f"Merged dataset split info saved to: {merged_split_file}")

    return {
        'train': merged_train,
        'validation': merged_val,
        'test': merged_test,
        'individual_datasets': {
            'train': all_train_datasets,
            'validation': all_val_datasets,
            'test': all_test_datasets
        }
    }


def load_and_process_data_8_1_1(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    output_dir: str = None,
    seed: int = 42
):
    """
    加载并处理数据，按 8:1:1 比例划分训练集、验证集和测试集

    Args:
        data_path: 数据文件路径
        tokenizer: Tokenizer
        max_length: 最大序列长度
        output_dir: 输出目录，用于保存划分信息
        seed: 随机种子

    Returns:
        dict: 包含 'train', 'validation', 'test' 的字典
    """
    import pickle
    from sklearn.model_selection import train_test_split

    # 设置随机种子
    torch.manual_seed(seed)

    dataset = CultureLLMNewFormatDataset(data_path, tokenizer, max_length)

    # 获取所有数据的索引和文化标签，用于分层采样
    indices = list(range(len(dataset)))
    culture_labels = []

    # 提取文化标签用于分层采样
    for i in range(len(dataset)):
        sample = dataset.data[i]
        culture_labels.append(sample.get('label', '0'))

    # 第一次划分：80% 训练，20% 临时（验证+测试）
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, culture_labels,
        test_size=0.2,
        random_state=seed,
        stratify=culture_labels
    )

    # 第二次划分：从20%中分出10%验证，10%测试
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels,
        test_size=0.5,  # 0.5 * 0.2 = 0.1 (10%)
        random_state=seed,
        stratify=temp_labels
    )

    # 创建子数据集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Train set size: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"Validation set size: {len(val_dataset)} ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"Test set size: {len(test_dataset)} ({len(test_dataset)/len(dataset)*100:.1f}%)")

    # 保存划分信息到pickle文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        split_info = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'total_size': len(dataset),
            'data_path': data_path,
            'seed': seed,
            'split_ratio': '8:1:1'
        }

        split_file = os.path.join(output_dir, 'data_split_8_1_1.pkl')
        with open(split_file, 'wb') as f:
            pickle.dump(split_info, f)
        print(f"Dataset split info saved to: {split_file}")

    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }


def extract_answer_from_text(text: str) -> str:
    """
    从生成的文本中提取答案

    使用正则表达式查找单个数字

    Args:
        text: 生成的文本

    Returns:
        提取的答案（数字字符串）
    """
    # 🔧 修复：清理文本中的多余空格和重复字符
    cleaned_text = ' '.join(text.split())

    # 🔧 新增：处理重复数字的情况（如"1111" -> "1"）
    # 如果文本主要由重复的同一数字组成，提取该数字
    if len(cleaned_text) > 1:
        # 检查是否是重复的单个数字
        first_char = cleaned_text[0]
        if first_char.isdigit() and first_char in ['1', '2', '3', '4']:
            # 如果文本主要是重复的同一数字，返回该数字
            if cleaned_text.count(first_char) / len(cleaned_text.replace(' ', '')) > 0.5:
                return first_char

    # 优先匹配单个数字1-4，避免提取"442"这种多位数字
    match = re.search(r'\b([1-4])\b', cleaned_text)
    if match:
        return match.group(1)

    # 如果没有找到1-4，尝试查找任意单个数字（兼容性）
    match = re.search(r'\b(\d)\b', cleaned_text)
    if match:
        return match.group(1)

    # 最后兜底：查找第一个数字字符
    match = re.search(r'(\d)', cleaned_text)
    if match:
        return match.group(1)

    return ""


def generate_answer(model, tokenizer, instruction: str, input_text: str, device: str = 'cuda', max_new_tokens: int = 5) -> str:
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
    # 🔧 修复：训练时的格式改为：full_input + " " + output，所以生成时也用空格
    # 训练时的格式：full_input = f"{instruction}\n{input_text}"，然后添加 {output}
    # 所以生成时应该给模型：full_input + " "，让它生成output

    if input_text:
        full_input = f"{instruction}\n{input_text}"
    else:
        full_input = instruction

    # 🔧 修复：确保生成时的输入格式与训练时完全一致，并添加引导
    # 训练时格式：full_input + output_text (不添加空格)
    # 生成时格式：full_input + 引导文本，让模型生成数字答案

    # 🔧 关键修复：保持与训练时完全一致的格式
    # 训练时的数据格式：instruction + input → output (直接是数字)
    # 数据集的instruction已经包含了完整的提示，例如：
    # "Give me the answer from 1 to 4: ... ### Answer:"
    # 所以不需要添加任何额外的引导文本
    full_input = full_input.rstrip()


    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512, padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 🔧 确保attention_mask存在且正确
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])


    with torch.no_grad():
        # 检查模型类型，确定使用哪种generate方法
        model_class_name = model.__class__.__name__

        # 🔧 修复：统一使用模型的generate方法
        # SimplifiedCultureMoEAdapter现在有自己的generate方法，可以直接调用
        if ('JointLoRAMoE' in model_class_name or
            hasattr(model, 'moe_layer') or
            'SimplifiedCultureMoEAdapter' in model_class_name or
            hasattr(model, 'base_model') or
            hasattr(model, 'generate')):

            # 🔧 统一调用：所有MoE模型都直接使用model.generate()

            # 🔧 简化生成参数：与训练时保持一致的生成策略
            # 不添加过多约束，让模型自然生成，就像训练时验证阶段一样
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=5,  # 🔧 给足够空间生成答案
                min_new_tokens=1,  # 🔧 至少生成1个token
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,  # 🔧 使用贪心解码，与训练时一致
                num_beams=1,
                early_stopping=True,  # 🔧 启用早停
                repetition_penalty=1.0  # 🔧 不添加重复惩罚
            )
        else:
            # 回退到标准generate方法
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,  # 🔧 与上面保持一致
                min_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
                early_stopping=True
            )

    # 解码
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]

    # 🔧 简化生成处理：直接解码生成的token，不添加调试输出
    if len(generated_ids) > 0:
        # 解码生成的token
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    else:
        generated_text = ""


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


def generate_and_evaluate_answers(model, val_dataset, tokenizer, device, output_dir, epoch=None, individual_datasets=None):
    """
    在验证集上生成答案并评估准确率（Post Eval）

    关键改进：
    - 只输入 instruction + input，不输入 output
    - 让模型生成答案
    - 通过正则表达式提取数字答案
    - 与真实答案比对
    - 支持合并数据集的分离统计

    Args:
        model: 模型
        val_dataset: 验证数据集（可能是合并的）
        tokenizer: tokenizer
        device: 设备
        output_dir: 输出目录
        epoch: 当前 epoch 数（用于保存文件名）
        individual_datasets: 各个数据集的验证集列表（用于分离统计）

    Returns:
        dict: 包含准确率等指标的字典
    """
    model.eval()

    # 合并数据集的统计
    correct = 0
    total = 0
    generated_data = []

    # 分离数据集的统计（如果有）
    individual_stats = {}

    # 1. 评估合并数据集（用于模型选择）
    for idx in tqdm(range(len(val_dataset)), desc="Evaluating merged dataset", disable=False, mininterval=1.0):
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
        is_correct = predicted_answer == true_output
        if is_correct:
            correct += 1
        total += 1

        # 保存生成的数据
        item = {
            'instruction': instruction,
            'input': input_text,
            'true_output': true_output,
            'label': label,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': is_correct
        }

        # 添加数据集来源信息（如果有）
        if 'source_dataset' in sample:
            item['source_dataset'] = sample['source_dataset']

        generated_data.append(item)

    merged_accuracy = correct / total if total > 0 else 0

    # 2. 分离评估各个数据集（如果有individual_datasets）
    if individual_datasets is not None:
        print(f"\n🔧 开始分离评估各个数据集...")

        for dataset_idx, individual_dataset in enumerate(individual_datasets):
            dataset_name = f"Dataset_{dataset_idx}"

            # 尝试从数据集中获取名称
            if hasattr(individual_dataset, 'dataset') and hasattr(individual_dataset.dataset, 'dataset_names'):
                if dataset_idx < len(individual_dataset.dataset.dataset_names):
                    dataset_name = individual_dataset.dataset.dataset_names[dataset_idx]

            print(f"  评估 {dataset_name}...")

            dataset_correct = 0
            dataset_total = 0
            dataset_generated_data = []

            for idx in tqdm(range(len(individual_dataset)), desc=f"Evaluating {dataset_name}", disable=False, mininterval=1.0, leave=False):
                # 获取样本
                if hasattr(individual_dataset, 'dataset'):
                    original_idx = individual_dataset.indices[idx]
                    sample = individual_dataset.dataset[original_idx]
                else:
                    sample = individual_dataset[idx]

                instruction = sample['instruction']
                input_text = sample['input']
                true_output = sample['output']
                label = sample['label']

                # 生成答案
                generated_text = generate_answer(model, tokenizer, instruction, input_text, device)
                predicted_answer = extract_answer_from_text(generated_text)

                # 比对答案
                is_correct = predicted_answer == true_output
                if is_correct:
                    dataset_correct += 1
                dataset_total += 1

                dataset_generated_data.append({
                    'instruction': instruction,
                    'input': input_text,
                    'true_output': true_output,
                    'label': label,
                    'generated_text': generated_text,
                    'predicted_answer': predicted_answer,
                    'correct': is_correct
                })

            dataset_accuracy = dataset_correct / dataset_total if dataset_total > 0 else 0
            individual_stats[dataset_name] = {
                'accuracy': dataset_accuracy,
                'correct': dataset_correct,
                'total': dataset_total,
                'generated_data': dataset_generated_data
            }

            print(f"    {dataset_name} - Accuracy: {dataset_accuracy:.4f} ({dataset_correct}/{dataset_total})")

    accuracy = merged_accuracy

    # 保存生成的答案（最新的）- 合并数据集结果
    with open(os.path.join(output_dir, 'generated_answers.json'), 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # ✅ 保存每个 epoch 的生成答案（不覆盖）- 合并数据集结果
    if epoch is not None:
        epoch_answers_file = os.path.join(output_dir, f'generated_answers_epoch_{epoch}.json')
        with open(epoch_answers_file, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=2, ensure_ascii=False)

    # 🔧 新增：保存分离数据集的结果（如果有）
    if individual_stats:
        # 保存分离统计结果
        individual_results_file = os.path.join(output_dir, 'individual_dataset_results.json')
        with open(individual_results_file, 'w', encoding='utf-8') as f:
            # 只保存统计信息，不保存详细的generated_data（避免文件过大）
            stats_only = {}
            for dataset_name, stats in individual_stats.items():
                stats_only[dataset_name] = {
                    'accuracy': stats['accuracy'],
                    'correct': stats['correct'],
                    'total': stats['total']
                }
            json.dump(stats_only, f, indent=2, ensure_ascii=False)

        # 保存每个epoch的分离统计结果
        if epoch is not None:
            epoch_individual_file = os.path.join(output_dir, f'individual_dataset_results_epoch_{epoch}.json')
            with open(epoch_individual_file, 'w', encoding='utf-8') as f:
                stats_only = {}
                for dataset_name, stats in individual_stats.items():
                    stats_only[dataset_name] = {
                        'accuracy': stats['accuracy'],
                        'correct': stats['correct'],
                        'total': stats['total']
                    }
                json.dump(stats_only, f, indent=2, ensure_ascii=False)

            # 为每个数据集保存详细的生成答案
            for dataset_name, stats in individual_stats.items():
                dataset_answers_file = os.path.join(output_dir, f'generated_answers_{dataset_name}_epoch_{epoch}.json')
                with open(dataset_answers_file, 'w', encoding='utf-8') as f:
                    json.dump(stats['generated_data'], f, indent=2, ensure_ascii=False)

    # 统计异常情况
    empty_generations = 0
    non_digit_generations = 0
    incorrect_predictions = 0

    for item in generated_data:
        if not item['generated_text'].strip():
            empty_generations += 1
        elif not item['predicted_answer']:
            non_digit_generations += 1
        elif not item['correct']:
            incorrect_predictions += 1

    # 打印汇总信息
    print(f"\n📊 生成评估汇总:")
    print(f"  📊 合并数据集统计（用于模型选择）:")
    print(f"    总样本数: {len(generated_data)}")
    print(f"    准确率: {accuracy:.4f}")
    print(f"    空生成: {empty_generations}")
    print(f"    未提取到数字: {non_digit_generations}")
    print(f"    错误预测: {incorrect_predictions}")

    # 打印分离统计信息
    if individual_stats:
        print(f"\n  📊 分离数据集统计（用于观察）:")
        for dataset_name, stats in individual_stats.items():
            print(f"    {dataset_name}:")
            print(f"      准确率: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        print(f"  💾 分离结果已保存到 individual_dataset_results*.json 文件")

    # 只在存在问题时打印详细示例
    if empty_generations > 0 or non_digit_generations > 0:
        print(f"\n⚠️ 发现生成问题，显示前3个有问题的样本:")
        print("-" * 80)
        problem_count = 0
        for idx, item in enumerate(generated_data):
            if problem_count >= 3:
                break
            # 只显示有问题的样本
            if (not item['generated_text'].strip() or
                not item['predicted_answer'] or
                item['predicted_answer'] not in ['1', '2', '3', '4']):
                print(f"\n问题样本 {idx + 1}:")
                print(f"  Instruction: {item['instruction'][:80]}...")
                print(f"  True Output: {item['true_output']}")
                print(f"  Generated Text: '{item['generated_text']}'")
                print(f"  Predicted Answer: '{item['predicted_answer']}'")
                problem_count += 1
        print("-" * 80)
    else:
        print("✅ 所有样本都成功生成了有效的数字答案")

    result = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

    # 添加分离统计信息（如果有）
    if individual_stats:
        result['individual_stats'] = {}
        for dataset_name, stats in individual_stats.items():
            result['individual_stats'][dataset_name] = {
                'accuracy': stats['accuracy'],
                'correct': stats['correct'],
                'total': stats['total']
            }

    return result


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

    # 🔧 修复Llama 3.1 tokenizer配置问题 - 使用官方padding token
    print(f"🔧 原始tokenizer状态: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")

    # 强制检查和修复pad_token配置，使用Llama 3.1官方的padding token
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128009:
        # Llama 3.1: 使用官方的finetune_right_pad_id
        print(f"🔧 检测到Llama 3.1模型，查找官方padding token...")

        # 查找Llama 3.1官方的padding token
        official_pad_token = "<|finetune_right_pad_id|>"
        try:
            pad_token_id = tokenizer.convert_tokens_to_ids(official_pad_token)

            # 检查这个token是否存在且有效
            if pad_token_id != tokenizer.unk_token_id and pad_token_id is not None:
                tokenizer.pad_token = official_pad_token
                tokenizer.pad_token_id = pad_token_id
                print(f"✅ Llama 3.1: 使用官方padding token: '{official_pad_token}' (id={pad_token_id})")
            else:
                raise ValueError("Official pad token not found or invalid")

        except Exception as e:
            print(f"⚠️ 无法找到官方padding token '{official_pad_token}': {e}")
            print(f"🔧 使用安全的低频字符作为fallback...")

            # 使用安全的低频字符作为fallback
            safe_tokens = ['~', '`', '|', '^', '§', '¶', '†', '‡']
            found_safe_token = False
            for safe_token in safe_tokens:
                try:
                    safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                    if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                        tokenizer.pad_token = safe_token
                        tokenizer.pad_token_id = safe_token_id
                        print(f"🔧 Llama 3.1: 使用安全字符 '{safe_token}' (id={safe_token_id}) 作为padding")
                        found_safe_token = True
                        break
                except:
                    continue

            if not found_safe_token:
                print(f"⚠️ 无法找到合适的padding token，将导致训练问题")

    elif tokenizer.pad_token is None:
        # 🔧 修复：避免将pad_token设置为eos_token
        print(f"🔧 标准配置: 设置安全的padding token")
        safe_tokens = ['~', '`', '|', '^', '§', '¶']
        found_safe_token = False
        for safe_token in safe_tokens:
            try:
                safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                    tokenizer.pad_token = safe_token
                    tokenizer.pad_token_id = safe_token_id
                    found_safe_token = True
                    break
            except:
                continue

        if not found_safe_token:
            tokenizer.pad_token_id = 0  # fallback to token_id=0
    else:
        # 对于已经有pad_token但可能配置错误的情况，也要检查
        if tokenizer.pad_token_id == 128009:
            print(f"🔧 检测到错误的pad_token配置(使用了<|eot_id|>)，强制修复...")

            # 对于Llama 3.1，优先尝试官方padding token
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == 128009:
                official_pad_token = "<|finetune_right_pad_id|>"
                try:
                    pad_token_id = tokenizer.convert_tokens_to_ids(official_pad_token)
                    if pad_token_id != tokenizer.unk_token_id and pad_token_id != 128009:
                        tokenizer.pad_token = official_pad_token
                        tokenizer.pad_token_id = pad_token_id
                        print(f"✅ 强制修复: 使用官方padding token '{official_pad_token}' (id={pad_token_id})")
                    else:
                        raise ValueError("Official pad token invalid")
                except:
                    # 如果官方token不可用，使用安全字符
                    safe_tokens = ['~', '`', '|', '^', '§', '¶']
                    for safe_token in safe_tokens:
                        try:
                            safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                            if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                                tokenizer.pad_token = safe_token
                                tokenizer.pad_token_id = safe_token_id
                                print(f"🔧 强制修复: 使用安全字符 '{safe_token}' (id={safe_token_id}) 作为padding")
                                break
                        except:
                            continue

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

        # 🔧 修复：使用安全的低频token作为padding，避免与eos_token冲突
        safe_tokens = ['~', '`', '|', '^', '§', '¶', '†', '‡']
        found_safe_token = False
        for safe_token in safe_tokens:
            try:
                safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                    tokenizer.pad_token = safe_token
                    tokenizer.pad_token_id = safe_token_id
                    print(f"   修复后pad_token_id: {tokenizer.pad_token_id}")
                    print(f"   修复后pad_token: {repr(tokenizer.pad_token)}")
                    found_safe_token = True
                    break
            except:
                continue

        if not found_safe_token:
            # 最后的fallback：使用token_id=0（通常是<unk>）
            tokenizer.pad_token_id = 0
            print(f"   Fallback修复后pad_token_id: {tokenizer.pad_token_id}")

    elif tokenizer.pad_token_id is None:
        print(f"🚨 错误: pad_token_id is None!")
        print(f"   强制设置专用padding token...")

        # 🔧 修复：避免使用eos_token作为padding
        safe_tokens = ['~', '`', '|', '^', '§', '¶', '†', '‡']
        found_safe_token = False
        for safe_token in safe_tokens:
            try:
                safe_token_id = tokenizer.convert_tokens_to_ids(safe_token)
                if safe_token_id != tokenizer.unk_token_id and safe_token_id != 128009:
                    tokenizer.pad_token = safe_token
                    tokenizer.pad_token_id = safe_token_id
                    print(f"   设置后pad_token_id: {tokenizer.pad_token_id}")
                    print(f"   设置后pad_token: {repr(tokenizer.pad_token)}")
                    found_safe_token = True
                    break
            except:
                continue

        if not found_safe_token:
            # 最后的fallback：使用token_id=0
            tokenizer.pad_token_id = 0
            print(f"   Fallback设置后pad_token_id: {tokenizer.pad_token_id}")

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
                # 🔧 修复：如果找不到合适的token，使用token_id=0作为fallback
                tokenizer.pad_token_id = 0
                chosen_pad_token = "fallback_token_0"

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

    # 🔧 新增：检查是否为合并数据集
    if args.train_file.startswith("MERGED:"):
        # 处理合并数据集
        merge_spec = args.train_file.replace("MERGED:", "")
        print(f"🔧 检测到合并数据集: {merge_spec}")

        if merge_spec == "CulturalBench+CultureLLM":
            # CulturalBench + CultureLLM 合并
            data_paths = [
                "/root/autodl-fs/CulturalBench_merge_gen.json",
                "/root/autodl-fs/cultureLLM_merge_gen.json"
            ]
            dataset_names = ["CulturalBench", "CultureLLM"]

            print(f"🔧 使用合并数据集: {dataset_names}")
            datasets = load_and_process_merged_data_8_1_1(
                data_paths=data_paths,
                dataset_names=dataset_names,
                tokenizer=tokenizer,
                max_length=args.max_length,
                output_dir=args.output_dir,
                seed=42
            )

            # 合并数据集的训练和验证集
            train_dataset = datasets['train']
            val_dataset = datasets['validation']

            # 保存个体数据集信息，用于分离统计
            individual_val_datasets = datasets['individual_datasets']['validation']

        else:
            raise ValueError(f"Unsupported merged dataset: {merge_spec}")

        print("✅ Merged data loaded")
    else:
        # 处理单个数据集
        datasets = load_and_process_data(
            args.train_file,
            tokenizer,
            max_length=args.max_length,
            val_split=args.val_split
        )
        train_dataset = datasets['train']
        val_dataset = datasets['validation']
        individual_val_datasets = None  # 单个数据集不需要分离统计

        print("✅ Data loaded")

    # 创建 collate_fn
    collate_fn = partial(dynamic_padding_collate_fn, tokenizer=tokenizer, max_seq_length=args.max_length)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
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
                model, val_dataset, tokenizer, args.device, args.output_dir, epoch=epoch+1,
                individual_datasets=individual_val_datasets
            )

            print(f"  Eval Loss: {val_metrics['loss']:.4f}")
            print(f"  Eval Accuracy (Merged): {gen_metrics['accuracy']:.4f} ({gen_metrics['correct']}/{gen_metrics['total']})")

            # 🔧 新增：显示分离统计结果
            if 'individual_stats' in gen_metrics:
                print(f"  Individual Dataset Accuracies:")
                for dataset_name, stats in gen_metrics['individual_stats'].items():
                    print(f"    {dataset_name}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

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
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'eval_loss': val_metrics['loss'],
                'eval_accuracy': gen_metrics['accuracy'],
                'correct': gen_metrics['correct'],
                'total': gen_metrics['total'],
                'is_best': gen_metrics['accuracy'] == best_eval_accuracy  # ✅ 标记是否为最佳
            }

            # 🔧 新增：添加分离统计信息
            if 'individual_stats' in gen_metrics:
                epoch_result['individual_stats'] = gen_metrics['individual_stats']

            epoch_results.append(epoch_result)
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

