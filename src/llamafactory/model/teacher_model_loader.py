# src/llamafactory/model/teacher_model_loader.py
"""
Teacher模型加载器
用于加载LoRA微调模型作为知识蒸馏的教师模型
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class TeacherModelLoader:
    """
    Teacher模型加载器：加载LoRA模型作为蒸馏教师
    """

    def __init__(self, base_model_path: str, lora_weights_path: str, device: str = 'cuda'):
        self.base_model_path = base_model_path
        self.lora_weights_path = lora_weights_path
        self.device = device
        self.teacher_model = None
        self.tokenizer = None

    def load_teacher_model(self) -> nn.Module:
        """
        加载并合并LoRA权重的教师模型

        Returns:
            合并后的教师模型
        """
        try:
            logging.info(f"Loading teacher model from {self.base_model_path}")

            # 1. 加载基础模型
            teacher_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map=None,
                trust_remote_code=True
            )

            # 2. 加载LoRA权重
            if self.lora_weights_path and os.path.exists(self.lora_weights_path):
                logging.info(f"Loading LoRA weights from {self.lora_weights_path}")
                teacher_model = PeftModel.from_pretrained(teacher_model, self.lora_weights_path)

                # 3. 合并LoRA权重
                teacher_model = teacher_model.merge_and_unload()
                logging.info("LoRA weights merged successfully")
            else:
                logging.warning(f"LoRA weights path not found: {self.lora_weights_path}")
                logging.info("Using base model as teacher")

            # 4. 移动到设备并设置为评估模式
            teacher_model = teacher_model.to(self.device)
            teacher_model.eval()

            # 5. 冻结所有参数
            for param in teacher_model.parameters():
                param.requires_grad = False

            self.teacher_model = teacher_model
            logging.info("Teacher model loaded and frozen successfully")

            return teacher_model

        except Exception as e:
            logging.error(f"Failed to load teacher model: {e}")
            raise e

    def load_tokenizer(self) -> AutoTokenizer:
        """
        加载tokenizer

        Returns:
            tokenizer实例
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def get_teacher_logits(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """
        获取教师模型的logits输出

        Args:
            input_ids: [B, L] 输入token IDs
            attention_mask: [B, L] 注意力mask

        Returns:
            teacher_logits: [B, L, V] 教师模型的logits
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded. Call load_teacher_model() first.")

        with torch.no_grad():
            # 确保输入在正确的设备上
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # 获取教师模型输出
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            return outputs.logits

    def get_teacher_hidden_states(self, input_ids: torch.Tensor,
                                 attention_mask: torch.Tensor,
                                 layer_idx: int = -1) -> torch.Tensor:
        """
        获取教师模型的隐藏状态

        Args:
            input_ids: [B, L] 输入token IDs
            attention_mask: [B, L] 注意力mask
            layer_idx: 要提取的层索引（-1表示最后一层）

        Returns:
            hidden_states: [B, L, H] 隐藏状态
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded. Call load_teacher_model() first.")

        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # 提取指定层的隐藏状态
            hidden_states = outputs.hidden_states[layer_idx]

            return hidden_states

    def cleanup(self):
        """清理资源"""
        if self.teacher_model is not None:
            del self.teacher_model
            self.teacher_model = None
            torch.cuda.empty_cache()
            logging.info("Teacher model cleaned up")


class TeacherLogitsCache:
    """
    Teacher模型logits缓存器
    用于预计算并缓存teacher logits，避免重复计算
    """

    def __init__(self, cache_dir: str, max_cache_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.cache = {}

        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
        """生成缓存键"""
        # 使用输入的hash作为键
        input_hash = hash((input_ids.cpu().numpy().tobytes(),
                          attention_mask.cpu().numpy().tobytes()))
        return str(input_hash)

    def get_cached_logits(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """获取缓存的logits"""
        cache_key = self.get_cache_key(input_ids, attention_mask)

        if cache_key in self.cache:
            return self.cache[cache_key]

        # 尝试从磁盘加载
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pt")
        if os.path.exists(cache_file):
            try:
                logits = torch.load(cache_file, map_location='cpu')
                self.cache[cache_key] = logits
                return logits
            except Exception as e:
                logging.warning(f"Failed to load cached logits: {e}")

        return None

    def cache_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    logits: torch.Tensor):
        """缓存logits"""
        cache_key = self.get_cache_key(input_ids, attention_mask)

        # 内存缓存
        if len(self.cache) < self.max_cache_size:
            self.cache[cache_key] = logits.cpu()

        # 磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pt")
        try:
            torch.save(logits.cpu(), cache_file)
        except Exception as e:
            logging.warning(f"Failed to save cached logits: {e}")

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        # 清理磁盘缓存文件
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pt'):
                os.remove(os.path.join(self.cache_dir, file))


def create_teacher_loader(base_model_path: str, lora_weights_path: str,
                         device: str = 'cuda') -> TeacherModelLoader:
    """
    创建Teacher模型加载器的工厂函数

    Args:
        base_model_path: 基础模型路径
        lora_weights_path: LoRA权重路径
        device: 设备

    Returns:
        TeacherModelLoader实例
    """
    return TeacherModelLoader(base_model_path, lora_weights_path, device)