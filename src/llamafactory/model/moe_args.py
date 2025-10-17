from dataclasses import dataclass, field  # 导入 field
from typing import Dict, Any

import torch


@dataclass
class ModelArgs:
    # Dataset configuration
    # dataset_type: str = "culturalbench"  # "culturalbench", "globalopinions", "culturebank"

    # Model paths
    llama_model_path: str = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"

    # Model architecture
    num_experts: int = 6
    experts_hidden_dim: int = 256
    # experts_output_dim: int = 128
    router_hidden_dim: int = 256
    shared_hidden_dim: int = 512
    num_heads: int = 8
    classification_hidden_dim: int = 256
    num_classes: int = 3  # TRUE/FALSE classification for culturalbench

    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: ["all"]) # 目标模块的选择将由 LoRA 的实现自动决定，=all：LoRA 将尝试在模型中的所有可支持的线性层上应用 LoRA

    # Training parameters
    learning_rate: float = 5e-8
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    dropout: float = 0.1 

    # Data parameters
    # max_length: int = 64
    # train_data_path: str = "/root/autodl-tmp/CultureMoE/Culture_Alignment/data/CulturalBench-Hard_train.json"
    # test_data_path: str = "/root/autodl-tmp/CultureMoE/Culture_Alignment/data/CulturalBench-Hard_test.json"
    # train_data_path: str = "/root/autodl-fs/CulturalBench-Hard_train.json"
    # test_data_path: str = "/root/autodl-fs/CulturalBench-Hard_test.json"
    # train_data_path: str = "/root/autodl-fs/global_opinions_train.json"
    # test_data_path: str = "/root/autodl-fs/global_opinions_test.json"

    # Device and training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10

    # 损失权重
    load_balance_weight: float = 0.0
    diversity_weight: float = 0.0

    # Output paths
    # output_dir: str = "/root/autodl-tmp/CultureMoE/Culture_Alignment/outputs/" + dataset_type
    # model_save_path: str = output_dir + "/CA_llama"
    # log_file: str = output_dir + "/CA_llama_training.log"

  