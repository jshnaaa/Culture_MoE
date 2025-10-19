# src/llamafactory/train/classification/workflow.py
# src/llamafactory/train/classification/workflow.py
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorWithPadding
)

from .metrics import compute_classification_metrics
from .trainer import ClassificationTrainer
from ...data.classification_processor import load_and_process_classification_data
from ...data.dual_classification_collator import DualClassificationDataCollator
from ...data.dual_classification_processor import load_and_process_dual_classification_data
from ...model.CultureMoE import LlamaSharedRouterExpertsModel
from ...model.moe_args import ModelArgs


@dataclass
class ClassificationTrainingArguments:
    """分类任务的训练参数"""

    # 模型参数
    model_name_or_path: str = field(
        metadata={"help": "预训练模型路径"}
    )

    # 数据参数
    train_file: str = field(
        metadata={"help": "训练数据文件路径 (json/jsonl)"}
    )
    val_file: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据文件路径，如果为 None 则从训练集分割"}
    )
    val_split: float = field(
        default=0.1,
        metadata={"help": "如果没有单独的验证集，从训练集分割的比例"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    use_dual_input: bool = field(
        default=False,
        metadata={"help": "是否使用双路输入（instruction + instruction_mask）"}
    )

    # MoE 参数
    num_experts: int = field(default=6, metadata={"help": "专家数量"})
    shared_hidden_dim: int = field(default=2048, metadata={"help": "Shared 层隐藏维度"})
    router_hidden_dim: int = field(default=1024, metadata={"help": "Router 隐藏维度"})
    experts_hidden_dim: int = field(default=2048, metadata={"help": "Experts 隐藏维度"})
    lora_rank: int = field(default=16, metadata={"help": "Experts 的 LoRA rank"})
    num_classes: int = field(default=3, metadata={"help": "分类类别数"})
    classification_hidden_dim: int = field(default=512, metadata={"help": "分类头隐藏维度"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout 率"})
    num_heads: int = field(default=8, metadata={"help": "注意力头数"})

    # ✅ LLaMA LoRA 参数
    freeze_llama: bool = field(
        default=True,
        metadata={"help": "是否冻结 LLaMA 基础模型参数"}
    )
    use_llama_lora: bool = field(
        default=False,
        metadata={"help": "是否对 LLaMA 使用 LoRA 微调（仅在 freeze_llama=False 时生效）"}
    )
    llama_lora_rank: int = field(
        default=8,
        metadata={"help": "LLaMA LoRA 的 rank"}
    )
    llama_lora_alpha: int = field(
        default=16,
        metadata={"help": "LLaMA LoRA 的 alpha"}
    )
    llama_lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LLaMA LoRA 的 dropout"}
    )
    llama_lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LLaMA LoRA 的目标模块，用逗号分隔"}
    )

    # 训练参数
    output_dir: str = field(
        default="/root/autodl-fs/output/classi_$(date +%Y%m%d_%H%M%S)",
        metadata={"help": "输出目录"}
    )
    num_train_epochs: int = field(default=3, metadata={"help": "训练轮数"})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "每个设备的训练批次大小"})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "每个设备的评估批次大小"})
    learning_rate: float = field(default=2e-5, metadata={"help": "学习率"})
    weight_decay: float = field(default=0.01, metadata={"help": "权重衰减"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "warmup 比例"})
    logging_steps: int = field(default=10, metadata={"help": "日志记录步数"})
    save_steps: int = field(default=500, metadata={"help": "模型保存步数"})
    eval_steps: int = field(default=500, metadata={"help": "评估步数"})
    save_total_limit: int = field(default=3, metadata={"help": "最多保存的检查点数量"})

    # GPU 优化参数
    fp16: bool = field(
        default=True,
        metadata={"help": "是否使用 FP16 混合精度训练"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "是否使用 BF16 混合精度训练（推荐用于 A100）"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数，增大可以模拟更大的 batch size"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "是否使用梯度检查点（节省显存但会降低速度）"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "数据加载器工作进程数"}
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "是否使用 pin_memory 加速 CPU->GPU 传输"}
    )

    # 其他参数
    seed: int = field(default=42, metadata={"help": "随机种子"})
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "评估策略: 'steps' 或 'epoch'"}
    )

    # 多卡训练参数
    local_rank: int = field(
        default=-1,
        metadata={"help": "分布式训练的 local rank，由 torch.distributed.launch 自动设置"}
    )


def run_classification_training(args: ClassificationTrainingArguments):
    """
    运行分类任务训练
    支持单卡和多卡训练，支持 LLaMA LoRA 微调

    Args:
        args: 训练参数
    """
    # 1. 设置分布式训练环境
    local_rank = args.local_rank
    is_distributed = local_rank != -1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
        print(f"[Rank {local_rank}] Initialized distributed training")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on device: {device}")

    # 2. 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 3. 加载 tokenizer
    if not is_distributed or local_rank == 0:
        print(f"\nLoading tokenizer from {args.model_name_or_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if not is_distributed or local_rank == 0:
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # 4. 加载和处理数据
    if not is_distributed or local_rank == 0:
        print(f"\nLoading and processing data from {args.train_file}...")
        if args.use_dual_input:
            print("Using dual input mode (instruction + instruction_mask)")

    # ✅ 根据 use_dual_input 选择数据处理器
    if args.use_dual_input:
        # 使用双路输入处理器
        load_func = load_and_process_dual_classification_data
    else:
        # 使用单路输入处理器
        load_func = load_and_process_classification_data

    if args.val_file:
        train_data = load_func(
            args.train_file,
            tokenizer,
            max_length=args.max_length,
            val_split=0,
            num_proc=args.dataloader_num_workers
        )
        val_data = load_func(
            args.val_file,
            tokenizer,
            max_length=args.max_length,
            val_split=0,
            num_proc=args.dataloader_num_workers
        )
        train_dataset = train_data["train"]
        val_dataset = val_data["train"]
    else:
        data = load_func(
            args.train_file,
            tokenizer,
            max_length=args.max_length,
            val_split=args.val_split,
            num_proc=args.dataloader_num_workers
        )
        train_dataset = data["train"]
        val_dataset = data["validation"]

    if not is_distributed or local_rank == 0:
        print(f"\nTrain dataset size: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation dataset size: {len(val_dataset)}")

    # 5. 加载基础模型
    if not is_distributed or local_rank == 0:
        print(f"\nLoading base model from {args.model_name_or_path}...")

    llama_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
        trust_remote_code=True
    )

    # ✅ 6. 如果不冻结且使用 LoRA，为 LLaMA 添加 LoRA
    if not args.freeze_llama and args.use_llama_lora:
        if not is_distributed or local_rank == 0:
            print("\n" + "=" * 60)
            print("Applying LoRA to LLaMA model...")
            print("=" * 60)

        from peft import get_peft_model, LoraConfig, TaskType

        # 解析目标模块
        target_modules = [m.strip() for m in args.llama_lora_target_modules.split(",")]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.llama_lora_rank,
            lora_alpha=args.llama_lora_alpha,
            lora_dropout=args.llama_lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )

        llama_model = get_peft_model(llama_model, lora_config)

        if not is_distributed or local_rank == 0:
            print("\nLoRA Configuration:")
            print(f"  Rank: {args.llama_lora_rank}")
            print(f"  Alpha: {args.llama_lora_alpha}")
            print(f"  Dropout: {args.llama_lora_dropout}")
            print(f"  Target modules: {target_modules}")
            print("\nLLaMA Trainable Parameters:")
            llama_model.print_trainable_parameters()
            print("=" * 60 + "\n")

    # 7. 创建 MoE 模型
    if not is_distributed or local_rank == 0:
        print("Creating CultureMoE model...")

    moe_args = ModelArgs(
        num_experts=args.num_experts,
        shared_hidden_dim=args.shared_hidden_dim,
        router_hidden_dim=args.router_hidden_dim,
        experts_hidden_dim=args.experts_hidden_dim,
        lora_rank=args.lora_rank,
        num_classes=args.num_classes,
        classification_hidden_dim=args.classification_hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads
    )

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    )

    # ✅ 8. 处理参数冻结（仅在完全冻结时）
    if args.freeze_llama:
        if args.use_llama_lora:
            if not is_distributed or local_rank == 0:
                print("️ Warning: freeze_llama=True but use_llama_lora=True")
                print(" LoRA parameters will remain trainable")
        else:
            # 只有在不使用 LoRA时才冻结
            for param in model.llama_model.parameters():
                param.requires_grad = False
            if not is_distributed or local_rank == 0:
                print(" Frozen LLaMA base model parameters")
    elif not args.use_llama_lora:
        # 不冻结且不使用 LoRA = 全参数微调
        if not is_distributed or local_rank == 0:
            print("⚠️  LLaMA will be fully fine-tuned (requires large GPU memory!)")

    # 9. 梯度检查点
    if not is_distributed or local_rank == 0:
        if args.gradient_checkpointing:
            print("Gradient checkpointing will be enabled by Trainer")

    # 10. 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 详细统计
    llama_params = sum(p.numel() for p in model.llama_model.parameters())
    llama_trainable = sum(p.numel() for p in model.llama_model.parameters() if p.requires_grad)

    shared_params = sum(p.numel() for p in model.shared.parameters())
    router_params = sum(p.numel() for p in model.router.parameters())
    experts_params = sum(p.numel() for p in model.experts_layer.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    if not is_distributed or local_rank == 0:
        print(f"\n{'=' * 60}")
        print("Model Parameter Statistics:")
        print(f"{'=' * 60}")
        print(f"LLaMA Base Model:")
        print(f"  Total: {llama_params:,}")
        print(f"  Trainable: {llama_trainable:,} ({llama_trainable / llama_params * 100:.2f}%)")
        print(f"\nMoE Components:")
        print(f"  Shared Layer: {shared_params:,}")
        print(f"  Router: {router_params:,}")
        print(f"  Experts (LoRA): {experts_params:,}")
        print(f"  Classifier: {classifier_params:,}")
        print(f"\nTotal Model:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
        print(f"{'=' * 60}\n")

    # 11. 创建 TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        # 日志和保存
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.evaluation_strategy if val_dataset else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,

        # 最佳模型
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="f1_macro" if val_dataset else None,
        greater_is_better=True,

        # GPU 优化和分布式训练
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,

        # 分布式训练
        local_rank=local_rank,
        ddp_find_unused_parameters=True,
        ddp_backend="nccl",

        # 其他
        remove_unused_columns=False,
        report_to=["tensorboard"] if (not is_distributed or local_rank == 0) else [],
        seed=args.seed,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )

    # 12. 创建 Data Collator
    # ✅ 根据 use_dual_input 选择 DataCollator
    if args.use_dual_input:
        data_collator = DualClassificationDataCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_length
        )
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_length
        )

    # 13. 创建 Trainer
    if not is_distributed or local_rank == 0:
        print("Creating trainer...")

    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_classification_metrics if (not is_distributed or local_rank == 0) else None,
    )

    # 14. 开始训练
    if not is_distributed or local_rank == 0:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

        if torch.cuda.is_available():
            print(f"\nGPU Information:")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")

            if is_distributed:
                print(f"\n  Using Distributed Data Parallel (DDP)")
                print(f"  World size: {torch.distributed.get_world_size()}")

    train_result = trainer.train()

    # 15. 保存模型
    if not is_distributed or local_rank == 0:
        print(f"\nSaving model to {args.output_dir}...")
        trainer.save_model()
        trainer.save_state()

        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # 最终评估
        if val_dataset:
            print("\nRunning final evaluation...")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

        print("\n" + "=" * 60)
        print("Training completed successfully! ✅")
        print("=" * 60)

    return trainer


# @dataclass
# class ClassificationTrainingArguments:
#     """分类任务的训练参数"""
#
#     # 模型参数
#     model_name_or_path: str = field(
#         metadata={"help": "预训练模型路径"}
#     )
#
#     # 数据参数
#     train_file: str = field(
#         metadata={"help": "训练数据文件路径 (json/jsonl)"}
#     )
#     val_file: Optional[str] = field(
#         default=None,
#         metadata={"help": "验证数据文件路径，如果为 None 则从训练集分割"}
#     )
#     val_split: float = field(
#         default=0.1,
#         metadata={"help": "如果没有单独的验证集，从训练集分割的比例"}
#     )
#     max_length: int = field(
#         default=512,
#         metadata={"help": "最大序列长度"}
#     )
#
#     # MoE 参数
#     num_experts: int = field(default=6, metadata={"help": "专家数量"})
#     shared_hidden_dim: int = field(default=2048, metadata={"help": "Shared 层隐藏维度"})
#     router_hidden_dim: int = field(default=1024, metadata={"help": "Router 隐藏维度"})
#     experts_hidden_dim: int = field(default=2048, metadata={"help": "Experts 隐藏维度"})
#     lora_rank: int = field(default=16, metadata={"help": "LoRA rank"})
#     num_classes: int = field(default=3, metadata={"help": "分类类别数"})
#     classification_hidden_dim: int = field(default=512, metadata={"help": "分类头隐藏维度"})
#     dropout: float = field(default=0.1, metadata={"help": "Dropout 率"})
#     num_heads: int = field(default=8, metadata={"help": "注意力头数"})
#
#     # 模型训练控制
#     freeze_llama: bool = field(
#         default=True,
#         metadata={"help": "是否冻结 LLaMA 基础模型参数"}
#     )
#
#     # 训练参数
#     output_dir: str = field(
#         default="./output/classification",
#         metadata={"help": "输出目录"}
#     )
#     num_train_epochs: int = field(default=3, metadata={"help": "训练轮数"})
#     per_device_train_batch_size: int = field(default=4, metadata={"help": "每个设备的训练批次大小"})
#     per_device_eval_batch_size: int = field(default=8, metadata={"help": "每个设备的评估批次大小"})
#     learning_rate: float = field(default=2e-5, metadata={"help": "学习率"})
#     weight_decay: float = field(default=0.01, metadata={"help": "权重衰减"})
#     warmup_ratio: float = field(default=0.1, metadata={"help": "warmup 比例"})
#     logging_steps: int = field(default=10, metadata={"help": "日志记录步数"})
#     save_steps: int = field(default=500, metadata={"help": "模型保存步数"})
#     eval_steps: int = field(default=500, metadata={"help": "评估步数"})
#     save_total_limit: int = field(default=3, metadata={"help": "最多保存的检查点数量"})
#
#     # GPU 优化参数
#     fp16: bool = field(
#         default=True,
#         metadata={"help": "是否使用 FP16 混合精度训练"}
#     )
#     bf16: bool = field(
#         default=False,
#         metadata={"help": "是否使用 BF16 混合精度训练（推荐用于 A100）"}
#     )
#     gradient_accumulation_steps: int = field(
#         default=4,
#         metadata={"help": "梯度累积步数，增大可以模拟更大的 batch size"}
#     )
#     gradient_checkpointing: bool = field(
#         default=True,
#         metadata={"help": "是否使用梯度检查点（节省显存但会降低速度）"}
#     )
#     dataloader_num_workers: int = field(
#         default=4,
#         metadata={"help": "数据加载器工作进程数"}
#     )
#     dataloader_pin_memory: bool = field(
#         default=True,
#         metadata={"help": "是否使用 pin_memory 加速 CPU->GPU 传输"}
#     )
#
#     # 其他参数
#     seed: int = field(default=42, metadata={"help": "随机种子"})
#     evaluation_strategy: str = field(
#         default="steps",
#         metadata={"help": "评估策略: 'steps' 或 'epoch'"}
#     )
#
#     # ✅ 多卡训练参数
#     local_rank: int = field(
#         default=-1,
#         metadata={"help": "分布式训练的 local rank，由 torch.distributed.launch 自动设置"}
#     )
#
#
# def run_classification_training(args: ClassificationTrainingArguments):
#     """
#     运行分类任务训练
#     支持单卡和多卡训练
#
#     Args:
#         args: 训练参数
#     """
#     # ✅ 1. 设置分布式训练环境
#     local_rank = args.local_rank
#     is_distributed = local_rank != -1
#
#     if is_distributed:
#         torch.cuda.set_device(local_rank)
#         torch.distributed.init_process_group(backend='nccl')
#         device = torch.device('cuda', local_rank)
#         print(f"[Rank {local_rank}] Initialized distributed training")
#     else:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Running on device: {device}")
#
#     # 2. 设置随机种子
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)
#
#     # 3. 加载 tokenizer（只在主进程打印）
#     if not is_distributed or local_rank == 0:
#         print(f"\nLoading tokenizer from {args.model_name_or_path}...")
#
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name_or_path,
#         trust_remote_code=True
#     )
#
#     # 确保有 pad_token
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         if not is_distributed or local_rank == 0:
#             print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
#
#     # 4. 加载和处理数据
#     if not is_distributed or local_rank == 0:
#         print(f"\nLoading and processing data from {args.train_file}...")
#
#     if args.val_file:
#         # 有单独的验证集
#         train_data = load_and_process_classification_data(
#             args.train_file,
#             tokenizer,
#             max_length=args.max_length,
#             val_split=0,
#             num_proc=args.dataloader_num_workers
#         )
#         val_data = load_and_process_classification_data(
#             args.val_file,
#             tokenizer,
#             max_length=args.max_length,
#             val_split=0,
#             num_proc=args.dataloader_num_workers
#         )
#         train_dataset = train_data["train"]
#         val_dataset = val_data["train"]
#     else:
#         # 从训练集分割
#         data = load_and_process_classification_data(
#             args.train_file,
#             tokenizer,
#             max_length=args.max_length,
#             val_split=args.val_split,
#             num_proc=args.dataloader_num_workers
#         )
#         train_dataset = data["train"]
#         val_dataset = data["validation"]
#
#     if not is_distributed or local_rank == 0:
#         print(f"\nTrain dataset size: {len(train_dataset)}")
#         if val_dataset:
#             print(f"Validation dataset size: {len(val_dataset)}")
#
#     # ✅ 5. 加载基础模型（不使用 device_map，让 Trainer 处理分布式）
#     if not is_distributed or local_rank == 0:
#         print(f"\nLoading base model from {args.model_name_or_path}...")
#
#     # ⚠️ 关键：不使用 device_map="auto"，让 DDP 自己管理设备
#     llama_model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
#         trust_remote_code=True
#     )
#
#     # ✅ 6. 创建 MoE 模型
#     if not is_distributed or local_rank == 0:
#         print("\nCreating CultureMoE model...")
#
#     moe_args = ModelArgs(
#         num_experts=args.num_experts,
#         shared_hidden_dim=args.shared_hidden_dim,
#         router_hidden_dim=args.router_hidden_dim,
#         experts_hidden_dim=args.experts_hidden_dim,
#         lora_rank=args.lora_rank,
#         num_classes=args.num_classes,
#         classification_hidden_dim=args.classification_hidden_dim,
#         dropout=args.dropout,
#         num_heads=args.num_heads
#     )
#
#     model = LlamaSharedRouterExpertsModel(
#         llama_model=llama_model,
#         config=llama_model.config,
#         args=moe_args
#     )
#
#     # ✅ 冻结 LLaMA 基础模型参数
#     if args.freeze_llama:
#         for param in model.llama_model.parameters():
#             param.requires_grad = False
#         if not is_distributed or local_rank == 0:
#             print("Frozen LLaMA base model parameters")
#
#     if not is_distributed or local_rank == 0:
#         if args.gradient_checkpointing:
#             print("Gradient checkpointing will be enabled by Trainer")
#
#     # ✅ 启用梯度检查点（节省显存）
#     # if args.gradient_checkpointing:
#     #     if hasattr(model.llama_model, 'gradient_checkpointing_enable'):
#     #         model.llama_model.gradient_checkpointing_enable()
#     #         if not is_distributed or local_rank == 0:
#     #             print("Enabled gradient checkpointing")
#
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     if not is_distributed or local_rank == 0:
#         print(f"\nModel created successfully!")
#         print(f"Total parameters: {total_params:,}")
#         print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
#
#     # ✅ 7. 创建 TrainingArguments（支持分布式）
#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         num_train_epochs=args.num_train_epochs,
#         per_device_train_batch_size=args.per_device_train_batch_size,
#         per_device_eval_batch_size=args.per_device_eval_batch_size,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         warmup_ratio=args.warmup_ratio,
#
#         # 日志和保存
#         logging_dir=os.path.join(args.output_dir, "logs"),
#         logging_steps=args.logging_steps,
#         save_steps=args.save_steps,
#         eval_steps=args.eval_steps,
#         eval_strategy=args.evaluation_strategy if val_dataset else "no",
#         save_strategy="steps",
#         save_total_limit=args.save_total_limit,
#
#         # 最佳模型
#         load_best_model_at_end=True if val_dataset else False,
#         metric_for_best_model="f1_macro" if val_dataset else None,
#         greater_is_better=True,
#
#         # ✅ GPU 优化和分布式训练
#         fp16=args.fp16,
#         bf16=args.bf16,
#         gradient_accumulation_steps=args.gradient_accumulation_steps,
#         gradient_checkpointing=args.gradient_checkpointing,
#         dataloader_num_workers=args.dataloader_num_workers,
#         dataloader_pin_memory=args.dataloader_pin_memory,
#
#         # ✅ 分布式训练关键参数
#         local_rank=local_rank,
#         ddp_find_unused_parameters=False,  # 设为 False 提高性能
#         ddp_backend="nccl",  # 使用 NCCL 后端
#
#         # 其他
#         remove_unused_columns=False,
#         report_to=["tensorboard"] if (not is_distributed or local_rank == 0) else [],
#         seed=args.seed,
#
#         # ✅ 显存优化
#         max_grad_norm=1.0,  # 梯度裁剪
#         optim="adamw_torch",  # 使用 PyTorch 原生 AdamW
#     )
#
#     # 8. 创建 Data Collator
#     data_collator = DataCollatorWithPadding(
#         tokenizer=tokenizer,
#         padding=True,
#         max_length=args.max_length
#     )
#
#     # 9. 创建 Trainer
#     if not is_distributed or local_rank == 0:
#         print("\nCreating trainer...")
#
#     trainer = ClassificationTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_classification_metrics if (not is_distributed or local_rank == 0) else None,
#     )
#
#     # 10. 开始训练
#     if not is_distributed or local_rank == 0:
#         print("\n" + "=" * 60)
#         print("Starting training...")
#         print("=" * 60)
#
#         # 打印 GPU 信息
#         if torch.cuda.is_available():
#             print(f"\nGPU Information:")
#             print(f"  Number of GPUs: {torch.cuda.device_count()}")
#             for i in range(torch.cuda.device_count()):
#                 print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
#                 print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
#
#             if is_distributed:
#                 print(f"\n  Using Distributed Data Parallel (DDP)")
#                 print(f"  World size: {torch.distributed.get_world_size()}")
#
#     train_result = trainer.train()
#
#     # 11. 保存模型（只在主进程）
#     if not is_distributed or local_rank == 0:
#         print(f"\nSaving model to {args.output_dir}...")
#         trainer.save_model()
#         trainer.save_state()
#
#         # 保存训练指标
#         metrics = train_result.metrics
#         trainer.log_metrics("train", metrics)
#         trainer.save_metrics("train", metrics)
#
#         # 12. 最终评估
#         if val_dataset:
#             print("\nRunning final evaluation...")
#             eval_metrics = trainer.evaluate()
#             trainer.log_metrics("eval", eval_metrics)
#             trainer.save_metrics("eval", eval_metrics)
#
#         print("\n" + "=" * 60)
#         print("Training completed successfully! ✅")
#         print("=" * 60)
#
#     return trainer