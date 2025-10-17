# src/llamafactory/data/classification_processor.py
from typing import Dict, List, Any, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer


class ClassificationDataProcessor:
    """
    三分类任务的数据处理器
    要求：数据集的 output 字段必须是整数类型 0/1/2
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        预处理函数，用于 dataset.map()

        Args:
            examples: 包含 instruction, input, output 字段的批次数据
                     - instruction: str, 问题描述
                     - input: str, 额外输入（可选）
                     - output: int, 标签 (0=no, 1=neutral, 2=yes)

        Returns:
            包含 input_ids, attention_mask, labels 的字典
        """
        # 1. 构建输入文本
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples.get("input", [""] * len(examples["instruction"]))[i]

            # 拼接 instruction 和 input
            if input_text and input_text.strip():
                text = f"{instruction}\n{input_text}"
            else:
                text = instruction

            texts.append(text)

        # 2. Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # 使用 DataCollator 动态 padding
            return_tensors=None
        )

        # 3. 使用整数标签（output 字段必须是 int 类型）
        tokenized["labels"] = examples["output"]

        return tokenized

    def process_dataset(self, dataset: Dataset, num_proc: int = 4) -> Dataset:
        """
        处理整个数据集

        Args:
            dataset: 原始数据集（output 字段必须是整数）
            num_proc: 并行处理的进程数

        Returns:
            处理后的数据集
        """
        # 验证数据格式
        first_sample = dataset[0]
        if "output" not in first_sample:
            raise ValueError("Dataset must contain 'output' field")

        if not isinstance(first_sample["output"], int):
            raise ValueError(
                f"'output' field must be int type, got {type(first_sample['output'])}. "
                "Please preprocess your data to convert labels to integers (0=no, 1=neutral, 2=yes)"
            )

        # 获取原始列名
        original_columns = dataset.column_names

        # 应用预处理
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
            desc="Tokenizing dataset"
        )

        return processed_dataset


def load_and_process_classification_data(
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        val_split: float = 0.1,
        num_proc: int = 4
) -> Dict[str, Dataset]:
    """
    加载并处理分类数据集

    要求：数据集必须包含以下字段
    - instruction: str, 问题描述
    - input: str, 额外输入（可选，可以为空字符串）
    - output: int, 标签 (0=no, 1=neutral, 2=yes)

    Args:
        data_path: 数据文件路径 (json/jsonl)
        tokenizer: tokenizer
        max_length: 最大序列长度
        val_split: 验证集比例
        num_proc: 并行处理进程数

    Returns:
        包含 train 和 validation 的数据集字典
    """
    from datasets import load_dataset

    # 1. 加载数据
    print(f"Loading dataset from {data_path}...")
    if data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    elif data_path.endswith('.json'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    print(f"Loaded {len(dataset)} samples")

    # 2. 验证数据格式
    first_sample = dataset[0]
    required_fields = ["instruction", "output"]
    for field in required_fields:
        if field not in first_sample:
            raise ValueError(f"Dataset must contain '{field}' field")

    if not isinstance(first_sample["output"], int):
        raise ValueError(
            f"'output' field must be int type (0/1/2), got {type(first_sample['output'])}. "
            "Please preprocess your data first."
        )

    # 打印数据统计
    from collections import Counter
    label_counts = Counter([sample["output"] for sample in dataset])
    print(f"\nLabel distribution:")
    label_names = {0: "no", 1: "neutral", 2: "yes"}
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = count / len(dataset) * 100
        label_name = label_names.get(label, f"unknown({label})")
        print(f"  {label_name}: {count} ({percentage:.1f}%)")

    # 3. 分割训练集和验证集
    if val_split > 0:
        print(f"\nSplitting dataset with val_split={val_split}...")
        split_dataset = dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    else:
        train_dataset = dataset
        val_dataset = None

    # 4. 处理数据（tokenization）
    print("\nTokenizing dataset...")
    processor = ClassificationDataProcessor(tokenizer, max_length)

    train_dataset = processor.process_dataset(train_dataset, num_proc)
    if val_dataset is not None:
        val_dataset = processor.process_dataset(val_dataset, num_proc)

    print("Data processing completed\n")

    return {
        "train": train_dataset,
        "validation": val_dataset
    }