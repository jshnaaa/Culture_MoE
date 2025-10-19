# src/llamafactory/data/dual_classification_processor.py
from typing import Dict, List, Any, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer


class DualClassificationDataProcessor:
    """
    双路输入的三分类任务数据处理器
    处理包含 instruction, instruction_mask, input, output 的数据
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        label_map: Optional[Dict[str, int]] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 标签映射
        self.label_map = label_map or {
            "no": 0,
            "neutral": 1,
            "yes": 2
        }

        # 反向映射
        self.id2label = {v: k for k, v in self.label_map.items()}

    def preprocess_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        预处理函数，用于 dataset.map()

        Args:
            examples: 包含 instruction, instruction_mask, input, output 字段的批次数据

        Returns:
            包含 input_ids, attention_mask, input_ids_mask, attention_mask_mask, labels 的字典
        """
        batch_size = len(examples["instruction"])

        # 1. 构建第一组输入文本（instruction + input）
        texts_full = []
        for i in range(batch_size):
            instruction = examples["instruction"][i]
            input_text = examples.get("input", [""] * batch_size)[i]

            # 拼接 instruction 和 input
            if input_text and input_text.strip():
                text = f"{instruction}\n{input_text}"
            else:
                text = instruction

            texts_full.append(text)

        # 2. 构建第二组输入文本（instruction_mask + input）
        texts_mask = []
        for i in range(batch_size):
            instruction_mask = examples["instruction_mask"][i]
            input_text = examples.get("input", [""] * batch_size)[i]

            # 拼接 instruction_mask 和 input
            if input_text and input_text.strip():
                text = f"{instruction_mask}\n{input_text}"
            else:
                text = instruction_mask

            texts_mask.append(text)

        # 3. Tokenize 第一组（instruction + input）
        tokenized_full = self.tokenizer(
            texts_full,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # 使用 DataCollator 动态 padding
            return_tensors=None
        )

        # 4. Tokenize 第二组（instruction_mask + input）
        tokenized_mask = self.tokenizer(
            texts_mask,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        # 5. 转换标签
        labels = []
        for output in examples["output"]:
            if isinstance(output, str):
                # 字符串标签转整数
                label = self.label_map.get(output.lower().strip(), 1)  # 默认 neutral
            elif isinstance(output, int):
                # 已经是整数
                label = output
            else:
                label = 1  # 默认 neutral

            labels.append(label)

        # 6. 返回结果（重命名字段以区分两组输入）
        result = {
            # 第一组：instruction + input (用于 h_all)
            "input_ids": tokenized_full["input_ids"],
            "attention_mask": tokenized_full["attention_mask"],

            # 第二组：instruction_mask + input (用于 h_no)
            "input_ids_mask": tokenized_mask["input_ids"],
            "attention_mask_mask": tokenized_mask["attention_mask"],

            # 标签
            "labels": labels
        }

        return result

    def process_dataset(self, dataset: Dataset, num_proc: int = 4) -> Dataset:
        """
        处理整个数据集

        Args:
            dataset: 原始数据集
            num_proc: 并行处理的进程数

        Returns:
            处理后的数据集
        """
        # 获取原始列名
        original_columns = dataset.column_names

        # 应用预处理
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
            desc="Tokenizing dual inputs and converting labels"
        )

        return processed_dataset


def load_and_process_dual_classification_data(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    val_split: float = 0.1,
    num_proc: int = 4
) -> Dict[str, Dataset]:
    """
    加载并处理双路输入的分类数据集

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

    # 打印数据统计
    if "output" in dataset.column_names:
        label_counts = {}
        for item in dataset:
            label = item["output"]
            if isinstance(label, int):
                label_name = {0: "no", 1: "neutral", 2: "yes"}.get(label, str(label))
            else:
                label_name = label
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} ({count/len(dataset)*100:.1f}%)")

    # 2. 分割训练集和验证集
    if val_split > 0:
        print(f"\nSplitting dataset with val_split={val_split}...")
        split_dataset = dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    else:
        train_dataset = dataset
        val_dataset = None

    # 3. 处理数据
    print("\nTokenizing dataset...")
    processor = DualClassificationDataProcessor(tokenizer, max_length)

    train_dataset = processor.process_dataset(train_dataset, num_proc)
    if val_dataset is not None:
        val_dataset = processor.process_dataset(val_dataset, num_proc)

    print("Data processing completed\n")

    return {
        "train": train_dataset,
        "validation": val_dataset
    }

