# examples/train_classification.py
# !/usr/bin/env python3
"""
三分类任务训练脚本
使用 CultureMoE 模型进行 yes/neutral/no 分类
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import HfArgumentParser
from src.llamafactory.train.classification.workflow import (
    ClassificationTrainingArguments,
    run_classification_training
)


def main():
    """主函数"""
    # 解析命令行参数
    parser = HfArgumentParser(ClassificationTrainingArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 从 JSON 文件加载参数
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 从命令行参数加载
        args, = parser.parse_args_into_dataclasses()

    # 运行训练
    run_classification_training(args)


if __name__ == "__main__":
    main()